"""Post-hoc sanity-check + override SAM3D's canonical_scale.

SAM3D's monocular scale estimate is unreliable on close-up clips (it lacks
our pipeline's true camera intrinsics, so it back-computes scale from a
generic FoV prior).  Result: pose_track_info[oid].scale_correction is
sometimes 3-5× too large, producing oversized meshes in the 3D viewer.

This tool computes an *independent* size estimate per oid from physical
first principles:

    real_size_max ≈ max(mask_W, mask_H) × median_MoGe_depth / dp_focal

If SAM3D's scale × canonical_max_axis exceeds N × real_size_max (default
N = 1.8), the scale is overridden to make the mesh match the physical
estimate.  Otherwise SAM3D's scale is left alone (it's already reasonable).

The override also corrects per-frame translation when the mesh centroid
is non-zero (otherwise rescaling around a non-origin centroid would shift
the mesh world position).

Usage
=====
  python -m egoinfinity.pipeline.post_tracking.scale_sanity --only=-MF7nEIDKLk_242.2_247.3 --dry-run
  python -m egoinfinity.pipeline.post_tracking.scale_sanity                         # all 105
  python -m egoinfinity.pipeline.post_tracking.scale_sanity --threshold 1.5         # tighter
"""
import argparse
import gzip
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import trimesh


def _read_ply_xyz(path: str) -> np.ndarray:
    """Read just the (x,y,z) vertex positions from a binary little-endian PLY.

    Handles SAM3D's Gaussian-Splat PLYs (17 props per vertex: xyz + nxyz +
    f_dc_3 + opacity + scale_3 + rot_4), which trimesh refuses to parse
    ("PLY is unexpected length") because it expects standard mesh PLYs
    with vertex/face elements only.
    """
    with open(path, "rb") as f:
        # Parse header
        n_verts = 0
        props: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"unexpected EOF in PLY header: {path}")
            s = line.decode("ascii", errors="replace").strip()
            if s.startswith("element vertex"):
                n_verts = int(s.split()[-1])
            elif s.startswith("property"):
                # property <type> <name>; collect float props in order
                parts = s.split()
                if len(parts) >= 3 and parts[1] in ("float", "float32"):
                    props.append(parts[-1])
            elif s == "end_header":
                break
        if n_verts == 0 or not props:
            raise ValueError(f"PLY missing vertex/property declarations: {path}")
        try:
            ix, iy, iz = props.index("x"), props.index("y"), props.index("z")
        except ValueError:
            raise ValueError(f"PLY does not declare x/y/z properties: {path}")
        n_floats = len(props)
        raw = np.frombuffer(f.read(n_verts * n_floats * 4), dtype=np.float32)
        if raw.size != n_verts * n_floats:
            raise ValueError(
                f"PLY body size mismatch: expected {n_verts}×{n_floats} floats, "
                f"got {raw.size} ({path})")
        rows = raw.reshape(n_verts, n_floats)
        return rows[:, [ix, iy, iz]].astype(np.float64)

REPO = Path(os.environ.get("EGOINFINITY_REPO") or Path(__file__).resolve().parents[3])
sys.path.insert(0, str(REPO))

FAV = Path(os.environ.get(
    "ACTION100M_CACHE", str(REPO / "cache"))) / "favorites"


def _compute_expected_size(fdata, oid, dp_focal, init_frame=None, n_sample_frames=5):
    """Return (expected_real_max_size_m, n_valid_samples).

    Samples are uniformly spaced + the SAM3D init_frame (so we always
    cross-reference against the frame SAM3D itself used for its scale
    estimate). Init_frame is added even if already in the uniform set
    (de-duplicated via set).
    """
    # Lazy import (only here because decode_depth_png is in scripts/)
    from scripts.pipeline_utils import decode_depth_png

    T = len(fdata)
    samples = []
    # Sample uniformly across the clip; always include SAM3D init_frame
    sample_ts = set(np.linspace(0, T - 1, min(n_sample_frames, T)).astype(int).tolist())
    if init_frame is not None and 0 <= int(init_frame) < T:
        sample_ts.add(int(init_frame))
    for t in sorted(sample_ts):
        fd = fdata[t]
        sd = fd.get("sam3_obj_data") or {}
        od = sd.get(oid) or sd.get(int(oid))
        if not isinstance(od, dict):
            continue
        mp = od.get("mask_packed")
        ms = od.get("mask_shape")
        if mp is None or ms is None:
            continue
        H, W = int(ms[0]), int(ms[1])
        m = np.unpackbits(np.asarray(mp, dtype=np.uint8))[: H * W].astype(bool).reshape(H, W)
        if not m.any():
            continue
        ys, xs = np.where(m)
        bw = float(xs.max() - xs.min())
        bh = float(ys.max() - ys.min())
        depth = decode_depth_png(fd.get("depth_png"))
        if depth is None:
            continue
        # Use median depth of mask region (robust to outliers)
        zs = depth[ys, xs]
        valid_d = zs[zs > 0.1]
        if len(valid_d) < 30:
            continue
        z_med = float(np.median(valid_d))
        # Real-world size at this depth
        real_w = bw * z_med / dp_focal
        real_h = bh * z_med / dp_focal
        samples.append(max(real_w, real_h))
    if not samples:
        return None, 0
    # Use 75th percentile of max-dim samples to bias toward the LARGER
    # appearances of the object (object size doesn't shrink, but bbox can
    # shrink under occlusion).
    return float(np.percentile(samples, 75)), len(samples)


def _process_clip(fav_dir, threshold, max_real_size_m, dry_run):
    pkl_path = fav_dir / "pipeline_result.pkl.gz"
    if not pkl_path.is_file():
        return "skip", "no pkl", []

    with gzip.open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fdata = data.get("frame_data") or []
    pti = data.get("pose_track_info") or {}
    if not fdata or not pti:
        return "skip", "no frame_data or pose_track_info", []
    dp_focal = float(data.get("dp_focal", 0))
    if dp_focal <= 0:
        return "fail", "no dp_focal", []

    mesh_info = data.get("sam3_mesh_info") or {}
    rows = []
    pti_changed = False
    for oid in sorted(pti.keys()):
        if not isinstance(oid, int):
            continue
        info = pti[oid]
        if not isinstance(info, dict):
            continue
        sc_old = info.get("scale_correction")
        if sc_old is None or sc_old <= 0:
            continue
        ply = fav_dir / "sam3_meshes" / f"obj_{oid}.ply"
        if not ply.exists():
            continue
        # Try trimesh first (standard mesh PLYs), fall back to xyz-only parser
        # for A100-generated Gaussian-Splat PLYs that trimesh chokes on.
        try:
            v = np.asarray(trimesh.load(str(ply), process=False).vertices)
        except (ValueError, Exception):
            v = _read_ply_xyz(str(ply))
        canonical_bb = v.max(axis=0) - v.min(axis=0)
        canon_max = float(canonical_bb.max())
        if canon_max < 1e-3:
            continue
        current_real_max = sc_old * canon_max

        init_f = mesh_info.get(oid, {}).get("init_frame") if isinstance(mesh_info.get(oid), dict) else None
        expected_max, n_samples = _compute_expected_size(fdata, oid, dp_focal, init_frame=init_f)
        if expected_max is None or n_samples < 2:
            rows.append({"oid": oid, "action": "skip", "reason": "no_pc_samples",
                         "sc_old": sc_old, "sc_new": sc_old})
            continue

        ratio = current_real_max / max(expected_max, 1e-6)
        if ratio <= threshold:
            rows.append({"oid": oid, "action": "keep", "sc_old": sc_old, "sc_new": sc_old,
                         "current_real_cm": current_real_max * 100,
                         "expected_real_cm": expected_max * 100, "ratio": ratio})
            continue

        # OVERRIDE
        sc_new = expected_max / canon_max
        # Safety cap: prevent absurdly small scale
        sc_new = max(sc_new, 0.01)

        if not dry_run:
            # Compute mesh centroid in canonical mesh frame
            mesh_centroid = v.mean(axis=0).astype(np.float64)
            # Update T_seq translation only if centroid isn't trivially at origin
            T_seq = info.get("T_seq")
            if T_seq is not None and np.linalg.norm(mesh_centroid) > 1e-4:
                T_seq = np.asarray(T_seq, dtype=np.float32).copy()
                R_anchor = T_seq[0, :3, :3].astype(np.float64)
                # Correction: t_new = c_obs - R @ (sc_new × c_canon)
                #             t_old = c_obs - R @ (sc_old × c_canon)
                # → t_new = t_old + R @ (sc_old - sc_new) × c_canon
                delta_t = R_anchor @ ((sc_old - sc_new) * mesh_centroid)
                T_seq[:, :3, 3] += delta_t.astype(np.float32)
                info["T_seq"] = T_seq
            info["scale_correction"] = float(sc_new)
            info["scale_correction_orig_sam3d"] = float(sc_old)
            pti_changed = True

        rows.append({
            "oid": oid, "action": "override",
            "sc_old": sc_old, "sc_new": sc_new,
            "current_real_cm": current_real_max * 100,
            "expected_real_cm": expected_max * 100,
            "ratio": ratio,
        })

    if pti_changed and not dry_run:
        # Provenance
        prov = data.get("scale_sanity_refresh") or {}
        history = prov.get("history") or []
        history.append({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "threshold": threshold,
            "n_overridden": sum(1 for r in rows if r["action"] == "override"),
            "details": [
                {"oid": r["oid"], "sc_old": r["sc_old"], "sc_new": r["sc_new"],
                 "ratio_before": r.get("ratio")}
                for r in rows if r["action"] == "override"
            ],
        })
        data["scale_sanity_refresh"] = {"history": history}
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    n_override = sum(1 for r in rows if r["action"] == "override")
    n_keep = sum(1 for r in rows if r["action"] == "keep")
    if n_override:
        return "ok", f"override {n_override}/{n_override+n_keep}", rows
    if rows:
        return "skip", f"all {len(rows)} within threshold", rows
    return "skip", "no oids", rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma-separated clip ids")
    ap.add_argument("--threshold", type=float, default=1.8,
                    help="override when sc×canonical_max > threshold × expected_max (default 1.8)")
    ap.add_argument("--max-real-size", type=float, default=2.0,
                    help="(reserved; not currently used as a hard cap)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    allow = set(s.strip() for s in args.only.split(",") if s.strip()) if args.only else None
    favs = sorted(p for p in FAV.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if allow:
        favs = [p for p in favs if p.name in allow]
    if not favs:
        print(f"no clips matched under {FAV}")
        return

    print(f"refresh_scale_sanity: {len(favs)} clips, dry_run={args.dry_run}, threshold={args.threshold}")
    n_ok = n_skip = n_fail = 0
    total_override = 0
    total_oids = 0
    t0 = time.time()
    for i, fav in enumerate(favs, 1):
        t_clip = time.time()
        status, summary, rows = _process_clip(
            fav, args.threshold, args.max_real_size, args.dry_run)
        dt = time.time() - t_clip
        n_over = sum(1 for r in rows if r["action"] == "override")
        total_override += n_over
        total_oids += len(rows)
        if status == "ok":
            n_ok += 1
            print(f"[{i:>3}/{len(favs)}] {'(dry) ' if args.dry_run else ''}+ {fav.name:<50}  {summary}  ({dt:.1f}s)")
            for r in rows:
                if r["action"] == "override":
                    print(f"      oid {r['oid']}: scale {r['sc_old']:.3f}→{r['sc_new']:.3f}  "
                          f"size {r['current_real_cm']:.0f}cm→{r['expected_real_cm']:.0f}cm  "
                          f"(ratio was {r['ratio']:.1f}×)")
        elif status == "skip":
            n_skip += 1
            # Print first 3 chars only to keep log clean — counts via summary
            if len(favs) <= 15:    # verbose when small batch
                print(f"[{i:>3}/{len(favs)}] - {fav.name}  SKIP ({summary})")
        else:
            n_fail += 1
            print(f"[{i:>3}/{len(favs)}] X {fav.name}  FAIL ({summary})")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min — "
          f"ok: {n_ok}  skip: {n_skip}  fail: {n_fail}  "
          f"overrides: {total_override}/{total_oids} oids")


if __name__ == "__main__":
    main()
