#!/usr/bin/env python3
"""
Action100M visual filter: hand detection + static background.

Usage:
    python -m action100m_filter.main --test_n_videos 500
    python -m action100m_filter.main --test_n_videos 10 --viz   # web preview on :8899
"""
import argparse
import logging
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from tqdm import tqdm

from .config import FilterConfig
from .detect import analyze_frames
from .stream import download_video, sample_frames_local, sample_motion_pairs, detect_shot_cuts
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── DB helpers ───────────────────────────────────────────────────

def _init_results_db(path: str) -> sqlite3.Connection:
    """Create/open the standalone results DB."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS visual_results (
            segment_id INTEGER PRIMARY KEY,
            visual_pass INTEGER NOT NULL,
            hand_ratio REAL,
            bg_flow REAL,
            avg_hand_size REAL,
            trunc_ratio REAL,
            reject_reason TEXT,
            n_cuts INTEGER
        )
    """)
    conn.commit()
    return conn


def _load_video_groups(
    src_conn: sqlite3.Connection,
    res_conn: sqlite3.Connection,
    limit: int | None = None,
) -> dict[str, list[dict]]:
    """Load exo_hoi=1 segments grouped by video_uid, skipping already processed."""
    # Get IDs already in results DB
    done_ids = {r[0] for r in res_conn.execute("SELECT segment_id FROM visual_results").fetchall()}

    sql = """
        SELECT DISTINCT video_uid
        FROM segments
    """
    if limit:
        sql += f" LIMIT {limit}"

    video_uids = [r[0] for r in src_conn.execute(sql).fetchall()]

    groups: dict[str, list[dict]] = {}
    for uid in video_uids:
        rows = src_conn.execute(
            "SELECT id, start_sec, end_sec, action_brief, actor "
            "FROM segments WHERE video_uid = ?",
            (uid,),
        ).fetchall()
        segs = [
            {"id": r[0], "start": r[1], "end": r[2], "action": r[3], "actor": r[4]}
            for r in rows if r[0] not in done_ids
        ]
        if segs:
            groups[uid] = segs
    return groups


# ── Visualization ────────────────────────────────────────────────

def _save_viz(frames: list[np.ndarray], metrics: dict, seg: dict,
              video_uid: str, accept: bool, reason: str | None,
              viz_dir: str, n_cuts: int = 0,
              used_range: tuple[float, float] | None = None) -> dict | None:
    """Draw annotated frame grid, save to viz_dir, return item dict for web server."""
    per_frame = metrics.get("per_frame", [])
    GREEN = (0, 220, 0)
    YELLOW = (0, 220, 220)
    RED = (0, 0, 220)
    WHITE = (255, 255, 255)
    CYAN = (255, 220, 0)

    annotated = []
    for i, frame in enumerate(frames):
        vis = frame.copy()
        if i < len(per_frame):
            pf = per_frame[i]
            for j, box in enumerate(pf["xyxy"]):
                x1, y1, x2, y2 = box.astype(int)
                conf = pf["confs"][j] if j < len(pf["confs"]) else 0
                color = YELLOW if pf["truncated"] else GREEN
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if len(pf["xyxy"]) == 0:
                cv2.putText(vis, "no hand", (5, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1)
        annotated.append(vis)

    # Grid: 2 rows x 4 cols
    h, w = annotated[0].shape[:2]
    cols = min(4, len(annotated))
    rows = (len(annotated) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(annotated):
        r, c = divmod(idx, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

    # Info bar
    bar_h = 50
    bar = np.zeros((bar_h, grid.shape[1], 3), dtype=np.uint8)
    status = "ACCEPT" if accept else f"REJECT: {reason}"
    cv2.putText(bar, status, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                GREEN if accept else RED, 2)
    info = (f"hand={metrics['hand_ratio']:.0%}  flow={metrics['bg_flow']:.3f}  "
            f"size={metrics['avg_hand_size']:.4f}  trunc={metrics['trunc_ratio']:.0%}"
            f"  cuts={n_cuts}")
    if used_range:
        info += f"  used={used_range[0]:.1f}-{used_range[1]:.1f}s"
    cv2.putText(bar, info, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)
    desc = f"{video_uid} {seg['start']:.1f}-{seg['end']:.1f}s"
    if used_range:
        desc += f" [used {used_range[0]:.1f}-{used_range[1]:.1f}s]"
    desc += f" {seg.get('actor','')} | {seg.get('action','')}"
    cv2.putText(bar, desc[:120], (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (160, 160, 160), 1)

    canvas = np.vstack([grid, bar])

    filename = f"{seg['id']}_{video_uid}.jpg"
    cv2.imwrite(os.path.join(viz_dir, "imgs", filename), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])

    return {
        "id": seg["id"],
        "filename": filename,
        "video_uid": video_uid,
        "status": "accept" if accept else "reject",
        "reason": reason or "",
        "metrics": info,
        "desc": desc[:140],
        "n_cuts": n_cuts,
    }


# ── Processing ───────────────────────────────────────────────────

def _find_longest_subseg(start: float, end: float, cut_times: list[float]) -> tuple[float, float]:
    """Return (start, end) of the longest sub-segment between cut points."""
    boundaries = [start] + cut_times + [end]
    best_i = max(range(len(boundaries) - 1),
                 key=lambda i: boundaries[i + 1] - boundaries[i])
    return boundaries[best_i], boundaries[best_i + 1]


def _judge(metrics: dict, cfg: FilterConfig) -> tuple[bool, str | None]:
    """Apply thresholds. Returns (accept, reject_reason)."""
    if metrics["hand_ratio"] < cfg.min_hand_ratio:
        return False, "low_hand_ratio"
    if metrics["bg_flow"] > cfg.max_bg_flow:
        return False, "bg_moving"
    if metrics["avg_hand_size"] < cfg.min_hand_size:
        return False, "hand_too_small"
    if metrics["avg_hand_size"] > cfg.max_hand_size:
        return False, "hand_too_large"
    if metrics["trunc_ratio"] > cfg.max_trunc_ratio:
        return False, "hand_truncated"
    return True, None


def process_one_video(
    video_uid: str,
    segments: list[dict],
    cfg: FilterConfig,
    viz_dir: str | None = None,
) -> list[dict]:
    """Download video once, then process all segments from local file."""
    video_path = download_video(video_uid, timeout=300)
    if video_path is None:
        log.warning("  ✗ %s: download failed, skipping %d segs", video_uid, len(segments))
        return [
            {"id": s["id"], "visual_pass": 0, "reason": "download_failed",
             "n_cuts": None,
             "hand_ratio": None, "bg_flow": None, "avg_hand_size": None, "trunc_ratio": None}
            for s in segments
        ]

    try:
        results = []
        dims_cache: dict = {}

        for seg in segments:
            try:
                # Step 0: Shot cut detection (fast, no frames piped)
                cut_times = detect_shot_cuts(
                    video_path, seg["start"], seg["end"],
                    threshold=cfg.scene_threshold, timeout=15,
                )
                n_cuts = len(cut_times)

                # Determine actual analysis range
                if n_cuts > 0:
                    seg_start, seg_end = _find_longest_subseg(
                        seg["start"], seg["end"], cut_times,
                    )
                    if seg_end - seg_start < cfg.min_subseg_duration:
                        results.append({
                            "id": seg["id"], "visual_pass": 0,
                            "reason": "cuts_too_short", "n_cuts": n_cuts,
                            "hand_ratio": None, "bg_flow": None,
                            "avg_hand_size": None, "trunc_ratio": None,
                        })
                        continue
                else:
                    seg_start, seg_end = seg["start"], seg["end"]

                frames = sample_frames_local(
                    video_path, seg_start, seg_end,
                    n_frames=cfg.n_frames,
                    max_height=cfg.max_height,
                    timeout=10,
                    _cached_dims=dims_cache,
                )
                if len(frames) < 2:
                    results.append({
                        "id": seg["id"], "visual_pass": 0, "reason": "no_frames",
                        "n_cuts": n_cuts,
                        "hand_ratio": None, "bg_flow": None, "avg_hand_size": None,
                        "trunc_ratio": None,
                    })
                    continue

                # Sample closely-spaced frame pairs for motion detection
                mpairs = sample_motion_pairs(
                    video_path, seg_start, seg_end,
                    max_height=cfg.max_height,
                    timeout=10,
                    _cached_dims=dims_cache,
                )

                metrics = analyze_frames(
                    frames,
                    detector_path=cfg.detector_path,
                    hand_conf=cfg.hand_conf,
                    trunc_edge_px=cfg.trunc_edge_px,
                    motion_pairs=mpairs or None,
                )
                accept, reason = _judge(metrics, cfg)

                # Save viz image + push to web server
                if viz_dir:
                    from . import viz_server
                    used_range = (seg_start, seg_end) if n_cuts > 0 else None
                    item = _save_viz(frames, metrics, seg, video_uid,
                                     accept, reason, viz_dir,
                                     n_cuts=n_cuts, used_range=used_range)
                    if item:
                        viz_server.add_item(item)

                results.append({
                    "id": seg["id"],
                    "visual_pass": 1 if accept else 0,
                    "reason": reason,
                    "n_cuts": n_cuts,
                    "hand_ratio": round(metrics["hand_ratio"], 3),
                    "bg_flow": round(metrics["bg_flow"], 3),
                    "avg_hand_size": round(metrics["avg_hand_size"], 4),
                    "trunc_ratio": round(metrics["trunc_ratio"], 3),
                })
            except Exception as e:
                results.append({
                    "id": seg["id"], "visual_pass": 0, "reason": str(e)[:100],
                    "n_cuts": None,
                    "hand_ratio": None, "bg_flow": None, "avg_hand_size": None,
                    "trunc_ratio": None,
                })
        return results
    finally:
        try:
            os.remove(video_path)
        except OSError:
            pass


def _write_results(conn: sqlite3.Connection, results: list[dict]):
    """Batch insert into standalone results DB."""
    conn.executemany(
        "INSERT OR REPLACE INTO visual_results "
        "(segment_id, visual_pass, hand_ratio, bg_flow, avg_hand_size, trunc_ratio, reject_reason, n_cuts) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (r["id"], r["visual_pass"], r["hand_ratio"], r["bg_flow"],
             r["avg_hand_size"], r["trunc_ratio"], r["reason"], r.get("n_cuts"))
            for r in results
        ],
    )
    conn.commit()


# ── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visual filter for Action100M HOI segments")
    parser.add_argument("--test_n_videos", type=int, default=500)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--db", type=str, default="")
    parser.add_argument("--detector_path", type=str, default="")
    parser.add_argument("--min_hand_ratio", type=float, default=0.75)
    parser.add_argument("--max_bg_flow", type=float, default=3.0)
    parser.add_argument("--min_hand_size", type=float, default=0.005)
    parser.add_argument("--max_hand_size", type=float, default=0.40)
    parser.add_argument("--max_trunc_ratio", type=float, default=0.5)
    parser.add_argument("--scene_threshold", type=float, default=0.4)
    parser.add_argument("--min_subseg_duration", type=float, default=5.0)
    parser.add_argument("--results_db", type=str, default="",
                        help="Path to results DB (default: visual_filter_results.db next to source DB)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset old visual_pass results before running")
    parser.add_argument("--viz", action="store_true",
                        help="Web preview at http://HOST:8899 (auto-updates)")
    parser.add_argument("--viz_port", type=int, default=8899)
    args = parser.parse_args()

    cfg = FilterConfig(
        test_n_videos=args.test_n_videos,
        n_workers=args.n_workers,
        min_hand_ratio=args.min_hand_ratio,
        max_bg_flow=args.max_bg_flow,
        min_hand_size=args.min_hand_size,
        max_hand_size=args.max_hand_size,
        max_trunc_ratio=args.max_trunc_ratio,
        scene_threshold=args.scene_threshold,
        min_subseg_duration=args.min_subseg_duration,
    )
    if args.db:
        cfg.db_path = args.db
    if args.detector_path:
        cfg.detector_path = args.detector_path

    viz_dir = None
    if args.viz:
        from . import viz_server
        viz_dir = os.path.join(os.path.dirname(cfg.db_path), "viz_output")
        viz_server.init(viz_dir)
        viz_server.start_server(port=args.viz_port)
        log.info("Viz server: http://0.0.0.0:%d", args.viz_port)

    # Source DB (read-only)
    src_conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)

    # Results DB (separate file)
    results_db_path = args.results_db or os.path.join(
        os.path.dirname(cfg.db_path), "visual_filter_results.db")
    res_conn = _init_results_db(results_db_path)
    log.info("Results DB: %s", results_db_path)

    if args.reset:
        n = res_conn.execute("SELECT COUNT(*) FROM visual_results").fetchone()[0]
        if n > 0:
            res_conn.execute("DELETE FROM visual_results")
            res_conn.commit()
            log.info("Reset %d previously processed segments", n)

    # Check exo_hoi exists
    cols = {r[1] for r in src_conn.execute("PRAGMA table_info(segments)").fetchall()}
    if "exo_hoi" not in cols:
        log.error("exo_hoi column not found! Run build_exo_filter.py first.")
        return

    # Load candidates
    groups = _load_video_groups(src_conn, res_conn, limit=cfg.test_n_videos)
    total_segments = sum(len(segs) for segs in groups.values())
    log.info(
        "Loaded %d videos, %d segments (test_n_videos=%d)",
        len(groups), total_segments, cfg.test_n_videos,
    )

    if not groups:
        log.info("Nothing to process.")
        conn.close()
        return

    # Stats
    stats = defaultdict(int)
    t0 = time.time()
    processed_videos = 0
    processed_segments = 0

    def _process_and_save(video_uid: str, segments: list[dict]) -> list[dict]:
        return process_one_video(video_uid, segments, cfg, viz_dir=viz_dir)

    if cfg.n_workers > 1:
        with ThreadPoolExecutor(max_workers=cfg.n_workers) as executor:
            futures = {
                executor.submit(_process_and_save, uid, segs): uid
                for uid, segs in groups.items()
            }
            pbar = tqdm(total=len(groups), desc="Videos", unit="vid")
            for future in as_completed(futures):
                uid = futures[future]
                try:
                    results = future.result()
                    _write_results(res_conn, results)
                    vid_accept = 0
                    vid_fail = 0
                    for r in results:
                        if r["visual_pass"] == 1:
                            stats["accept"] += 1
                            vid_accept += 1
                        else:
                            stats[r["reason"] or "unknown"] += 1
                            vid_fail += 1
                    processed_segments += len(results)
                    log.info(
                        "  ✓ %s: %d segs → %d accept, %d reject",
                        uid, len(results), vid_accept, vid_fail,
                    )
                except Exception as e:
                    log.error("Error processing %s: %s", uid, e)
                processed_videos += 1
                pbar.update(1)
                reject_total = processed_segments - stats["accept"]
                pbar.set_postfix(
                    segs=processed_segments,
                    accept=stats["accept"],
                    reject=reject_total,
                    dl_fail=stats.get("download_failed", 0),
                )
            pbar.close()
    else:
        pbar = tqdm(groups.items(), desc="Videos", unit="vid")
        for uid, segs in pbar:
            results = _process_and_save(uid, segs)
            _write_results(res_conn, results)
            vid_accept = 0
            vid_fail = 0
            for r in results:
                if r["visual_pass"] == 1:
                    stats["accept"] += 1
                    vid_accept += 1
                else:
                    stats[r["reason"] or "unknown"] += 1
                    vid_fail += 1
            processed_segments += len(results)
            log.info(
                "  ✓ %s: %d segs → %d accept, %d reject",
                uid, len(results), vid_accept, vid_fail,
            )
            pbar.set_postfix(
                segs=processed_segments,
                accept=stats["accept"],
                dl_fail=stats.get("download_failed", 0),
            )

    src_conn.close()
    res_conn.close()
    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*50}")
    print(f"Visual filter done — {elapsed:.1f}s")
    print(f"{'='*50}")
    print(f"Videos processed: {processed_videos}")
    print(f"Segments processed: {processed_segments}")
    print(f"Accepted (visual_pass=1): {stats['accept']}")
    print(f"\nReject reasons:")
    for reason, count in sorted(stats.items(), key=lambda x: -x[1]):
        if reason == "accept":
            continue
        pct = count / max(processed_segments, 1) * 100
        print(f"  {reason:25s} {count:>6}  ({pct:.1f}%)")

    if viz_dir:
        print(f"\nViz server still running at http://0.0.0.0:{args.viz_port}")
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
