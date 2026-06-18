#!/usr/bin/env python3
"""Generate statistics report for visual filter results."""
import argparse
import os
import sqlite3

from .config import FilterConfig


def main():
    parser = argparse.ArgumentParser(description="Visual filter report")
    parser.add_argument("--db", type=str, default="")
    parser.add_argument("--results_db", type=str, default="")
    parser.add_argument("--sample", type=int, default=30, help="Number of accept samples to show")
    args = parser.parse_args()

    cfg = FilterConfig()
    if args.db:
        cfg.db_path = args.db

    results_db_path = args.results_db or os.path.join(
        os.path.dirname(cfg.db_path), "visual_filter_results.db")

    conn = sqlite3.connect(cfg.db_path)
    conn.row_factory = sqlite3.Row
    # Attach results DB as 'res'
    conn.execute(f"ATTACH DATABASE ? AS res", (results_db_path,))

    # Basic counts
    total_exo = conn.execute("SELECT COUNT(*) FROM segments WHERE exo_hoi=1").fetchone()[0]
    processed = conn.execute("SELECT COUNT(*) FROM res.visual_results").fetchone()[0]
    accepted = conn.execute("SELECT COUNT(*) FROM res.visual_results WHERE visual_pass=1").fetchone()[0]
    rejected = conn.execute("SELECT COUNT(*) FROM res.visual_results WHERE visual_pass=0").fetchone()[0]

    print(f"{'='*60}")
    print(f"Action100M Visual Filter Report")
    print(f"{'='*60}")
    print(f"Results DB: {results_db_path}")
    print(f"Total exo_hoi=1 segments:   {total_exo:>10,}")
    print(f"Processed (visual checked): {processed:>10,}")
    print(f"  Accepted (visual_pass=1): {accepted:>10,}  ({accepted/max(processed,1)*100:.1f}%)")
    print(f"  Rejected (visual_pass=0): {rejected:>10,}  ({rejected/max(processed,1)*100:.1f}%)")
    print(f"  Not yet checked:          {total_exo - processed:>10,}")

    # Reject reasons
    print(f"\n--- Reject Reasons ---")
    rows = conn.execute(
        "SELECT reject_reason, COUNT(*) as cnt FROM res.visual_results "
        "WHERE visual_pass=0 AND reject_reason IS NOT NULL "
        "GROUP BY reject_reason ORDER BY cnt DESC"
    ).fetchall()
    for r in rows:
        pct = r["cnt"] / max(rejected, 1) * 100
        print(f"  {r['reject_reason']:25s} {r['cnt']:>8,}  ({pct:.1f}%)")

    # Metric distributions for accepted
    print(f"\n--- Accepted Segment Metrics (mean ± std) ---")
    metrics = conn.execute(
        "SELECT hand_ratio, bg_flow, avg_hand_size, trunc_ratio FROM res.visual_results "
        "WHERE visual_pass=1 AND hand_ratio IS NOT NULL"
    ).fetchall()
    if metrics:
        import numpy as np
        hr = np.array([m["hand_ratio"] for m in metrics])
        bf = np.array([m["bg_flow"] for m in metrics])
        hs = np.array([m["avg_hand_size"] for m in metrics])
        tr = np.array([m["trunc_ratio"] for m in metrics])
        print(f"  hand_ratio:    {hr.mean():.3f} ± {hr.std():.3f}  (min={hr.min():.3f}, max={hr.max():.3f})")
        print(f"  bg_flow:       {bf.mean():.3f} ± {bf.std():.3f}  (min={bf.min():.3f}, max={bf.max():.3f})")
        print(f"  avg_hand_size: {hs.mean():.4f} ± {hs.std():.4f}  (min={hs.min():.4f}, max={hs.max():.4f})")
        print(f"  trunc_ratio:   {tr.mean():.3f} ± {tr.std():.3f}  (min={tr.min():.3f}, max={tr.max():.3f})")

    # Unique video count
    n_videos_accepted = conn.execute(
        "SELECT COUNT(DISTINCT s.video_uid) FROM segments s "
        "JOIN res.visual_results r ON s.id = r.segment_id WHERE r.visual_pass=1"
    ).fetchone()[0]
    print(f"\n--- Unique Videos ---")
    print(f"  Videos with accepted segments: {n_videos_accepted}")

    # Sample accepted segments with YouTube links
    print(f"\n--- Sample Accepted Segments (click to verify) ---")
    samples = conn.execute(
        "SELECT s.video_uid, s.start_sec, s.end_sec, s.action_brief, s.actor, "
        "r.hand_ratio, r.bg_flow, r.avg_hand_size, r.trunc_ratio "
        "FROM segments s JOIN res.visual_results r ON s.id = r.segment_id "
        "WHERE r.visual_pass=1 ORDER BY RANDOM() LIMIT ?",
        (args.sample,),
    ).fetchall()
    for i, s in enumerate(samples, 1):
        t = int(s["start_sec"])
        url = f"https://www.youtube.com/watch?v={s['video_uid']}&t={t}"
        print(f"\n  [{i:2d}] {url}")
        print(f"       {s['start_sec']:.1f}s - {s['end_sec']:.1f}s | {s['actor']} → {s['action_brief']}")
        print(f"       hand={s['hand_ratio']:.2f}  flow={s['bg_flow']:.2f}  size={s['avg_hand_size']:.4f}  trunc={s['trunc_ratio']:.2f}")

    conn.close()


if __name__ == "__main__":
    main()
