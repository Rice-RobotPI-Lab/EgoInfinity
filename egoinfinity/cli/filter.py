"""`egoinfinity filter` — batch / standalone filter mode.

Two use cases:

  1. Pipeline-stage mode:  handled by ``stages/filter.py``, invoked as
     part of the full pipeline (a Stage). One clip at a time. Not this
     CLI's concern.

  2. Curation mode (this CLI):  filter MANY videos to identify HOI
     candidates. Wraps the existing batch script
     ``python -m action100m_filter.main`` with sensible defaults and
     optional viz-server attachment.

Examples::

    # Filter from an Action100M sqlite index (the historical workflow)
    egoinfinity filter --from-db data/action100m_index.db --test_n_videos 500 --viz :8899

    # Filter an explicit list of YouTube URLs / video paths (planned; not yet wired)
    egoinfinity filter --input-list ./candidate_videos.txt --viz :8899
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="egoinfinity filter")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-db", type=Path,
                     help="Path to Action100M sqlite index db.")
    src.add_argument("--input-list", type=Path,
                     help="Newline-separated list of video URIs (path or URL).")
    ap.add_argument("--test_n_videos", type=int, default=500,
                    help="Process at most N videos (from-db mode).")
    ap.add_argument("--n_workers", type=int, default=1)
    ap.add_argument("--detector_path", type=Path, default=None,
                    help="YOLO hand detector .pt (default: pretrained_models/detector.pt).")
    ap.add_argument("--viz", default=None, metavar="[HOST]:PORT",
                    help="Launch viz server (default port :8899). E.g. --viz :8899")
    ap.add_argument("--viz-dir", type=Path, default=None,
                    help="Where to write viz HTML + sample images (default: ./filter_viz/).")
    ap.add_argument("--", dest="dashdash", nargs=argparse.REMAINDER, default=[],
                    help="Pass-through args forwarded to action100m_filter.main.")
    args = ap.parse_args(argv)

    if args.input_list is not None:
        print("ERROR: --input-list is not yet wired. Use --from-db with an "
              "Action100M sqlite index for now; see docs/THIRD_PARTY.md.",
              file=sys.stderr)
        return 2

    # Curation mode via existing action100m_filter.main
    cmd = [sys.executable, "-m", "action100m_filter.main",
           "--test_n_videos", str(args.test_n_videos),
           "--n_workers", str(args.n_workers)]
    if args.from_db is not None:
        os.environ.setdefault("ACTION100M_DB", str(args.from_db))
    if args.detector_path is not None:
        cmd += ["--detector_path", str(args.detector_path)]
    if args.viz is not None:
        cmd += ["--viz", "--viz_port", args.viz.lstrip(":").split(":")[-1]]
    if args.dashdash:
        cmd += args.dashdash
    print(" ".join(cmd))
    return subprocess.run(cmd).returncode
