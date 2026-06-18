"""
Per-clip pipeline phase tracking for the multi-host workflow.

A small JSON file ``pipeline_state.json`` lives alongside
``pipeline_result.pkl.gz`` in each clip's cache directory.  It records
which pipeline phases have been completed so the multi-host workflow
(e.g. 4070 Ti runs A..D-sam3, A100 runs D-sam3d, either runs D-track)
can decide what to do next without re-parsing the pkl.

Schema (version 1)::

    {
      "schema_version": 1,
      "phases_done":    ["A", "A-grav", "B", "C", "D-sam3"],
      "phases_pending": ["D-sam3d", "D-track"],
      "host_history":   [
        {"host": "host-a", "phase": "D-sam3",
         "ts": "2026-05-12T10:00:00+00:00"}
      ]
    }

Callers must treat absence (``read_state() is None``) as "unknown —
assume nothing done", so the file is silently backwards compatible
with every existing clip.
"""
from __future__ import annotations

import json
import os
import socket as _socket
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Union

PIPELINE_STATE_FILENAME = "pipeline_state.json"
SCHEMA_VERSION = 1

# Canonical phase ordering (informational; not enforced).  Mirrors
# scripts/exo_pipeline.py ordering — keep in sync if a phase is added.
PHASE_ORDER = [
    "A", "A-grav", "B", "C",
    "D-sam3", "D-sam3d", "D-track",
    "E",
]


PathLike = Union[str, Path]


def state_path(clip_dir: PathLike) -> Path:
    """Return the path where the state file would live (whether or not it exists)."""
    return Path(clip_dir) / PIPELINE_STATE_FILENAME


def read_state(clip_dir: PathLike) -> Optional[dict]:
    """Load the state dict, or return None if absent / unreadable / wrong schema."""
    p = state_path(clip_dir)
    if not p.exists():
        return None
    try:
        s = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(s, dict) or s.get("schema_version") != SCHEMA_VERSION:
        return None
    return s


def phases_done(clip_dir: PathLike) -> List[str]:
    s = read_state(clip_dir)
    return list(s.get("phases_done", [])) if s else []


def phases_pending(clip_dir: PathLike) -> List[str]:
    s = read_state(clip_dir)
    return list(s.get("phases_pending", [])) if s else []


def is_phase_done(clip_dir: PathLike, phase: str) -> bool:
    return phase in phases_done(clip_dir)


def host_history(clip_dir: PathLike) -> List[dict]:
    s = read_state(clip_dir)
    return list(s.get("host_history", [])) if s else []


# ── Mutators ──────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _current_host() -> str:
    """Identifier for the machine that just ran a phase.

    Honours ``EGOINFINITY_HOST`` override (lets the user pick a nicer name
    than the raw hostname), otherwise falls back to ``socket.gethostname()``.
    """
    return os.environ.get("EGOINFINITY_HOST", "").strip() or _socket.gethostname()


def _write(clip_dir: PathLike, s: dict) -> None:
    """Atomic write via tmp + rename; preserves prior file on failure."""
    p = state_path(clip_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(s, indent=2) + "\n")
    os.replace(tmp, p)


def init_state(clip_dir: PathLike, pending: List[str]) -> dict:
    """Create a fresh state file with ``pending`` phases listed.

    Used at the start of a pipeline run.  Overwrites any prior file —
    callers that want to preserve history should ``read_state`` first
    and pass the surviving host_history forward.

    Returns the freshly written dict.
    """
    s = {
        "schema_version": SCHEMA_VERSION,
        "phases_done": [],
        "phases_pending": list(pending),
        "host_history": [],
    }
    _write(clip_dir, s)
    return s


def mark_done(clip_dir: PathLike, phase: str,
              host: Optional[str] = None) -> dict:
    """Append ``phase`` to ``phases_done`` and append a host-history entry.

    **Invalidates downstream phases**: any phase that appears AFTER
    ``phase`` in ``PHASE_ORDER`` is moved from ``phases_done`` to
    ``phases_pending`` because, by definition, re-running an upstream
    phase makes downstream outputs stale.  This is how a 4070 Ti rerun
    of D-sam3 correctly signals "I changed the SAM3 tracks, A100 needs
    to redo D-sam3d, and the old D-track is also stale."

    Idempotent — re-calling for an already-done phase still records a
    fresh host-history entry but does NOT duplicate the phases_done
    list.

    If the state file is absent (or stale schema), creates a fresh one
    with no pending list.  Callers wanting a full pending list should
    have called ``init_state`` first.

    Returns the resulting dict (already on disk).
    """
    s = read_state(clip_dir) or {
        "schema_version": SCHEMA_VERSION,
        "phases_done": [],
        "phases_pending": [],
        "host_history": [],
    }
    if phase not in s["phases_done"]:
        s["phases_done"].append(phase)
    if phase in s["phases_pending"]:
        s["phases_pending"].remove(phase)
    # Invalidate downstream phases.  Any phase strictly after `phase` in
    # PHASE_ORDER becomes stale once we touch `phase`.
    if phase in PHASE_ORDER:
        idx = PHASE_ORDER.index(phase)
        downstream = PHASE_ORDER[idx + 1:]
        for ds in downstream:
            if ds in s["phases_done"]:
                s["phases_done"].remove(ds)
            if ds not in s["phases_pending"]:
                s["phases_pending"].append(ds)
    s["host_history"].append({
        "host": host or _current_host(),
        "phase": phase,
        "ts": _now_iso(),
    })
    _write(clip_dir, s)
    return s
