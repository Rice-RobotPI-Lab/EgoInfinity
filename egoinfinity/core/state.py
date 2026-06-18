"""ClipState — per-clip stage provenance + done-tracking.

Schema (state.json)::

    {
      "version": 1,
      "clip_id": "...",
      "stages": {
        "<stage_name>": {
          "ts_start": "<ISO8601>",
          "ts_end": "<ISO8601>",
          "duration_s": 41.2,
          "backend": "moge2",
          "args": { ... },
          "outputs_hash": { "depth.npz": "<sha1>", "focal.json": "<sha1>" }
        },
        ...
      }
    }

The runner queries ``is_done()`` to decide whether to skip a stage; the
stage code calls ``mark_done()`` after writing its outputs. ``forget()``
+ ``cascade_forget()`` implement ``--force`` and ``--force --cascade``.

Distinct from the older ``egoinfinity/pipeline/pipeline_state.py`` (which
tracks multi-host coordination). They coexist for now; eventually the
older file becomes a thin shim over this.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore


_SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ClipState:
    """State.json reader/writer for one clip."""

    def __init__(self, store: ArtifactStore) -> None:
        self.store = store
        self.path = store.state_path()
        self._data = self._load()

    # ── Persistence ──────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                if d.get("version") == _SCHEMA_VERSION:
                    d.setdefault("stages", {})
                    return d
            except Exception:
                pass   # corrupt → start fresh
        return {
            "version": _SCHEMA_VERSION,
            "clip_id": self.store.clip_id,
            "stages": {},
        }

    def save(self) -> None:
        # Atomic write via temp + rename
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True))
        tmp.replace(self.path)

    # ── Query ────────────────────────────────────────────────────────────

    def is_done(self, stage: str) -> bool:
        """A stage is "done" iff it has a stages[stage].ts_end entry."""
        e = self._data["stages"].get(stage)
        return e is not None and e.get("ts_end") is not None

    def get(self, stage: str) -> dict[str, Any] | None:
        return self._data["stages"].get(stage)

    def stages_done(self) -> list[str]:
        return [s for s, e in self._data["stages"].items() if e.get("ts_end") is not None]

    # ── Mutate ───────────────────────────────────────────────────────────

    def mark_running(self, stage: str, *, backend: str, args: dict[str, Any]) -> None:
        self._data["stages"][stage] = {
            "ts_start": _now_iso(),
            "ts_end": None,
            "duration_s": None,
            "backend": backend,
            "args": dict(args),
            "outputs_hash": {},
        }
        self.save()

    def mark_done(
        self,
        stage: str,
        *,
        backend: str,
        args: dict[str, Any],
        duration_s: float,
        outputs: dict[str, str],
    ) -> None:
        """Record stage as completed with hashes of its outputs."""
        existing = self._data["stages"].get(stage, {})
        self._data["stages"][stage] = {
            "ts_start": existing.get("ts_start", _now_iso()),
            "ts_end": _now_iso(),
            "duration_s": float(duration_s),
            "backend": backend,
            "args": dict(args),
            "outputs_hash": dict(outputs),
        }
        self.save()

    def forget(self, stage: str) -> None:
        """Drop a stage's record (used by ``--force``)."""
        self._data["stages"].pop(stage, None)
        self.save()

    def cascade_forget(self, root_stage: str, downstream_map: dict[str, list[str]]) -> None:
        """Drop root_stage + every stage that lists it (transitively) as upstream.

        ``downstream_map`` is ``{stage: [stages_that_depend_on_it]}``.
        """
        to_forget: set[str] = {root_stage}
        changed = True
        while changed:
            changed = False
            for s in list(to_forget):
                for d in downstream_map.get(s, []):
                    if d not in to_forget:
                        to_forget.add(d)
                        changed = True
        for s in to_forget:
            self._data["stages"].pop(s, None)
        self.save()
