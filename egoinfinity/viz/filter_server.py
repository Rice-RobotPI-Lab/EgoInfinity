"""Programmatic entry to the filter visualizer (port of action100m_filter/viz_server.py).

The actual server implementation lives at
``action100m_filter/viz_server.py`` (~250 lines of stdlib ``http.server``
+ embedded HTML). This module re-exports its public API so callers can
use the filter viz without importing from ``action100m_filter`` directly.

Use from CLI: ``egoinfinity filter --viz :8899`` (see ``egoinfinity/cli/filter.py``)

Programmatic use::

    from egoinfinity.viz import filter_server
    filter_server.init("/tmp/filter_viz")
    server = filter_server.start_server(port=8899)
    # ... add items as the filter runs:
    filter_server.add_item({
        "id": "abc",
        "filename": "abc.mp4",
        "video_uid": "abc",
        "status": "accept",
        "reason": "static-cam + 2 hands",
        "metrics": {...},
        "desc": "Cleaning a countertop.",
    })
    # Open http://localhost:8899/ in a browser
"""
from __future__ import annotations

# Re-export from the canonical implementation in action100m_filter.
# The split is purely organizational: action100m_filter holds the
# filter algorithm itself, egoinfinity.viz holds the visualization
# entrypoints surfaced by the unified `egoinfinity` CLI.
from action100m_filter.viz_server import (   # noqa: F401
    init,
    add_item,
    start_server,
)

__all__ = ["init", "add_item", "start_server"]
