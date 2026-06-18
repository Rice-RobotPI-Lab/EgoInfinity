"""Unified pipeline orchestration layer.

Wraps the existing egoinfinity.pipeline (incl. post_tracking/), retarget,
action100m_filter, and tools/ libraries into a config-driven Stage DAG
with artifact-aware resume.

See docs/ARCHITECTURE.md for the design overview.
"""
__version__ = "0.1.0"
