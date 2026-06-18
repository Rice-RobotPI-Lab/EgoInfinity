"""LLM-based SAM-3 prompt extraction — DEPRECATED.

This module previously hosted a Qwen3.5-4B int4 extractor that ran in the
egoinfinity env. It has been removed in favour of using Claude (Anthropic) for
prompt extraction; see :file:`object-extraction-prompt.md` in this directory for the
system prompt + I/O format.

The function below is preserved as a clear failure path so callers that
still try to auto-extract trigger an explicit error instead of silently
loading an old model.

Curated workflow (post-Qwen):
    1. For each clip's manifest, hand the (action_brief, action_detailed,
       actor, summary) text to Claude with the prompt in object-extraction-prompt.md.
    2. Write the returned JSON array to ``manifest['objects']`` and set
       ``manifest['objects_source'] = 'claude'`` (or ``'manual'`` if hand
       edited afterwards).
    3. Pipeline reprocess (``tools/batch_reprocess_trimmed.py``) treats
       any ``objects_source != 'manual'`` clip as needing extraction —
       so for ``'claude'`` clips it would still try to call the (now
       removed) extractor; flip those to ``'manual'`` once curated.
"""
from __future__ import annotations

from typing import List, Optional, Tuple


_DEPRECATION_MSG = (
    "egoinfinity.pipeline.llm_extractor.extract_object_prompts has been "
    "removed. SAM-3 prompts are now produced via Claude — see "
    "egoinfinity/pipeline/object-extraction-prompt.md and pre-fill manifest.objects "
    "with objects_source='manual'."
)


def is_loaded() -> bool:
    return False


def load_model(model_id: Optional[str] = None) -> None:
    raise NotImplementedError(_DEPRECATION_MSG)


def unload_model() -> None:
    return None


def model_info() -> dict:
    return {"loaded": False, "deprecated": True, "message": _DEPRECATION_MSG}


def extract_object_prompts(
    action_brief: str = "",
    action_detailed: str = "",
    actor: str = "",
    summary: str = "",
    max_prompts: int = 5,
) -> Tuple[List[str], dict]:
    raise NotImplementedError(_DEPRECATION_MSG)
