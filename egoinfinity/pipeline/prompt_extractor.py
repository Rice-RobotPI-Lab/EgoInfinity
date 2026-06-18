"""
SAM3 prompt extractor.

Historically this module used spaCy for noun chunk extraction, but spaCy's
pure-syntactic approach has fundamental limitations (no world knowledge,
can't merge synonyms, can't filter abstractions, can't fill in implied
tools).  It has been **replaced with an LLM-based extractor** using
Qwen3.5-4B int4 which runs entirely offline in the main egoinfinity env.

This file now is just a thin facade over :mod:`llm_extractor`.  All the
real work happens there.
"""
from __future__ import annotations

from typing import List, Tuple

from egoinfinity.pipeline.llm_extractor import extract_object_prompts as _llm_extract


def extract_object_prompts(
    action_brief: str = "",
    action_detailed: str = "",
    actor: str = "",
    summary: str = "",
    max_prompts: int = 5,
) -> Tuple[List[str], dict]:
    """Extract SAM3 text prompts from an Action100M segment's text fields.

    Uses the globally-loaded Qwen3.5-4B int4 model in
    :mod:`egoinfinity.pipeline.llm_extractor`.  The model is loaded lazily on
    first call and stays resident for the lifetime of the process.

    Args:
        action_brief: gpt.action.brief
        action_detailed: gpt.action.detailed
        actor: gpt.action.actor
        summary: gpt.summary.detailed (optional, improves coverage)
        max_prompts: cap on returned list length

    Returns:
        (prompts, debug_info) where *prompts* is a list of short noun
        phrases suitable as SAM3 text prompts.
    """
    return _llm_extract(
        action_brief=action_brief,
        action_detailed=action_detailed,
        summary=summary,
        actor=actor,
        max_prompts=max_prompts,
    )
