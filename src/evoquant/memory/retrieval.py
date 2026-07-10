"""Bounded retrieval for LLM context.

Never ships raw history: the context is a compact, size-bounded digest —
top lessons for the context, dominant failure labels, operator outcome
summary — and it is passed through the lockbox-content guard before any
agent may see it.
"""
from __future__ import annotations

from typing import Any

from evoquant.llm.contracts import assert_no_lockbox_content
from evoquant.memory.bandit import context_key
from evoquant.memory.lessons import LessonStore
from evoquant.memory.store import MemoryStore
from evoquant.regimes.fingerprints import PairFingerprint

MAX_LESSONS = 5
MAX_FAILURE_LABELS = 6
MAX_OPERATORS = 8


def build_llm_context(
    fingerprint: PairFingerprint | None,
    regime_name: str,
    memory: MemoryStore,
    lessons: LessonStore,
) -> dict[str, Any]:
    """Compact retrieval context for hypothesis/diagnosis agents."""
    ctx = context_key(fingerprint, regime_name)

    label_counts = memory.failure_label_counts(context=ctx)
    top_failures = sorted(label_counts.items(), key=lambda kv: -kv[1])[:MAX_FAILURE_LABELS]

    op_outcomes = memory.operator_outcomes()
    op_summary = [
        {
            "operator": op,
            "successes": v["successes"],
            "trials": v["trials"],
        }
        for (c, op), v in sorted(op_outcomes.items())
        if c == ctx
    ][:MAX_OPERATORS]

    context = {
        "context_key": ctx,
        "regime": regime_name,
        "fingerprint": fingerprint.to_dict() if fingerprint is not None else None,
        "dominant_failure_labels": [
            {"label": label, "count": count} for label, count in top_failures
        ],
        "operator_outcomes": op_summary,
        "lessons": [
            {
                "lesson_id": le.lesson_id,
                "hypothesis": le.hypothesis,
                "observed_outcome": le.observed_outcome,
                "evidence_count": le.evidence_count,
                "confidence": le.confidence,
            }
            for le in lessons.active_for_context(ctx, k=MAX_LESSONS)
        ],
    }
    assert_no_lockbox_content(context)
    return context
