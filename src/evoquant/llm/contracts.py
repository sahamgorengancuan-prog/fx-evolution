"""Strict JSON contracts for LLM agents.

Two non-negotiable rules, both enforced here and tested:

1. **No fitness authority.** Response schemas have no channel for scores,
   fitness, gate overrides, or rankings; a response that smuggles such a
   field is rejected outright (never repaired, never replaced randomly).
2. **No lockbox visibility.** Anything headed for an LLM prompt passes
   :func:`assert_no_lockbox_content`; forbidden keys anywhere in the
   payload raise :class:`LockboxLeakError`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from evoquant.errors import EvoquantError
from evoquant.features.proposals import IndicatorProposal
from evoquant.genome.mutations import OPERATOR_NAMES


class LLMContractError(EvoquantError):
    code = "LLM_CONTRACT"


class LockboxLeakError(EvoquantError):
    code = "LOCKBOX_LEAK"


#: keys that must never appear in LLM-bound payloads (case-insensitive,
#: substring match on mapping keys at any depth)
FORBIDDEN_CONTEXT_KEYS = ("lockbox", "holdout", "sealed_test", "final_test")

#: response fields that would constitute fitness authority
FORBIDDEN_RESPONSE_KEYS = ("fitness", "score", "rank", "gate_override", "feasible")


def assert_no_lockbox_content(obj: Any, _path: str = "$") -> None:
    """Recursively scan mapping keys for forbidden terms; raise on hit."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            for term in FORBIDDEN_CONTEXT_KEYS:
                if term in lowered:
                    raise LockboxLeakError(
                        "LLM-bound payload contains lockbox-scoped content; "
                        "agents must never see final-test information.",
                        key=str(key),
                        path=_path,
                    )
            assert_no_lockbox_content(value, f"{_path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            assert_no_lockbox_content(item, f"{_path}[{i}]")


@dataclass(frozen=True)
class HypothesisRequest:
    """What the search sends to the hypothesis agent."""

    context: dict[str, Any]  # from memory.retrieval.build_llm_context
    question: str

    def __post_init__(self) -> None:
        assert_no_lockbox_content(self.context)
        if not self.question.strip():
            raise LLMContractError("HypothesisRequest.question must be non-empty")

    def to_messages(self) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": (
                    "You are a quantitative research hypothesis agent. You may "
                    "propose mechanisms, indicator ideas, and mutation-operator "
                    "suggestions. You have NO authority over fitness, gates, or "
                    "rankings; numerical evaluation is done elsewhere. Respond "
                    "with a single JSON object with keys: hypothesis (string), "
                    "suggested_operators (array of strings), "
                    "indicator_proposal (object or null), notes (string)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"context": self.context, "question": self.question},
                    sort_keys=True,
                ),
            },
        ]


@dataclass(frozen=True)
class HypothesisResponse:
    hypothesis: str
    suggested_operators: tuple[str, ...] = field(default_factory=tuple)
    indicator_proposal: IndicatorProposal | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.indicator_proposal is not None:
            d["indicator_proposal"] = self.indicator_proposal.to_dict()
        return d


def parse_hypothesis_response(raw: str) -> HypothesisResponse:
    """Parse + validate an LLM reply. Invalid replies raise — no fallback."""
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMContractError(
            "LLM response is not valid JSON", reason=str(exc), raw=raw[:400]
        ) from exc
    if not isinstance(doc, dict):
        raise LLMContractError("LLM response must be a JSON object", raw=raw[:400])

    for key in doc:
        lowered = str(key).lower()
        if any(term == lowered or term in lowered for term in FORBIDDEN_RESPONSE_KEYS):
            raise LLMContractError(
                "LLM response attempts to assign fitness/ranking — rejected. "
                "The LLM has no fitness authority (truth constraint #4).",
                offending_key=str(key),
            )

    hypothesis = doc.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise LLMContractError("LLM response missing 'hypothesis' string")

    ops_raw = doc.get("suggested_operators", [])
    if not isinstance(ops_raw, list):
        raise LLMContractError("'suggested_operators' must be an array")
    unknown = [op for op in ops_raw if op not in OPERATOR_NAMES]
    if unknown:
        raise LLMContractError(
            "LLM suggested unknown mutation operators",
            unknown=unknown,
            allowed=list(OPERATOR_NAMES),
        )

    proposal_doc = doc.get("indicator_proposal")
    proposal = (
        IndicatorProposal.from_dict(proposal_doc) if isinstance(proposal_doc, dict) else None
    )

    notes = doc.get("notes", "")
    if not isinstance(notes, str):
        raise LLMContractError("'notes' must be a string")

    return HypothesisResponse(
        hypothesis=hypothesis.strip(),
        suggested_operators=tuple(str(op) for op in ops_raw),
        indicator_proposal=proposal,
        notes=notes,
    )
