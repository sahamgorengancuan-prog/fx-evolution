from __future__ import annotations

import json

import pytest

from evoquant.llm.contracts import (
    HypothesisRequest,
    LLMContractError,
    LockboxLeakError,
    assert_no_lockbox_content,
    parse_hypothesis_response,
)
from evoquant.memory.lessons import LessonStore
from evoquant.memory.retrieval import build_llm_context
from evoquant.memory.store import MemoryStore


class TestLockboxGuard:
    def test_clean_payload_passes(self):
        assert_no_lockbox_content({"regime": "trend", "lessons": [{"id": 1}]})

    @pytest.mark.parametrize(
        "payload",
        [
            {"lockbox_sharpe": 1.2},
            {"nested": {"deep": {"LOCKBOX_METRICS": {}}}},
            {"items": [{"holdout_apr": -0.1}]},
            {"final_test_result": "pass"},
        ],
    )
    def test_forbidden_keys_raise_at_any_depth(self, payload):
        with pytest.raises(LockboxLeakError):
            assert_no_lockbox_content(payload)

    def test_request_construction_enforces_guard(self):
        with pytest.raises(LockboxLeakError):
            HypothesisRequest(
                context={"lockbox": {"sharpe": 2.0}}, question="what next?"
            )


class TestResponseParsing:
    def _valid(self) -> str:
        return json.dumps(
            {
                "hypothesis": "Range compression precedes expansion",
                "suggested_operators": ["jitter_periods", "replace_allow"],
                "indicator_proposal": None,
                "notes": "try volatility gates",
            }
        )

    def test_valid_response_parses(self):
        resp = parse_hypothesis_response(self._valid())
        assert resp.hypothesis.startswith("Range compression")
        assert resp.suggested_operators == ("jitter_periods", "replace_allow")
        assert resp.indicator_proposal is None

    def test_invalid_json_rejected_without_fallback(self):
        with pytest.raises(LLMContractError):
            parse_hypothesis_response("I think you should buy low and sell high")

    @pytest.mark.parametrize(
        "extra_key", ["fitness", "score", "rank", "gate_override", "feasible"]
    )
    def test_fitness_authority_rejected(self, extra_key):
        doc = json.loads(self._valid())
        doc[extra_key] = 0.99
        with pytest.raises(LLMContractError) as exc:
            parse_hypothesis_response(json.dumps(doc))
        assert "fitness" in str(exc.value).lower()

    def test_unknown_operators_rejected(self):
        doc = json.loads(self._valid())
        doc["suggested_operators"] = ["jitter_periods", "delete_lockbox"]
        with pytest.raises(LLMContractError) as exc:
            parse_hypothesis_response(json.dumps(doc))
        assert "delete_lockbox" in str(exc.value.details["unknown"])

    def test_missing_hypothesis_rejected(self):
        with pytest.raises(LLMContractError):
            parse_hypothesis_response(json.dumps({"suggested_operators": []}))

    def test_embedded_proposal_is_schema_validated(self):
        doc = json.loads(self._valid())
        doc["indicator_proposal"] = {"name": "bad proposal"}  # missing fields
        from evoquant.features.proposals import ProposalValidationError

        with pytest.raises(ProposalValidationError):
            parse_hypothesis_response(json.dumps(doc))

    def test_request_messages_shape(self):
        req = HypothesisRequest(context={"regime": "trend"}, question="why failing?")
        messages = req.to_messages()
        assert messages[0]["role"] == "system"
        assert "NO authority" in messages[0]["content"]
        assert "why failing?" in messages[1]["content"]


class TestRetrievalIntegration:
    def test_built_context_is_bounded_and_guard_clean(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        lessons = LessonStore(store)
        ctx_key = "anyfp|trend_high_vol"
        for i in range(20):
            store.append(
                "candidate_evaluated",
                {
                    "genome_hash": f"g{i}",
                    "operator": "jitter_risk",
                    "context": ctx_key,
                    "reward": i % 3 == 0,
                    "failure_labels": ["NO_EDGE_UNDER_REALISTIC_COSTS", "OVERFIRE"],
                },
            )
        for i in range(10):
            lessons.add(
                hypothesis=f"hypothesis {i}",
                observed_outcome="failed under costs",
                evidence_count=i + 1,
                confidence=0.1 * (i % 10),
                applicable_contexts=(ctx_key,),
            )
        context = build_llm_context(None, "trend_high_vol", store, lessons)
        assert context["context_key"] == ctx_key
        assert len(context["lessons"]) <= 5  # bounded, not raw history
        assert len(context["dominant_failure_labels"]) <= 6
        assert_no_lockbox_content(context)  # explicit re-check
        # usable directly as a request
        HypothesisRequest(context=context, question="what mechanism is missing?")
