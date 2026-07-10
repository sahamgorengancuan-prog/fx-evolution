"""LLM indicator-proposal schema (validation only in Phase 2).

An LLM may *propose* a new primitive, but never executes code. A proposal
is a strict JSON document; anything failing validation is rejected with a
structured error — never replaced with a random fallback.

Phase 2 ships schema validation and basic formula hygiene checks.
Whitelisted formula compilation, sandbox execution, causality/parity/
ablation testing, and grammar promotion arrive in Phase 6 — until then no
proposal can enter the grammar (fail closed by construction).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from evoquant.errors import EvoquantError
from evoquant.features.types import Dimension

VALID_INPUTS = frozenset({"open", "high", "low", "close", "volume", "atr"})

#: Substrings that immediately disqualify a formula string.
_FORBIDDEN = ("__", "import", "exec", "eval", "lambda", "open(", "os.", "sys.", ";")

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,40}$")


class ProposalValidationError(EvoquantError):
    code = "PROPOSAL_INVALID"


@dataclass(frozen=True)
class IndicatorProposal:
    name: str
    hypothesis: str
    inputs: tuple[str, ...]
    output_type: str
    lookback_parameters: tuple[str, ...]
    causal: bool
    incremental_formula: str
    expected_regimes: tuple[str, ...]
    failure_modes: tuple[str, ...]
    complexity_cost: int
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> IndicatorProposal:
        required = (
            "name",
            "hypothesis",
            "inputs",
            "output_type",
            "lookback_parameters",
            "causal",
            "incremental_formula",
            "expected_regimes",
            "failure_modes",
            "complexity_cost",
        )
        missing = [k for k in required if k not in doc]
        if missing:
            raise ProposalValidationError(
                "Proposal missing required fields", missing_fields=missing
            )
        proposal = cls(
            name=str(doc["name"]),
            hypothesis=str(doc["hypothesis"]),
            inputs=tuple(doc["inputs"]),
            output_type=str(doc["output_type"]),
            lookback_parameters=tuple(doc["lookback_parameters"]),
            causal=bool(doc["causal"]),
            incremental_formula=str(doc["incremental_formula"]),
            expected_regimes=tuple(doc["expected_regimes"]),
            failure_modes=tuple(doc["failure_modes"]),
            complexity_cost=int(doc["complexity_cost"]),
            extra={k: v for k, v in doc.items() if k not in required},
        )
        proposal.validate()
        return proposal

    def validate(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ProposalValidationError(
                "Proposal name must be snake_case, 3-41 chars", name=self.name
            )
        if not self.causal:
            raise ProposalValidationError(
                "Non-causal proposals are rejected outright", name=self.name
            )
        bad_inputs = [i for i in self.inputs if i not in VALID_INPUTS]
        if bad_inputs:
            raise ProposalValidationError(
                "Unknown proposal inputs", name=self.name, bad_inputs=bad_inputs
            )
        try:
            dim = Dimension(self.output_type)
        except ValueError:
            raise ProposalValidationError(
                "Unknown output_type", name=self.name, output_type=self.output_type
            ) from None
        if dim in (Dimension.BOOL, Dimension.UNSCALED):
            raise ProposalValidationError(
                "Proposals must output a comparable numeric dimension",
                name=self.name,
                output_type=self.output_type,
            )
        if not self.lookback_parameters:
            raise ProposalValidationError(
                "Proposal must declare at least one lookback parameter", name=self.name
            )
        if self.complexity_cost < 1:
            raise ProposalValidationError(
                "complexity_cost must be >= 1", name=self.name
            )
        if not self.hypothesis.strip() or not self.failure_modes:
            raise ProposalValidationError(
                "Proposal must state a hypothesis and at least one failure mode",
                name=self.name,
            )
        lowered = self.incremental_formula.lower()
        hits = [tok for tok in _FORBIDDEN if tok in lowered]
        if hits:
            raise ProposalValidationError(
                "Formula contains forbidden constructs",
                name=self.name,
                forbidden=hits,
            )
        if not self.incremental_formula.strip():
            raise ProposalValidationError("Empty incremental_formula", name=self.name)
