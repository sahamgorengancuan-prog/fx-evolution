"""Constraint-first candidate evaluation records.

Hard constraints and Pareto objectives stay separate vectors all the way
through selection — nothing is scalarized (v8 audit D3 fix). A candidate
is *feasible* iff every constraint violation is zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

#: Objective sense: everything below is MAXIMIZED (negate costs upstream).
OBJECTIVE_NAMES = (
    "median_fold_return",
    "neg_max_drawdown",
    "profit_factor",
    "neg_fold_dispersion",
    "neg_complexity",
)


@dataclass(frozen=True)
class TargetConfig:
    """Shared target vector (same for every pair; pass is never forced)."""

    min_trades_total: int = 30
    max_drawdown: float = 0.25
    min_profit_factor: float = 1.0
    min_median_fold_return: float = 0.0
    max_fold_dispersion: float = 0.5
    max_complexity: int = 120

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateEvaluation:
    genome_hash: str
    fold_returns: tuple[float, ...]
    n_trades_total: int
    max_drawdown: float
    profit_factor: float | None
    complexity: int
    #: extra diagnostics for descriptors/failure labels
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #

    def objectives(self) -> np.ndarray:
        rets = np.asarray(self.fold_returns, dtype=np.float64)
        finite = rets[np.isfinite(rets)]
        median_ret = float(np.median(finite)) if len(finite) else -1.0
        dispersion = float(np.std(finite)) if len(finite) > 1 else 0.0
        pf = self.profit_factor if self.profit_factor is not None else 0.0
        return np.array(
            [
                median_ret,
                -self.max_drawdown,
                min(pf, 10.0),  # cap: a huge PF on 3 trades must not dominate
                -dispersion,
                -float(self.complexity),
            ]
        )

    def violations(self, targets: TargetConfig) -> dict[str, float]:
        """Zero-clipped constraint violations. Empty values == feasible."""
        rets = np.asarray(self.fold_returns, dtype=np.float64)
        finite = rets[np.isfinite(rets)]
        median_ret = float(np.median(finite)) if len(finite) else -1.0
        dispersion = float(np.std(finite)) if len(finite) > 1 else 0.0
        pf = self.profit_factor if self.profit_factor is not None else 0.0
        v = {
            "min_trades_total": max(0.0, targets.min_trades_total - self.n_trades_total)
            / max(targets.min_trades_total, 1),
            "max_drawdown": max(0.0, self.max_drawdown - targets.max_drawdown),
            "min_profit_factor": max(0.0, targets.min_profit_factor - pf),
            "min_median_fold_return": max(
                0.0, targets.min_median_fold_return - median_ret
            ),
            "max_fold_dispersion": max(0.0, dispersion - targets.max_fold_dispersion),
            "max_complexity": max(0.0, (self.complexity - targets.max_complexity))
            / max(targets.max_complexity, 1),
        }
        return {k: val for k, val in v.items() if val > 0.0}

    def total_violation(self, targets: TargetConfig) -> float:
        return float(sum(self.violations(targets).values()))

    def is_feasible(self, targets: TargetConfig) -> bool:
        return not self.violations(targets)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
