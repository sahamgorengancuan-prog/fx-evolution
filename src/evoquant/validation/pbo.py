"""Probability of Backtest Overfitting via CSCV (Bailey et al.).

Requires what the definition requires: a **candidate-performance matrix**
(T periods × N candidates). v8 reported a per-single-strategy "pbo",
which has no defined meaning (audit S1) — that API shape is impossible
here: one column raises.

CSCV: rows are split into S contiguous groups; for every C(S, S/2)
combination, half the groups form the in-sample set. The IS-best
candidate's *relative rank* ω in the complementary out-of-sample set is
recorded; PBO = P(ω < 0.5) — how often the IS winner is an OOS loser.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from evoquant.validation.sharpe import ValidationInputError


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_combinations: int
    n_candidates: int
    lambdas: tuple[float, ...]  # logit of OOS relative ranks of IS winners

    def to_dict(self) -> dict[str, float | int | list[float]]:
        return {
            "pbo": self.pbo,
            "n_combinations": self.n_combinations,
            "n_candidates": self.n_candidates,
            "lambda_median": float(np.median(self.lambdas)) if self.lambdas else 0.0,
        }


def _metric(rows: np.ndarray) -> np.ndarray:
    """Per-candidate Sharpe over the given rows (mean/std, std guard)."""
    mean = rows.mean(axis=0)
    std = rows.std(axis=0, ddof=1)
    out = np.where(std > 0, mean / np.where(std > 0, std, 1.0), -np.inf)
    return np.asarray(out, dtype=np.float64)


def pbo_cscv(candidate_matrix: np.ndarray, n_partitions: int = 8) -> PBOResult:
    """CSCV PBO over a (T, N) per-period performance matrix."""
    m = np.asarray(candidate_matrix, dtype=np.float64)
    if m.ndim != 2:
        raise ValidationInputError("PBO requires a 2-D (T, N) matrix", shape=str(m.shape))
    t, n = m.shape
    if n < 2:
        raise ValidationInputError(
            "PBO is undefined for a single candidate — a self temporal split "
            "is not PBO (v8 audit S1). Provide the full candidate battery.",
            n_candidates=n,
        )
    if n_partitions < 2 or n_partitions % 2 != 0:
        raise ValidationInputError("n_partitions must be an even integer >= 2")
    if t < 2 * n_partitions:
        raise ValidationInputError(
            "Not enough rows for the requested partitions",
            rows=t,
            n_partitions=n_partitions,
        )

    groups = np.array_split(np.arange(t), n_partitions)
    lambdas: list[float] = []
    for combo in combinations(range(n_partitions), n_partitions // 2):
        is_rows = np.concatenate([groups[g] for g in combo])
        oos_rows = np.concatenate(
            [groups[g] for g in range(n_partitions) if g not in combo]
        )
        is_perf = _metric(m[is_rows])
        oos_perf = _metric(m[oos_rows])
        best = int(np.argmax(is_perf))
        # relative rank of the IS winner among OOS performances, in (0, 1)
        rank = float(np.sum(oos_perf < oos_perf[best]))
        ties = float(np.sum(oos_perf == oos_perf[best])) - 1.0
        omega = (rank + 0.5 * ties + 1.0) / (n + 1.0)
        omega = min(max(omega, 1e-9), 1.0 - 1e-9)
        lambdas.append(math.log(omega / (1.0 - omega)))

    lam = np.asarray(lambdas)
    return PBOResult(
        pbo=float(np.mean(lam <= 0.0)),
        n_combinations=len(lambdas),
        n_candidates=n,
        lambdas=tuple(float(x) for x in lam),
    )
