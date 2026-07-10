"""Contextual Thompson sampling over mutation operators.

For each (context, operator) pair a Beta(successes+a, failures+b)
posterior is maintained; selection samples all operators' posteriors and
picks the argmax. Contexts are coarse buckets of (pair fingerprint,
regime), so what worked on a trending high-vol market does not silently
bias a ranging low-vol one.

Reward is decided by the caller (island): a mutation earns 1 when its
child moves toward feasibility, Pareto-improves its parent, or wins a
MAP-Elites cell — never from a raw scalar score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evoquant.regimes.fingerprints import PairFingerprint


def context_key(fingerprint: PairFingerprint | None, regime_name: str) -> str:
    """Coarse, stable context bucket. Thresholds are fixed constants."""
    if fingerprint is None:
        return f"anyfp|{regime_name}"
    vol = "hivol" if fingerprint.atr_pct_median > 0.01 else "lovol"
    trend = "trendy" if fingerprint.efficiency_median > 0.3 else "choppy"
    return f"{vol}_{trend}|{regime_name}"


@dataclass
class ThompsonOperatorSelector:
    operators: tuple[str, ...]
    seed: int = 0
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    #: (context, operator) -> [successes, failures]
    counts: dict[tuple[str, str], list[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def select(self, context: str) -> str:
        samples = []
        for op in self.operators:
            s, f = self.counts.get((context, op), [0, 0])
            samples.append(
                self._rng.beta(self.prior_alpha + s, self.prior_beta + f)
            )
        return self.operators[int(np.argmax(samples))]

    def update(self, context: str, operator: str, reward: bool) -> None:
        if operator not in self.operators:
            return  # unknown operators (e.g. crossover path) are not tracked
        slot = self.counts.setdefault((context, operator), [0, 0])
        slot[0 if reward else 1] += 1

    def posterior_mean(self, context: str, operator: str) -> float:
        s, f = self.counts.get((context, operator), [0, 0])
        return (self.prior_alpha + s) / (self.prior_alpha + self.prior_beta + s + f)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operators": list(self.operators),
            "seed": self.seed,
            "counts": {f"{c}::{o}": v for (c, o), v in self.counts.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ThompsonOperatorSelector:
        sel = cls(operators=tuple(d["operators"]), seed=int(d["seed"]))
        for key, v in d["counts"].items():
            c, o = key.split("::", 1)
            sel.counts[(c, o)] = [int(v[0]), int(v[1])]
        return sel
