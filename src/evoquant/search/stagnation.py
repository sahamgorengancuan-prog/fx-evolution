"""Multi-signal stagnation detection + logged escalation ladder.

v8 counted "best score unchanged for N generations" and did nothing with
it. Here stagnation is a conjunction of weaker signals (hypervolume,
feasibility rate, archive coverage, lineage concentration) and every
response is an explicit, logged state transition. The ladder ends in
``NO_EDGE_FOUND`` — a legal, first-class outcome. More generations are
never assumed to help.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Intervention(str, Enum):  # noqa: UP042 - plain-str JSON round-trip
    LOCAL_REFINEMENT = "LOCAL_REFINEMENT"
    NOVELTY_INCREASE = "NOVELTY_INCREASE"
    STRUCTURE_REPLACEMENT = "STRUCTURE_REPLACEMENT"
    ISLAND_RESTART = "ISLAND_RESTART"
    NO_EDGE_FOUND = "NO_EDGE_FOUND"


#: escalation order — each firing advances one rung; success resets
LADDER: tuple[Intervention, ...] = (
    Intervention.LOCAL_REFINEMENT,
    Intervention.NOVELTY_INCREASE,
    Intervention.STRUCTURE_REPLACEMENT,
    Intervention.ISLAND_RESTART,
    Intervention.NO_EDGE_FOUND,
)


@dataclass(frozen=True)
class GenerationSignals:
    generation: int
    hypervolume: float  # 2D proxy over (median return, -maxDD) vs ref point
    feasibility_rate: float
    archive_coverage: int
    lineage_concentration: float  # share of population sharing the modal ancestor
    #: smallest total constraint violation in the population (0 if any feasible)
    min_violation: float = 0.0


def hypervolume_2d(points: np.ndarray, ref: tuple[float, float] = (-1.0, -1.0)) -> float:
    """Exact 2D hypervolume (maximization) against a reference point."""
    pts = points[np.all(points > np.asarray(ref), axis=1)] if len(points) else points
    if len(pts) == 0:
        return 0.0
    order = np.argsort(-pts[:, 0])
    hv = 0.0
    best_y = ref[1]
    for i in order:
        x, y = float(pts[i, 0]), float(pts[i, 1])
        if y > best_y:
            hv += (x - ref[0]) * (y - best_y)
            best_y = y
    return hv


@dataclass
class StagnationController:
    """Detects stagnation and walks the escalation ladder. Fully logged."""

    patience: int = 3  # stagnant generations before a rung fires
    min_improvement: float = 1e-9
    #: violation must shrink by this relative fraction to count as progress
    #: (an epsilon drift toward an unreachable target is not progress)
    min_rel_improvement: float = 1e-3
    history: list[GenerationSignals] = field(default_factory=list)
    log: list[dict[str, Any]] = field(default_factory=list)
    _rung: int = 0
    _stagnant: int = 0
    _best_hv: float = float("-inf")
    _best_feas: float = float("-inf")
    _best_cov: int = -1
    _best_minviol: float = float("inf")

    def observe(self, s: GenerationSignals) -> Intervention | None:
        """Feed one generation's signals; maybe returns an intervention.

        Progress semantics: while the population is entirely infeasible,
        only movement TOWARD feasibility counts (feasibility rate up or
        smallest violation down). Hypervolume/diversity growth among
        candidates that all fail the gates is churn, not progress —
        otherwise a hopeless island could evade `NO_EDGE_FOUND` forever
        by shuffling infeasible variety (the v8 failure mode, D3/D4).
        """
        self.history.append(s)
        viol_bar = self._best_minviol * (1.0 - self.min_rel_improvement)
        toward_feasibility = (
            s.feasibility_rate > self._best_feas + self.min_improvement
            or s.min_violation < viol_bar
        )
        any_feasible = s.feasibility_rate > 0.0
        quality_progress = any_feasible and (
            s.hypervolume > self._best_hv + self.min_improvement
            or s.archive_coverage > self._best_cov
        )
        improved = toward_feasibility or quality_progress
        self._best_hv = max(self._best_hv, s.hypervolume)
        self._best_feas = max(self._best_feas, s.feasibility_rate)
        self._best_cov = max(self._best_cov, s.archive_coverage)
        self._best_minviol = min(self._best_minviol, s.min_violation)

        if improved:
            if self._stagnant or self._rung:
                self.log.append(
                    {
                        "generation": s.generation,
                        "event": "progress_resumed",
                        "rung_reset_from": self._rung,
                    }
                )
            self._stagnant = 0
            self._rung = 0
            return None

        self._stagnant += 1
        if self._stagnant < self.patience:
            return None

        intervention = LADDER[min(self._rung, len(LADDER) - 1)]
        self.log.append(
            {
                "generation": s.generation,
                "event": "intervention",
                "intervention": intervention.value,
                "rung": self._rung,
                "signals": asdict(s),
                "stagnant_generations": self._stagnant,
            }
        )
        self._rung = min(self._rung + 1, len(LADDER) - 1)
        self._stagnant = 0
        return intervention

    @property
    def terminal(self) -> bool:
        return any(
            e.get("intervention") == Intervention.NO_EDGE_FOUND.value for e in self.log
        )
