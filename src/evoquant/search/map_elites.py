"""MAP-Elites quality-diversity archive.

Cells are behavior-descriptor bins; each holds at most one elite. A
challenger replaces the incumbent only if it constraint-dominates it —
the archive can therefore hold many *different* mechanisms instead of one
lineage's neighborhood (v8 audit D4 fix).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evoquant.search.nsga import constraint_dominates
from evoquant.search.objectives import CandidateEvaluation, TargetConfig

#: descriptor axes: (name, lo, hi, n_bins) — values clipped into range
DESCRIPTOR_AXES: tuple[tuple[str, float, float, int], ...] = (
    ("trade_frequency", 0.0, 0.2, 6),  # trades per bar
    ("avg_holding_bars", 0.0, 60.0, 5),
    ("long_share", 0.0, 1.0, 4),
    ("win_rate", 0.0, 1.0, 4),
)


def behavior_descriptor(ev: CandidateEvaluation) -> tuple[float, ...]:
    d = ev.diagnostics
    return (
        float(d.get("trade_frequency", 0.0)),
        float(d.get("avg_holding_bars", 0.0)),
        float(d.get("long_share", 0.5)),
        float(d.get("win_rate", 0.0)),
    )


def descriptor_cell(desc: tuple[float, ...]) -> tuple[int, ...]:
    cell = []
    for value, (_, lo, hi, bins) in zip(desc, DESCRIPTOR_AXES, strict=True):
        x = min(max(value, lo), hi - 1e-12)
        cell.append(int((x - lo) / (hi - lo) * bins))
    return tuple(cell)


@dataclass
class MapElitesArchive:
    targets: TargetConfig
    cells: dict[tuple[int, ...], CandidateEvaluation] = field(default_factory=dict)
    #: total insertion attempts / accepted replacements (novelty telemetry)
    attempts: int = 0
    accepted: int = 0

    def try_insert(self, ev: CandidateEvaluation) -> bool:
        self.attempts += 1
        cell = descriptor_cell(behavior_descriptor(ev))
        incumbent = self.cells.get(cell)
        if incumbent is None or constraint_dominates(ev, incumbent, self.targets):
            self.cells[cell] = ev
            self.accepted += 1
            return True
        return False

    @property
    def coverage(self) -> int:
        return len(self.cells)

    def coverage_fraction(self) -> float:
        total = float(np.prod([a[3] for a in DESCRIPTOR_AXES]))
        return self.coverage / total

    def elites(self) -> list[CandidateEvaluation]:
        return list(self.cells.values())

    def summary(self) -> dict[str, Any]:
        return {
            "coverage_cells": self.coverage,
            "coverage_fraction": self.coverage_fraction(),
            "attempts": self.attempts,
            "accepted": self.accepted,
            "axes": [asdict_axis(a) for a in DESCRIPTOR_AXES],
        }


def asdict_axis(axis: tuple[str, float, float, int]) -> dict[str, Any]:
    name, lo, hi, bins = axis
    return {"name": name, "lo": lo, "hi": hi, "bins": bins}
