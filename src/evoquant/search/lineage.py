"""Lineage records — every evaluated candidate, kept forever.

In-memory + JSON export for Phase 5; the event-sourced store (DuckDB)
arrives in Phase 6 and replays these records unchanged.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LineageRecord:
    genome_hash: str
    parent_hashes: tuple[str, ...]
    origin: str  # random | mutation | crossover | local_refine
    operator: str | None  # mutation operator name, if any
    generation: int
    feasible: bool
    total_violation: float
    objectives: tuple[float, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineageStore:
    records: list[LineageRecord] = field(default_factory=list)
    _by_hash: dict[str, LineageRecord] = field(default_factory=dict)

    def add(self, record: LineageRecord) -> None:
        self.records.append(record)  # append-only; duplicates allowed (re-evals)
        self._by_hash.setdefault(record.genome_hash, record)

    def __len__(self) -> int:
        return len(self.records)

    def root_ancestor(self, genome_hash: str, max_depth: int = 64) -> str:
        h = genome_hash
        for _ in range(max_depth):
            rec = self._by_hash.get(h)
            if rec is None or not rec.parent_hashes:
                return h
            h = rec.parent_hashes[0]
        return h

    def lineage_concentration(self, population_hashes: list[str]) -> float:
        """Share of the population descending from the modal root ancestor."""
        if not population_hashes:
            return 0.0
        roots = Counter(self.root_ancestor(h) for h in population_hashes)
        return roots.most_common(1)[0][1] / len(population_hashes)

    def export_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps([asdict(r) for r in self.records], indent=2, sort_keys=True)
        )
        return path
