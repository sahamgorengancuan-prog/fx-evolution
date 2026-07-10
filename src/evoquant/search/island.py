"""One pair-regime search island: the generational loop.

Wires together: typed genome generation/mutation (grammar-guided GP),
constraint-dominance NSGA-II survival, the MAP-Elites archive, ES-style
local parameter refinement gated on structural stability, the stagnation
controller's escalation ladder, and lineage recording for every evaluated
candidate. Deterministic under its seed.

Deliberately absent until Phase 6: the contextual operator bandit
(operators are drawn uniformly here — the hook is the `operator` field on
every lineage record) and the persistent event store.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evoquant.genome.genome import StrategyGenome, random_genome
from evoquant.genome.mutations import OPERATOR_NAMES, crossover, mutate
from evoquant.search.evaluator import EvalContext, evaluate_genome
from evoquant.search.lineage import LineageRecord, LineageStore
from evoquant.search.map_elites import MapElitesArchive
from evoquant.search.nsga import rank_population
from evoquant.search.objectives import CandidateEvaluation, TargetConfig
from evoquant.search.stagnation import (
    GenerationSignals,
    Intervention,
    StagnationController,
    hypervolume_2d,
)

#: mutation operators considered "parameter-local" for the refinement rung
_LOCAL_OPS = ("jitter_periods", "jitter_constants", "jitter_risk")
_STRUCTURAL_OPS = ("replace_entry_long", "replace_entry_short", "replace_allow")


@dataclass
class IslandConfig:
    symbol: str
    population_size: int = 16
    n_generations: int = 8
    seed: int = 0
    random_fraction: float = 0.2  # fresh random genomes per generation
    crossover_fraction: float = 0.2
    stagnation_patience: int = 3


@dataclass
class IslandState:
    generation: int = 0
    population: list[StrategyGenome] = field(default_factory=list)
    evaluations: list[CandidateEvaluation] = field(default_factory=list)
    #: forced operator pool override from an escalation rung (one generation)
    forced_ops: tuple[str, ...] | None = None
    outcome: str = "RUNNING"  # RUNNING | BUDGET_EXHAUSTED | NO_EDGE_FOUND
    telemetry: list[dict[str, Any]] = field(default_factory=list)


class SearchIsland:
    def __init__(self, config: IslandConfig, ctx: EvalContext, targets: TargetConfig) -> None:
        self.config = config
        self.ctx = ctx
        self.targets = targets
        self.rng = np.random.default_rng(config.seed)
        self.lineage = LineageStore()
        self.archive = MapElitesArchive(targets=targets)
        self.stagnation = StagnationController(patience=config.stagnation_patience)
        self.state = IslandState()
        self._eval_cache: dict[str, CandidateEvaluation] = {}

    # ------------------------------------------------------------------ #

    def _evaluate(self, genome: StrategyGenome, operator: str | None) -> CandidateEvaluation:
        h = genome.genome_hash()
        ev = self._eval_cache.get(h)
        if ev is None:
            ev = evaluate_genome(genome, self.ctx)
            self._eval_cache[h] = ev
        self.lineage.add(
            LineageRecord(
                genome_hash=h,
                parent_hashes=genome.parent_hashes,
                origin=genome.origin,
                operator=operator,
                generation=self.state.generation,
                feasible=ev.is_feasible(self.targets),
                total_violation=ev.total_violation(self.targets),
                objectives=tuple(float(x) for x in ev.objectives()),
                diagnostics=dict(ev.diagnostics),
            )
        )
        self.archive.try_insert(ev)
        return ev

    def _init_population(self) -> None:
        self.state.population = [
            random_genome(self.rng, self.config.symbol, generation=0)
            for _ in range(self.config.population_size)
        ]
        self.state.evaluations = [self._evaluate(g, None) for g in self.state.population]

    def _signals(self) -> GenerationSignals:
        evals = self.state.evaluations
        objs = np.vstack([e.objectives() for e in evals])
        pts = objs[:, :2]  # (median return, -maxDD)
        feas = [e.is_feasible(self.targets) for e in evals]
        conc = self.lineage.lineage_concentration([g.genome_hash() for g in self.state.population])
        return GenerationSignals(
            generation=self.state.generation,
            hypervolume=hypervolume_2d(pts),
            feasibility_rate=float(np.mean(feas)),
            archive_coverage=self.archive.coverage,
            lineage_concentration=conc,
            min_violation=float(min(e.total_violation(self.targets) for e in evals)),
        )

    def _operator_pool(self) -> tuple[str, ...]:
        if self.state.forced_ops is not None:
            pool = self.state.forced_ops
            self.state.forced_ops = None
            return pool
        return OPERATOR_NAMES[:-1]  # crossover handled separately

    def _apply_intervention(self, intervention: Intervention) -> None:
        if intervention is Intervention.LOCAL_REFINEMENT:
            self.state.forced_ops = _LOCAL_OPS
        elif intervention is Intervention.NOVELTY_INCREASE:
            self.config.random_fraction = min(0.6, self.config.random_fraction * 2)
        elif intervention is Intervention.STRUCTURE_REPLACEMENT:
            self.state.forced_ops = _STRUCTURAL_OPS
        elif intervention is Intervention.ISLAND_RESTART:
            keep = self.archive.elites()
            keep_hashes = {e.genome_hash for e in keep}
            survivors = [
                g for g in self.state.population if g.genome_hash() in keep_hashes
            ]
            fresh = [
                random_genome(self.rng, self.config.symbol, generation=self.state.generation)
                for _ in range(self.config.population_size - len(survivors))
            ]
            self.state.population = (survivors + fresh)[: self.config.population_size]
            self.state.evaluations = [self._evaluate(g, None) for g in self.state.population]
        elif intervention is Intervention.NO_EDGE_FOUND:
            self.state.outcome = "NO_EDGE_FOUND"

    def _next_generation(self) -> None:
        cfg = self.config
        gen = self.state.generation
        ranked = rank_population(self.state.evaluations, self.targets)
        elite_count = max(2, cfg.population_size // 4)
        elites = [self.state.population[r.index] for r in ranked[:elite_count]]

        n_random = int(round(cfg.population_size * cfg.random_fraction))
        n_cross = int(round(cfg.population_size * cfg.crossover_fraction))
        n_mutants = cfg.population_size - len(elites) - n_random - n_cross

        children: list[tuple[StrategyGenome, str | None]] = [(g, None) for g in elites]
        pool = self._operator_pool()
        for _ in range(max(n_mutants, 0)):
            parent = elites[int(self.rng.integers(len(elites)))]
            op = str(self.rng.choice(pool))
            child, applied = mutate(parent, op, self.rng, gen)
            children.append((child, op if applied else None))
        for _ in range(n_cross):
            a = elites[int(self.rng.integers(len(elites)))]
            b = elites[int(self.rng.integers(len(elites)))]
            child, applied = crossover(a, b, self.rng, gen)
            children.append((child, "crossover" if applied else None))
        while len(children) < cfg.population_size:
            children.append(
                (random_genome(self.rng, cfg.symbol, generation=gen), None)
            )

        self.state.population = [c[0] for c in children[: cfg.population_size]]
        self.state.evaluations = [
            self._evaluate(g, op) for g, op in children[: cfg.population_size]
        ]

    # ------------------------------------------------------------------ #

    def run(self) -> IslandState:
        self._init_population()
        for gen in range(1, self.config.n_generations + 1):
            self.state.generation = gen
            self._next_generation()
            signals = self._signals()
            intervention = self.stagnation.observe(signals)
            self.state.telemetry.append(
                {
                    "generation": gen,
                    "hypervolume": signals.hypervolume,
                    "feasibility_rate": signals.feasibility_rate,
                    "archive_coverage": signals.archive_coverage,
                    "lineage_concentration": signals.lineage_concentration,
                    "intervention": intervention.value if intervention else None,
                    "n_evaluated": len(self.lineage),
                }
            )
            if intervention is not None:
                self._apply_intervention(intervention)
            if self.state.outcome == "NO_EDGE_FOUND":
                return self.state
        # Budget backstop: the compute budget IS the evidence threshold.
        # Exhausting it without a single feasible candidate ever appearing
        # is a NO_EDGE_FOUND verdict, not a neutral "ran out of time".
        any_feasible_ever = any(r.feasible for r in self.lineage.records)
        self.state.outcome = "BUDGET_EXHAUSTED" if any_feasible_ever else "NO_EDGE_FOUND"
        return self.state

    def champion(self) -> tuple[StrategyGenome, CandidateEvaluation] | None:
        """Best candidate under constraint dominance, or None if population empty."""
        if not self.state.evaluations:
            return None
        ranked = rank_population(self.state.evaluations, self.targets)
        best = ranked[0]
        return self.state.population[best.index], self.state.evaluations[best.index]
