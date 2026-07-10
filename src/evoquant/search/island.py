"""One pair-regime search island: the generational loop.

Wires together: typed genome generation/mutation (grammar-guided GP),
constraint-dominance NSGA-II survival, the MAP-Elites archive, ES-style
local parameter refinement gated on structural stability, the stagnation
controller's escalation ladder, and lineage recording for every evaluated
candidate. Deterministic under its seed.

Phase-6 integrations (both optional, injected): a contextual
Thompson-sampling operator selector (uniform draw when absent) and the
event-sourced MemoryStore, which receives a `candidate_evaluated` event —
context, operator, reward, failure labels — for every evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from evoquant.genome.genome import StrategyGenome, random_genome
from evoquant.genome.mutations import OPERATOR_NAMES, crossover, mutate
from evoquant.search.evaluator import EvalContext, evaluate_genome

if TYPE_CHECKING:  # avoid an import cycle at runtime
    from evoquant.memory.bandit import ThompsonOperatorSelector
    from evoquant.memory.store import MemoryStore
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
    def __init__(
        self,
        config: IslandConfig,
        ctx: EvalContext,
        targets: TargetConfig,
        *,
        operator_selector: ThompsonOperatorSelector | None = None,
        memory: MemoryStore | None = None,
        context_key: str = "anyfp|all",
    ) -> None:
        self.config = config
        self.ctx = ctx
        self.targets = targets
        self.rng = np.random.default_rng(config.seed)
        self.lineage = LineageStore()
        self.archive = MapElitesArchive(targets=targets)
        self.stagnation = StagnationController(patience=config.stagnation_patience)
        self.state = IslandState()
        self.selector = operator_selector
        self.memory = memory
        self.context_key = context_key
        self._eval_cache: dict[str, CandidateEvaluation] = {}

    # ------------------------------------------------------------------ #

    def _evaluate(
        self, genome: StrategyGenome, operator: str | None
    ) -> tuple[CandidateEvaluation, bool]:
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
        accepted = self.archive.try_insert(ev)
        return ev, accepted

    def _reward(
        self,
        parent_ev: CandidateEvaluation | None,
        child_ev: CandidateEvaluation,
        archive_accepted: bool,
    ) -> bool:
        """Operator reward: movement toward feasibility, a Pareto win over
        the parent, or a MAP-Elites cell win — never a raw scalar."""
        if parent_ev is None:
            return archive_accepted
        child_feas = child_ev.is_feasible(self.targets)
        parent_feas = parent_ev.is_feasible(self.targets)
        if child_feas and not parent_feas:
            return True
        if not child_feas and not parent_feas:
            improved = child_ev.total_violation(self.targets) < parent_ev.total_violation(
                self.targets
            ) * (1.0 - 1e-6)
            return improved or archive_accepted
        from evoquant.search.nsga import pareto_dominates

        return archive_accepted or bool(
            pareto_dominates(child_ev.objectives(), parent_ev.objectives())
        )

    def _record_event(
        self,
        genome: StrategyGenome,
        ev: CandidateEvaluation,
        operator: str | None,
        reward: bool | None,
    ) -> None:
        if self.memory is None:
            return
        from evoquant.memory.failures import failure_labels

        self.memory.append(
            "candidate_evaluated",
            {
                "genome_hash": genome.genome_hash(),
                "parent_hashes": list(genome.parent_hashes),
                "origin": genome.origin,
                "operator": operator,
                "context": self.context_key,
                "generation": self.state.generation,
                "feasible": ev.is_feasible(self.targets),
                "total_violation": ev.total_violation(self.targets),
                "objectives": [float(x) for x in ev.objectives()],
                "failure_labels": failure_labels(ev, self.targets),
                "reward": bool(reward) if reward is not None else None,
                "diagnostics": dict(ev.diagnostics),
            },
        )

    def _init_population(self) -> None:
        self.state.population = [
            random_genome(self.rng, self.config.symbol, generation=0)
            for _ in range(self.config.population_size)
        ]
        self.state.evaluations = []
        for g in self.state.population:
            ev, _ = self._evaluate(g, None)
            self._record_event(g, ev, None, None)
            self.state.evaluations.append(ev)

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
            self.state.evaluations = [
                self._evaluate(g, None)[0] for g in self.state.population
            ]
        elif intervention is Intervention.NO_EDGE_FOUND:
            self.state.outcome = "NO_EDGE_FOUND"

    def _next_generation(self) -> None:
        cfg = self.config
        gen = self.state.generation
        ranked = rank_population(self.state.evaluations, self.targets)
        elite_count = max(2, cfg.population_size // 4)
        elite_pairs = [
            (self.state.population[r.index], self.state.evaluations[r.index])
            for r in ranked[:elite_count]
        ]
        elites = [g for g, _ in elite_pairs]

        n_random = int(round(cfg.population_size * cfg.random_fraction))
        n_cross = int(round(cfg.population_size * cfg.crossover_fraction))
        n_mutants = cfg.population_size - len(elites) - n_random - n_cross

        forced = self.state.forced_ops is not None
        pool = self._operator_pool()  # note: consumes forced_ops
        children: list[tuple[StrategyGenome, str | None, CandidateEvaluation | None]] = [
            (g, None, ev) for g, ev in elite_pairs
        ]
        for _ in range(max(n_mutants, 0)):
            idx = int(self.rng.integers(len(elites)))
            parent, parent_ev = elite_pairs[idx]
            if self.selector is not None and not forced:
                op = self.selector.select(self.context_key)
                if op not in pool:
                    op = str(self.rng.choice(pool))
            else:
                op = str(self.rng.choice(pool))
            child, applied = mutate(parent, op, self.rng, gen)
            children.append((child, op if applied else None, parent_ev))
        for _ in range(n_cross):
            a = elites[int(self.rng.integers(len(elites)))]
            b = elites[int(self.rng.integers(len(elites)))]
            child, applied = crossover(a, b, self.rng, gen)
            children.append((child, "crossover" if applied else None, None))
        while len(children) < cfg.population_size:
            children.append(
                (random_genome(self.rng, cfg.symbol, generation=gen), None, None)
            )

        population: list[StrategyGenome] = []
        evaluations: list[CandidateEvaluation] = []
        for genome, child_op, child_parent_ev in children[: cfg.population_size]:
            ev, accepted = self._evaluate(genome, child_op)
            reward: bool | None = None
            if child_op is not None and child_op != "crossover":
                reward = self._reward(child_parent_ev, ev, accepted)
                if self.selector is not None:
                    self.selector.update(self.context_key, child_op, reward)
            self._record_event(genome, ev, child_op, reward)
            population.append(genome)
            evaluations.append(ev)
        self.state.population = population
        self.state.evaluations = evaluations

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
