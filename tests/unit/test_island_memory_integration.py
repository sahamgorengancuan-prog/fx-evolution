"""Island + bandit + memory-store integration on cheap synthetic bars.

Synthetic data is unit-test-only per policy; this exercises the wiring,
not market behavior.
"""
from __future__ import annotations

from evoquant.backtest.costs import ExecutionAssumptions, resolve_costs
from evoquant.data.splits import IndexRange
from evoquant.genome.mutations import OPERATOR_NAMES
from evoquant.memory.bandit import ThompsonOperatorSelector
from evoquant.memory.store import MemoryStore
from evoquant.search.evaluator import EvalContext
from evoquant.search.island import IslandConfig, SearchIsland
from evoquant.search.objectives import TargetConfig
from tests.conftest import make_bars, make_spec

OPS = tuple(op for op in OPERATOR_NAMES if op != "crossover")

TARGETS = TargetConfig(
    min_trades_total=1,
    max_drawdown=0.9,
    min_profit_factor=0.0,
    min_median_fold_return=-1.0,
    max_fold_dispersion=10.0,
    max_complexity=500,
)


def _ctx() -> EvalContext:
    bars = make_bars(1_200, seed=77)
    spec = make_spec()
    costs = resolve_costs(
        spec,
        bars,
        ExecutionAssumptions(
            label="synthetic unit fixture",
            initial_equity=100_000.0,
            commission_pct_notional=0.001,
            assumed_spread_points=2.0,
            funding_mode="not_applicable_spot",
            min_notional=10.0,
        ),
    )
    return EvalContext(
        bars=bars,
        spec=spec,
        costs=costs,
        eval_ranges=(IndexRange(700, 950), IndexRange(950, 1_200)),
        initial_equity=100_000.0,
        warmup_bars=150,
    )


def _run(tmp_path, seed=11):
    memory = MemoryStore(tmp_path / f"mem_{seed}.sqlite")
    selector = ThompsonOperatorSelector(operators=OPS, seed=seed)
    island = SearchIsland(
        IslandConfig(
            symbol="TESTUSDT",
            population_size=6,
            n_generations=3,
            seed=seed,
            stagnation_patience=2,
        ),
        _ctx(),
        TARGETS,
        operator_selector=selector,
        memory=memory,
        context_key="lovol_choppy|all",
    )
    island.run()
    return island, memory, selector


def test_memory_receives_every_evaluation(tmp_path):
    island, memory, _ = _run(tmp_path)
    assert memory.count("candidate_evaluated") == len(island.lineage)
    for e in memory.events("candidate_evaluated"):
        assert e["context"] == "lovol_choppy|all"
        assert "failure_labels" in e and "objectives" in e


def test_bandit_posteriors_updated_from_real_rewards(tmp_path):
    island, memory, selector = _run(tmp_path)
    total_trials = sum(v[0] + v[1] for v in selector.counts.values())
    assert total_trials > 0, "bandit must receive updates from mutations"
    # replayed outcomes from the store match the live bandit counts
    replayed = memory.operator_outcomes()
    replay_trials = sum(v["trials"] for v in replayed.values())
    assert replay_trials == total_trials


def test_island_with_bandit_is_seed_deterministic(tmp_path):
    a, _, _ = _run(tmp_path / "a", seed=11)
    b, _, _ = _run(tmp_path / "b", seed=11)
    assert [r.genome_hash for r in a.lineage.records] == [
        r.genome_hash for r in b.lineage.records
    ]
    ca, cb = a.champion(), b.champion()
    assert ca is not None and cb is not None
    assert ca[0].genome_hash() == cb[0].genome_hash()


def test_bandit_replay_reconstructs_selector(tmp_path):
    _, memory, selector = _run(tmp_path)
    rebuilt = ThompsonOperatorSelector(operators=OPS, seed=0)
    for e in memory.events("candidate_evaluated"):
        if e.get("operator") and e.get("reward") is not None:
            rebuilt.update(e["context"], e["operator"], bool(e["reward"]))
    for key, counts in selector.counts.items():
        assert rebuilt.counts.get(key) == counts


def test_forced_escalation_pool_overrides_bandit(tmp_path):
    memory = MemoryStore(tmp_path / "mem.sqlite")
    selector = ThompsonOperatorSelector(operators=OPS, seed=5)
    # bias the bandit hard toward a structural operator
    for _ in range(200):
        selector.update("lovol_choppy|all", "replace_entry_long", True)
    island = SearchIsland(
        IslandConfig(symbol="TESTUSDT", population_size=6, n_generations=1, seed=5),
        _ctx(),
        TARGETS,
        operator_selector=selector,
        memory=memory,
        context_key="lovol_choppy|all",
    )
    island._init_population()
    island.state.generation = 1
    island.state.forced_ops = ("jitter_risk",)  # escalation rung override
    island._next_generation()
    ops_used = {
        r.operator
        for r in island.lineage.records
        if r.operator is not None and r.generation == 1
    }
    assert ops_used <= {"jitter_risk"}, "forced pool must outrank the bandit"
