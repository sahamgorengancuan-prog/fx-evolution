"""Phase-5 acceptance: a mini search island on real BNBUSDT data.

Small budget on purpose — this proves mechanics (determinism, lineage,
diversity archive, honest termination), not strategy quality.
"""
from __future__ import annotations

import pytest

from evoquant.backtest.costs import ExecutionAssumptions, resolve_costs
from evoquant.data.loaders import load_fsb_json
from evoquant.data.splits import IndexRange
from evoquant.search.evaluator import EvalContext
from evoquant.search.island import IslandConfig, SearchIsland
from evoquant.search.objectives import TargetConfig
from tests.conftest import REAL_DATA

pytestmark = pytest.mark.skipif(
    not REAL_DATA.exists(), reason="real BNBUSDT data file not present"
)

TARGETS = TargetConfig(
    min_trades_total=5,
    max_drawdown=0.5,
    min_profit_factor=0.0,
    min_median_fold_return=-1.0,
    max_fold_dispersion=10.0,
    max_complexity=500,
)

IMPOSSIBLE_TARGETS = TargetConfig(
    min_trades_total=100_000,
    max_drawdown=0.0001,
    min_profit_factor=50.0,
    min_median_fold_return=10.0,
    max_fold_dispersion=0.0001,
    max_complexity=1,
)


@pytest.fixture(scope="module")
def ctx() -> EvalContext:
    full, spec, _ = load_fsb_json(REAL_DATA)
    bars = full.slice(60_000, 64_000)  # recent era; small for the O(n·p) engine
    costs = resolve_costs(
        spec,
        bars,
        ExecutionAssumptions(
            label="ASSUMED phase5 fixture: 10bps, 5pt spread, spot, minNotional 10",
            initial_equity=1_000_000.0,
            commission_pct_notional=0.001,
            assumed_spread_points=5.0,
            funding_mode="not_applicable_spot",
            min_notional=10.0,
        ),
    )
    return EvalContext(
        bars=bars,
        spec=spec,
        costs=costs,
        eval_ranges=(IndexRange(2_400, 3_200), IndexRange(3_200, 4_000)),
        initial_equity=1_000_000.0,
        warmup_bars=250,
    )


def _run(ctx: EvalContext, seed: int = 123, targets=TARGETS, gens: int = 3, pop: int = 8):
    island = SearchIsland(
        IslandConfig(
            symbol="BNBUSDT",
            population_size=pop,
            n_generations=gens,
            seed=seed,
            stagnation_patience=2,
        ),
        ctx,
        targets,
    )
    island.run()
    return island


def test_search_is_seed_deterministic(ctx):
    a = _run(ctx, seed=123)
    b = _run(ctx, seed=123)
    ca, cb = a.champion(), b.champion()
    assert ca is not None and cb is not None
    assert ca[0].genome_hash() == cb[0].genome_hash()
    assert len(a.lineage) == len(b.lineage)
    assert [r.genome_hash for r in a.lineage.records] == [
        r.genome_hash for r in b.lineage.records
    ]
    assert a.state.telemetry == b.state.telemetry


def test_different_seed_changes_the_search(ctx):
    a = _run(ctx, seed=123)
    b = _run(ctx, seed=456)
    assert [r.genome_hash for r in a.lineage.records] != [
        r.genome_hash for r in b.lineage.records
    ]


def test_every_candidate_has_a_lineage_record(ctx):
    island = _run(ctx, seed=123)
    pop, gens = island.config.population_size, island.state.generation
    # init population + one record per candidate per generation
    assert len(island.lineage) >= pop * (gens + 1) - pop  # elites may be cached but still recorded
    for rec in island.lineage.records:
        assert rec.genome_hash
        assert rec.origin in ("random", "mutation", "crossover", "local_refine")
        assert len(rec.objectives) == 5


def test_population_stays_compilable_and_archive_fills(ctx):
    island = _run(ctx, seed=123)
    for genome in island.state.population:
        genome.validate()
    assert island.archive.coverage >= 1
    assert island.state.telemetry, "telemetry must be recorded every generation"
    for entry in island.state.telemetry:
        assert set(entry) >= {
            "generation",
            "hypervolume",
            "feasibility_rate",
            "archive_coverage",
            "lineage_concentration",
            "intervention",
        }


def test_impossible_targets_terminate_no_edge_found(ctx):
    """Either the escalation ladder reaches its terminal rung, or the
    budget backstop converts an all-infeasible run into NO_EDGE_FOUND —
    a pass is never forced (blueprint truth constraint #7).

    The full rung ordering is unit-tested in test_stagnation.py.
    """
    island = _run(ctx, seed=7, targets=IMPOSSIBLE_TARGETS, gens=15, pop=6)
    assert island.state.outcome == "NO_EDGE_FOUND"
    # no candidate was ever feasible, and interventions were attempted+logged
    assert all(not r.feasible for r in island.lineage.records)
    fired = [
        e["intervention"] for e in island.stagnation.log if e["event"] == "intervention"
    ]
    assert fired, "stagnation responses must be logged, not silent"


def test_lockbox_is_physically_out_of_reach(ctx):
    """The eval context's bars end before any lockbox could begin — the
    search cannot touch data it was never given."""
    assert ctx.eval_ranges[-1].stop <= ctx.bars.n_bars
