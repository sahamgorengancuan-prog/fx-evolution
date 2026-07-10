from __future__ import annotations

import numpy as np

from evoquant.search.nsga import (
    constraint_dominates,
    crowding_distance,
    fast_nondominated_sort,
    pareto_dominates,
    rank_population,
    select_survivors,
)
from evoquant.search.objectives import CandidateEvaluation, TargetConfig

TARGETS = TargetConfig(
    min_trades_total=10,
    max_drawdown=0.3,
    min_profit_factor=1.0,
    min_median_fold_return=0.0,
    max_fold_dispersion=1.0,
    max_complexity=1000,
)


def ev(ret=0.05, dd=0.1, pf=1.5, trades=50, complexity=20, tag="x") -> CandidateEvaluation:
    return CandidateEvaluation(
        genome_hash=tag,
        fold_returns=(ret, ret),
        n_trades_total=trades,
        max_drawdown=dd,
        profit_factor=pf,
        complexity=complexity,
    )


def test_pareto_dominates():
    assert pareto_dominates(np.array([1.0, 1.0]), np.array([0.5, 1.0]))
    assert not pareto_dominates(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert not pareto_dominates(np.array([1.0, 1.0]), np.array([1.0, 1.0]))


def test_feasibility_and_violations():
    good = ev()
    assert good.is_feasible(TARGETS)
    bad = ev(trades=3, dd=0.5, pf=0.4)
    v = bad.violations(TARGETS)
    assert {"min_trades_total", "max_drawdown", "min_profit_factor"} <= set(v)
    assert bad.total_violation(TARGETS) > 0


def test_feasible_always_beats_infeasible():
    # infeasible candidate has spectacular objectives; feasible modest ones
    stellar_but_infeasible = ev(ret=5.0, dd=0.05, pf=9.0, trades=3, tag="a")
    modest_feasible = ev(ret=0.01, dd=0.2, pf=1.1, trades=50, tag="b")
    assert constraint_dominates(modest_feasible, stellar_but_infeasible, TARGETS)
    assert not constraint_dominates(stellar_but_infeasible, modest_feasible, TARGETS)

    ranked = rank_population([stellar_but_infeasible, modest_feasible], TARGETS)
    assert ranked[0].index == 1 and ranked[0].feasible


def test_infeasible_ranked_by_violation():
    worse = ev(trades=0, tag="worse")
    better = ev(trades=8, tag="better")  # closer to min_trades=10
    ranked = rank_population([worse, better], TARGETS)
    assert ranked[0].index == 1


def test_nondominated_sort_on_planted_fronts():
    objs = np.array(
        [
            [1.0, 1.0],  # front 0
            [0.9, 1.1],  # front 0
            [0.5, 0.5],  # front 1 (dominated by [1,1])
            [0.1, 0.1],  # front 2
        ]
    )
    fronts = fast_nondominated_sort(objs)
    assert sorted(fronts[0]) == [0, 1]
    assert fronts[1] == [2]
    assert fronts[2] == [3]


def test_crowding_extremes_infinite():
    objs = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    crowd = crowding_distance(objs, [0, 1, 2])
    assert crowd[0] == float("inf") and crowd[2] == float("inf")
    assert np.isfinite(crowd[1])


def test_selection_deterministic_and_bounded():
    pop = [ev(ret=0.01 * i, tag=str(i)) for i in range(8)]
    a = select_survivors(pop, TARGETS, 4)
    b = select_survivors(pop, TARGETS, 4)
    assert a == b and len(a) == 4


def test_pf_cap_prevents_tiny_sample_domination():
    lucky = ev(ret=0.02, pf=500.0, trades=50, tag="lucky")
    solid = ev(ret=0.02, pf=11.0, trades=50, tag="solid")
    assert lucky.objectives()[2] == solid.objectives()[2] == 10.0
