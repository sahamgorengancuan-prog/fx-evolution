from __future__ import annotations

from evoquant.search.map_elites import (
    DESCRIPTOR_AXES,
    MapElitesArchive,
    behavior_descriptor,
    descriptor_cell,
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


def ev(ret=0.05, dd=0.1, pf=1.5, trades=50, freq=0.05, hold=10.0, long_share=0.5, wr=0.6, tag="x"):
    return CandidateEvaluation(
        genome_hash=tag,
        fold_returns=(ret, ret),
        n_trades_total=trades,
        max_drawdown=dd,
        profit_factor=pf,
        complexity=20,
        diagnostics={
            "trade_frequency": freq,
            "avg_holding_bars": hold,
            "long_share": long_share,
            "win_rate": wr,
        },
    )


def test_descriptor_cell_binning_and_clipping():
    d = behavior_descriptor(ev(freq=0.05, hold=10.0, long_share=0.5, wr=0.6))
    cell = descriptor_cell(d)
    assert len(cell) == len(DESCRIPTOR_AXES)
    # out-of-range values clip into the last bin, never overflow
    extreme = descriptor_cell((99.0, 9_999.0, 5.0, 2.0))
    for idx, (_, _, _, bins) in zip(extreme, DESCRIPTOR_AXES, strict=True):
        assert 0 <= idx < bins


def test_insert_new_cell_accepts():
    archive = MapElitesArchive(targets=TARGETS)
    assert archive.try_insert(ev(tag="a"))
    assert archive.coverage == 1


def test_same_cell_replacement_requires_constraint_dominance():
    archive = MapElitesArchive(targets=TARGETS)
    incumbent = ev(ret=0.05, dd=0.1, tag="incumbent")
    assert archive.try_insert(incumbent)

    # identical behavior cell, strictly worse objectives -> rejected
    worse = ev(ret=0.01, dd=0.2, pf=1.1, tag="worse")
    assert not archive.try_insert(worse)
    assert archive.cells[descriptor_cell(behavior_descriptor(incumbent))].genome_hash == "incumbent"

    # strictly better on every objective -> replaces
    better = ev(ret=0.10, dd=0.05, pf=2.0, tag="better")
    assert archive.try_insert(better)
    assert archive.cells[descriptor_cell(behavior_descriptor(better))].genome_hash == "better"


def test_feasible_replaces_infeasible_in_cell():
    archive = MapElitesArchive(targets=TARGETS)
    infeasible = ev(ret=1.0, dd=0.9, tag="infeasible")  # dd breaches target
    feasible = ev(ret=0.01, dd=0.1, tag="feasible")
    assert archive.try_insert(infeasible)
    assert archive.try_insert(feasible)
    cell = descriptor_cell(behavior_descriptor(feasible))
    assert archive.cells[cell].genome_hash == "feasible"


def test_coverage_grows_with_behavioral_diversity():
    archive = MapElitesArchive(targets=TARGETS)
    archive.try_insert(ev(freq=0.01, hold=5.0, tag="a"))
    archive.try_insert(ev(freq=0.15, hold=5.0, tag="b"))
    archive.try_insert(ev(freq=0.01, hold=55.0, tag="c"))
    archive.try_insert(ev(freq=0.15, hold=55.0, long_share=0.9, tag="d"))
    assert archive.coverage == 4
    assert 0 < archive.coverage_fraction() < 1
    assert archive.summary()["accepted"] == 4
