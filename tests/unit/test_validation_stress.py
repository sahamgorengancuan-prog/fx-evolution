from __future__ import annotations

import numpy as np

from evoquant.backtest.costs import ExecutionAssumptions, resolve_costs
from evoquant.backtest.signals import RiskBlock
from evoquant.data.splits import IndexRange
from evoquant.features.ast import Node
from evoquant.genome.genome import StrategyGenome
from evoquant.search.evaluator import EvalContext
from evoquant.validation.stress import (
    cost_stress,
    parameter_perturbation,
    start_offset_stress,
)
from tests.conftest import make_bars, make_spec


def _n(op: str, *children: Node, **kw) -> Node:
    return Node(op=op, children=tuple(children), **kw)


def _genome() -> StrategyGenome:
    close = _n("close")
    # frequent entries; huge SL/TP so exits are max_bars-only => the trade
    # list is identical across cost multipliers (isolates pure cost effect)
    entry_long = _n("cross_above", close, _n("sma", close, period=5))
    entry_short = _n("cross_below", close, _n("sma", close, period=5))
    allow = _n(
        "gt",
        _n("rank", close, period=8),
        Node(op="constant", value=0.0, unit="OSCILLATOR_0_1"),
    )
    return StrategyGenome(
        symbol="TESTUSDT",
        entry_long=entry_long,
        entry_short=entry_short,
        allow=allow,
        risk=RiskBlock(
            sl_atr=6.0,
            tp_atr=6.0,
            risk_per_trade=0.01,
            max_bars=12,
            cooldown_bars=2,
            atr_period=14,
        ),
    )


def _ctx() -> EvalContext:
    bars = make_bars(1_200, seed=99)
    spec = make_spec()
    costs = resolve_costs(
        spec,
        bars,
        ExecutionAssumptions(
            label="stress fixture (synthetic, unit test only)",
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


def test_cost_stress_erodes_returns_monotonically():
    report = cost_stress(_genome(), _ctx(), spread_multipliers=(1.0, 2.0, 4.0))
    assert report["kind"] == "cost_stress"
    medians = [row["median_fold_return"] for row in report["rows"]]
    trades = [row["n_trades"] for row in report["rows"]]
    assert trades[0] > 5, "fixture must actually trade"
    assert trades == sorted(trades, reverse=True) or len(set(trades)) == 1
    # wider spreads can only hurt when the trade list is cost-independent
    assert medians[0] >= medians[1] >= medians[2]
    assert report["return_erosion"] >= 0.0


def test_parameter_perturbation_reports_dispersion_and_is_seeded():
    a = parameter_perturbation(_genome(), _ctx(), n_perturbations=4, seed=5)
    b = parameter_perturbation(_genome(), _ctx(), n_perturbations=4, seed=5)
    assert a == b  # deterministic under seed
    assert a["kind"] == "parameter_perturbation"
    assert a["perturbed_min"] <= a["perturbed_median"] <= a["perturbed_max"]
    assert a["dispersion"] >= 0.0
    assert a["n_perturbations"] == 4


def test_start_offset_stress_shape():
    report = start_offset_stress(_genome(), _ctx(), offsets=(0, 12, 24))
    assert report["kind"] == "start_offset_stress"
    assert len(report["rows"]) == 3
    assert report["dispersion"] >= 0.0
    offsets = [row["offset_bars"] for row in report["rows"]]
    assert offsets == [0.0, 12.0, 24.0]


def test_stress_reports_are_json_safe():
    import json

    report = cost_stress(_genome(), _ctx(), spread_multipliers=(1.0, 2.0))
    json.dumps(report)  # must not raise (artifact requirement)
    report2 = start_offset_stress(_genome(), _ctx(), offsets=(0, 12))
    json.dumps(report2)
    report3 = parameter_perturbation(_genome(), _ctx(), n_perturbations=2, seed=1)
    json.dumps(report3)


def test_numpy_free_of_nan_leakage():
    report = cost_stress(_genome(), _ctx(), spread_multipliers=(1.0,))
    assert np.isfinite(report["rows"][0]["median_fold_return"])
