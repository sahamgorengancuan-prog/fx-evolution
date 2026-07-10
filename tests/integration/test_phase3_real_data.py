"""Phase-3 acceptance on real BNBUSDT data.

Wires Phase 2 (typed compiled signals) into the Tier-A engine with
explicitly labeled cost assumptions, on a recent slice of the real file.
Asserts determinism, labeling, fail-closed cost behavior, and sane
mechanics — makes NO claim about strategy quality.
"""
from __future__ import annotations

import numpy as np
import pytest

from evoquant.backtest.costs import ExecutionAssumptions, resolve_costs
from evoquant.backtest.fast_engine import ENGINE_LABEL, run_backtest
from evoquant.backtest.metrics import compute_metrics
from evoquant.backtest.signals import RiskBlock, SignalPolicy
from evoquant.data.loaders import load_fsb_json
from evoquant.errors import MissingMetadataError
from evoquant.features.ast import Node
from evoquant.features.compiler import compile_feature
from tests.conftest import REAL_DATA

pytestmark = pytest.mark.skipif(
    not REAL_DATA.exists(), reason="real BNBUSDT data file not present"
)


def n(op: str, *children: Node, **kw) -> Node:
    return Node(op=op, children=tuple(children), **kw)


@pytest.fixture(scope="module")
def setup():
    full, spec, _ = load_fsb_json(REAL_DATA)
    bars = full.slice(60_000, 66_000)  # recent, non-quantized era

    close = n("close")
    macd_like = n("sub", n("ema", close, period=13), n("ema", close, period=55))
    entry_long = compile_feature(
        n(
            "cross_above",
            n("zscore", macd_like, period=89),
            Node(op="constant", value=1.0, unit="ZSCORE"),
        )
    )
    entry_short = compile_feature(
        n(
            "cross_below",
            n("zscore", macd_like, period=89),
            Node(op="constant", value=-1.0, unit="ZSCORE"),
        )
    )
    allow = compile_feature(
        n(
            "lt",
            Node(op="chop", period=144),
            Node(op="constant", value=0.55, unit="OSCILLATOR_0_1"),
        )
    )
    atr = compile_feature(Node(op="atr", period=14))

    policy = SignalPolicy(
        entry_long=np.asarray(entry_long.evaluate(bars), dtype=bool),
        entry_short=np.asarray(entry_short.evaluate(bars), dtype=bool),
        allow=np.asarray(allow.evaluate(bars), dtype=bool),
        atr=np.asarray(atr.evaluate(bars), dtype=np.float64),
        risk=RiskBlock(
            sl_atr=2.5,
            tp_atr=3.5,
            risk_per_trade=0.01,
            max_bars=48,
            cooldown_bars=4,
            atr_period=14,
        ),
        meta={"note": "phase3 acceptance strategy; no quality claim"},
    )
    assumptions = ExecutionAssumptions(
        label=(
            "ASSUMED (not historical): 10bps taker commission, 5-point "
            "constant spread, spot (no funding), Binance min notional 10 USDT"
        ),
        initial_equity=1_000_000.0,
        commission_pct_notional=0.001,
        assumed_spread_points=5.0,
        funding_mode="not_applicable_spot",
        min_notional=10.0,
        max_leverage=1.0,
    )
    costs = resolve_costs(spec, bars, assumptions)
    return bars, spec, policy, costs


def test_engine_runs_and_is_labeled_approximate(setup):
    bars, spec, policy, costs = setup
    result = run_backtest(bars, spec, policy, costs, 1_000_000.0)
    assert result.engine == ENGINE_LABEL == "BAR_APPROXIMATION"
    assert result.cost_provenance["spread_is_assumed"] is True
    assert "ASSUMED" in result.cost_provenance["label"]
    assert len(result.equity) == bars.n_bars
    assert np.all(np.isfinite(result.equity))
    assert len(result.trades) > 10  # strategy fires on this slice
    m = compute_metrics(result, timeframe_minutes=60, n_bars=bars.n_bars)
    assert m.n_trades == len(result.trades)
    assert 0.0 <= m.max_drawdown < 1.0
    assert m.total_costs_quote > 0.0


def test_real_run_is_deterministic(setup):
    bars, spec, policy, costs = setup
    a = run_backtest(bars, spec, policy, costs, 1_000_000.0)
    b = run_backtest(bars, spec, policy, costs, 1_000_000.0)
    assert a.trades_hash() == b.trades_hash()
    assert np.array_equal(a.equity, b.equity)


def test_costs_fail_closed_on_real_spec(setup):
    bars, spec, _, _ = setup
    # the real FSB spec has an untrusted spread and unknown funding/min_notional:
    with pytest.raises(MissingMetadataError):
        resolve_costs(
            spec,
            bars,
            ExecutionAssumptions(
                label="incomplete", initial_equity=1e6, commission_pct_notional=0.001,
                funding_mode="not_applicable_spot", min_notional=10.0,
                # no spread assumption, spec spread untrusted -> must raise
            ),
        )
    with pytest.raises(MissingMetadataError):
        resolve_costs(
            spec,
            bars,
            ExecutionAssumptions(
                label="incomplete", initial_equity=1e6, commission_pct_notional=0.001,
                assumed_spread_points=5.0, min_notional=10.0,
                # no funding decision -> must raise
            ),
        )


def test_every_trade_respects_contract_mechanics(setup):
    bars, spec, policy, costs = setup
    result = run_backtest(bars, spec, policy, costs, 1_000_000.0)
    for t in result.trades:
        assert t.exit_index >= t.entry_index
        assert t.exit_reason in ("sl", "tp", "max_bars", "end_of_data")
        # lots on the venue grid
        steps = round((t.lots - spec.min_lot) / spec.lot_step)
        assert abs(spec.min_lot + steps * spec.lot_step - t.lots) < 1e-9
        assert t.qty_base == pytest.approx(t.lots * spec.lot_size)
        assert t.qty_base * t.entry_price >= costs.min_notional
        # SL/TP on the correct side
        if t.direction == 1:
            assert t.sl_price < t.entry_price < t.tp_price
        else:
            assert t.tp_price < t.entry_price < t.sl_price
