"""Tier A engine acceptance tests, mostly hand-computed fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from evoquant.backtest.costs import ExecutionAssumptions, resolve_costs
from evoquant.backtest.fast_engine import ENGINE_LABEL, run_backtest
from evoquant.backtest.metrics import compute_metrics, max_drawdown
from evoquant.backtest.signals import PolicyError, RiskBlock, SignalPolicy
from evoquant.data.loaders import BarData
from tests.conftest import make_spec

TICK = 0.01
HS = 0.01  # half spread: 2 points * 0.01 / 2

SPEC = make_spec()


def bars_from_ohlc(o, h, l, c) -> BarData:
    n = len(o)
    ts = (np.arange(n, dtype=np.int64) * 3600) + 1_600_000_000
    return BarData(
        symbol="TESTUSDT",
        timeframe_minutes=60,
        ts_utc_s=ts,
        open=np.asarray(o, dtype=np.float64),
        high=np.asarray(h, dtype=np.float64),
        low=np.asarray(l, dtype=np.float64),
        close=np.asarray(c, dtype=np.float64),
        volume=np.ones(n),
        spread_points=None,
    )


def flat_bars(n: int, price: float = 100.0) -> BarData:
    p = [price] * n
    return bars_from_ohlc(p, p, p, p)


def make_policy(
    n: int,
    long_bars=(),
    short_bars=(),
    *,
    sl_atr=2.0,
    tp_atr=3.0,
    risk=0.01,
    max_bars=3,
    cooldown=0,
    atr_value=1.0,
    allow=None,
) -> SignalPolicy:
    entry_long = np.zeros(n, dtype=bool)
    entry_long[list(long_bars)] = True
    entry_short = np.zeros(n, dtype=bool)
    entry_short[list(short_bars)] = True
    return SignalPolicy(
        entry_long=entry_long,
        entry_short=entry_short,
        allow=np.ones(n, dtype=bool) if allow is None else allow,
        atr=np.full(n, atr_value, dtype=np.float64),
        risk=RiskBlock(
            sl_atr=sl_atr,
            tp_atr=tp_atr,
            risk_per_trade=risk,
            max_bars=max_bars,
            cooldown_bars=cooldown,
            atr_period=14,
        ),
    )


def default_costs(bars: BarData, **overrides):
    base = dict(
        label="fast-engine unit fixture",
        initial_equity=100_000.0,
        commission_pct_notional=0.001,
        assumed_spread_points=2.0,
        funding_mode="not_applicable_spot",
        min_notional=10.0,
    )
    base.update(overrides)
    return resolve_costs(SPEC, bars, ExecutionAssumptions(**base))


def run(bars, policy, costs=None, equity=100_000.0):
    return run_backtest(bars, SPEC, policy, costs or default_costs(bars), equity)


class TestHandComputedTrade:
    def test_exact_pnl_and_costs(self):
        bars = flat_bars(20)
        policy = make_policy(20, long_bars=[5], max_bars=3)
        result = run(bars, policy)

        assert result.engine == ENGINE_LABEL
        assert len(result.trades) == 1
        t = result.trades[0]
        assert t.entry_index == 6  # signal bar 5 executes at next open (causal)
        assert t.entry_price == pytest.approx(100.01)  # ask = 100 + half spread
        assert t.exit_index == 9  # max_bars = 3
        assert t.exit_reason == "max_bars"
        assert t.exit_price == pytest.approx(99.99)  # bid close
        assert t.qty_base == pytest.approx(500.0)  # 1% of 100k / (2 * ATR 1.0)
        assert t.lots == pytest.approx(0.5)
        # costs: entry 50_005*0.001 + exit 49_995*0.001 = 100.0
        assert t.costs_quote == pytest.approx(100.0)
        # gross = (99.99 - 100.01) * 500 = -10; net = -110
        assert t.pnl_quote == pytest.approx(-110.0)
        assert result.equity[-1] == pytest.approx(100_000.0 - 110.0)

    def test_mark_to_market_during_open_position(self):
        bars = flat_bars(20)
        policy = make_policy(20, long_bars=[5], max_bars=3)
        result = run(bars, policy)
        # bar 6 close: cash 100k - 50.005 commission, unrealized -10
        assert result.equity[6] == pytest.approx(100_000.0 - 50.005 - 10.0)


class TestMtmExcursion:
    def test_equity_dips_while_winning_trade_is_open(self):
        # entry at bar 3 open (100.01); dip to 97 for 3 bars; then rally to TP
        o = [100, 100, 100, 100, 97, 97, 106, 100, 100, 100]
        h = [100, 100, 100, 100, 97, 97, 106, 106, 106, 106]
        l = [100, 100, 100, 96, 96, 96, 100, 100, 100, 100]
        c = [100, 100, 100, 97, 97, 97, 106, 106, 106, 106]
        bars = bars_from_ohlc(o, h, l, c)
        policy = make_policy(10, long_bars=[2], sl_atr=8.0, tp_atr=4.0, max_bars=8)
        result = run(bars, policy)
        assert len(result.trades) == 1
        t = result.trades[0]
        assert t.exit_reason == "tp"
        assert t.pnl_quote > 0
        # audit D9 fix: the drawdown while holding is visible in equity
        assert np.min(result.equity) < 100_000.0 - 100.0
        assert max_drawdown(result.equity) > 0.001


class TestGapThroughStop:
    def test_gap_fills_at_open_not_at_stop(self):
        # long from bar 3 open at 100.01, SL = 98.01; bar 5 gaps to open 95
        o = [100, 100, 100, 100, 100, 95, 95, 95]
        h = [100, 100, 100, 100, 100, 95, 95, 95]
        l = [100, 100, 100, 100, 100, 94, 95, 95]
        c = [100, 100, 100, 100, 100, 95, 95, 95]
        bars = bars_from_ohlc(o, h, l, c)
        policy = make_policy(8, long_bars=[2], sl_atr=1.0, tp_atr=50.0, max_bars=50)
        result = run(bars, policy)
        t = result.trades[0]
        assert t.exit_reason == "sl"
        assert t.sl_price == pytest.approx(99.01)
        # filled at first executable price (bid open 94.99), NOT the stop level
        assert t.exit_price == pytest.approx(94.99)
        assert t.exit_price < t.sl_price


class TestWorstCaseOrdering:
    def test_sl_and_tp_same_bar_takes_sl(self):
        # long from bar 3 open (100.01): SL 98.01, TP 103.01; bar 4 spans both
        o = [100, 100, 100, 100, 100, 100]
        h = [100, 100, 100, 100, 104, 100]
        l = [100, 100, 100, 100, 97, 100]
        c = [100, 100, 100, 100, 100, 100]
        bars = bars_from_ohlc(o, h, l, c)
        policy = make_policy(6, long_bars=[2], sl_atr=2.0, tp_atr=3.0, max_bars=50)
        result = run(bars, policy)
        t = result.trades[0]
        assert t.exit_reason == "sl"  # conservative: never assume TP first
        assert t.exit_price == pytest.approx(t.sl_price)


class TestShortSide:
    def test_short_trade_prices_use_correct_book_side(self):
        bars = flat_bars(20)
        policy = make_policy(20, short_bars=[5], max_bars=3)
        result = run(bars, policy)
        t = result.trades[0]
        assert t.direction == -1
        assert t.entry_price == pytest.approx(99.99)  # sell at bid
        assert t.exit_price == pytest.approx(100.01)  # buy back at ask
        assert t.sl_price == pytest.approx(101.99)
        assert t.tp_price == pytest.approx(96.99)


class TestGuards:
    def test_conflicting_signals_produce_no_trade(self):
        bars = flat_bars(10)
        policy = make_policy(10, long_bars=[4], short_bars=[4])
        assert len(run(bars, policy).trades) == 0

    def test_allow_gate_blocks_entries(self):
        bars = flat_bars(10)
        allow = np.zeros(10, dtype=bool)
        policy = make_policy(10, long_bars=[4], allow=allow)
        assert len(run(bars, policy).trades) == 0

    def test_cooldown_blocks_immediate_reentry(self):
        bars = flat_bars(30)
        policy = make_policy(30, long_bars=list(range(30)), max_bars=2, cooldown=10)
        result = run(bars, policy)
        assert len(result.trades) >= 2
        first, second = result.trades[0], result.trades[1]
        assert second.entry_index - first.exit_index > 10

    def test_nan_atr_skips_entry(self):
        bars = flat_bars(10)
        policy = make_policy(10, long_bars=[4])
        atr = np.array(policy.atr)
        atr[4] = np.nan
        policy = SignalPolicy(
            entry_long=policy.entry_long,
            entry_short=policy.entry_short,
            allow=policy.allow,
            atr=atr,
            risk=policy.risk,
        )
        result = run(bars, policy)
        assert len(result.trades) == 0
        assert any(s["reason"] == "atr_invalid" for s in result.skipped_entries)

    def test_end_of_data_force_close(self):
        bars = flat_bars(10)
        policy = make_policy(10, long_bars=[7], max_bars=50)
        result = run(bars, policy)
        assert result.trades[0].exit_reason == "end_of_data"
        assert result.trades[0].exit_index == 9

    def test_policy_length_mismatch_raises(self):
        bars = flat_bars(10)
        policy = make_policy(12, long_bars=[4])
        with pytest.raises(PolicyError):
            run(bars, policy)


class TestFundingAndDeterminism:
    def test_funding_accrues_per_bar_held(self):
        import dataclasses

        # convention: positive swap points = holder PAYS (cost-positive)
        funded_spec = dataclasses.replace(
            SPEC, swap_long=24.0, swap_short=0.0, swap_mode="points_per_day"
        )
        bars = flat_bars(20)
        policy = make_policy(20, long_bars=[5], max_bars=4)
        costs = resolve_costs(
            funded_spec,
            bars,
            ExecutionAssumptions(
                label="funding fixture",
                initial_equity=100_000.0,
                commission_pct_notional=0.0,
                assumed_spread_points=2.0,
                funding_mode="points_per_day_from_spec",
                min_notional=10.0,
            ),
        )
        result = run_backtest(bars, funded_spec, policy, costs, 100_000.0)
        t = result.trades[0]
        # held bars 6..10 (5 accruals) at -24 pts/day * 0.01 * 500 qty * (1/24 day)
        expected_funding = 24.0 * 0.01 * 500.0 * (1.0 / 24.0) * 5
        assert t.costs_quote == pytest.approx(expected_funding)

    def test_deterministic_trade_hash(self):
        bars = flat_bars(50)
        policy = make_policy(50, long_bars=[5, 20, 35], max_bars=3)
        a = run(bars, policy)
        b = run(bars, policy)
        assert a.trades_hash() == b.trades_hash()
        assert np.array_equal(a.equity, b.equity)


class TestMetrics:
    def test_max_drawdown_hand_computed(self):
        eq = np.array([100.0, 110.0, 99.0, 120.0, 90.0])
        assert max_drawdown(eq) == pytest.approx(0.25)  # 120 -> 90

    def test_metrics_shape(self):
        bars = flat_bars(50)
        policy = make_policy(50, long_bars=[5, 20, 35], max_bars=3)
        result = run(bars, policy)
        m = compute_metrics(result, timeframe_minutes=60, n_bars=50)
        assert m.engine == ENGINE_LABEL
        assert m.n_trades == len(result.trades)
        assert 0 <= m.exposure <= 1
        assert m.total_costs_quote > 0
