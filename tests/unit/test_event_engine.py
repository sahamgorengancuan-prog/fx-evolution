"""Tier B event engine tests + Tier A/B differential suite.

Synthetic ticks appear ONLY here (unit tests) — never in research paths.
Tick path per bar is O -> H -> L -> C, which for a long makes the tick
engine resolve SL/TP ambiguity in favor of TP; Tier A must then be the
conservative (less optimistic) engine.
"""
from __future__ import annotations

import numpy as np
import pytest

from evoquant.backtest.event_engine import ENGINE_LABEL as TICK_LABEL
from evoquant.backtest.event_engine import TickData, run_event_backtest
from evoquant.backtest.fast_engine import run_backtest
from evoquant.data.loaders import BarData
from evoquant.errors import DataFormatError
from tests.conftest import make_spec
from tests.unit.test_fast_engine import (
    HS,
    bars_from_ohlc,
    default_costs,
    flat_bars,
    make_policy,
)

SPEC = make_spec()
TF_S = 3600


def ticks_from_bars(bars: BarData, half_spread: float = HS) -> tuple[TickData, np.ndarray]:
    """4 ticks per bar (O,H,L,C) + a final boundary tick. Test-only synthetic."""
    ts_list: list[float] = []
    bid: list[float] = []
    ask: list[float] = []
    for i in range(bars.n_bars):
        t0 = float(bars.ts_utc_s[i])
        for k, px in enumerate((bars.open[i], bars.high[i], bars.low[i], bars.close[i])):
            ts_list.append(t0 + k * TF_S / 4)
            bid.append(float(px) - half_spread)
            ask.append(float(px) + half_spread)
    # boundary tick completing the last bar, at the last close price
    ts_list.append(float(bars.ts_utc_s[-1]) + TF_S)
    bid.append(float(bars.close[-1]) - half_spread)
    ask.append(float(bars.close[-1]) + half_spread)
    ticks = TickData(
        ts_utc_s=np.asarray(ts_list), bid=np.asarray(bid), ask=np.asarray(ask)
    )
    bar_close_ts = np.asarray(bars.ts_utc_s, dtype=np.float64) + TF_S
    return ticks, bar_close_ts


def run_tick(bars, policy, *, latency_s=0.0, rejection_prob=0.0, seed=1):
    ticks, grid = ticks_from_bars(bars)
    return run_event_backtest(
        ticks,
        grid,
        SPEC,
        policy,
        default_costs(bars),
        100_000.0,
        latency_s=latency_s,
        rejection_prob=rejection_prob,
        seed=seed,
        timeframe_minutes=60,
    )


class TestTickDataValidation:
    def test_crossed_book_rejected(self):
        with pytest.raises(DataFormatError):
            TickData(
                ts_utc_s=np.array([1.0, 2.0]),
                bid=np.array([100.0, 100.0]),
                ask=np.array([99.0, 100.1]),
            )

    def test_decreasing_timestamps_rejected(self):
        with pytest.raises(DataFormatError):
            TickData(
                ts_utc_s=np.array([2.0, 1.0]),
                bid=np.array([100.0, 100.0]),
                ask=np.array([100.1, 100.1]),
            )


class TestZeroLatencyParityWithTierA:
    def test_unambiguous_trade_matches_bar_engine(self):
        bars = flat_bars(20)
        policy = make_policy(20, long_bars=[5], max_bars=3)
        a = run_backtest(bars, SPEC, policy, default_costs(bars), 100_000.0)
        b = run_tick(bars, policy)
        assert b.engine == TICK_LABEL
        assert len(a.trades) == len(b.trades) == 1
        ta, tb = a.trades[0], b.trades[0]
        assert tb.entry_price == pytest.approx(ta.entry_price)  # next-bar open ask
        assert tb.exit_reason == ta.exit_reason == "max_bars"
        assert tb.pnl_quote == pytest.approx(ta.pnl_quote, abs=1e-9)
        assert b.equity[-1] == pytest.approx(a.equity[-1], abs=1e-6)

    def test_sl_only_trade_matches_bar_engine(self):
        # price falls steadily; SL hit unambiguously
        o = [100, 100, 100, 100, 98.5, 98, 98, 98]
        h = [100, 100, 100, 100, 98.5, 98, 98, 98]
        l = [100, 100, 100, 99, 98, 98, 98, 98]
        c = [100, 100, 100, 99, 98, 98, 98, 98]
        bars = bars_from_ohlc(o, h, l, c)
        policy = make_policy(8, long_bars=[2], sl_atr=1.5, tp_atr=50.0, max_bars=50)
        a = run_backtest(bars, SPEC, policy, default_costs(bars), 100_000.0)
        b = run_tick(bars, policy)
        assert len(a.trades) == len(b.trades) == 1
        assert a.trades[0].exit_reason == b.trades[0].exit_reason == "sl"


class TestConservativeInequality:
    def test_tier_a_never_more_optimistic_on_ambiguous_bar(self):
        # long entered bar 3 at 100.01; bar 4 touches TP (H first in tick
        # path) AND SL. Tick path O->H->L->C exits at TP; Tier A assumes SL.
        o = [100, 100, 100, 100, 100, 100]
        h = [100, 100, 100, 100, 104, 100]
        l = [100, 100, 100, 100, 97, 100]
        c = [100, 100, 100, 100, 100, 100]
        bars = bars_from_ohlc(o, h, l, c)
        policy = make_policy(6, long_bars=[2], sl_atr=2.0, tp_atr=3.0, max_bars=50)
        a = run_backtest(bars, SPEC, policy, default_costs(bars), 100_000.0)
        b = run_tick(bars, policy)
        assert a.trades[0].exit_reason == "sl"
        assert b.trades[0].exit_reason == "tp"
        assert a.trades[0].pnl_quote <= b.trades[0].pnl_quote


class TestLatencyAndRejection:
    def test_latency_shifts_fill_to_later_tick(self):
        # entry bar's tick path: O(0s)=100, H(900s)=103, L(1800s)=99, C=100
        o = [100, 100, 100, 100, 100]
        h = [100, 100, 100, 103, 100]
        l = [100, 100, 100, 99, 100]
        c = [100, 100, 100, 100, 100]
        bars = bars_from_ohlc(o, h, l, c)
        policy = make_policy(5, long_bars=[2], sl_atr=50.0, tp_atr=60.0, max_bars=50)
        instant = run_tick(bars, policy, latency_s=0.0)
        delayed = run_tick(bars, policy, latency_s=1000.0)
        # zero latency fills at bar-3 open ask (100.01); 1000s latency skips
        # the 900s tick and fills at the 1800s tick (low 99 -> ask 99.01)
        assert instant.trades[0].entry_price == pytest.approx(100.01)
        assert delayed.trades[0].entry_price == pytest.approx(99.01)

    def test_full_rejection_produces_no_trades(self):
        bars = flat_bars(20)
        policy = make_policy(20, long_bars=[5], max_bars=3)
        result = run_tick(bars, policy, rejection_prob=0.999999, seed=7)
        assert len(result.trades) == 0
        assert any(s["reason"] == "rejected" for s in result.skipped_entries)

    def test_rejection_is_seed_deterministic(self):
        bars = flat_bars(200)
        policy = make_policy(200, long_bars=list(range(0, 200, 10)), max_bars=2)
        r1 = run_tick(bars, policy, rejection_prob=0.5, seed=42)
        r2 = run_tick(bars, policy, rejection_prob=0.5, seed=42)
        assert r1.trades_hash() == r2.trades_hash()


class TestGapThroughStopOnTicks:
    def test_stop_fills_at_first_tick_beyond_level(self):
        # bar 4 gaps: its open tick is already far below the stop
        o = [100, 100, 100, 100, 93, 93]
        h = [100, 100, 100, 100, 93, 93]
        l = [100, 100, 100, 100, 92, 93]
        c = [100, 100, 100, 100, 93, 93]
        bars = bars_from_ohlc(o, h, l, c)
        policy = make_policy(6, long_bars=[2], sl_atr=2.0, tp_atr=50.0, max_bars=50)
        b = run_tick(bars, policy)
        t = b.trades[0]
        assert t.exit_reason == "sl"
        assert t.sl_price == pytest.approx(98.01)
        assert t.exit_price == pytest.approx(93.0 - HS)  # first executable bid
        assert t.exit_price < t.sl_price


class TestMarkToMarketGrid:
    def test_equity_marked_on_bar_grid(self):
        bars = flat_bars(20)
        policy = make_policy(20, long_bars=[5], max_bars=3)
        b = run_tick(bars, policy)
        assert len(b.equity) == 20
        assert np.all(np.isfinite(b.equity))
        # open position bars reflect unrealized PnL + entry commission
        assert b.equity[6] < 100_000.0
