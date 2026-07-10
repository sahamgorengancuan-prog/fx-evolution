"""Tier B — chronological event/tick engine. Label: ``TICK_EVENT``.

Processes a strictly increasing bid/ask tick stream and models what Tier A
cannot: order latency, seeded rejection, and the *actual* intra-bar path
through SL/TP levels. Resting stop/take-profit orders trigger on the
correct side of the book (long exits on bid, short exits on ask) and gaps
fill at the first executable tick price beyond the level — never at the
level itself.

Signal timing matches Tier A: the per-bar :class:`SignalPolicy` arrays are
read when a bar completes (first tick at/after the bar boundary), and the
resulting market order becomes executable ``latency_s`` later.

Determinism: all randomness (rejection) flows from one seeded generator.

Real tick history for BNBUSDT is NOT attached to this repository; this
engine is exercised by synthetic-tick unit tests only and real-tick
validation remains PENDING (see VALIDATION_STATUS.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evoquant.backtest.costs import ResolvedCosts, size_position
from evoquant.backtest.fast_engine import LONG, SHORT, BacktestResult, Trade, _Position
from evoquant.backtest.signals import PolicyError, SignalPolicy
from evoquant.data.contracts import InstrumentSpec
from evoquant.errors import DataFormatError

ENGINE_LABEL = "TICK_EVENT"


@dataclass(frozen=True)
class TickData:
    """Chronological bid/ask ticks. Timestamps: UTC epoch seconds (float)."""

    ts_utc_s: np.ndarray
    bid: np.ndarray
    ask: np.ndarray

    def __post_init__(self) -> None:
        if not (len(self.ts_utc_s) == len(self.bid) == len(self.ask)):
            raise DataFormatError("TickData array length mismatch")
        if len(self.ts_utc_s) and np.any(np.diff(self.ts_utc_s) < 0):
            raise DataFormatError("TickData timestamps must be non-decreasing")
        if np.any(self.ask < self.bid):
            raise DataFormatError("TickData has ask < bid (crossed book)")

    @property
    def n_ticks(self) -> int:
        return len(self.ts_utc_s)


def run_event_backtest(
    ticks: TickData,
    bar_close_ts: np.ndarray,
    spec: InstrumentSpec,
    policy: SignalPolicy,
    costs: ResolvedCosts,
    initial_equity: float,
    *,
    latency_s: float = 0.0,
    rejection_prob: float = 0.0,
    seed: int = 0,
    timeframe_minutes: int = 60,
) -> BacktestResult:
    """Run the event engine.

    ``bar_close_ts[i]`` is the UTC epoch second at which bar ``i`` closes
    (i.e. its last included instant); signals of bar ``i`` become
    actionable at the first tick with ``ts >= bar_close_ts[i]``.
    Equity is marked at every bar completion (same grid as Tier A).
    """
    n_bars = len(bar_close_ts)
    if policy.n_bars != n_bars:
        raise PolicyError(
            "SignalPolicy length does not match bar grid",
            policy_bars=policy.n_bars,
            grid_bars=n_bars,
        )
    if initial_equity <= 0:
        raise PolicyError("initial_equity must be > 0", value=initial_equity)

    rng = np.random.default_rng(seed)
    risk = policy.risk
    bar_days = timeframe_minutes / 1440.0
    slip = costs.slippage_quote()

    cash = float(initial_equity)
    equity_curve = np.full(n_bars, np.nan)
    trades: list[Trade] = []
    skipped: list[dict[str, Any]] = []

    pos: _Position | None = None
    #: (direction, executable_from_ts, signal_bar) or None
    pending: tuple[int, float, int] | None = None
    cooldown_until_bar = -1
    next_bar = 0  # index of the next bar boundary to complete
    last_bid = np.nan
    last_ask = np.nan

    def _close_position(exit_price: float, exit_reason: str, bar_index: int) -> None:
        nonlocal cash, pos, cooldown_until_bar
        assert pos is not None
        commission = abs(exit_price) * pos.qty_base * costs.commission_pct
        cash -= commission
        pos.costs_accrued += commission
        gross = (exit_price - pos.entry_price) * pos.qty_base * pos.direction
        cash += gross
        trades.append(
            Trade(
                direction=pos.direction,
                entry_index=pos.entry_index,
                exit_index=bar_index,
                entry_price=pos.entry_price,
                exit_price=float(exit_price),
                qty_base=pos.qty_base,
                lots=pos.lots,
                sl_price=pos.sl_price,
                tp_price=pos.tp_price,
                pnl_quote=float(gross - pos.costs_accrued),
                costs_quote=float(pos.costs_accrued),
                exit_reason=exit_reason,
            )
        )
        cooldown_until_bar = bar_index + risk.cooldown_bars
        pos = None

    def _try_execute_pending(ts_now: float, bid_now: float, ask_now: float) -> None:
        nonlocal pending, pos, cash
        if pending is None or pos is not None or ts_now < pending[1]:
            return
        direction, _, signal_bar = pending
        pending = None
        if rejection_prob > 0.0 and rng.random() < rejection_prob:
            skipped.append({"bar": signal_bar, "direction": direction, "reason": "rejected"})
            return
        atr_sig = float(policy.atr[signal_bar])
        if not (np.isfinite(atr_sig) and atr_sig > 0.0):
            skipped.append(
                {"bar": signal_bar, "direction": direction, "reason": "atr_invalid"}
            )
            return
        fill = (ask_now + slip) if direction == LONG else (bid_now - slip)
        stop_dist = risk.sl_atr * atr_sig
        sizing = size_position(
            equity=cash,
            risk_per_trade=risk.risk_per_trade,
            stop_distance_quote=stop_dist,
            entry_price=fill,
            spec=spec,
            costs=costs,
        )
        if sizing.lots <= 0.0:
            skipped.append(
                {"bar": signal_bar, "direction": direction, "reason": sizing.reason_no_trade}
            )
            return
        commission = sizing.notional_quote * costs.commission_pct
        cash -= commission
        sl = fill - direction * stop_dist
        tp = fill + direction * risk.tp_atr * atr_sig
        pos = _Position(
            direction=direction,
            entry_index=min(signal_bar + 1, n_bars - 1),
            entry_price=fill,
            qty_base=sizing.qty_base,
            lots=sizing.lots,
            sl_price=sl,
            tp_price=tp,
            costs_accrued=commission,
        )

    for t in range(ticks.n_ticks):
        ts = float(ticks.ts_utc_s[t])
        bid = float(ticks.bid[t])
        ask = float(ticks.ask[t])
        last_bid, last_ask = bid, ask
        current_bar = min(next_bar, n_bars - 1)

        # ---- resting SL/TP orders (checked before new entries) --------- #
        if pos is not None:
            if pos.direction == LONG:
                if bid <= pos.sl_price:
                    _close_position(bid - slip, "sl", current_bar)
                elif bid >= pos.tp_price:
                    _close_position(bid - slip, "tp", current_bar)
            else:
                if ask >= pos.sl_price:
                    _close_position(ask + slip, "sl", current_bar)
                elif ask <= pos.tp_price:
                    _close_position(ask + slip, "tp", current_bar)

        # ---- pending market order after latency ------------------------- #
        _try_execute_pending(ts, bid, ask)

        # ---- bar completions --------------------------------------------- #
        while next_bar < n_bars and ts >= float(bar_close_ts[next_bar]):
            i = next_bar
            # funding + time-based exit at bar boundary
            if pos is not None:
                funding_points = (
                    costs.funding_long_points_per_day
                    if pos.direction == LONG
                    else costs.funding_short_points_per_day
                )
                if funding_points != 0.0:
                    funding_cost = (
                        funding_points * costs.tick_size * pos.qty_base * bar_days
                    )
                    cash -= funding_cost
                    pos.costs_accrued += funding_cost
                if i - pos.entry_index >= risk.max_bars:
                    px = (bid - slip) if pos.direction == LONG else (ask + slip)
                    _close_position(px, "max_bars", i)

            # mark-to-market on the bar grid
            if pos is not None:
                mark = bid if pos.direction == LONG else ask
                equity_curve[i] = cash + (mark - pos.entry_price) * pos.qty_base * pos.direction
            else:
                equity_curve[i] = cash

            # read signals of the completed bar
            if (
                pos is None
                and pending is None
                and i < n_bars - 1
                and bool(policy.allow[i])
                and i >= cooldown_until_bar
            ):
                want_long = bool(policy.entry_long[i])
                want_short = bool(policy.entry_short[i])
                if want_long != want_short:
                    direction = LONG if want_long else SHORT
                    pending = (direction, ts + latency_s, i)
            next_bar += 1

        # zero-latency orders placed at this bar boundary fill on this tick
        _try_execute_pending(ts, bid, ask)

    # ---- end of stream: force-close and fill trailing equity ------------- #
    if pos is not None and np.isfinite(last_bid):
        px = (last_bid - slip) if pos.direction == LONG else (last_ask + slip)
        _close_position(px, "end_of_data", min(next_bar, n_bars) - 1)
    prev = float(initial_equity)
    for i in range(n_bars):
        if np.isnan(equity_curve[i]):
            # bars beyond the tick stream reflect final cash; interior gaps
            # (should not occur with complete tick coverage) carry forward
            equity_curve[i] = cash if i >= next_bar else prev
        prev = float(equity_curve[i])

    return BacktestResult(
        engine=ENGINE_LABEL,
        equity=equity_curve,
        trades=tuple(trades),
        cost_provenance={**costs.provenance(), "latency_s": latency_s, "seed": seed},
        skipped_entries=tuple(skipped),
    )
