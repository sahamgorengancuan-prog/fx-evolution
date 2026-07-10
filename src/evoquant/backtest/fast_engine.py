"""Tier A — fast bar-level engine. Label: ``BAR_APPROXIMATION``.

Guarantees (each backed by a Phase-3 test):

* **Causal**: signals from bar ``i`` execute at bar ``i+1`` open.
* **Mark-to-market**: equity is revalued at every bar close including open
  positions (audit D9 fix).
* **Bid/ask aware**: source prices are treated as mid; longs buy at ask,
  sell at bid (and vice versa) using the resolved spread series.
* **Gap-through-stop**: if a bar opens beyond SL/TP, the fill is the open
  price (first executable), never the stop level (audit B5 fix).
* **Conservative intrabar ambiguity**: when SL and TP are both touched in
  one bar, the SL is assumed to hit first. Tier A must never be more
  optimistic than the tick path (differential-tested against Tier B).
* **Contract-aware**: lot step, min lot, min notional, leverage cap.
* **Deterministic**: no randomness; identical inputs => identical trades.

Not modeled here (Tier B / KNOWN_LIMITATIONS): latency, rejection,
partial fills, margin/liquidation, session outages.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from evoquant.backtest.costs import ResolvedCosts, size_position
from evoquant.backtest.signals import PolicyError, SignalPolicy
from evoquant.data.contracts import InstrumentSpec
from evoquant.data.loaders import BarData

ENGINE_LABEL = "BAR_APPROXIMATION"

LONG = 1
SHORT = -1


@dataclass(frozen=True)
class Trade:
    direction: int  # +1 long, -1 short
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    qty_base: float
    lots: float
    sl_price: float
    tp_price: float
    pnl_quote: float  # net of all costs
    costs_quote: float  # commissions + funding (spread/slippage are in prices)
    exit_reason: str  # "sl" | "tp" | "max_bars" | "end_of_data"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    engine: str
    equity: np.ndarray  # per-bar mark-to-market equity, quote currency
    trades: tuple[Trade, ...]
    cost_provenance: dict[str, Any]
    skipped_entries: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def trades_hash(self) -> str:
        doc = [t.to_dict() for t in self.trades]
        return hashlib.sha256(
            json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "n_trades": len(self.trades),
            "final_equity": float(self.equity[-1]) if len(self.equity) else None,
            "trades_hash": self.trades_hash(),
            "cost_provenance": self.cost_provenance,
            "n_skipped_entries": len(self.skipped_entries),
        }


@dataclass
class _Position:
    direction: int
    entry_index: int
    entry_price: float
    qty_base: float
    lots: float
    sl_price: float
    tp_price: float
    costs_accrued: float  # commissions + funding so far


def run_backtest(
    bars: BarData,
    spec: InstrumentSpec,
    policy: SignalPolicy,
    costs: ResolvedCosts,
    initial_equity: float,
) -> BacktestResult:
    if policy.n_bars != bars.n_bars:
        raise PolicyError(
            "SignalPolicy length does not match bars",
            policy_bars=policy.n_bars,
            data_bars=bars.n_bars,
        )
    n = bars.n_bars
    o = np.asarray(bars.open, dtype=np.float64)
    h = np.asarray(bars.high, dtype=np.float64)
    l = np.asarray(bars.low, dtype=np.float64)
    c = np.asarray(bars.close, dtype=np.float64)

    risk = policy.risk
    bar_days = bars.timeframe_minutes / 1440.0
    slip = costs.slippage_quote()

    if initial_equity <= 0:
        raise PolicyError("initial_equity must be > 0", value=initial_equity)
    equity_curve = np.empty(n, dtype=np.float64)
    cash = float(initial_equity)
    pos: _Position | None = None
    pending_dir = 0  # signal awaiting next-open execution
    pending_signal_bar = -1
    cooldown_until = -1
    trades: list[Trade] = []
    skipped: list[dict[str, Any]] = []

    for i in range(n):
        hs = costs.half_spread_quote(i)

        # ---- 1. execute pending entry at this bar's open ---------------- #
        if pending_dir != 0 and pos is None and i > cooldown_until:
            atr_sig = float(policy.atr[pending_signal_bar])
            if np.isfinite(atr_sig) and atr_sig > 0.0:
                if pending_dir == LONG:
                    fill = o[i] + hs + slip  # buy at ask + slippage
                else:
                    fill = o[i] - hs - slip  # sell at bid - slippage
                stop_dist = risk.sl_atr * atr_sig
                sizing = size_position(
                    equity=cash,
                    risk_per_trade=risk.risk_per_trade,
                    stop_distance_quote=stop_dist,
                    entry_price=fill,
                    spec=spec,
                    costs=costs,
                )
                if sizing.lots > 0.0:
                    commission = sizing.notional_quote * costs.commission_pct
                    cash -= commission
                    if pending_dir == LONG:
                        sl = fill - stop_dist
                        tp = fill + risk.tp_atr * atr_sig
                    else:
                        sl = fill + stop_dist
                        tp = fill - risk.tp_atr * atr_sig
                    pos = _Position(
                        direction=pending_dir,
                        entry_index=i,
                        entry_price=fill,
                        qty_base=sizing.qty_base,
                        lots=sizing.lots,
                        sl_price=sl,
                        tp_price=tp,
                        costs_accrued=commission,
                    )
                else:
                    skipped.append(
                        {
                            "bar": i,
                            "direction": pending_dir,
                            "reason": sizing.reason_no_trade,
                        }
                    )
            else:
                skipped.append({"bar": i, "direction": pending_dir, "reason": "atr_invalid"})
        pending_dir = 0

        # ---- 2. manage open position over bar i ------------------------- #
        if pos is not None:
            exit_price: float | None = None
            exit_reason = ""
            if pos.direction == LONG:
                bid_open, bid_high, bid_low = o[i] - hs, h[i] - hs, l[i] - hs
                sl_hit = bid_low <= pos.sl_price
                tp_hit = bid_high >= pos.tp_price
                if sl_hit:  # worst case first: SL dominates TP ambiguity
                    exit_reason = "sl"
                    # gap through stop: first executable price is the open
                    exit_price = min(bid_open, pos.sl_price) - slip
                elif tp_hit:
                    exit_reason = "tp"
                    exit_price = (
                        bid_open if bid_open >= pos.tp_price else pos.tp_price
                    ) - slip
            else:
                ask_open, ask_high, ask_low = o[i] + hs, h[i] + hs, l[i] + hs
                sl_hit = ask_high >= pos.sl_price
                tp_hit = ask_low <= pos.tp_price
                if sl_hit:
                    exit_reason = "sl"
                    exit_price = max(ask_open, pos.sl_price) + slip
                elif tp_hit:
                    exit_reason = "tp"
                    exit_price = (
                        ask_open if ask_open <= pos.tp_price else pos.tp_price
                    ) + slip

            if exit_price is None and i - pos.entry_index >= risk.max_bars:
                exit_reason = "max_bars"
                exit_price = (c[i] - hs - slip) if pos.direction == LONG else (c[i] + hs + slip)
            if exit_price is None and i == n - 1:
                exit_reason = "end_of_data"
                exit_price = (c[i] - hs - slip) if pos.direction == LONG else (c[i] + hs + slip)

            # funding accrual for holding through this bar
            funding_points = (
                costs.funding_long_points_per_day
                if pos.direction == LONG
                else costs.funding_short_points_per_day
            )
            if funding_points != 0.0:
                funding_cost = funding_points * costs.tick_size * pos.qty_base * bar_days
                cash -= funding_cost
                pos.costs_accrued += funding_cost

            if exit_price is not None:
                notional_exit = abs(exit_price) * pos.qty_base
                commission = notional_exit * costs.commission_pct
                cash -= commission
                pos.costs_accrued += commission
                gross = (exit_price - pos.entry_price) * pos.qty_base * pos.direction
                cash += gross
                trades.append(
                    Trade(
                        direction=pos.direction,
                        entry_index=pos.entry_index,
                        exit_index=i,
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
                cooldown_until = i + risk.cooldown_bars
                pos = None

        # ---- 3. mark-to-market at bar close ------------------------------ #
        if pos is not None:
            mark = (c[i] - hs) if pos.direction == LONG else (c[i] + hs)
            unrealized = (mark - pos.entry_price) * pos.qty_base * pos.direction
            equity_curve[i] = cash + unrealized
        else:
            equity_curve[i] = cash

        # ---- 4. read new signals at bar close ---------------------------- #
        if pos is None and i < n - 1 and bool(policy.allow[i]) and i >= cooldown_until:
            want_long = bool(policy.entry_long[i])
            want_short = bool(policy.entry_short[i])
            if want_long != want_short:  # conflicting signals => no trade
                pending_dir = LONG if want_long else SHORT
                pending_signal_bar = i

    return BacktestResult(
        engine=ENGINE_LABEL,
        equity=equity_curve,
        trades=tuple(trades),
        cost_provenance=costs.provenance(),
        skipped_entries=tuple(skipped),
    )
