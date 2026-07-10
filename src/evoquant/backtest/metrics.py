"""Performance metrics from mark-to-market equity and trade lists.

Plain, correctly named quantities only. Anything inferential (deflated
Sharpe, bootstrap CIs, PBO) belongs to the Phase-7 validation framework —
deliberately not here, so a raw backtest report can never masquerade as a
statistically validated result.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from evoquant.backtest.fast_engine import BacktestResult

MINUTES_PER_YEAR = 525_600.0


@dataclass(frozen=True)
class BacktestMetrics:
    engine: str
    n_trades: int
    total_return: float
    max_drawdown: float  # fraction of peak equity, from MTM curve
    sharpe_annualized: float | None
    profit_factor: float | None
    win_rate: float | None
    exposure: float  # fraction of bars with an open position (approx: from trades)
    total_costs_quote: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    return float(np.max(dd)) if len(dd) else 0.0


def compute_metrics(
    result: BacktestResult, timeframe_minutes: int, n_bars: int
) -> BacktestMetrics:
    eq = np.asarray(result.equity, dtype=np.float64)
    rets = np.diff(eq) / eq[:-1]
    rets = rets[np.isfinite(rets)]

    sharpe: float | None = None
    if len(rets) > 1:
        sd = float(np.std(rets, ddof=1))
        if sd > 0:
            bars_per_year = MINUTES_PER_YEAR / timeframe_minutes
            sharpe = float(np.mean(rets)) / sd * math.sqrt(bars_per_year)

    wins = [t.pnl_quote for t in result.trades if t.pnl_quote > 0]
    losses = [-t.pnl_quote for t in result.trades if t.pnl_quote < 0]
    pf: float | None = None
    if losses and sum(losses) > 0:
        pf = float(sum(wins) / sum(losses)) if wins else 0.0
    win_rate = (len(wins) / len(result.trades)) if result.trades else None

    bars_in_position = sum(t.exit_index - t.entry_index + 1 for t in result.trades)

    return BacktestMetrics(
        engine=result.engine,
        n_trades=len(result.trades),
        total_return=float(eq[-1] / eq[0] - 1.0) if len(eq) > 1 and eq[0] > 0 else 0.0,
        max_drawdown=max_drawdown(eq),
        sharpe_annualized=sharpe,
        profit_factor=pf,
        win_rate=win_rate,
        exposure=bars_in_position / n_bars if n_bars else 0.0,
        total_costs_quote=float(sum(t.costs_quote for t in result.trades)),
    )
