"""Return-concentration diagnostics.

An "edge" whose PnL lives in two lucky trades is a tail event wearing a
strategy costume. These are plain descriptive statistics over the trade
list — no inference implied.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from evoquant.backtest.fast_engine import Trade
from evoquant.validation.sharpe import ValidationInputError


def return_concentration(trades: tuple[Trade, ...] | list[Trade]) -> dict[str, Any]:
    if not trades:
        raise ValidationInputError("concentration needs at least one trade")
    pnls = np.asarray([t.pnl_quote for t in trades], dtype=np.float64)
    positive = pnls[pnls > 0]
    total_positive = float(np.sum(positive))

    if total_positive <= 0:
        top1_share = top3_share = float("nan")
        hhi = float("nan")
    else:
        ordered = np.sort(positive)[::-1]
        top1_share = float(ordered[0] / total_positive)
        top3_share = float(np.sum(ordered[:3]) / total_positive)
        weights = positive / total_positive
        hhi = float(np.sum(weights**2))

    return {
        "n_trades": len(trades),
        "n_winners": int(len(positive)),
        "top1_winner_share": top1_share,
        "top3_winner_share": top3_share,
        "winner_hhi": hhi,
        "net_pnl": float(np.sum(pnls)),
    }
