"""Structured failure attribution (minimal deterministic subset).

Every rejected candidate gets machine-readable labels from the blueprint
taxonomy instead of a bare "reject". Deeper attribution (ablation,
MFE/MAE, cost decomposition) extends this module later; labels emitted
here are stable identifiers the memory and LLM layers rely on.
"""
from __future__ import annotations

import numpy as np

from evoquant.search.objectives import CandidateEvaluation, TargetConfig

#: subset of the blueprint failure taxonomy derivable from an evaluation
KNOWN_LABELS = (
    "NO_SIGNAL",
    "INSUFFICIENT_TRADES",
    "OVERFIRE",
    "FOLD_INSTABILITY",
    "STOP_GAP_RISK",
    "NO_EDGE_UNDER_REALISTIC_COSTS",
    "EXCESS_COMPLEXITY",
)


def failure_labels(ev: CandidateEvaluation, targets: TargetConfig) -> list[str]:
    """Deterministic labels for an (in)feasible evaluation. Empty == clean."""
    labels: list[str] = []
    v = ev.violations(targets)

    if ev.n_trades_total == 0:
        labels.append("NO_SIGNAL")
    elif "min_trades_total" in v:
        labels.append("INSUFFICIENT_TRADES")

    freq = float(ev.diagnostics.get("trade_frequency", 0.0))
    if freq > 0.15:  # > ~1 trade per 7 bars on H1 is churn
        labels.append("OVERFIRE")

    if "max_fold_dispersion" in v:
        labels.append("FOLD_INSTABILITY")

    if "max_drawdown" in v:
        labels.append("STOP_GAP_RISK")

    rets = np.asarray(ev.fold_returns, dtype=np.float64)
    finite = rets[np.isfinite(rets)]
    pf = ev.profit_factor if ev.profit_factor is not None else 0.0
    if ev.n_trades_total > 0 and (
        "min_profit_factor" in v
        or "min_median_fold_return" in v
        or (len(finite) and float(np.median(finite)) <= 0.0 and pf < 1.0)
    ):
        labels.append("NO_EDGE_UNDER_REALISTIC_COSTS")

    if "max_complexity" in v:
        labels.append("EXCESS_COMPLEXITY")

    return labels
