"""Causal pair fingerprints.

A fingerprint summarizes *how a market behaves* over a data slice using
only causal statistics. It is fitted per fold on the training partition
only (v8 leaked full-history characterization into search seeding —
audit D7) and later becomes the context vector for contextual operator
learning (Phase 6).

Every underlying series is produced by the Phase-2 compiler, whose
causality is test-enforced; the aggregation (medians/quantiles over the
slice) happens strictly within the slice passed in.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from evoquant.data.loaders import BarData
from evoquant.data.manifest import sha256_of_dict
from evoquant.features.ast import Node
from evoquant.features.compiler import compile_feature


def _series(op_node: Node, bars: BarData) -> np.ndarray:
    return np.asarray(compile_feature(op_node).evaluate(bars), dtype=np.float64)


def _n(op: str, *children: Node, **kw: Any) -> Node:
    return Node(op=op, children=tuple(children), **kw)


@dataclass(frozen=True)
class PairFingerprint:
    symbol: str
    n_bars: int
    #: median ATR as a fraction of price (volatility level)
    atr_pct_median: float
    #: interquartile spread of ATR%, relative (vol-of-vol proxy)
    atr_pct_iqr_rel: float
    #: median trend efficiency ratio (0..1)
    efficiency_median: float
    #: 90th percentile efficiency (how trendy the trendy periods get)
    efficiency_p90: float
    #: median absolute 1-bar return
    abs_ret_median: float
    #: tail ratio: p99 |ret| / median |ret| (jumpiness)
    ret_tail_ratio: float
    #: fraction of bars with a gap vs previous close > 1 ATR (jump frequency)
    gap_over_atr_freq: float
    #: lag-1 autocorrelation of returns (sign persistence)
    ret_autocorr_lag1: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint_hash(self) -> str:
        return sha256_of_dict(self.to_dict())


def compute_fingerprint(
    bars: BarData, *, er_period: int = 24, atr_period: int = 14
) -> PairFingerprint:
    """Aggregate causal descriptors over exactly the slice given.

    The caller is responsible for passing a TRAIN-ONLY slice; this
    function cannot see anything else by construction.
    """
    close = _n("close")
    atr = _series(Node(op="atr", period=atr_period), bars)
    er = _series(_n("efficiency_ratio", close, period=er_period), bars)
    ret = _series(_n("ret", close, period=1), bars)

    c = np.asarray(bars.close, dtype=np.float64)
    atr_pct = atr / c

    def med(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        return float(np.median(x)) if len(x) else float("nan")

    def q(x: np.ndarray, p: float) -> float:
        x = x[np.isfinite(x)]
        return float(np.quantile(x, p)) if len(x) else float("nan")

    abs_ret = np.abs(ret)
    atr_med = med(atr_pct)
    iqr_rel = (
        (q(atr_pct, 0.75) - q(atr_pct, 0.25)) / atr_med if atr_med and atr_med > 0 else float("nan")
    )
    med_abs = med(abs_ret)
    tail_ratio = q(abs_ret, 0.99) / med_abs if med_abs and med_abs > 0 else float("nan")

    # gap frequency: |open - prev close| > 1 ATR
    prev_close = np.concatenate([[np.nan], c[:-1]])
    gap = np.abs(np.asarray(bars.open, dtype=np.float64) - prev_close)
    valid = np.isfinite(gap) & np.isfinite(atr) & (atr > 0)
    gap_freq = float(np.mean(gap[valid] > atr[valid])) if valid.any() else float("nan")

    r = ret[np.isfinite(ret)]
    if len(r) > 2 and np.std(r) > 0:
        autocorr = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    else:
        autocorr = float("nan")

    return PairFingerprint(
        symbol=bars.symbol,
        n_bars=bars.n_bars,
        atr_pct_median=atr_med,
        atr_pct_iqr_rel=iqr_rel,
        efficiency_median=med(er),
        efficiency_p90=q(er, 0.9),
        abs_ret_median=med_abs,
        ret_tail_ratio=tail_ratio,
        gap_over_atr_freq=gap_freq,
        ret_autocorr_lag1=autocorr,
    )
