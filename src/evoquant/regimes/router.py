"""Train-only causal regime router.

Discipline (ADR-002, blueprint §8.2 — each rule is a test):

1. ``fit()`` sees the training slice only; its parameters are a pure
   function of that slice (deterministic, hashable).
2. ``infer()`` labels bars one-sided: label[i] uses bars[0..i] only —
   proven by prefix-invariance and future-perturbation tests.
3. Warmup and low-confidence bars get ``UNKNOWN`` (policy: stay flat).
4. Min-dwell smoothing is causal: a switch is committed only after the
   raw regime has persisted ``min_dwell`` consecutive bars.

Baseline implementation: quantile thresholds on two causal features —
trend efficiency (efficiency_ratio) and relative volatility (ATR%) —
yielding {trend,range} × {high_vol,low_vol} + UNKNOWN. HMM / change-point
routers plug in later behind the same interface (interchangeable-router
requirement); they must pass the same test battery.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any

import numpy as np

from evoquant.data.loaders import BarData
from evoquant.data.manifest import sha256_of_dict
from evoquant.errors import EvoquantError
from evoquant.features.ast import Node
from evoquant.features.compiler import compile_feature


class RouterError(EvoquantError):
    code = "REGIME_ROUTER"


class Regime(IntEnum):
    UNKNOWN = 0
    TREND_HIGH_VOL = 1
    TREND_LOW_VOL = 2
    RANGE_HIGH_VOL = 3
    RANGE_LOW_VOL = 4


REGIME_NAMES = {r: r.name.lower() for r in Regime}


@dataclass(frozen=True)
class RouterParams:
    """Everything ``fit`` learned. Pure function of the training slice."""

    er_period: int
    atr_period: int
    er_threshold: float  # train median efficiency ratio
    vol_threshold: float  # train median ATR%
    min_dwell: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def params_hash(self) -> str:
        return sha256_of_dict(self.to_dict())


def _features(bars: BarData, er_period: int, atr_period: int) -> tuple[np.ndarray, np.ndarray]:
    close = Node(op="close")
    er = np.asarray(
        compile_feature(
            Node(op="efficiency_ratio", period=er_period, children=(close,))
        ).evaluate(bars),
        dtype=np.float64,
    )
    atr = np.asarray(
        compile_feature(Node(op="atr", period=atr_period)).evaluate(bars),
        dtype=np.float64,
    )
    atr_pct = atr / np.asarray(bars.close, dtype=np.float64)
    return er, atr_pct


class QuantileRegimeRouter:
    """Rule-based baseline router. Fit on train only; infer one-sided."""

    def __init__(self, er_period: int = 24, atr_period: int = 14, min_dwell: int = 12) -> None:
        if min_dwell < 1:
            raise RouterError("min_dwell must be >= 1")
        self._er_period = er_period
        self._atr_period = atr_period
        self._min_dwell = min_dwell
        self._params: RouterParams | None = None

    # ------------------------------------------------------------------ #

    @property
    def params(self) -> RouterParams:
        if self._params is None:
            raise RouterError("Router is not fitted; call fit(train_bars) first")
        return self._params

    def fit(self, train_bars: BarData) -> RouterParams:
        """Learn quantile thresholds from the TRAINING slice only."""
        er, atr_pct = _features(train_bars, self._er_period, self._atr_period)
        er_valid = er[np.isfinite(er)]
        vol_valid = atr_pct[np.isfinite(atr_pct)]
        if len(er_valid) < 10 * self._min_dwell or len(vol_valid) < 10 * self._min_dwell:
            raise RouterError(
                "Training slice too short to fit regime thresholds",
                n_bars=train_bars.n_bars,
                valid_er=len(er_valid),
            )
        self._params = RouterParams(
            er_period=self._er_period,
            atr_period=self._atr_period,
            er_threshold=float(np.median(er_valid)),
            vol_threshold=float(np.median(vol_valid)),
            min_dwell=self._min_dwell,
        )
        return self._params

    # ------------------------------------------------------------------ #

    def infer(self, bars: BarData) -> np.ndarray:
        """One-sided regime labels for every bar. Never re-labels the past."""
        p = self.params
        er, atr_pct = _features(bars, p.er_period, p.atr_period)

        raw = np.full(bars.n_bars, int(Regime.UNKNOWN), dtype=np.int64)
        valid = np.isfinite(er) & np.isfinite(atr_pct)
        trend = er > p.er_threshold
        high_vol = atr_pct > p.vol_threshold
        raw[valid & trend & high_vol] = int(Regime.TREND_HIGH_VOL)
        raw[valid & trend & ~high_vol] = int(Regime.TREND_LOW_VOL)
        raw[valid & ~trend & high_vol] = int(Regime.RANGE_HIGH_VOL)
        raw[valid & ~trend & ~high_vol] = int(Regime.RANGE_LOW_VOL)

        return _causal_dwell_smooth(raw, p.min_dwell)


def _causal_dwell_smooth(raw: np.ndarray, min_dwell: int) -> np.ndarray:
    """Commit a regime switch only after `min_dwell` consecutive raw bars.

    Strictly causal: the committed label at bar i depends on raw[0..i].
    UNKNOWN raw bars reset the challenge counter and keep the committed
    regime (a burst of missing features doesn't fabricate a switch), but
    the sequence starts as UNKNOWN until the first commitment.
    """
    n = len(raw)
    out = np.empty(n, dtype=np.int64)
    committed = int(Regime.UNKNOWN)
    challenger = int(Regime.UNKNOWN)
    streak = 0
    for i in range(n):
        r = int(raw[i])
        if r != int(Regime.UNKNOWN) and r != committed:
            if r == challenger:
                streak += 1
            else:
                challenger = r
                streak = 1
            if streak >= min_dwell:
                committed = r
                streak = 0
                challenger = int(Regime.UNKNOWN)
        else:
            streak = 0
            challenger = int(Regime.UNKNOWN)
        out[i] = committed
    return out
