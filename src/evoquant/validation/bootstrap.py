"""Circular block bootstrap for serially dependent returns.

Used for confidence intervals (Sharpe, mean) and **path statistics**
(drawdown distribution). Permutation/resampling is legitimate for paths;
it is NOT used to produce a "terminal return percentile" — compounding
identical returns is order-invariant, so that number is undefined (audit
S2), and no such API exists in this package (asserted by a test).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evoquant.backtest.metrics import max_drawdown
from evoquant.validation.sharpe import ValidationInputError, sharpe_ratio


def circular_block_bootstrap(
    returns: np.ndarray, block: int, n_samples: int, seed: int
) -> np.ndarray:
    """(n_samples, T) matrix of circular-block resampled return paths."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    t = len(r)
    if t < 4:
        raise ValidationInputError("bootstrap needs >= 4 returns", n=t)
    if not (1 <= block <= t):
        raise ValidationInputError("block must be in [1, T]", block=block, t=t)
    rng = np.random.default_rng(seed)
    n_blocks = -(-t // block)  # ceil
    starts = rng.integers(0, t, size=(n_samples, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]) % t
    return np.asarray(r[idx.reshape(n_samples, -1)[:, :t]], dtype=np.float64)


@dataclass(frozen=True)
class BootstrapCI:
    p05: float
    p50: float
    p95: float
    n_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return {"p05": self.p05, "p50": self.p50, "p95": self.p95, "n": self.n_samples}


def sharpe_confidence(
    returns: np.ndarray, block: int = 24, n_samples: int = 500, seed: int = 0
) -> BootstrapCI:
    paths = circular_block_bootstrap(returns, block, n_samples, seed)
    sharpes = np.array([sharpe_ratio(p) for p in paths])
    sharpes = sharpes[np.isfinite(sharpes)]
    if len(sharpes) == 0:
        raise ValidationInputError("all bootstrap Sharpe estimates degenerate")
    return BootstrapCI(
        p05=float(np.quantile(sharpes, 0.05)),
        p50=float(np.quantile(sharpes, 0.50)),
        p95=float(np.quantile(sharpes, 0.95)),
        n_samples=len(sharpes),
    )


def drawdown_distribution(
    returns: np.ndarray, block: int = 24, n_samples: int = 500, seed: int = 0
) -> BootstrapCI:
    """Max-drawdown distribution over resampled paths (a PATH statistic —
    the legitimate use of resampling, unlike terminal returns)."""
    paths = circular_block_bootstrap(returns, block, n_samples, seed)
    equity = np.cumprod(1.0 + paths, axis=1)
    dds = np.array([max_drawdown(eq) for eq in equity])
    return BootstrapCI(
        p05=float(np.quantile(dds, 0.05)),
        p50=float(np.quantile(dds, 0.50)),
        p95=float(np.quantile(dds, 0.95)),
        n_samples=n_samples,
    )


def white_reality_check(
    candidate_returns: np.ndarray,
    block: int = 24,
    n_samples: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """White (2000) Reality Check for data snooping across a battery.

    H0: the best candidate's mean return is <= 0. The statistic is
    max_k mean(r_k); the bootstrap distribution recenters each candidate
    at zero mean. Returns the p-value of the observed max.
    """
    m = np.asarray(candidate_returns, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] < 1:
        raise ValidationInputError("reality check needs a (T, N) matrix")
    t, n = m.shape
    means = m.mean(axis=0)
    observed = float(np.max(means)) * np.sqrt(t)

    rng = np.random.default_rng(seed)
    n_blocks = -(-t // block)
    exceed = 0
    for _ in range(n_samples):
        starts = rng.integers(0, t, size=n_blocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % t).reshape(-1)[:t]
        resampled = m[idx] - means[None, :]  # recentered under H0
        stat = float(np.max(resampled.mean(axis=0))) * np.sqrt(t)
        if stat >= observed:
            exceed += 1
    return {
        "p_value": (exceed + 1.0) / (n_samples + 1.0),
        "observed_stat": observed,
        "n_candidates": float(n),
        "n_samples": float(n_samples),
    }
