"""Probabilistic and deflated Sharpe ratios, with effective trial counts.

The v8 system fed the *raw* evolutionary evaluation count (118,000) into
its deflated Sharpe, guaranteeing DSR=0 forever — a gate that can never
pass is not a test (audit S3). Here the number of trials is the
*effective* count estimated from the correlation structure of candidate
returns (participation ratio of the correlation spectrum): identical
candidates collapse to 1 trial, independent ones count fully.

References: Bailey & López de Prado, "The Sharpe Ratio Efficient
Frontier" (PSR) and "The Deflated Sharpe Ratio" (DSR). Implemented from
the published formulas; no claims beyond them.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from evoquant.errors import EvoquantError

_NORM = NormalDist()
_EULER_GAMMA = 0.5772156649015329


class ValidationInputError(EvoquantError):
    code = "VALIDATION_INPUT"


def sharpe_ratio(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(np.mean(r)) / sd


def probabilistic_sharpe(returns: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark) accounting for skew/kurtosis of returns."""
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 4:
        raise ValidationInputError("PSR needs >= 4 returns", n=n)
    sr = sharpe_ratio(r)
    if not np.isfinite(sr):
        return 0.0
    sd = float(np.std(r, ddof=1))
    centered = (r - np.mean(r)) / sd
    skew = float(np.mean(centered**3))
    kurt = float(np.mean(centered**4))  # non-excess
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom <= 0.0:
        return 0.0  # pathological higher moments: no confidence
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(_NORM.cdf(z))


def effective_trials(candidate_returns: np.ndarray) -> float:
    """Effective number of independent trials from a (T, N) return matrix.

    Participation ratio of the correlation-matrix spectrum:
    N_eff = (sum λ)^2 / sum(λ^2). Identical columns -> 1; orthogonal -> N.
    """
    m = np.asarray(candidate_returns, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] < 3 or m.shape[1] < 1:
        raise ValidationInputError(
            "effective_trials needs a (T>=3, N>=1) matrix", shape=str(m.shape)
        )
    n = m.shape[1]
    if n == 1:
        return 1.0
    stds = m.std(axis=0, ddof=1)
    keep = stds > 0
    if keep.sum() <= 1:
        return 1.0
    corr = np.corrcoef(m[:, keep], rowvar=False)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(np.sum(eigvals))
    sq = float(np.sum(eigvals**2))
    if sq <= 0.0:
        return 1.0
    n_eff = total * total / sq
    return float(np.clip(n_eff, 1.0, n))


def expected_max_sharpe(n_trials: float, sr_variance: float) -> float:
    """E[max SR] across n_trials of zero-true-SR strategies (Bailey/LdP)."""
    if n_trials < 1.0:
        raise ValidationInputError("n_trials must be >= 1", n_trials=n_trials)
    if sr_variance < 0.0:
        raise ValidationInputError("sr_variance must be >= 0")
    if n_trials <= 1.0 or sr_variance == 0.0:
        return 0.0
    z1 = _NORM.inv_cdf(1.0 - 1.0 / n_trials)
    z2 = _NORM.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2)


def deflated_sharpe(
    returns: np.ndarray,
    n_trials: float,
    sr_variance_across_trials: float | None = None,
) -> dict[str, float]:
    """DSR = PSR against the expected-max-Sharpe benchmark.

    ``n_trials`` should be the EFFECTIVE count (see :func:`effective_trials`).
    ``sr_variance_across_trials`` is the variance of SR estimates across the
    candidate battery; when unavailable, the estimator variance of this
    strategy's SR is used as a documented approximation.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 4:
        raise ValidationInputError("DSR needs >= 4 returns", n=len(r))
    sr = sharpe_ratio(r)
    if sr_variance_across_trials is None:
        # approximation: variance of the SR estimator itself
        sr_variance_across_trials = (1.0 + 0.5 * sr * sr) / (len(r) - 1)
    sr0 = expected_max_sharpe(n_trials, sr_variance_across_trials)
    return {
        "sharpe": sr,
        "n_trials": float(n_trials),
        "expected_max_sharpe": sr0,
        "deflated_sharpe": probabilistic_sharpe(r, sr_benchmark=sr0),
    }
