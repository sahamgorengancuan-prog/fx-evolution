from __future__ import annotations

import numpy as np
import pytest

from evoquant.validation.sharpe import (
    ValidationInputError,
    deflated_sharpe,
    effective_trials,
    expected_max_sharpe,
    probabilistic_sharpe,
    sharpe_ratio,
)


class TestPSR:
    def test_strong_edge_high_confidence(self):
        rng = np.random.default_rng(0)
        r = rng.normal(0.01, 0.01, 500)  # SR ~ 1 per period
        assert probabilistic_sharpe(r, 0.0) > 0.99

    def test_psr_at_own_sharpe_is_half(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.001, 0.01, 500)
        sr = sharpe_ratio(r)
        assert probabilistic_sharpe(r, sr) == pytest.approx(0.5, abs=1e-9)

    def test_noise_low_confidence(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.0, 0.01, 500)
        assert probabilistic_sharpe(r, 0.5) < 0.05  # nowhere near SR 0.5

    def test_too_few_returns_rejected(self):
        with pytest.raises(ValidationInputError):
            probabilistic_sharpe(np.array([0.01, 0.02]))


class TestEffectiveTrials:
    def test_identical_candidates_collapse_to_one(self):
        rng = np.random.default_rng(3)
        col = rng.normal(0, 1, 300)
        m = np.column_stack([col] * 10)
        assert effective_trials(m) == pytest.approx(1.0, abs=1e-6)

    def test_independent_candidates_count_fully(self):
        rng = np.random.default_rng(4)
        m = rng.normal(0, 1, (2_000, 10))
        assert effective_trials(m) > 8.5  # near N=10 modulo sampling noise

    def test_correlated_battery_lands_in_between(self):
        rng = np.random.default_rng(5)
        base = rng.normal(0, 1, (1_000, 1))
        noise = rng.normal(0, 1, (1_000, 12))
        m = 0.8 * base + 0.6 * noise  # strongly cross-correlated candidates
        n_eff = effective_trials(m)
        assert 1.5 < n_eff < 11.0

    def test_single_column_is_one(self):
        m = np.random.default_rng(6).normal(0, 1, (100, 1))
        assert effective_trials(m) == 1.0


class TestDSR:
    def test_expected_max_sharpe_grows_with_trials(self):
        v = 0.05
        values = [expected_max_sharpe(n, v) for n in (1, 10, 100, 1000)]
        assert values[0] == 0.0
        assert values == sorted(values)
        assert values[-1] > values[1]

    def test_dsr_decreases_with_trial_count(self):
        rng = np.random.default_rng(7)
        r = rng.normal(0.004, 0.01, 400)
        d_small = deflated_sharpe(r, n_trials=2)["deflated_sharpe"]
        d_big = deflated_sharpe(r, n_trials=10_000)["deflated_sharpe"]
        assert d_big < d_small

    def test_effective_n_deflates_less_than_raw_n(self):
        """The v8 mistake in reverse: raw trial counts over-deflate.

        A correlated battery's effective N is far below its raw N, so the
        DSR computed with effective N must be >= the raw-N DSR.
        """
        rng = np.random.default_rng(8)
        base = rng.normal(0, 1, (500, 1))
        battery = 0.9 * base + 0.3 * rng.normal(0, 1, (500, 50))
        n_eff = effective_trials(battery)
        assert n_eff < 50 * 0.5  # far below raw N

        strategy_returns = rng.normal(0.003, 0.01, 400)
        with_raw = deflated_sharpe(strategy_returns, n_trials=50)["deflated_sharpe"]
        with_eff = deflated_sharpe(strategy_returns, n_trials=n_eff)["deflated_sharpe"]
        assert with_eff >= with_raw

    def test_dsr_report_shape(self):
        rng = np.random.default_rng(9)
        report = deflated_sharpe(rng.normal(0.002, 0.01, 300), n_trials=25)
        assert set(report) == {
            "sharpe",
            "n_trials",
            "expected_max_sharpe",
            "deflated_sharpe",
        }
        assert 0.0 <= report["deflated_sharpe"] <= 1.0
