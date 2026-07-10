"""Phase-4 acceptance: train-only fitting, one-sided inference, causal dwell."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from evoquant.regimes.fingerprints import compute_fingerprint
from evoquant.regimes.router import (
    QuantileRegimeRouter,
    Regime,
    RouterError,
    _causal_dwell_smooth,
)
from tests.conftest import make_bars


def _fitted(bars=None):
    bars = bars or make_bars(2_000, seed=31)
    router = QuantileRegimeRouter(er_period=24, atr_period=14, min_dwell=6)
    router.fit(bars)
    return router, bars


class TestFitDiscipline:
    def test_unfitted_router_refuses_inference(self):
        router = QuantileRegimeRouter()
        with pytest.raises(RouterError):
            router.infer(make_bars(500))

    def test_refit_is_bit_identical(self):
        bars = make_bars(2_000, seed=31)
        p1 = QuantileRegimeRouter(min_dwell=6).fit(bars)
        p2 = QuantileRegimeRouter(min_dwell=6).fit(bars)
        assert p1.params_hash() == p2.params_hash()

    def test_params_depend_only_on_train_slice(self):
        """The leakage probe (audit D7 regression).

        Perturbing TEST-period data must not change fitted parameters;
        perturbing TRAIN-period data must.
        """
        full = make_bars(3_000, seed=31)
        train = full.slice(0, 2_000)

        p_before = QuantileRegimeRouter(min_dwell=6).fit(train)

        # mutate the test period (bars 2000+) heavily
        close = np.array(full.close)
        close[2_000:] *= 5.0
        high = np.array(full.high)
        high[2_000:] = np.maximum(high[2_000:], close[2_000:] + 1)
        perturbed_full = replace(full, close=close, high=high)
        p_after_test_perturb = QuantileRegimeRouter(min_dwell=6).fit(
            perturbed_full.slice(0, 2_000)
        )
        assert p_before.params_hash() == p_after_test_perturb.params_hash()

        # mutate the train period => parameters must change
        close2 = np.array(full.close)
        close2[500:1_500] *= 5.0
        high2 = np.array(full.high)
        high2[500:1_500] = np.maximum(high2[500:1_500], close2[500:1_500] + 1)
        p_after_train_perturb = QuantileRegimeRouter(min_dwell=6).fit(
            replace(full, close=close2, high=high2).slice(0, 2_000)
        )
        assert p_before.params_hash() != p_after_train_perturb.params_hash()

    def test_too_short_train_slice_fails_closed(self):
        with pytest.raises(RouterError):
            QuantileRegimeRouter(min_dwell=12).fit(make_bars(50))


class TestOneSidedInference:
    def test_prefix_invariance_of_labels(self):
        router, bars = _fitted()
        full_labels = router.infer(bars)
        prefix_labels = router.infer(bars.slice(0, 1_200))
        assert np.array_equal(full_labels[:1_200], prefix_labels)

    def test_future_perturbation_cannot_relabel_the_past(self):
        router, bars = _fitted()
        base = router.infer(bars)
        close = np.array(bars.close)
        high = np.array(bars.high)
        close[1_500:] *= 4.0
        high[1_500:] = np.maximum(high[1_500:], close[1_500:] + 1)
        shifted = router.infer(replace(bars, close=close, high=high))
        assert np.array_equal(base[:1_500], shifted[:1_500])

    def test_inference_is_deterministic(self):
        router, bars = _fitted()
        assert np.array_equal(router.infer(bars), router.infer(bars))

    def test_warmup_is_unknown(self):
        router, bars = _fitted()
        labels = router.infer(bars)
        # features need ~24 bars + 6 dwell before anything can commit
        assert np.all(labels[:24] == int(Regime.UNKNOWN))
        assert np.any(labels != int(Regime.UNKNOWN))


class TestDwellSmoothing:
    def test_switch_requires_min_dwell_consecutive_bars(self):
        raw = np.array([1] * 20 + [3] * 3 + [1] * 20, dtype=np.int64)
        out = _causal_dwell_smooth(raw, min_dwell=6)
        assert np.all(out[6:] == 1)  # 3-bar excursion never commits
        assert set(out[:6]) <= {0, 1}

    def test_persistent_change_commits_after_dwell(self):
        raw = np.array([1] * 20 + [3] * 20, dtype=np.int64)
        out = _causal_dwell_smooth(raw, min_dwell=6)
        assert out[19] == 1
        assert np.all(out[20:25] == 1)  # not yet committed during challenge
        assert np.all(out[25:] == 3)  # commits at the 6th consecutive bar

    def test_unknown_gap_does_not_fabricate_switch(self):
        raw = np.array([1] * 20 + [0] * 10 + [1] * 10, dtype=np.int64)
        out = _causal_dwell_smooth(raw, min_dwell=3)
        assert np.all(out[10:] == 1)

    def test_committed_regime_runs_respect_dwell(self):
        rng = np.random.default_rng(3)
        raw = rng.integers(0, 5, size=2_000).astype(np.int64)
        out = _causal_dwell_smooth(raw, min_dwell=5)
        # committed switches only happen after 5 consecutive raw bars, so
        # any change in `out` must be preceded by such a run
        changes = np.nonzero(np.diff(out) != 0)[0] + 1
        for idx in changes:
            new = out[idx]
            assert np.all(raw[idx - 4 : idx + 1] == new)


class TestFingerprint:
    def test_fingerprint_deterministic_and_hashable(self):
        bars = make_bars(1_500, seed=9)
        f1 = compute_fingerprint(bars)
        f2 = compute_fingerprint(bars)
        assert f1.fingerprint_hash() == f2.fingerprint_hash()
        assert f1.symbol == "TESTUSDT"

    def test_fingerprint_depends_only_on_its_slice(self):
        full = make_bars(3_000, seed=9)
        base = compute_fingerprint(full.slice(0, 2_000))
        close = np.array(full.close)
        close[2_500:] *= 10
        high = np.maximum(np.array(full.high), close)
        perturbed = replace(full, close=close, high=high)
        again = compute_fingerprint(perturbed.slice(0, 2_000))
        assert base.fingerprint_hash() == again.fingerprint_hash()

    def test_fingerprint_fields_sane(self):
        f = compute_fingerprint(make_bars(1_500, seed=9))
        assert 0.0 <= f.efficiency_median <= 1.0
        assert f.atr_pct_median > 0
        assert f.ret_tail_ratio >= 1.0
        assert -1.0 <= f.ret_autocorr_lag1 <= 1.0
        assert 0.0 <= f.gap_over_atr_freq <= 1.0
