from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from evoquant.data.quality import run_quality_checks, validate_or_raise
from evoquant.errors import DataQualityError
from tests.conftest import make_bars, make_spec


def _mutate(bars, **arrays):
    new = {}
    for name, arr in arrays.items():
        a = np.array(getattr(bars, name))  # writable copy
        for idx, val in arr:
            a[idx] = val
        new[name] = a
    return replace(bars, **new)


def _failed(report, name):
    return [c for c in report.checks if c.name == name and not c.passed]


def test_clean_data_passes(bars, spec):
    report = validate_or_raise(bars, spec)
    assert report.passed


def test_non_monotonic_time_is_error(bars, spec):
    bad = _mutate(bars, ts_utc_s=[(50, bars.ts_utc_s[10])])
    report = run_quality_checks(bad, spec)
    assert _failed(report, "monotonic_time")
    assert not report.passed


def test_duplicate_timestamps_is_error(bars, spec):
    bad = _mutate(bars, ts_utc_s=[(51, bars.ts_utc_s[50])])
    report = run_quality_checks(bad, spec)
    assert _failed(report, "no_duplicate_timestamps")


def test_ohlc_violation_is_error(bars, spec):
    bad = _mutate(bars, high=[(20, float(bars.low[20]) - 1.0)])
    report = run_quality_checks(bad, spec)
    assert _failed(report, "ohlc_invariants")


def test_nan_is_error(bars, spec):
    bad = _mutate(bars, close=[(30, np.nan)])
    report = run_quality_checks(bad, spec)
    assert _failed(report, "finite_values")


def test_non_positive_price_is_error(bars, spec):
    bad = _mutate(bars, low=[(40, 0.0)], open=[(40, 0.0)])
    report = run_quality_checks(bad, spec)
    assert _failed(report, "positive_prices")


def test_negative_volume_is_error(bars, spec):
    bad = _mutate(bars, volume=[(5, -1.0)])
    report = run_quality_checks(bad, spec)
    assert _failed(report, "volume_non_negative")


def test_validate_or_raise_produces_structured_artifact(bars, spec):
    bad = _mutate(bars, close=[(30, np.nan)])
    with pytest.raises(DataQualityError) as exc:
        validate_or_raise(bad, spec)
    artifact = exc.value.to_artifact()
    assert artifact["error_code"] == "DATA_QUALITY"
    assert "finite_values" in [c["name"] for c in artifact["details"]["failed_checks"]]


def test_gap_is_warning_not_error(spec):
    bars = make_bars(100)
    ts = np.array(bars.ts_utc_s)
    ts[60:] += 5 * 3600  # 5-hour hole, still aligned to H1 grid
    gapped = replace(bars, ts_utc_s=ts)
    report = run_quality_checks(gapped, spec)
    assert report.passed  # no hard failure
    gap = _failed(report, "gap_census")
    assert gap and gap[0].details["n_gaps"] == 1
    assert gap[0].details["total_missing_bars"] == 5


def test_constant_spread_flagged_as_synthetic(bars, spec):
    report = run_quality_checks(bars, spec)  # fixture spread is constant 10
    flagged = _failed(report, "spread_series_informative")
    assert flagged and flagged[0].details["constant_value_points"] == 10.0
    assert report.passed  # warning only


def test_quantization_warning():
    spec = make_spec()
    bars = make_bars(100)
    # push prices below 100 ticks => relative tick > 1%
    scale = 0.005 / np.array(bars.close)
    quantized = replace(
        bars,
        open=bars.open * scale,
        high=bars.high * scale + 0.001,
        low=bars.low * scale - 0.0001,
        close=bars.close * scale,
    )
    report = run_quality_checks(quantized, spec)
    q = _failed(report, "price_quantization")
    assert q and q[0].details["bars_above_threshold"] == 100
