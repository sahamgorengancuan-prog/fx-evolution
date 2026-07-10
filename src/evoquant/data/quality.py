"""Data-quality validation.

Hard checks (severity=error) abort research runs via
:func:`validate_or_raise` — fail closed, no repair, no synthetic fill.
Soft checks (severity=warning) are recorded in the QA report and travel
with the data manifest so downstream stages can make explicit decisions
(e.g. exclude the quantized 2017 BNB era) instead of inheriting silent
assumptions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from evoquant.data.contracts import InstrumentSpec
from evoquant.data.loaders import BarData
from evoquant.errors import DataQualityError

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class QACheck:
    name: str
    severity: str  # "error" | "warning"
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QAReport:
    symbol: str
    n_bars: int
    checks: tuple[QACheck, ...]

    @property
    def errors(self) -> list[QACheck]:
        return [c for c in self.checks if c.severity == ERROR and not c.passed]

    @property
    def warnings(self) -> list[QACheck]:
        return [c for c in self.checks if c.severity == WARNING and not c.passed]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_bars": self.n_bars,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
        }

    def summary(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "failed_checks": [c.name for c in self.checks if not c.passed],
        }


def _check(name: str, severity: str, passed: bool, **details: Any) -> QACheck:
    return QACheck(name=name, severity=severity, passed=bool(passed), details=details)


def run_quality_checks(
    bars: BarData,
    spec: InstrumentSpec,
    *,
    max_gap_report: int = 20,
    quantization_rel_tick: float = 0.01,
) -> QAReport:
    """Run the full QA battery. Never mutates or repairs data."""
    checks: list[QACheck] = []
    ts = bars.ts_utc_s
    o, h, l, c, v = bars.open, bars.high, bars.low, bars.close, bars.volume
    n = bars.n_bars

    # --- hard checks -------------------------------------------------- #
    checks.append(_check("non_empty", ERROR, n > 0, n_bars=n))
    if n == 0:
        return QAReport(symbol=bars.symbol, n_bars=0, checks=tuple(checks))

    dt = np.diff(ts)
    dup = int(np.sum(dt == 0))
    backwards = int(np.sum(dt < 0))
    checks.append(_check("monotonic_time", ERROR, backwards == 0, backwards_steps=backwards))
    checks.append(_check("no_duplicate_timestamps", ERROR, dup == 0, duplicates=dup))

    tf_s = bars.timeframe_minutes * 60
    misaligned = int(np.sum((ts - ts[0]) % tf_s != 0))
    checks.append(
        _check("timeframe_alignment", ERROR, misaligned == 0, misaligned_bars=misaligned)
    )

    finite = all(np.isfinite(arr).all() for arr in (o, h, l, c, v))
    checks.append(_check("finite_values", ERROR, finite))

    if finite:
        ohlc_bad = int(
            np.sum(
                (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)
            )
        )
        checks.append(_check("ohlc_invariants", ERROR, ohlc_bad == 0, violations=ohlc_bad))
        nonpos = int(np.sum((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)))
        checks.append(_check("positive_prices", ERROR, nonpos == 0, non_positive=nonpos))
        neg_vol = int(np.sum(v < 0))
        checks.append(_check("volume_non_negative", ERROR, neg_vol == 0, negative=neg_vol))
    else:
        checks.append(_check("ohlc_invariants", ERROR, False, reason="non-finite values"))
        checks.append(_check("positive_prices", ERROR, False, reason="non-finite values"))
        checks.append(_check("volume_non_negative", ERROR, False, reason="non-finite values"))

    # --- soft checks --------------------------------------------------- #
    gap_idx = np.nonzero(dt > tf_s)[0]
    gaps = [
        {
            "after_ts_utc_s": int(ts[i]),
            "missing_bars": int(dt[i] // tf_s - 1),
            "gap_seconds": int(dt[i]),
        }
        for i in gap_idx[:max_gap_report]
    ]
    checks.append(
        _check(
            "gap_census",
            WARNING,
            len(gap_idx) == 0,
            n_gaps=int(len(gap_idx)),
            total_missing_bars=int(np.sum(dt[gap_idx] // tf_s - 1)) if len(gap_idx) else 0,
            examples=gaps,
        )
    )

    zero_vol = int(np.sum(v == 0))
    checks.append(
        _check("zero_volume_bars", WARNING, zero_vol == 0, zero_volume_bars=zero_vol)
    )

    if bars.spread_points is not None:
        sp = bars.spread_points
        constant = bool(np.all(sp == sp[0]))
        checks.append(
            _check(
                "spread_series_informative",
                WARNING,
                not constant,
                constant_value_points=float(sp[0]) if constant else None,
                note=(
                    "Spread series is a single constant value across all bars; "
                    "this is a synthetic placeholder, not historical spread. "
                    "Cost models must not treat it as trusted (fail-closed)."
                    if constant
                    else ""
                ),
            )
        )
    else:
        checks.append(
            _check(
                "spread_series_informative",
                WARNING,
                False,
                note="No per-bar spread series present.",
            )
        )

    # Price quantization: bars whose price is so low relative to the tick
    # size that the price grid exceeds `quantization_rel_tick` of price.
    rel_tick = spec.tick_size / np.maximum(c, spec.tick_size)
    n_quantized = int(np.sum(rel_tick > quantization_rel_tick))
    first_ok = int(np.argmax(rel_tick <= quantization_rel_tick)) if n_quantized < n else n
    checks.append(
        _check(
            "price_quantization",
            WARNING,
            n_quantized == 0,
            bars_above_threshold=n_quantized,
            threshold_rel_tick=quantization_rel_tick,
            first_unaffected_index=first_ok,
            note=(
                "Bars where tick size exceeds "
                f"{quantization_rel_tick:.1%} of price; price grid too coarse for "
                "signal/execution research on those bars."
                if n_quantized
                else ""
            ),
        )
    )

    return QAReport(symbol=bars.symbol, n_bars=n, checks=tuple(checks))


def validate_or_raise(bars: BarData, spec: InstrumentSpec, **kwargs: Any) -> QAReport:
    """Run QA; raise :class:`DataQualityError` on any hard failure."""
    report = run_quality_checks(bars, spec, **kwargs)
    if not report.passed:
        raise DataQualityError(
            f"Data quality validation failed for {bars.symbol}: "
            f"{[c.name for c in report.errors]}",
            symbol=bars.symbol,
            failed_checks=[asdict(c) for c in report.errors],
            qa_summary=report.summary(),
        )
    return report
