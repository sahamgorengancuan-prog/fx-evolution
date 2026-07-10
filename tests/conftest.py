"""Shared fixtures.

Synthetic data appears ONLY here, in unit tests — never in research or
production paths (blueprint truth-constraint #5).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from evoquant.data.contracts import InstrumentSpec
from evoquant.data.loaders import FSB_EPOCH_UNIX_S, BarData

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA = REPO_ROOT / "data" / "raw" / "BNBUSDT_H1.json"


def make_bars(
    n: int = 500,
    tf_minutes: int = 60,
    symbol: str = "TESTUSDT",
    start_minute: int = 9_000_000,
    seed: int = 7,
    spread: float | None = 10.0,
) -> BarData:
    rng = np.random.default_rng(seed)
    t = start_minute + np.arange(n, dtype=np.int64) * tf_minutes
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)
    o = np.concatenate([[close[0]], close[:-1]])
    h = np.maximum(o, close) + rng.uniform(0, 0.5, n)
    l = np.minimum(o, close) - rng.uniform(0, 0.5, n)
    l = np.maximum(l, 0.5)
    v = rng.uniform(1, 1000, n)
    return BarData(
        symbol=symbol,
        timeframe_minutes=tf_minutes,
        ts_utc_s=t * 60 + FSB_EPOCH_UNIX_S,
        open=o,
        high=h,
        low=l,
        close=close,
        volume=v,
        spread_points=np.full(n, spread) if spread is not None else None,
    )


def make_spec(symbol: str = "TESTUSDT") -> InstrumentSpec:
    return InstrumentSpec(
        symbol=symbol,
        base_currency="TEST",
        quote_currency="USD",
        tick_size=0.01,
        digits=2,
        lot_size=1000.0,
        min_lot=0.01,
        max_lot=1000.0,
        lot_step=0.01,
        commission=0.1,
        commission_mode="percent_notional",
        static_spread_points=10.0,
        spread_source="static_header",
        swap_long=None,
        swap_short=None,
        swap_mode=None,
        min_notional=None,
    )


def make_fsb_doc(n: int = 200) -> dict:
    bars = make_bars(n)
    minutes = ((bars.ts_utc_s - FSB_EPOCH_UNIX_S) // 60).tolist()
    return {
        "ver": 3,
        "terminal": "test",
        "company": "test",
        "server": "test",
        "symbol": "TESTUSDT",
        "description": "synthetic unit-test data",
        "period": 60,
        "baseCurrency": "TEST",
        "priceIn": "USD",
        "lotSize": 1000,
        "stopLevel": 0,
        "minLot": 0.01,
        "maxLot": 1000,
        "lotStep": 0.01,
        "spread": 10,
        "digits": 2,
        "bars": n,
        "swapLong": 0,
        "swapShort": 0,
        "commissionType": 5,
        "commission": 0.1,
        "pointValue": 1,
        "point": 0.01,
        "pip": 0.01,
        "time": minutes,
        "open": bars.open.tolist(),
        "high": bars.high.tolist(),
        "low": bars.low.tolist(),
        "close": bars.close.tolist(),
        "volume": bars.volume.tolist(),
        "spreads": [10] * n,
        "meta": [],
    }


@pytest.fixture
def fsb_file(tmp_path: Path) -> Path:
    path = tmp_path / "test_fsb.json"
    path.write_text(json.dumps(make_fsb_doc()))
    return path


@pytest.fixture
def bars() -> BarData:
    return make_bars()


@pytest.fixture
def spec() -> InstrumentSpec:
    return make_spec()
