"""Market-data loaders.

Currently supported: ForexSB JSON bar export (the format of the attached
``BNBUSDT_H1.json``). Its ``time`` array is minutes since 2000-01-01 00:00 UTC;
we convert to explicit UTC epoch seconds at load time so no downstream code
ever guesses a time base.

No synthetic fallback: a missing or malformed file raises; nothing is
generated in its place.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from evoquant.data.contracts import InstrumentSpec
from evoquant.errors import DataFormatError

#: Unix epoch seconds of 2000-01-01T00:00:00Z, the ForexSB time base.
FSB_EPOCH_UNIX_S = 946_684_800

_FSB_ARRAY_KEYS = ("time", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class BarData:
    """Immutable OHLCV bar container. Timestamps are UTC epoch seconds."""

    symbol: str
    timeframe_minutes: int
    ts_utc_s: np.ndarray  # int64, epoch seconds, strictly increasing expected
    open: np.ndarray  # float64
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    #: Spread per bar in points; may be a static placeholder (see spec.spread_source).
    spread_points: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.ts_utc_s)
        for name in ("open", "high", "low", "close", "volume"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise DataFormatError(
                    f"BarData array length mismatch: {name} has {len(arr)}, time has {n}",
                    field=name,
                )
        if self.spread_points is not None and len(self.spread_points) != n:
            raise DataFormatError("BarData spread length mismatch")
        for arr in (self.ts_utc_s, self.open, self.high, self.low, self.close, self.volume):
            arr.setflags(write=False)
        if self.spread_points is not None:
            self.spread_points.setflags(write=False)

    @property
    def n_bars(self) -> int:
        return len(self.ts_utc_s)

    def slice(self, start: int, stop: int) -> BarData:
        """Half-open positional slice returning a new immutable view."""
        if not (0 <= start <= stop <= self.n_bars):
            raise DataFormatError(
                "BarData.slice out of bounds", start=start, stop=stop, n_bars=self.n_bars
            )
        return replace(
            self,
            ts_utc_s=self.ts_utc_s[start:stop],
            open=self.open[start:stop],
            high=self.high[start:stop],
            low=self.low[start:stop],
            close=self.close[start:stop],
            volume=self.volume[start:stop],
            spread_points=(
                self.spread_points[start:stop] if self.spread_points is not None else None
            ),
        )


def load_fsb_json(path: str | Path) -> tuple[BarData, InstrumentSpec, dict[str, Any]]:
    """Load a ForexSB JSON export.

    Returns ``(bars, instrument_spec, raw_header)``. The raw header is
    preserved verbatim for the data manifest.
    """
    path = Path(path)
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError:
        raise DataFormatError("Data file not found", path=str(path)) from None
    except json.JSONDecodeError as exc:
        raise DataFormatError(
            "Data file is not valid JSON", path=str(path), reason=str(exc)
        ) from exc

    missing = [k for k in _FSB_ARRAY_KEYS if k not in doc]
    if missing:
        raise DataFormatError(
            "ForexSB document missing arrays", path=str(path), missing=missing
        )
    if "period" not in doc or "symbol" not in doc:
        raise DataFormatError("ForexSB document missing 'period'/'symbol'", path=str(path))

    n_declared = doc.get("bars")
    time_minutes = np.asarray(doc["time"], dtype=np.int64)
    if n_declared is not None and n_declared != len(time_minutes):
        raise DataFormatError(
            "Declared bar count does not match time array",
            declared=n_declared,
            actual=len(time_minutes),
        )

    spreads = doc.get("spreads")
    bars = BarData(
        symbol=str(doc["symbol"]),
        timeframe_minutes=int(doc["period"]),
        ts_utc_s=time_minutes * 60 + FSB_EPOCH_UNIX_S,
        open=np.asarray(doc["open"], dtype=np.float64),
        high=np.asarray(doc["high"], dtype=np.float64),
        low=np.asarray(doc["low"], dtype=np.float64),
        close=np.asarray(doc["close"], dtype=np.float64),
        volume=np.asarray(doc["volume"], dtype=np.float64),
        spread_points=(
            np.asarray(spreads, dtype=np.float64)
            if spreads is not None and len(spreads) == len(time_minutes)
            else None
        ),
    )
    header = {k: v for k, v in doc.items() if not isinstance(v, list)}
    spec = InstrumentSpec.from_fsb_header(header)
    return bars, spec, header
