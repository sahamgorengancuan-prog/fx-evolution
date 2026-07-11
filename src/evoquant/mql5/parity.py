"""Python <-> MQL5 differential parity.

Comparison happens at every level the blueprint demands — signal, order
direction, position size, SL/TP levels, exit reason, and trade-by-trade —
never final equity only. Every discrepancy is itemized; nothing is
silently tolerated or netted away.

Until a real Strategy Tester run's CSVs are supplied, a bundle's parity
status is ``PENDING_REAL_TICK`` — pending is an honest state, success is
not assumed (blueprint work protocol).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from evoquant.backtest.fast_engine import Trade
from evoquant.backtest.signals import SignalPolicy
from evoquant.mql5.parser import EASignalRow, EATradeRow

PASS = "PASS"
FAIL = "FAIL"
PENDING_REAL_TICK = "PENDING_REAL_TICK"


@dataclass(frozen=True)
class Discrepancy:
    level: str  # "signal" | "trade" | "structure"
    where: str  # bar timestamp / trade ordinal
    fieldname: str
    python_value: str
    ea_value: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class ParityReport:
    status: str
    n_signals_compared: int = 0
    n_trades_compared: int = 0
    discrepancies: list[Discrepancy] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "n_signals_compared": self.n_signals_compared,
            "n_trades_compared": self.n_trades_compared,
            "n_discrepancies": len(self.discrepancies),
            "discrepancies": [d.to_dict() for d in self.discrepancies],
        }


def compare_signals(
    bar_ts_utc_s: np.ndarray,
    policy: SignalPolicy,
    ea_rows: list[EASignalRow],
    *,
    skip_warmup_bars: int,
    atr_atol: float = 1e-6,
) -> list[Discrepancy]:
    """Bar-by-bar signal parity after the declared warmup region."""
    out: list[Discrepancy] = []
    ea_by_ts = {r.ts_utc_s: r for r in ea_rows}
    for i in range(skip_warmup_bars, policy.n_bars):
        ts = int(bar_ts_utc_s[i])
        row = ea_by_ts.get(ts)
        if row is None:
            out.append(
                Discrepancy("structure", str(ts), "missing_ea_bar", "present", "absent")
            )
            continue
        for name, py_val in (
            ("entry_long", bool(policy.entry_long[i])),
            ("entry_short", bool(policy.entry_short[i])),
            ("allow", bool(policy.allow[i])),
        ):
            ea_val = getattr(row, name)
            if py_val != ea_val:
                out.append(Discrepancy("signal", str(ts), name, str(py_val), str(ea_val)))
        py_atr = float(policy.atr[i])
        if math.isfinite(py_atr) and abs(py_atr - row.atr) > atr_atol:
            out.append(
                Discrepancy("signal", str(ts), "atr", f"{py_atr:.10f}", f"{row.atr:.10f}")
            )
    return out


def compare_trades(
    bar_ts_utc_s: np.ndarray,
    python_trades: tuple[Trade, ...] | list[Trade],
    ea_trades: list[EATradeRow],
    *,
    price_atol: float,
    lots_atol: float = 1e-6,
) -> list[Discrepancy]:
    """Ordinal trade-by-trade parity on direction/size/prices/levels/reason."""
    out: list[Discrepancy] = []
    if len(python_trades) != len(ea_trades):
        out.append(
            Discrepancy(
                "structure", "trade_count", "n_trades",
                str(len(python_trades)), str(len(ea_trades)),
            )
        )
    for k, (pt, et) in enumerate(zip(python_trades, ea_trades, strict=False)):
        where = f"trade[{k}]"
        if pt.direction != et.direction:
            out.append(
                Discrepancy("trade", where, "direction", str(pt.direction), str(et.direction))
            )
        py_open_ts = int(bar_ts_utc_s[pt.entry_index])
        if py_open_ts != et.open_ts_utc_s:
            out.append(
                Discrepancy("trade", where, "open_time", str(py_open_ts), str(et.open_ts_utc_s))
            )
        if abs(pt.lots - et.lots) > lots_atol:
            out.append(Discrepancy("trade", where, "lots", f"{pt.lots:.2f}", f"{et.lots:.2f}"))
        for name, py_val, ea_val in (
            ("entry_price", pt.entry_price, et.entry_price),
            ("exit_price", pt.exit_price, et.exit_price),
            ("sl", pt.sl_price, et.sl),
            ("tp", pt.tp_price, et.tp),
        ):
            if abs(py_val - ea_val) > price_atol:
                out.append(
                    Discrepancy("trade", where, name, f"{py_val:.10f}", f"{ea_val:.10f}")
                )
        if pt.exit_reason != et.exit_reason:
            out.append(
                Discrepancy("trade", where, "exit_reason", pt.exit_reason, et.exit_reason)
            )
    return out


def compare_signal_rows(
    python_rows: list[EASignalRow],
    ea_rows: list[EASignalRow],
    *,
    skip_warmup_bars: int,
    atr_atol: float = 1e-6,
) -> list[Discrepancy]:
    """Row-wise signal parity between two dumps in the same CSV contract."""
    out: list[Discrepancy] = []
    ea_by_ts = {r.ts_utc_s: r for r in ea_rows}
    for py in python_rows[skip_warmup_bars:]:
        row = ea_by_ts.get(py.ts_utc_s)
        if row is None:
            out.append(
                Discrepancy(
                    "structure", str(py.ts_utc_s), "missing_ea_bar", "present", "absent"
                )
            )
            continue
        for name in ("entry_long", "entry_short", "allow"):
            if getattr(py, name) != getattr(row, name):
                out.append(
                    Discrepancy(
                        "signal", str(py.ts_utc_s), name,
                        str(getattr(py, name)), str(getattr(row, name)),
                    )
                )
        if math.isfinite(py.atr) and abs(py.atr - row.atr) > atr_atol:
            out.append(
                Discrepancy(
                    "signal", str(py.ts_utc_s), "atr", f"{py.atr:.10f}", f"{row.atr:.10f}"
                )
            )
    return out


def compare_trade_rows(
    python_rows: list[EATradeRow],
    ea_rows: list[EATradeRow],
    *,
    price_atol: float,
    lots_atol: float = 1e-6,
) -> list[Discrepancy]:
    out: list[Discrepancy] = []
    if len(python_rows) != len(ea_rows):
        out.append(
            Discrepancy(
                "structure", "trade_count", "n_trades",
                str(len(python_rows)), str(len(ea_rows)),
            )
        )
    for k, (pt, et) in enumerate(zip(python_rows, ea_rows, strict=False)):
        where = f"trade[{k}]"
        if pt.direction != et.direction:
            out.append(
                Discrepancy("trade", where, "direction", str(pt.direction), str(et.direction))
            )
        if pt.open_ts_utc_s != et.open_ts_utc_s:
            out.append(
                Discrepancy(
                    "trade", where, "open_time",
                    str(pt.open_ts_utc_s), str(et.open_ts_utc_s),
                )
            )
        if abs(pt.lots - et.lots) > lots_atol:
            out.append(Discrepancy("trade", where, "lots", f"{pt.lots:.2f}", f"{et.lots:.2f}"))
        for name in ("entry_price", "exit_price", "sl", "tp"):
            pv, ev = getattr(pt, name), getattr(et, name)
            if abs(pv - ev) > price_atol:
                out.append(Discrepancy("trade", where, name, f"{pv:.10f}", f"{ev:.10f}"))
        if pt.exit_reason != et.exit_reason:
            out.append(
                Discrepancy("trade", where, "exit_reason", pt.exit_reason, et.exit_reason)
            )
    return out


def parity_from_bundle(
    bundle_dir: str,
    ea_signals_path: str,
    ea_trades_path: str,
    *,
    price_atol: float = 1e-6,
) -> ParityReport:
    """Compare a bundle's Python expectation dumps against real EA CSVs and
    persist parity_report.json + updated parity_status.json in the bundle."""
    import json
    from pathlib import Path

    from evoquant.mql5.parser import parse_signals_csv, parse_trades_csv

    bundle = Path(bundle_dir)
    status_doc = json.loads((bundle / "parity_status.json").read_text())
    skip = int(status_doc.get("skip_warmup_bars", 0))

    py_signals = parse_signals_csv(bundle / "python_signals.csv")
    py_trades = parse_trades_csv(bundle / "python_trades.csv")
    ea_signals = parse_signals_csv(ea_signals_path)
    ea_trades = parse_trades_csv(ea_trades_path)

    discrepancies = compare_signal_rows(py_signals, ea_signals, skip_warmup_bars=skip)
    discrepancies += compare_trade_rows(py_trades, ea_trades, price_atol=price_atol)
    report = ParityReport(
        status=PASS if not discrepancies else FAIL,
        n_signals_compared=max(len(py_signals) - skip, 0),
        n_trades_compared=min(len(py_trades), len(ea_trades)),
        discrepancies=discrepancies,
    )
    (bundle / "parity_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True)
    )
    status_doc["status"] = report.status
    (bundle / "parity_status.json").write_text(
        json.dumps(status_doc, indent=2, sort_keys=True)
    )
    return report


def run_parity(
    bar_ts_utc_s: np.ndarray,
    policy: SignalPolicy,
    python_trades: tuple[Trade, ...] | list[Trade],
    ea_signals: list[EASignalRow] | None,
    ea_trades: list[EATradeRow] | None,
    *,
    skip_warmup_bars: int,
    price_atol: float,
) -> ParityReport:
    """Full differential run. Missing EA artifacts => PENDING, not PASS."""
    if ea_signals is None or ea_trades is None:
        return ParityReport(status=PENDING_REAL_TICK)
    discrepancies = compare_signals(
        bar_ts_utc_s, policy, ea_signals, skip_warmup_bars=skip_warmup_bars
    )
    discrepancies += compare_trades(
        bar_ts_utc_s, python_trades, ea_trades, price_atol=price_atol
    )
    return ParityReport(
        status=PASS if not discrepancies else FAIL,
        n_signals_compared=max(policy.n_bars - skip_warmup_bars, 0),
        n_trades_compared=min(len(python_trades), len(ea_trades)),
        discrepancies=discrepancies,
    )
