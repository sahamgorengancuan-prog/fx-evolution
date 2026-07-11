"""Parsers for the EA's signal-dump and trade-log CSV contracts.

Format (semicolon-separated, written by the generated EA in dump mode):

    evoquant_signals.csv: time;entry_long;entry_short;allow;atr
    evoquant_trades.csv:  open_time;close_time;direction;lots;
                          entry_price;exit_price;sl;tp;exit_reason
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from evoquant.errors import EvoquantError


class ParseError(EvoquantError):
    code = "MQL5_PARSE"


@dataclass(frozen=True)
class EASignalRow:
    ts_utc_s: int
    entry_long: bool
    entry_short: bool
    allow: bool
    atr: float


@dataclass(frozen=True)
class EATradeRow:
    open_ts_utc_s: int
    close_ts_utc_s: int
    direction: int
    lots: float
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    exit_reason: str


def parse_signals_csv(path: str | Path) -> list[EASignalRow]:
    rows: list[EASignalRow] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader, None)
        if header is None or [h.strip() for h in header[:5]] != [
            "time", "entry_long", "entry_short", "allow", "atr",
        ]:
            raise ParseError("Unexpected signals CSV header", header=str(header), path=str(path))
        for lineno, rec in enumerate(reader, start=2):
            if not rec or all(not c.strip() for c in rec):
                continue
            try:
                rows.append(
                    EASignalRow(
                        ts_utc_s=int(rec[0]),
                        entry_long=rec[1].strip() == "1",
                        entry_short=rec[2].strip() == "1",
                        allow=rec[3].strip() == "1",
                        atr=float(rec[4]),
                    )
                )
            except (ValueError, IndexError) as exc:
                raise ParseError(
                    "Malformed signals CSV row", line=lineno, row=str(rec), path=str(path)
                ) from exc
    return rows


def parse_trades_csv(path: str | Path) -> list[EATradeRow]:
    rows: list[EATradeRow] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter=";")
        header = next(reader, None)
        expected = [
            "open_time", "close_time", "direction", "lots",
            "entry_price", "exit_price", "sl", "tp", "exit_reason",
        ]
        if header is None or [h.strip() for h in header[:9]] != expected:
            raise ParseError("Unexpected trades CSV header", header=str(header), path=str(path))
        for lineno, rec in enumerate(reader, start=2):
            if not rec or all(not c.strip() for c in rec):
                continue
            try:
                rows.append(
                    EATradeRow(
                        open_ts_utc_s=int(rec[0]),
                        close_ts_utc_s=int(rec[1]),
                        direction=int(rec[2]),
                        lots=float(rec[3]),
                        entry_price=float(rec[4]),
                        exit_price=float(rec[5]),
                        sl=float(rec[6]),
                        tp=float(rec[7]),
                        exit_reason=rec[8].strip(),
                    )
                )
            except (ValueError, IndexError) as exc:
                raise ParseError(
                    "Malformed trades CSV row", line=lineno, row=str(rec), path=str(path)
                ) from exc
    return rows


def write_signals_csv(path: str | Path, rows: list[EASignalRow]) -> Path:
    """Writer for the same contract (used for Python expectation dumps)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["time", "entry_long", "entry_short", "allow", "atr"])
        for r in rows:
            writer.writerow(
                [r.ts_utc_s, int(r.entry_long), int(r.entry_short), int(r.allow), f"{r.atr:.10f}"]
            )
    return path


def write_trades_csv(path: str | Path, rows: list[EATradeRow]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(
            ["open_time", "close_time", "direction", "lots",
             "entry_price", "exit_price", "sl", "tp", "exit_reason"]
        )
        for r in rows:
            writer.writerow(
                [r.open_ts_utc_s, r.close_ts_utc_s, r.direction, f"{r.lots:.2f}",
                 f"{r.entry_price:.10f}", f"{r.exit_price:.10f}",
                 f"{r.sl:.10f}", f"{r.tp:.10f}", r.exit_reason]
            )
    return path
