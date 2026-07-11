"""Export bundle: everything a human needs to run real-tick validation.

A bundle directory contains the generated EA, the Strategy Tester
config (Model=4 — every tick based on real ticks), the archived
instrument spec and genome, the Python-side expectation dumps
(signals + trades in the EA's own CSV contract), the exact manual run
instructions, and a parity status file that starts PENDING_REAL_TICK.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from evoquant.backtest.fast_engine import BacktestResult
from evoquant.backtest.signals import SignalPolicy
from evoquant.data.contracts import InstrumentSpec
from evoquant.data.loaders import BarData
from evoquant.genome.genome import StrategyGenome
from evoquant.mql5.exporter import RUN_INSTRUCTIONS, export_ea, strategy_tester_ini
from evoquant.mql5.parity import PENDING_REAL_TICK
from evoquant.mql5.parser import EASignalRow, EATradeRow, write_signals_csv, write_trades_csv


def write_bundle(
    out_dir: str | Path,
    genome: StrategyGenome,
    spec: InstrumentSpec,
    bars: BarData,
    policy: SignalPolicy,
    result: BacktestResult,
    deposit: float,
    skip_warmup_bars: int,
) -> dict[str, str]:
    """Write the full export bundle; returns {artifact: path}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ea_name = f"Evoquant_{genome.symbol}_{genome.genome_hash()[:10]}"

    (out / f"{ea_name}.mq5").write_text(export_ea(genome, spec))
    iso = _dt.datetime.fromtimestamp(int(bars.ts_utc_s[0]), _dt.UTC)
    iso_end = _dt.datetime.fromtimestamp(int(bars.ts_utc_s[-1]), _dt.UTC)
    (out / "tester.ini").write_text(
        strategy_tester_ini(
            symbol=genome.symbol,
            ea_name=ea_name,
            from_date=iso.strftime("%Y.%m.%d"),
            to_date=iso_end.strftime("%Y.%m.%d"),
            deposit=deposit,
        )
    )
    (out / "RUN_INSTRUCTIONS.md").write_text(
        RUN_INSTRUCTIONS.format(symbol=genome.symbol, ea_name=ea_name)
    )
    (out / "genome.json").write_text(json.dumps(genome.to_dict(), indent=2, sort_keys=True))
    (out / "instrument_spec.json").write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True)
    )

    # Python-side expectations, serialized in the EA's own CSV contract
    signal_rows = [
        EASignalRow(
            ts_utc_s=int(bars.ts_utc_s[i]),
            entry_long=bool(policy.entry_long[i]),
            entry_short=bool(policy.entry_short[i]),
            allow=bool(policy.allow[i]),
            atr=float(policy.atr[i]) if np.isfinite(policy.atr[i]) else 0.0,
        )
        for i in range(policy.n_bars)
    ]
    write_signals_csv(out / "python_signals.csv", signal_rows)
    trade_rows = [
        EATradeRow(
            open_ts_utc_s=int(bars.ts_utc_s[t.entry_index]),
            close_ts_utc_s=int(bars.ts_utc_s[t.exit_index]),
            direction=t.direction,
            lots=t.lots,
            entry_price=t.entry_price,
            exit_price=t.exit_price,
            sl=t.sl_price,
            tp=t.tp_price,
            exit_reason=t.exit_reason,
        )
        for t in result.trades
    ]
    write_trades_csv(out / "python_trades.csv", trade_rows)

    (out / "parity_status.json").write_text(
        json.dumps(
            {
                "status": PENDING_REAL_TICK,
                "reason": (
                    "No MetaTrader terminal available in this environment; "
                    "real-tick validation must be executed manually per "
                    "RUN_INSTRUCTIONS.md. Success is never assumed."
                ),
                "skip_warmup_bars": skip_warmup_bars,
                "genome_hash": genome.genome_hash(),
                "created_at_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return {
        "ea": str(out / f"{ea_name}.mq5"),
        "strategy_tester_ini": str(out / "tester.ini"),
        "instructions": str(out / "RUN_INSTRUCTIONS.md"),
        "genome": str(out / "genome.json"),
        "instrument_spec": str(out / "instrument_spec.json"),
        "python_signals": str(out / "python_signals.csv"),
        "python_trades": str(out / "python_trades.csv"),
        "parity_status": str(out / "parity_status.json"),
    }
