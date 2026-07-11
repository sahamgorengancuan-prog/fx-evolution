"""Parity comparator tests against simulated EA output.

The comparator itself is what gets proven here; true EA-vs-Python
equivalence requires a real Strategy Tester run and remains
PENDING_REAL_TICK (see bundle tests).
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np

from evoquant.backtest.costs import ExecutionAssumptions, resolve_costs
from evoquant.backtest.fast_engine import run_backtest
from evoquant.backtest.signals import RiskBlock, SignalPolicy
from evoquant.features.ast import Node
from evoquant.genome.genome import StrategyGenome
from evoquant.mql5.bundle import write_bundle
from evoquant.mql5.parity import (
    FAIL,
    PASS,
    PENDING_REAL_TICK,
    run_parity,
)
from evoquant.mql5.parser import (
    parse_signals_csv,
    parse_trades_csv,
    write_signals_csv,
    write_trades_csv,
)
from tests.conftest import make_bars, make_spec

SPEC = make_spec()
WARMUP = 100


def _n(op: str, *children: Node, **kw) -> Node:
    return Node(op=op, children=tuple(children), **kw)


def _genome() -> StrategyGenome:
    close = _n("close")
    return StrategyGenome(
        symbol="TESTUSDT",
        entry_long=_n("cross_above", close, _n("sma", close, period=8)),
        entry_short=_n("cross_below", close, _n("sma", close, period=8)),
        allow=_n(
            "gt",
            _n("rank", close, period=13),
            Node(op="constant", value=0.1, unit="OSCILLATOR_0_1"),
        ),
        risk=RiskBlock(
            sl_atr=3.0, tp_atr=3.0, risk_per_trade=0.01,
            max_bars=10, cooldown_bars=2, atr_period=14,
        ),
    )


def _setup():
    from evoquant.features.compiler import compile_feature

    bars = make_bars(600, seed=42)
    genome = _genome()
    policy = SignalPolicy(
        entry_long=np.asarray(compile_feature(genome.entry_long).evaluate(bars), dtype=bool),
        entry_short=np.asarray(compile_feature(genome.entry_short).evaluate(bars), dtype=bool),
        allow=np.asarray(compile_feature(genome.allow).evaluate(bars), dtype=bool),
        atr=np.asarray(
            compile_feature(Node(op="atr", period=14)).evaluate(bars), dtype=np.float64
        ),
        risk=genome.risk,
    )
    costs = resolve_costs(
        SPEC,
        bars,
        ExecutionAssumptions(
            label="parity fixture", initial_equity=100_000.0,
            commission_pct_notional=0.001, assumed_spread_points=2.0,
            funding_mode="not_applicable_spot", min_notional=10.0,
        ),
    )
    result = run_backtest(bars, SPEC, policy, costs, 100_000.0)
    assert len(result.trades) > 3, "fixture must trade"
    return bars, genome, policy, result


def _ea_rows_from_python(bars, policy, result, tmp_path):
    """Simulate EA output that matches Python exactly (via the bundle dumps)."""
    paths = write_bundle(
        tmp_path / "bundle", _genome(), SPEC, bars, policy, result,
        deposit=100_000.0, skip_warmup_bars=WARMUP,
    )
    return (
        parse_signals_csv(paths["python_signals"]),
        parse_trades_csv(paths["python_trades"]),
        paths,
    )


class TestParityComparator:
    def test_identical_output_passes_with_zero_discrepancies(self, tmp_path):
        bars, genome, policy, result = _setup()
        ea_signals, ea_trades, _ = _ea_rows_from_python(bars, policy, result, tmp_path)
        report = run_parity(
            bars.ts_utc_s, policy, list(result.trades), ea_signals, ea_trades,
            skip_warmup_bars=WARMUP, price_atol=1e-9,
        )
        assert report.status == PASS
        assert report.discrepancies == []
        assert report.n_trades_compared == len(result.trades)

    def test_flipped_signal_is_itemized(self, tmp_path):
        bars, genome, policy, result = _setup()
        ea_signals, ea_trades, _ = _ea_rows_from_python(bars, policy, result, tmp_path)
        k = WARMUP + 25
        ea_signals[k] = dataclasses.replace(
            ea_signals[k], entry_long=not ea_signals[k].entry_long
        )
        report = run_parity(
            bars.ts_utc_s, policy, list(result.trades), ea_signals, ea_trades,
            skip_warmup_bars=WARMUP, price_atol=1e-9,
        )
        assert report.status == FAIL
        hits = [d for d in report.discrepancies if d.fieldname == "entry_long"]
        assert len(hits) == 1
        assert hits[0].where == str(int(bars.ts_utc_s[k]))

    def test_price_lot_and_reason_divergences_all_reported(self, tmp_path):
        bars, genome, policy, result = _setup()
        ea_signals, ea_trades, _ = _ea_rows_from_python(bars, policy, result, tmp_path)
        ea_trades[0] = dataclasses.replace(ea_trades[0], entry_price=ea_trades[0].entry_price + 0.5)
        ea_trades[1] = dataclasses.replace(ea_trades[1], lots=ea_trades[1].lots + 1.0)
        ea_trades[2] = dataclasses.replace(ea_trades[2], exit_reason="stopped_out")
        report = run_parity(
            bars.ts_utc_s, policy, list(result.trades), ea_signals, ea_trades,
            skip_warmup_bars=WARMUP, price_atol=1e-6,
        )
        assert report.status == FAIL
        fields = {d.fieldname for d in report.discrepancies}
        assert {"entry_price", "lots", "exit_reason"} <= fields

    def test_trade_count_mismatch_is_structural(self, tmp_path):
        bars, genome, policy, result = _setup()
        ea_signals, ea_trades, _ = _ea_rows_from_python(bars, policy, result, tmp_path)
        report = run_parity(
            bars.ts_utc_s, policy, list(result.trades), ea_signals, ea_trades[:-1],
            skip_warmup_bars=WARMUP, price_atol=1e-9,
        )
        assert report.status == FAIL
        assert any(d.level == "structure" and d.fieldname == "n_trades"
                   for d in report.discrepancies)

    def test_missing_ea_artifacts_are_pending_not_pass(self, tmp_path):
        bars, genome, policy, result = _setup()
        report = run_parity(
            bars.ts_utc_s, policy, list(result.trades), None, None,
            skip_warmup_bars=WARMUP, price_atol=1e-9,
        )
        assert report.status == PENDING_REAL_TICK
        assert report.n_signals_compared == 0


class TestCsvRoundTrip:
    def test_signals_and_trades_roundtrip(self, tmp_path):
        bars, genome, policy, result = _setup()
        ea_signals, ea_trades, _ = _ea_rows_from_python(bars, policy, result, tmp_path)
        s2 = parse_signals_csv(write_signals_csv(tmp_path / "s.csv", ea_signals))
        t2 = parse_trades_csv(write_trades_csv(tmp_path / "t.csv", ea_trades))
        assert s2 == ea_signals
        assert t2 == ea_trades


class TestParityCli:
    def test_bundle_parity_cli_pass_and_fail(self, tmp_path):
        from evoquant.cli import main as cli_main

        bars, genome, policy, result = _setup()
        ea_signals, ea_trades, paths = _ea_rows_from_python(bars, policy, result, tmp_path)
        bundle = str(tmp_path / "bundle")

        # identical CSVs => PASS, exit code 0, status file flips to PASS
        rc = cli_main(
            ["mql5-parity", "--bundle", bundle,
             "--signals", paths["python_signals"], "--trades", paths["python_trades"]]
        )
        assert rc == 0
        status = json.loads(open(paths["parity_status"]).read())
        assert status["status"] == PASS
        report = json.loads(open(f"{bundle}/parity_report.json").read())
        assert report["n_discrepancies"] == 0

        # corrupted EA trades => FAIL, exit code 1, itemized report persisted
        bad = write_trades_csv(
            tmp_path / "bad_trades.csv",
            [dataclasses.replace(ea_trades[0], lots=ea_trades[0].lots + 5.0)]
            + ea_trades[1:],
        )
        rc = cli_main(
            ["mql5-parity", "--bundle", bundle,
             "--signals", paths["python_signals"], "--trades", str(bad)]
        )
        assert rc == 1
        report = json.loads(open(f"{bundle}/parity_report.json").read())
        assert any(d["fieldname"] == "lots" for d in report["discrepancies"])


class TestBundle:
    def test_bundle_contents_and_pending_status(self, tmp_path):
        bars, genome, policy, result = _setup()
        _, _, paths = _ea_rows_from_python(bars, policy, result, tmp_path)
        status = json.loads(open(paths["parity_status"]).read())
        assert status["status"] == PENDING_REAL_TICK
        assert "never assumed" in status["reason"]
        instructions = open(paths["instructions"]).read()
        assert "Every tick based on real ticks" in instructions
        assert "terminal64.exe /config:tester.ini" in instructions
        ini = open(paths["strategy_tester_ini"]).read()
        assert "Model=4" in ini
        genome_doc = json.loads(open(paths["genome"]).read())
        assert genome_doc["genome_hash"] == _genome().genome_hash()
