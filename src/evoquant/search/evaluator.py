"""Candidate evaluation: genome -> compiled signals -> Tier-A folds.

Evaluation runs on inner-validation blocks of the sealed split plan's
training region only. Each block is backtested on a window that starts
``warmup`` bars earlier so features are converged, but *metrics are taken
from the validation segment only*. The engine is the Phase-3 reference
Tier A — approximate by label, honest by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evoquant.backtest.costs import ResolvedCosts
from evoquant.backtest.fast_engine import BacktestResult, run_backtest
from evoquant.backtest.metrics import max_drawdown
from evoquant.backtest.signals import SignalPolicy
from evoquant.data.contracts import InstrumentSpec
from evoquant.data.loaders import BarData
from evoquant.data.splits import IndexRange
from evoquant.features.ast import Node
from evoquant.features.compiler import compile_feature
from evoquant.genome.genome import StrategyGenome
from evoquant.search.objectives import CandidateEvaluation


@dataclass(frozen=True)
class EvalContext:
    bars: BarData  # research view ONLY (lockbox physically excluded)
    spec: InstrumentSpec
    costs: ResolvedCosts
    eval_ranges: tuple[IndexRange, ...]  # inner validation blocks
    initial_equity: float
    warmup_bars: int = 300


def _compile_policy(genome: StrategyGenome, bars: BarData) -> SignalPolicy:
    entry_long = np.asarray(compile_feature(genome.entry_long).evaluate(bars), dtype=bool)
    entry_short = np.asarray(compile_feature(genome.entry_short).evaluate(bars), dtype=bool)
    allow = np.asarray(compile_feature(genome.allow).evaluate(bars), dtype=bool)
    atr = np.asarray(
        compile_feature(Node(op="atr", period=genome.risk.atr_period)).evaluate(bars),
        dtype=np.float64,
    )
    return SignalPolicy(
        entry_long=entry_long,
        entry_short=entry_short,
        allow=allow,
        atr=atr,
        risk=genome.risk,
        meta={"genome_hash": genome.genome_hash()},
    )


def evaluate_genome(genome: StrategyGenome, ctx: EvalContext) -> CandidateEvaluation:
    fold_returns: list[float] = []
    n_trades = 0
    dds: list[float] = []
    wins_sum = 0.0
    losses_sum = 0.0
    win_count = 0
    long_count = 0
    holding_bars = 0
    eval_bars_total = 0
    engine_label = "NONE"

    for rng_ in ctx.eval_ranges:
        start = max(0, rng_.start - ctx.warmup_bars)
        window = ctx.bars.slice(start, rng_.stop)
        policy = _compile_policy(genome, window)
        result: BacktestResult = run_backtest(
            window, ctx.spec, policy, _slice_costs(ctx.costs, start, rng_.stop), ctx.initial_equity
        )
        engine_label = result.engine
        offset = rng_.start - start
        eq = result.equity[offset:]
        eval_bars_total += len(eq)
        fold_returns.append(
            float(eq[-1] / eq[0] - 1.0) if len(eq) > 1 and eq[0] > 0 else 0.0
        )
        dds.append(max_drawdown(eq))
        for t in result.trades:
            if t.entry_index < offset:
                continue  # trade belongs to the warmup segment
            n_trades += 1
            holding_bars += t.exit_index - t.entry_index + 1
            if t.direction == 1:
                long_count += 1
            if t.pnl_quote > 0:
                wins_sum += t.pnl_quote
                win_count += 1
            else:
                losses_sum += -t.pnl_quote

    pf = (wins_sum / losses_sum) if losses_sum > 0 else (None if wins_sum == 0 else 10.0)
    return CandidateEvaluation(
        genome_hash=genome.genome_hash(),
        fold_returns=tuple(fold_returns),
        n_trades_total=n_trades,
        max_drawdown=float(max(dds)) if dds else 1.0,
        profit_factor=pf,
        complexity=genome.complexity(),
        diagnostics={
            "trade_frequency": n_trades / eval_bars_total if eval_bars_total else 0.0,
            "avg_holding_bars": holding_bars / n_trades if n_trades else 0.0,
            "long_share": long_count / n_trades if n_trades else 0.5,
            "win_rate": win_count / n_trades if n_trades else 0.0,
            "engine": engine_label,
        },
    )


def _slice_costs(costs: ResolvedCosts, start: int, stop: int) -> ResolvedCosts:
    from dataclasses import replace

    return replace(costs, spread_points=costs.spread_points[start:stop])
