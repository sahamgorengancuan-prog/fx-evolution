"""Execution and parameter stress tests.

These re-run the *actual* Tier-A reference engine under perturbed
conditions — costs scaled up, risk parameters jittered, start offset —
and report metric dispersion. They perturb economics and paths, unlike
the v8 return-permutation which changed neither (audit S2/B-series).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from evoquant.backtest.signals import RiskBlock
from evoquant.genome.genome import StrategyGenome
from evoquant.search.evaluator import EvalContext, evaluate_genome


def cost_stress(
    genome: StrategyGenome,
    ctx: EvalContext,
    spread_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0),
) -> dict[str, Any]:
    """Median fold return as execution costs scale. Fail = edge evaporates."""
    rows: list[dict[str, float]] = []
    for mult in spread_multipliers:
        stressed_costs = replace(
            ctx.costs,
            spread_points=ctx.costs.spread_points * mult,
            slippage_points=ctx.costs.slippage_points * mult,
        )
        ev = evaluate_genome(genome, replace(ctx, costs=stressed_costs))
        rets = np.asarray(ev.fold_returns, dtype=np.float64)
        finite = rets[np.isfinite(rets)]
        rows.append(
            {
                "spread_multiplier": mult,
                "median_fold_return": float(np.median(finite)) if len(finite) else 0.0,
                "n_trades": float(ev.n_trades_total),
                "max_drawdown": ev.max_drawdown,
            }
        )
    base = rows[0]["median_fold_return"]
    worst = rows[-1]["median_fold_return"]
    return {
        "kind": "cost_stress",
        "rows": rows,
        "return_erosion": base - worst,
        "survives_worst_case": worst > 0.0,
    }


def parameter_perturbation(
    genome: StrategyGenome,
    ctx: EvalContext,
    n_perturbations: int = 8,
    scale: float = 0.15,
    seed: int = 0,
) -> dict[str, Any]:
    """Metric dispersion under risk-parameter jitter (cliff detection)."""
    rng = np.random.default_rng(seed)
    base_ev = evaluate_genome(genome, ctx)
    base_rets = np.asarray(base_ev.fold_returns, dtype=np.float64)
    base_median = float(np.median(base_rets[np.isfinite(base_rets)]))

    medians: list[float] = []
    for _ in range(n_perturbations):
        r = genome.risk
        jittered = RiskBlock(
            sl_atr=float(np.clip(r.sl_atr * rng.uniform(1 - scale, 1 + scale), 0.5, 6.0)),
            tp_atr=float(np.clip(r.tp_atr * rng.uniform(1 - scale, 1 + scale), 0.5, 6.0)),
            risk_per_trade=r.risk_per_trade,
            max_bars=max(6, int(r.max_bars * rng.uniform(1 - scale, 1 + scale))),
            cooldown_bars=r.cooldown_bars,
            atr_period=r.atr_period,
        )
        ev = evaluate_genome(replace(genome, risk=jittered), ctx)
        rets = np.asarray(ev.fold_returns, dtype=np.float64)
        finite = rets[np.isfinite(rets)]
        medians.append(float(np.median(finite)) if len(finite) else 0.0)

    arr = np.asarray(medians)
    return {
        "kind": "parameter_perturbation",
        "base_median_return": base_median,
        "perturbed_min": float(np.min(arr)),
        "perturbed_median": float(np.median(arr)),
        "perturbed_max": float(np.max(arr)),
        "dispersion": float(np.std(arr)),
        "sign_flips": int(np.sum(np.sign(arr) != np.sign(base_median)))
        if base_median != 0
        else 0,
        "n_perturbations": n_perturbations,
    }


def start_offset_stress(
    genome: StrategyGenome,
    ctx: EvalContext,
    offsets: tuple[int, ...] = (0, 8, 16, 24),
) -> dict[str, Any]:
    """Sensitivity to evaluation-window start (calendar-luck detection)."""
    rows: list[dict[str, float]] = []
    for off in offsets:
        shifted = tuple(
            type(r)(r.start + off, r.stop) for r in ctx.eval_ranges if r.start + off < r.stop
        )
        if not shifted:
            continue
        ev = evaluate_genome(genome, replace(ctx, eval_ranges=shifted))
        rets = np.asarray(ev.fold_returns, dtype=np.float64)
        finite = rets[np.isfinite(rets)]
        rows.append(
            {
                "offset_bars": float(off),
                "median_fold_return": float(np.median(finite)) if len(finite) else 0.0,
            }
        )
    values = np.asarray([r["median_fold_return"] for r in rows])
    return {
        "kind": "start_offset_stress",
        "rows": rows,
        "dispersion": float(np.std(values)) if len(values) else 0.0,
    }
