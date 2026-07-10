"""Type-safe mutation and crossover operators.

Every operator returns a genome that compiles (validated before return;
an invalid product is discarded and the parent returned unchanged with a
flag — never silently replaced by a random genome).

Operator names are stable identifiers: the Phase-6 contextual bandit
learns posteriors over exactly these names.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from evoquant.backtest.signals import RiskBlock
from evoquant.features.ast import Node
from evoquant.features.registry import CompileError, FeatureTypeError
from evoquant.genome.generator import PERIODS, generate_bool
from evoquant.genome.genome import StrategyGenome

OPERATOR_NAMES = (
    "jitter_periods",
    "jitter_constants",
    "jitter_risk",
    "swap_comparison",
    "replace_entry_long",
    "replace_entry_short",
    "replace_allow",
    "crossover",
)

_SWAPS = {
    "gt": "lt",
    "lt": "gt",
    "cross_above": "cross_below",
    "cross_below": "cross_above",
    "sma": "ema",
    "ema": "sma",
    "highest": "lowest",
    "lowest": "highest",
}


def _map_nodes(node: Node, fn: Callable[[Node], Node]) -> Node:
    children = tuple(_map_nodes(c, fn) for c in node.children)
    return fn(replace(node, children=children))


def _jitter_periods(node: Node, rng: np.random.Generator, prob: float = 0.4) -> Node:
    def fn(n: Node) -> Node:
        if n.period is not None and rng.random() < prob:
            idx = int(np.argmin(np.abs(np.array(PERIODS) - n.period)))
            step = int(rng.choice([-1, 1]))
            new_idx = int(np.clip(idx + step, 0, len(PERIODS) - 1))
            return replace(n, period=PERIODS[new_idx])
        return n

    return _map_nodes(node, fn)


def _jitter_constants(node: Node, rng: np.random.Generator, scale: float = 0.15) -> Node:
    def fn(n: Node) -> Node:
        if n.op == "constant" and n.value is not None and rng.random() < 0.7:
            spread = max(abs(n.value) * scale, 0.01)
            return replace(n, value=round(float(n.value + rng.normal(0, spread)), 4))
        return n

    return _map_nodes(node, fn)


def _swap_ops(node: Node, rng: np.random.Generator, prob: float = 0.3) -> Node:
    def fn(n: Node) -> Node:
        if n.op in _SWAPS and rng.random() < prob:
            return replace(n, op=_SWAPS[n.op])
        return n

    return _map_nodes(node, fn)


def _finalize(
    parent: StrategyGenome,
    candidate: StrategyGenome,
) -> tuple[StrategyGenome, bool]:
    """Validate; on failure return the parent with applied=False.

    Never substitutes a random genome for a failed product (truth
    constraint: no silent random fallback).
    """
    try:
        candidate.validate()
    except (CompileError, FeatureTypeError):
        return parent, False
    if candidate.genome_hash() == parent.genome_hash():
        return parent, False
    return candidate, True


def mutate(
    genome: StrategyGenome,
    operator: str,
    rng: np.random.Generator,
    generation: int,
) -> tuple[StrategyGenome, bool]:
    """Apply a named operator. Returns (genome, applied)."""
    base = replace(
        genome,
        origin="mutation",
        parent_hashes=(genome.genome_hash(),),
        created_generation=generation,
    )
    if operator == "jitter_periods":
        cand = replace(
            base,
            entry_long=_jitter_periods(genome.entry_long, rng),
            entry_short=_jitter_periods(genome.entry_short, rng),
            allow=_jitter_periods(genome.allow, rng),
        )
    elif operator == "jitter_constants":
        cand = replace(
            base,
            entry_long=_jitter_constants(genome.entry_long, rng),
            entry_short=_jitter_constants(genome.entry_short, rng),
            allow=_jitter_constants(genome.allow, rng),
        )
    elif operator == "jitter_risk":
        r = genome.risk
        cand = replace(
            base,
            risk=RiskBlock(
                sl_atr=round(float(np.clip(r.sl_atr * rng.uniform(0.8, 1.25), 0.5, 6.0)), 4),
                tp_atr=round(float(np.clip(r.tp_atr * rng.uniform(0.8, 1.25), 0.5, 6.0)), 4),
                risk_per_trade=r.risk_per_trade,
                max_bars=int(np.clip(r.max_bars + rng.choice([-12, 0, 12]), 6, 96)),
                cooldown_bars=int(np.clip(r.cooldown_bars + rng.choice([-2, 0, 2]), 0, 24)),
                atr_period=r.atr_period,
            ),
        )
    elif operator == "swap_comparison":
        cand = replace(
            base,
            entry_long=_swap_ops(genome.entry_long, rng),
            entry_short=_swap_ops(genome.entry_short, rng),
        )
    elif operator == "replace_entry_long":
        cand = replace(base, entry_long=generate_bool(rng, depth=3))
    elif operator == "replace_entry_short":
        cand = replace(base, entry_short=generate_bool(rng, depth=3))
    elif operator == "replace_allow":
        cand = replace(base, allow=generate_bool(rng, depth=2))
    else:
        raise ValueError(f"unknown mutation operator {operator!r}")
    return _finalize(genome, cand)


def crossover(
    a: StrategyGenome, b: StrategyGenome, rng: np.random.Generator, generation: int
) -> tuple[StrategyGenome, bool]:
    """Exchange whole components between two parents (type-safe by design)."""
    cand = StrategyGenome(
        symbol=a.symbol,
        entry_long=a.entry_long if rng.random() < 0.5 else b.entry_long,
        entry_short=a.entry_short if rng.random() < 0.5 else b.entry_short,
        allow=a.allow if rng.random() < 0.5 else b.allow,
        risk=a.risk if rng.random() < 0.5 else b.risk,
        origin="crossover",
        parent_hashes=(a.genome_hash(), b.genome_hash()),
        created_generation=generation,
    )
    return _finalize(a, cand)
