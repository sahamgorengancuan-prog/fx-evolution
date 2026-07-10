"""Type-directed random expression generation.

Trees are grown *by dimension*: to produce a BOOL we pick a comparison
whose two sides are generated in the same comparable dimension, so every
generated tree compiles under the Phase-2 type system by construction
(and a property test re-verifies compilation anyway — belt and braces).

All randomness flows from the numpy Generator passed in; identical seeds
give identical genomes.
"""
from __future__ import annotations

import numpy as np

from evoquant.features.ast import Node
from evoquant.features.types import Dimension

#: Fibonacci-ish period menu (bounded so lookbacks stay sane on H1)
PERIODS = (3, 5, 8, 13, 21, 34, 55, 89, 144)

#: constant ranges per dimension for comparison thresholds
CONST_RANGES: dict[Dimension, tuple[float, float]] = {
    Dimension.ZSCORE: (-2.5, 2.5),
    Dimension.OSCILLATOR_0_1: (0.05, 0.95),
    Dimension.OSCILLATOR_0_100: (10.0, 90.0),
    Dimension.RETURN: (-0.05, 0.05),
}

_COMPARISON_OPS = ("gt", "lt", "cross_above", "cross_below")
_COMPARABLE_DIMS = (
    Dimension.PRICE,
    Dimension.RETURN,
    Dimension.ZSCORE,
    Dimension.OSCILLATOR_0_1,
    Dimension.OSCILLATOR_0_100,
    Dimension.VOLATILITY,
)


def _period(rng: np.random.Generator) -> int:
    return int(rng.choice(PERIODS))


def _price(rng: np.random.Generator, depth: int) -> Node:
    close = Node(op="close")
    if depth <= 0 or rng.random() < 0.4:
        return close
    op = str(rng.choice(["sma", "ema", "highest", "lowest"]))
    return Node(op=op, period=_period(rng), children=(_price(rng, depth - 1),))


def _unscaled(rng: np.random.Generator, depth: int) -> Node:
    if rng.random() < 0.5:
        return Node(
            op="sub", children=(_price(rng, depth - 1), _price(rng, depth - 1))
        )
    return Node(op="delta", period=_period(rng), children=(_price(rng, depth - 1),))


def generate_series(rng: np.random.Generator, dim: Dimension, depth: int = 3) -> Node:
    """Generate a random series node of the requested dimension."""
    if dim is Dimension.PRICE:
        return _price(rng, depth)
    if dim is Dimension.RETURN:
        return Node(op="ret", period=_period(rng), children=(_price(rng, depth - 1),))
    if dim is Dimension.ZSCORE:
        inner_dim = Dimension.UNSCALED if rng.random() < 0.5 else Dimension.PRICE
        inner = (
            _unscaled(rng, depth - 1)
            if inner_dim is Dimension.UNSCALED
            else _price(rng, depth - 1)
        )
        return Node(op="zscore", period=_period(rng), children=(inner,))
    if dim is Dimension.OSCILLATOR_0_1:
        r = rng.random()
        if r < 0.34:
            return Node(op="chop", period=_period(rng))
        if r < 0.67:
            return Node(
                op="efficiency_ratio", period=_period(rng), children=(_price(rng, depth - 1),)
            )
        inner = _unscaled(rng, depth - 1) if rng.random() < 0.5 else _price(rng, depth - 1)
        return Node(op="rank", period=_period(rng), children=(inner,))
    if dim is Dimension.OSCILLATOR_0_100:
        return Node(op="rsi", period=_period(rng), children=(_price(rng, depth - 1),))
    if dim is Dimension.VOLATILITY:
        if rng.random() < 0.5:
            return Node(op="atr", period=_period(rng))
        return Node(op="stdev", period=_period(rng), children=(_price(rng, depth - 1),))
    if dim is Dimension.UNSCALED:
        return _unscaled(rng, depth)
    raise ValueError(f"cannot generate series of dimension {dim}")


def generate_bool(rng: np.random.Generator, depth: int = 3) -> Node:
    """Generate a random BOOL expression (entry rule / filter)."""
    if depth > 1 and rng.random() < 0.25:
        combiner = str(rng.choice(["and", "or"]))
        return Node(
            op=combiner,
            children=(generate_bool(rng, depth - 1), generate_bool(rng, depth - 1)),
        )
    dim = Dimension(rng.choice([d.value for d in _COMPARABLE_DIMS]))
    left = generate_series(rng, dim, depth)
    op = str(rng.choice(_COMPARISON_OPS))
    # compare against a constant when the dimension has a natural scale,
    # otherwise against another series of the same dimension
    if dim in CONST_RANGES and rng.random() < 0.6 and op in ("gt", "lt"):
        lo, hi = CONST_RANGES[dim]
        value = round(float(rng.uniform(lo, hi)), 4)
        right: Node = Node(op="constant", value=value, unit=dim.value)
    else:
        right = generate_series(rng, dim, depth - 1)
    return Node(op=op, children=(left, right))
