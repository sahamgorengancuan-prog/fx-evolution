"""Compiler: type-check an expression tree and produce evaluators.

Compilation performs, in order:
1. structural validation (known op, arity, period presence/bounds,
   no stray value/unit fields);
2. dimension inference bottom-up — dimensionally invalid trees are
   rejected here with :class:`FeatureTypeError`;
3. lookback and complexity accounting.

The result exposes two independent evaluation paths:
* ``evaluate(bars)`` — causal batch evaluation over arrays;
* ``stream()`` — a bar-by-bar incremental evaluator.
Their agreement (to 1e-12) is enforced by the Phase-2 parity tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evoquant.data.loaders import BarData
from evoquant.features import primitives  # noqa: F401 - populates the registry
from evoquant.features.ast import Node
from evoquant.features.registry import CompileError, StreamOp, get_spec
from evoquant.features.types import Dimension


def infer_dimension(node: Node) -> Dimension:
    """Bottom-up dimension inference; raises on invalid trees."""
    spec = get_spec(node.op)

    if spec.arity >= 0 and len(node.children) != spec.arity:
        raise CompileError(
            f"Operator {node.op!r} expects {spec.arity} children, got {len(node.children)}",
            op=node.op,
        )
    if spec.arity == -1 and len(node.children) < 2:
        raise CompileError(
            f"Variadic operator {node.op!r} expects >= 2 children", op=node.op
        )
    if spec.needs_period:
        if node.period is None:
            raise CompileError(f"Operator {node.op!r} requires a period", op=node.op)
        if node.period < spec.min_period:
            raise CompileError(
                f"Operator {node.op!r} requires period >= {spec.min_period}",
                op=node.op,
                period=node.period,
            )
    elif node.period is not None:
        raise CompileError(
            f"Operator {node.op!r} does not take a period", op=node.op, period=node.period
        )
    if node.op != "constant" and (node.value is not None or node.unit is not None):
        raise CompileError(
            f"Operator {node.op!r} does not take value/unit fields", op=node.op
        )

    child_dims = [infer_dimension(c) for c in node.children]
    return spec.infer_dim(node, child_dims)


def total_lookback(node: Node) -> int:
    spec = get_spec(node.op)
    own = spec.own_lookback(node)
    child_max = max((total_lookback(c) for c in node.children), default=0)
    return own + child_max


def total_complexity(node: Node) -> int:
    spec = get_spec(node.op)
    return spec.complexity + sum(total_complexity(c) for c in node.children)


class StreamEvaluator:
    """Bar-by-bar evaluator for a compiled expression tree."""

    def __init__(self, node: Node) -> None:
        self._node = node
        self._op: StreamOp = get_spec(node.op).stream_factory(node)
        self._children = [StreamEvaluator(c) for c in node.children]

    def update(self, o: float, h: float, l: float, c: float, v: float) -> Any:
        bar = {"open": o, "high": h, "low": l, "close": c, "volume": v}
        return self._update(bar)

    def _update(self, bar: dict[str, float]) -> Any:
        child_values = [child._update(bar) for child in self._children]
        return self._op.update(bar, child_values)


@dataclass(frozen=True)
class CompiledFeature:
    node: Node
    dimension: Dimension
    lookback: int
    complexity: int

    def evaluate(self, bars: BarData) -> np.ndarray:
        """Causal batch evaluation. Bool array for BOOL, float64 otherwise."""
        memo: dict[str, np.ndarray] = {}
        return self._eval(self.node, bars, memo)

    @staticmethod
    def _eval(node: Node, bars: BarData, memo: dict[str, np.ndarray]) -> np.ndarray:
        key = node.serialize()
        cached = memo.get(key)
        if cached is not None:
            return cached
        spec = get_spec(node.op)
        child_arrays = [CompiledFeature._eval(c, bars, memo) for c in node.children]
        out = spec.batch(bars, child_arrays, node)
        memo[key] = out
        return out

    def stream(self) -> StreamEvaluator:
        return StreamEvaluator(self.node)


def compile_feature(node: Node) -> CompiledFeature:
    dim = infer_dimension(node)
    return CompiledFeature(
        node=node,
        dimension=dim,
        lookback=total_lookback(node),
        complexity=total_complexity(node),
    )
