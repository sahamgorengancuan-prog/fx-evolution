"""Operator registry.

Every operator declares: arity, period requirements, complexity cost, its
own causal lookback, a dimension-inference rule, a batch (numpy)
implementation, and a factory for an independent streaming implementation.

Batch and stream share the same window formulas (single source of truth
for semantics) but consume data through different paths — arrays vs.
bar-by-bar buffers — so the parity tests exercise a real difference.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evoquant.data.loaders import BarData
from evoquant.errors import EvoquantError
from evoquant.features.ast import Node
from evoquant.features.types import Dimension


class FeatureTypeError(EvoquantError):
    """Expression is dimensionally invalid (the D5 defect class)."""

    code = "FEATURE_TYPE"


class CompileError(EvoquantError):
    """Expression is structurally invalid for the registry."""

    code = "FEATURE_COMPILE"


class StreamOp:
    """One node's incremental evaluator. Subclasses hold their own state."""

    def update(self, bar: dict[str, float], child_values: Sequence[Any]) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class OpSpec:
    name: str
    #: number of children; -1 = variadic (>= 2)
    arity: int
    needs_period: bool
    min_period: int
    complexity: int
    #: (node, child_dims) -> output dim; raises FeatureTypeError
    infer_dim: Callable[[Node, list[Dimension]], Dimension]
    #: own causal lookback in bars (excluding children's lookback)
    own_lookback: Callable[[Node], int]
    #: (bars, child_arrays, node) -> output array
    batch: Callable[[BarData, list[np.ndarray], Node], np.ndarray]
    #: node -> StreamOp
    stream_factory: Callable[[Node], StreamOp]


REGISTRY: dict[str, OpSpec] = {}


def register(spec: OpSpec) -> None:
    if spec.name in REGISTRY:
        raise CompileError(f"Operator {spec.name!r} already registered")
    REGISTRY[spec.name] = spec


def get_spec(name: str) -> OpSpec:
    spec = REGISTRY.get(name)
    if spec is None:
        raise CompileError(
            f"Unknown operator {name!r}; not in the approved grammar",
            op=name,
            known=sorted(REGISTRY),
        )
    return spec
