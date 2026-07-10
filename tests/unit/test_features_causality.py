"""Causality acceptance tests: future data must never alter past outputs.

Runs prefix-invariance and future-perturbation probes for EVERY operator in
the registry — a new primitive cannot ship without passing these.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from evoquant.features.ast import Node
from evoquant.features.compiler import compile_feature
from evoquant.features.registry import REGISTRY
from tests.conftest import make_bars


def n(op: str, *children: Node, **kw) -> Node:
    return Node(op=op, children=tuple(children), **kw)


CLOSE = n("close")
UNSCALED = n("delta", CLOSE, period=2)  # signed input for abs/neg
BOOL_A = n("gt", CLOSE, n("sma", CLOSE, period=3))
BOOL_B = n("lt", CLOSE, n("sma", CLOSE, period=8))

#: A minimal valid expression exercising each registered operator.
MINIMAL_NODES: dict[str, Node] = {
    "close": n("close"),
    "open": n("open"),
    "high": n("high"),
    "low": n("low"),
    "volume": n("volume"),
    "constant": Node(op="constant", value=1.5, unit="ZSCORE"),
    "abs": n("abs", UNSCALED),
    "neg": n("neg", UNSCALED),
    "add": n("add", CLOSE, CLOSE),
    "sub": n("sub", CLOSE, n("sma", CLOSE, period=3)),
    "gt": n("gt", CLOSE, n("sma", CLOSE, period=3)),
    "lt": n("lt", CLOSE, n("sma", CLOSE, period=3)),
    "and": n("and", BOOL_A, BOOL_B),
    "or": n("or", BOOL_A, BOOL_B),
    "not": n("not", BOOL_A),
    "cross_above": n("cross_above", CLOSE, n("sma", CLOSE, period=5)),
    "cross_below": n("cross_below", CLOSE, n("sma", CLOSE, period=5)),
    "sma": n("sma", CLOSE, period=5),
    "ema": n("ema", CLOSE, period=5),
    "highest": n("highest", CLOSE, period=5),
    "lowest": n("lowest", CLOSE, period=5),
    "stdev": n("stdev", CLOSE, period=5),
    "zscore": n("zscore", CLOSE, period=5),
    "rank": n("rank", CLOSE, period=5),
    "delta": n("delta", CLOSE, period=5),
    "ret": n("ret", CLOSE, period=5),
    "efficiency_ratio": n("efficiency_ratio", CLOSE, period=5),
    "rsi": n("rsi", CLOSE, period=5),
    "atr": Node(op="atr", period=5),
    "chop": Node(op="chop", period=5),
}


def test_every_registered_op_has_a_causality_probe():
    """A registered primitive without a probe here must fail CI."""
    assert set(MINIMAL_NODES) == set(REGISTRY)


def _assert_equal_nanaware(a: np.ndarray, b: np.ndarray, op: str) -> None:
    if a.dtype == bool:
        assert np.array_equal(a, b), f"{op}: boolean outputs diverged"
    else:
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{op}: NaN pattern diverged"
        mask = ~np.isnan(a)
        assert np.array_equal(a[mask], b[mask]), f"{op}: values diverged"


@pytest.mark.parametrize("op", sorted(MINIMAL_NODES))
def test_prefix_invariance(op: str):
    """Evaluating on a prefix must equal the prefix of the full evaluation."""
    bars = make_bars(300, seed=11)
    compiled = compile_feature(MINIMAL_NODES[op])
    full = compiled.evaluate(bars)
    prefix = compiled.evaluate(bars.slice(0, 200))
    _assert_equal_nanaware(np.asarray(full[:200]), np.asarray(prefix), op)


@pytest.mark.parametrize("op", sorted(MINIMAL_NODES))
def test_future_perturbation_cannot_change_the_past(op: str):
    """Rewriting the last 50 bars must leave the first 250 outputs intact."""
    bars = make_bars(300, seed=11)
    compiled = compile_feature(MINIMAL_NODES[op])
    base = np.asarray(compiled.evaluate(bars))

    perturbed_close = np.array(bars.close)
    perturbed_high = np.array(bars.high)
    perturbed_close[250:] *= 3.7
    perturbed_high[250:] = np.maximum(perturbed_high[250:], perturbed_close[250:] + 1.0)
    perturbed = replace(bars, close=perturbed_close, high=perturbed_high)
    shifted = np.asarray(compiled.evaluate(perturbed))

    _assert_equal_nanaware(base[:250], shifted[:250], op)
