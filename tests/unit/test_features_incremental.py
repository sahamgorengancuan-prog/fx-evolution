"""Incremental (streaming) vs batch parity to 1e-12.

The streaming evaluator consumes data bar-by-bar through independent state
(deques/recursions); the batch path uses arrays. Agreement demonstrates the
batch implementations are causal and both paths share one semantics.
"""
from __future__ import annotations

import numpy as np
import pytest

from evoquant.features.ast import Node
from evoquant.features.compiler import compile_feature
from tests.conftest import make_bars

ATOL = 1e-12


def n(op: str, *children: Node, **kw) -> Node:
    return Node(op=op, children=tuple(children), **kw)


CLOSE = n("close")
MACD_LIKE = n("sub", n("ema", CLOSE, period=13), n("ema", CLOSE, period=55))

EXPRESSIONS = {
    "blueprint_zscore_macd": n("zscore", MACD_LIKE, period=89),
    "cross_zscore_const": n(
        "cross_above",
        n("zscore", MACD_LIKE, period=21),
        Node(op="constant", value=0.75, unit="ZSCORE"),
    ),
    "regime_and_rsi": n(
        "and",
        n(
            "lt",
            Node(op="chop", period=14),
            Node(op="constant", value=0.55, unit="OSCILLATOR_0_1"),
        ),
        n(
            "gt",
            n("rsi", CLOSE, period=14),
            Node(op="constant", value=55.0, unit="OSCILLATOR_0_100"),
        ),
    ),
    "rank_of_delta": n("rank", n("delta", MACD_LIKE, period=21), period=50),
    "atr_vs_stdev": n("gt", Node(op="atr", period=14), n("stdev", CLOSE, period=14)),
    "ret_rank": n(
        "gt",
        n("rank", n("ret", CLOSE, period=8), period=34),
        Node(op="constant", value=0.9, unit="OSCILLATOR_0_1"),
    ),
    "highest_lowest_er": n(
        "and",
        n("gt", CLOSE, n("highest", n("sma", CLOSE, period=5), period=20)),
        n(
            "gt",
            n("efficiency_ratio", CLOSE, period=10),
            Node(op="constant", value=0.3, unit="OSCILLATOR_0_1"),
        ),
    ),
}


def _stream_all(compiled, bars) -> np.ndarray:
    ev = compiled.stream()
    out = [
        ev.update(
            float(bars.open[i]),
            float(bars.high[i]),
            float(bars.low[i]),
            float(bars.close[i]),
            float(bars.volume[i]),
        )
        for i in range(bars.n_bars)
    ]
    return np.asarray(out)


@pytest.mark.parametrize("name", sorted(EXPRESSIONS))
def test_stream_matches_batch(name: str):
    bars = make_bars(400, seed=23)
    compiled = compile_feature(EXPRESSIONS[name])
    batch = np.asarray(compiled.evaluate(bars))
    stream = _stream_all(compiled, bars)

    if batch.dtype == bool:
        assert np.array_equal(batch, stream.astype(bool)), name
    else:
        stream = stream.astype(np.float64)
        assert np.array_equal(np.isnan(batch), np.isnan(stream)), f"{name}: NaN pattern"
        mask = ~np.isnan(batch)
        np.testing.assert_allclose(
            batch[mask], stream[mask], atol=ATOL, rtol=0.0, err_msg=name
        )


def test_stream_is_restartable_and_deterministic():
    bars = make_bars(200, seed=5)
    compiled = compile_feature(EXPRESSIONS["blueprint_zscore_macd"])
    a = _stream_all(compiled, bars)
    b = _stream_all(compiled, bars)  # fresh evaluator each call
    assert np.array_equal(np.isnan(a), np.isnan(b))
    mask = ~np.isnan(a)
    assert np.array_equal(a[mask], b[mask])
