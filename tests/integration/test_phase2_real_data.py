"""Phase-2 acceptance on real BNBUSDT data.

Compiles typed rewrites of the v8 champion's *valid* parts, evaluates them
on real bars, and proves determinism, causality, and stream parity outside
synthetic fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest

from evoquant.data.loaders import load_fsb_json
from evoquant.features.ast import Node
from evoquant.features.compiler import compile_feature
from evoquant.features.registry import FeatureTypeError
from tests.conftest import REAL_DATA

pytestmark = pytest.mark.skipif(
    not REAL_DATA.exists(), reason="real BNBUSDT data file not present"
)


def n(op: str, *children: Node, **kw) -> Node:
    return Node(op=op, children=tuple(children), **kw)


@pytest.fixture(scope="module")
def bars():
    full, _, _ = load_fsb_json(REAL_DATA)
    # A recent, non-quantized slice keeps the O(n·p) reference engine quick.
    return full.slice(40_000, 46_000)


CLOSE = n("close")
# v8 regime filter, unchanged: chop(144) < 0.4489
V8_FILTER = n(
    "lt", Node(op="chop", period=144), Node(op="constant", value=0.4489, unit="OSCILLATOR_0_1")
)
# v8 short entry, re-typed: zscore(abs(close - ema(close,233)), 144) > 1.06
KELTNER_LIKE = n("abs", n("sub", CLOSE, n("ema", CLOSE, period=233)))
V8_SHORT = n(
    "gt",
    n("zscore", KELTNER_LIKE, period=144),
    Node(op="constant", value=1.06, unit="ZSCORE"),
)


def test_v8_champion_long_entry_still_rejected_on_principle():
    macd_like = n("sub", n("ema", CLOSE, period=3), n("ema", CLOSE, period=6))
    champion_long = n(
        "cross_above",
        n("rank", n("delta", macd_like, period=21), period=233),
        n("ret", CLOSE, period=8),
    )
    with pytest.raises(FeatureTypeError):
        compile_feature(champion_long)


def test_filter_and_entry_evaluate_deterministically(bars):
    for node in (V8_FILTER, V8_SHORT):
        compiled = compile_feature(node)
        a = np.asarray(compiled.evaluate(bars))
        b = np.asarray(compiled.evaluate(bars))
        assert np.array_equal(a, b)
        assert a.dtype == bool
        density = float(np.mean(a))
        assert 0.0 < density < 0.9, f"degenerate signal density {density}"


def test_prefix_invariance_on_real_data(bars):
    compiled = compile_feature(V8_SHORT)
    full = np.asarray(compiled.evaluate(bars))
    prefix = np.asarray(compiled.evaluate(bars.slice(0, 3_000)))
    assert np.array_equal(full[:3_000], prefix)


def test_stream_parity_on_real_data(bars):
    view = bars.slice(0, 1_500)
    compiled = compile_feature(V8_FILTER)
    batch = np.asarray(compiled.evaluate(view))
    ev = compiled.stream()
    stream = np.asarray(
        [
            ev.update(
                float(view.open[i]),
                float(view.high[i]),
                float(view.low[i]),
                float(view.close[i]),
                float(view.volume[i]),
            )
            for i in range(view.n_bars)
        ]
    )
    assert np.array_equal(batch, stream.astype(bool))
