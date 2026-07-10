"""Type-system tests. The centerpiece: the v8 champion's entry rules are
compile-time errors in the new grammar."""
from __future__ import annotations

import pytest

from evoquant.features.ast import Node
from evoquant.features.compiler import compile_feature, infer_dimension
from evoquant.features.registry import CompileError, FeatureTypeError
from evoquant.features.types import Dimension


def n(op: str, *children: Node, **kw) -> Node:
    return Node(op=op, children=tuple(children), **kw)


CLOSE = n("close")

# v8's `macd(3)` expressed in the typed grammar: a difference of two EMAs.
MACD_LIKE = n("sub", n("ema", CLOSE, period=3), n("ema", CLOSE, period=6))


class TestV8ChampionRejection:
    def test_champion_entry_long_is_a_type_error(self):
        """cross_above(rank(delta(macd(3),21),233), ret(8)) — audit D5.

        A [0,1] rank crossed against a raw return must not compile.
        """
        champion_long = n(
            "cross_above",
            n("rank", n("delta", MACD_LIKE, period=21), period=233),
            n("ret", CLOSE, period=8),
        )
        with pytest.raises(FeatureTypeError) as exc:
            compile_feature(champion_long)
        assert "OSCILLATOR_0_1" in str(exc.value)
        assert "RETURN" in str(exc.value)

    def test_unnormalized_macd_cannot_be_compared(self):
        """sub(ema,ema) is UNSCALED; comparing it directly is rejected."""
        with pytest.raises(FeatureTypeError) as exc:
            compile_feature(n("gt", MACD_LIKE, n("constant", value=0.0, unit="PRICE")))
        assert "same dimension" in str(exc.value) or "not comparable" in str(exc.value)

    def test_normalized_macd_is_fine(self):
        """zscore(sub(ema,ema)) > const ZSCORE — the blueprint's example."""
        rule = n(
            "gt",
            n("zscore", MACD_LIKE, period=89),
            n("constant", value=0.75, unit="ZSCORE"),
        )
        compiled = compile_feature(rule)
        assert compiled.dimension is Dimension.BOOL

    def test_champion_regime_filter_is_valid(self):
        """chop(144) < 0.4489 was dimensionally sound in v8; still compiles."""
        rule = n(
            "lt",
            Node(op="chop", period=144),
            n("constant", value=0.4489, unit="OSCILLATOR_0_1"),
        )
        assert compile_feature(rule).dimension is Dimension.BOOL


class TestDimensionRules:
    def test_price_vs_zscore_constant_rejected(self):
        with pytest.raises(FeatureTypeError):
            infer_dimension(n("gt", CLOSE, n("constant", value=1.2, unit="ZSCORE")))

    def test_add_different_dims_rejected(self):
        with pytest.raises(FeatureTypeError):
            infer_dimension(n("add", CLOSE, n("volume")))

    def test_logic_requires_bool(self):
        with pytest.raises(FeatureTypeError):
            infer_dimension(n("and", CLOSE, n("gt", CLOSE, n("sma", CLOSE, period=5))))

    def test_abs_of_price_rejected(self):
        # |price| is meaningless; abs is for signed dims only
        with pytest.raises(FeatureTypeError):
            infer_dimension(n("abs", CLOSE))

    def test_abs_of_unscaled_ok(self):
        assert infer_dimension(n("abs", MACD_LIKE)) is Dimension.UNSCALED

    def test_ret_requires_price(self):
        with pytest.raises(FeatureTypeError):
            infer_dimension(n("ret", n("volume"), period=5))

    def test_expected_output_dims(self):
        cases = {
            Dimension.PRICE: n("sma", CLOSE, period=5),
            Dimension.RETURN: n("ret", CLOSE, period=5),
            Dimension.ZSCORE: n("zscore", CLOSE, period=5),
            Dimension.OSCILLATOR_0_1: n("rank", CLOSE, period=5),
            Dimension.OSCILLATOR_0_100: n("rsi", CLOSE, period=5),
            Dimension.VOLATILITY: Node(op="atr", period=5),
            Dimension.UNSCALED: n("delta", CLOSE, period=5),
            Dimension.BOOL: n("gt", CLOSE, n("sma", CLOSE, period=5)),
        }
        for expected, node in cases.items():
            assert infer_dimension(node) is expected, node.op

    def test_constant_bool_and_unscaled_rejected(self):
        with pytest.raises(FeatureTypeError):
            infer_dimension(Node(op="constant", value=1.0, unit="BOOL"))
        with pytest.raises(FeatureTypeError):
            infer_dimension(Node(op="constant", value=1.0, unit="UNSCALED"))
        with pytest.raises(FeatureTypeError):
            infer_dimension(Node(op="constant", value=1.0, unit="FURLONGS"))


class TestStructuralValidation:
    def test_unknown_op_rejected(self):
        with pytest.raises(CompileError) as exc:
            infer_dimension(Node(op="keltner", period=233))
        assert "approved grammar" in str(exc.value)

    def test_missing_period_rejected(self):
        with pytest.raises(CompileError):
            infer_dimension(n("sma", CLOSE))

    def test_period_below_minimum_rejected(self):
        with pytest.raises(CompileError):
            infer_dimension(n("zscore", CLOSE, period=1))

    def test_stray_period_rejected(self):
        with pytest.raises(CompileError):
            infer_dimension(Node(op="close", period=5))

    def test_stray_value_rejected(self):
        with pytest.raises(CompileError):
            infer_dimension(Node(op="sma", period=5, value=1.0, children=(CLOSE,)))

    def test_wrong_arity_rejected(self):
        with pytest.raises(CompileError):
            infer_dimension(n("gt", CLOSE))
        with pytest.raises(CompileError):
            infer_dimension(Node(op="and", children=(n("gt", CLOSE, CLOSE),)))


class TestAccounting:
    def test_lookback_accumulates_along_paths(self):
        assert compile_feature(n("sma", CLOSE, period=5)).lookback == 4
        assert compile_feature(n("delta", CLOSE, period=5)).lookback == 5
        nested = n("zscore", n("sma", CLOSE, period=10), period=20)
        assert compile_feature(nested).lookback == 19 + 9

    def test_complexity_deterministic(self):
        rule = n(
            "gt",
            n("zscore", MACD_LIKE, period=89),
            n("constant", value=0.75, unit="ZSCORE"),
        )
        c1 = compile_feature(rule).complexity
        c2 = compile_feature(rule).complexity
        assert c1 == c2 > 0
