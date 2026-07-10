from __future__ import annotations

from evoquant.features.ast import Node
from evoquant.features.compiler import compile_feature
from evoquant.features.simplifier import canonicalize, semantic_hash, simplify


def n(op: str, *children: Node, **kw) -> Node:
    return Node(op=op, children=tuple(children), **kw)


CLOSE = n("close")
MACD_LIKE = n("sub", n("ema", CLOSE, period=3), n("ema", CLOSE, period=6))
BOOL_A = n("gt", CLOSE, n("sma", CLOSE, period=5))
BOOL_B = n("lt", CLOSE, n("sma", CLOSE, period=13))


def test_abs_abs_collapses():
    """The v8 champion contained zscore(abs(abs(keltner(233))),144)."""
    tree = n("abs", n("abs", MACD_LIKE))
    assert simplify(tree) == n("abs", MACD_LIKE)


def test_v8_hall_of_fame_duplicates_detected():
    """v8 HOF rank 1 used abs(abs(x)); rank 2 used abs(x) — same mechanism.

    Semantic hashing must identify them so search never wastes budget on
    syntactic no-ops again.
    """
    with_redundancy = n("zscore", n("abs", n("abs", MACD_LIKE)), period=144)
    without = n("zscore", n("abs", MACD_LIKE), period=144)
    assert semantic_hash(with_redundancy) == semantic_hash(without)


def test_neg_neg_and_abs_neg():
    assert simplify(n("neg", n("neg", MACD_LIKE))) == MACD_LIKE
    assert simplify(n("abs", n("neg", MACD_LIKE))) == n("abs", MACD_LIKE)


def test_not_not():
    assert simplify(n("not", n("not", BOOL_A))) == BOOL_A


def test_commutative_hash_equality():
    assert semantic_hash(n("and", BOOL_A, BOOL_B)) == semantic_hash(n("and", BOOL_B, BOOL_A))
    assert semantic_hash(n("or", BOOL_A, BOOL_B)) == semantic_hash(n("or", BOOL_B, BOOL_A))
    assert semantic_hash(n("add", CLOSE, n("sma", CLOSE, period=5))) == semantic_hash(
        n("add", n("sma", CLOSE, period=5), CLOSE)
    )


def test_direction_normalization():
    a, b = CLOSE, n("sma", CLOSE, period=5)
    assert semantic_hash(n("lt", a, b)) == semantic_hash(n("gt", b, a))
    assert semantic_hash(n("cross_below", a, b)) == semantic_hash(n("cross_above", b, a))
    # but a genuine direction difference stays distinct
    assert semantic_hash(n("gt", a, b)) != semantic_hash(n("gt", b, a))


def test_nested_and_flattens_and_dedupes():
    nested = n("and", n("and", BOOL_A, BOOL_B), BOOL_A)
    flat = simplify(nested)
    assert flat.op == "and"
    assert set(c.serialize() for c in flat.children) == {
        BOOL_A.serialize(),
        BOOL_B.serialize(),
    }
    # and(x, x) -> x
    assert simplify(n("and", BOOL_A, BOOL_A)) == BOOL_A


def test_simplified_tree_still_compiles_with_same_dimension():
    tree = n("zscore", n("abs", n("abs", MACD_LIKE)), period=144)
    before = compile_feature(tree)
    after = compile_feature(simplify(tree))
    assert after.dimension is before.dimension
    assert after.complexity < before.complexity  # redundancy removed


def test_canonicalize_is_idempotent():
    tree = n("and", n("lt", CLOSE, n("sma", CLOSE, period=5)), BOOL_A)
    once = canonicalize(tree)
    twice = canonicalize(once)
    assert once == twice
