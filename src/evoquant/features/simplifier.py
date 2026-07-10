"""AST simplification and canonicalization.

``simplify`` removes semantics-preserving redundancy (safe to apply to
stored genomes):

* ``abs(abs(x))  -> abs(x)``      (the v8 champion contained this)
* ``abs(neg(x))  -> abs(x)``
* ``neg(neg(x))  -> x``
* ``not(not(x))  -> x``
* nested ``and``/``or`` flattening

``canonicalize`` additionally normalizes representation so semantically
equal trees hash equally:

* ``lt(a, b)         -> gt(b, a)``
* ``cross_below(a,b) -> cross_above(b, a)``
* commutative operators (``and``/``or``/``add``) sort children by their
  canonical serialization

``semantic_hash`` is the identity used for duplicate detection in search
archives and the evolutionary memory.
"""
from __future__ import annotations

from evoquant.features.ast import Node

_INVOLUTIVE = {("neg", "neg"), ("not", "not")}
_ABS_ABSORBS = {"abs", "neg"}
_COMMUTATIVE = {"and", "or", "add"}
_FLATTEN = {"and", "or"}
_DIRECTION_SWAP = {"lt": "gt", "cross_below": "cross_above"}


def simplify(node: Node) -> Node:
    children = tuple(simplify(c) for c in node.children)

    # involutions: neg(neg(x)) -> x ; not(not(x)) -> x
    if children and (node.op, children[0].op) in _INVOLUTIVE:
        return children[0].children[0]

    # abs absorbs abs and neg: abs(abs(x)) -> abs(x); abs(neg(x)) -> abs(x)
    if node.op == "abs" and children and children[0].op in _ABS_ABSORBS:
        return simplify(Node(op="abs", children=(children[0].children[0],)))

    # flatten nested and/or: and(and(a,b),c) -> and(a,b,c)
    if node.op in _FLATTEN:
        flat: list[Node] = []
        for c in children:
            if c.op == node.op:
                flat.extend(c.children)
            else:
                flat.append(c)
        # drop exact duplicates: and(x, x) -> and(x) -> x
        seen: dict[str, Node] = {}
        for c in flat:
            seen.setdefault(c.serialize(), c)
        deduped = tuple(seen.values())
        if len(deduped) == 1:
            return deduped[0]
        children = deduped

    return Node(
        op=node.op,
        period=node.period,
        value=node.value,
        unit=node.unit,
        children=children,
    )


def canonicalize(node: Node) -> Node:
    node = simplify(node)
    return _canonical(node)


def _canonical(node: Node) -> Node:
    children = tuple(_canonical(c) for c in node.children)

    if node.op in _DIRECTION_SWAP and len(children) == 2:
        return Node(op=_DIRECTION_SWAP[node.op], children=(children[1], children[0]))

    if node.op in _COMMUTATIVE:
        children = tuple(sorted(children, key=lambda c: c.serialize()))

    return Node(
        op=node.op,
        period=node.period,
        value=node.value,
        unit=node.unit,
        children=children,
    )


def semantic_hash(node: Node) -> str:
    return canonicalize(node).structural_hash()
