from __future__ import annotations

import pytest

from evoquant.features.ast import ExpressionError, Node


def _tree() -> Node:
    return Node(
        op="gt",
        children=(
            Node(op="zscore", period=20, children=(Node(op="close"),)),
            Node(op="constant", value=0.75, unit="ZSCORE"),
        ),
    )


def test_roundtrip_dict():
    node = _tree()
    clone = Node.from_dict(node.to_dict())
    assert clone == node
    assert clone.serialize() == node.serialize()
    assert clone.structural_hash() == node.structural_hash()


def test_serialize_deterministic():
    assert _tree().serialize() == _tree().serialize()


def test_hash_changes_with_structure():
    a = _tree()
    b = Node(
        op="gt",
        children=(
            Node(op="zscore", period=21, children=(Node(op="close"),)),  # period changed
            Node(op="constant", value=0.75, unit="ZSCORE"),
        ),
    )
    assert a.structural_hash() != b.structural_hash()


def test_invalid_period_rejected():
    with pytest.raises(ExpressionError):
        Node(op="sma", period=0, children=(Node(op="close"),))


def test_from_dict_requires_op():
    with pytest.raises(ExpressionError):
        Node.from_dict({"period": 5})


def test_size_and_depth():
    node = _tree()
    assert node.size() == 4
    assert node.depth() == 3
