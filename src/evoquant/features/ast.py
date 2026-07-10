"""Typed expression AST.

Nodes are immutable and serialize to/from the genome JSON shape:

    {"op": "zscore", "period": 89, "children": [...]}
    {"op": "constant", "value": 0.75, "unit": "ZSCORE"}

Canonical serialization is deterministic (sorted keys, fixed float
formatting) so structural hashes are stable across processes.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from evoquant.errors import EvoquantError


class ExpressionError(EvoquantError):
    """Malformed expression tree (structure, not types)."""

    code = "EXPRESSION"


@dataclass(frozen=True)
class Node:
    op: str
    period: int | None = None
    value: float | None = None
    unit: str | None = None
    children: tuple[Node, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.op or not isinstance(self.op, str):
            raise ExpressionError("Node.op must be a non-empty string")
        if self.period is not None and (not isinstance(self.period, int) or self.period < 1):
            raise ExpressionError(
                "Node.period must be a positive integer", op=self.op, period=self.period
            )

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"op": self.op}
        if self.period is not None:
            d["period"] = self.period
        if self.value is not None:
            d["value"] = self.value
        if self.unit is not None:
            d["unit"] = self.unit
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Node:
        if not isinstance(d, dict) or "op" not in d:
            raise ExpressionError("Expression dict must contain 'op'", doc=str(d)[:200])
        value = d.get("value")
        return cls(
            op=d["op"],
            period=d.get("period"),
            value=float(value) if value is not None else None,
            unit=d.get("unit"),
            children=tuple(cls.from_dict(c) for c in d.get("children", [])),
        )

    # ------------------------------------------------------------------ #

    def serialize(self) -> str:
        """Deterministic canonical string of this exact tree."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def structural_hash(self) -> str:
        return hashlib.sha256(self.serialize().encode()).hexdigest()

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def depth(self) -> int:
        return 1 + max((c.depth() for c in self.children), default=0)

    def walk(self) -> Iterator[Node]:
        yield self
        for c in self.children:
            yield from c.walk()
