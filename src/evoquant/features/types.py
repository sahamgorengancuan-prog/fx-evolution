"""Dimension system for the feature DSL.

The v8 grammar accepted `cross_above(rank(...), ret(8))` — a [0,1] rank
crossed against a raw return (audit D5). The dimension system makes that
class of expression a compile-time error: comparisons require both sides
to share a dimension, and that dimension must be *comparable*.

``UNSCALED`` is the deliberate escape hatch for intermediates whose units
are no longer level-meaningful (e.g. a difference of two prices — the
MACD line). UNSCALED values cannot be compared to anything; they must
first pass through a normalizer (``zscore``, ``rank``) to re-enter
comparable space. This is conservative on purpose.
"""
from __future__ import annotations

from enum import Enum


class Dimension(str, Enum):  # noqa: UP042 - keep plain-str .value semantics for JSON round-trips
    PRICE = "PRICE"
    RETURN = "RETURN"
    VOLATILITY = "VOLATILITY"
    VOLUME = "VOLUME"
    OSCILLATOR_0_1 = "OSCILLATOR_0_1"
    OSCILLATOR_0_100 = "OSCILLATOR_0_100"
    ZSCORE = "ZSCORE"
    TIME = "TIME"
    BOOL = "BOOL"
    #: Unit-ambiguous intermediate; must be normalized before comparison.
    UNSCALED = "UNSCALED"


#: Dimensions whose values may appear in comparisons (gt/lt/cross).
COMPARABLE: frozenset[Dimension] = frozenset(
    {
        Dimension.PRICE,
        Dimension.RETURN,
        Dimension.VOLATILITY,
        Dimension.VOLUME,
        Dimension.OSCILLATOR_0_1,
        Dimension.OSCILLATOR_0_100,
        Dimension.ZSCORE,
        Dimension.TIME,
    }
)

#: Everything numeric (feedable into transforms/normalizers).
NUMERIC: frozenset[Dimension] = frozenset(
    {
        Dimension.PRICE,
        Dimension.RETURN,
        Dimension.VOLATILITY,
        Dimension.VOLUME,
        Dimension.OSCILLATOR_0_1,
        Dimension.OSCILLATOR_0_100,
        Dimension.ZSCORE,
        Dimension.TIME,
        Dimension.UNSCALED,
    }
)

#: Dimensions where negation keeps a meaningful unit.
SIGNED: frozenset[Dimension] = frozenset(
    {Dimension.RETURN, Dimension.ZSCORE, Dimension.UNSCALED}
)
