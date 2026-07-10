"""Primitive operators: batch + streaming implementations and type rules.

Semantics notes (documented choices, see DECISIONS.md):

* Window operators emit NaN during warmup; NaN inputs propagate to NaN
  outputs (never silently skipped or filled).
* ``ema`` seeds on the first finite input; a NaN input yields NaN output
  and leaves the recursion state unchanged.
* ``rsi`` is Cutler's RSI (simple-mean gains/losses) — window-computable,
  hence exactly reproducible incrementally.
* ``atr`` is the simple mean of true range over the period (not Wilder).
* ``chop`` is the Choppiness Index normalized to [0, 1].
* Comparisons involving NaN are False (no signal), never True.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from evoquant.data.loaders import BarData
from evoquant.features.ast import Node
from evoquant.features.registry import (
    FeatureTypeError,
    OpSpec,
    StreamOp,
    register,
)
from evoquant.features.types import COMPARABLE, NUMERIC, SIGNED, Dimension

# --------------------------------------------------------------------- #
# dimension-rule helpers
# --------------------------------------------------------------------- #


def _dim_err(node: Node, msg: str, child_dims: list[Dimension]) -> FeatureTypeError:
    return FeatureTypeError(
        f"{msg} (op={node.op}, child dims={[d.value for d in child_dims]})",
        op=node.op,
        child_dims=[d.value for d in child_dims],
    )


def _same_numeric_passthrough(node: Node, dims: list[Dimension]) -> Dimension:
    if dims[0] not in NUMERIC:
        raise _dim_err(node, "operand must be numeric", dims)
    return dims[0]


def _numeric_to(target: Dimension) -> Callable[[Node, list[Dimension]], Dimension]:
    def rule(node: Node, dims: list[Dimension]) -> Dimension:
        if dims[0] not in NUMERIC:
            raise _dim_err(node, "operand must be numeric", dims)
        return target

    return rule


def _signed_passthrough(node: Node, dims: list[Dimension]) -> Dimension:
    if dims[0] not in SIGNED:
        raise _dim_err(
            node,
            "operand must be a signed dimension (RETURN/ZSCORE/UNSCALED); "
            "levels and bounded oscillators cannot be negated/rectified meaningfully",
            dims,
        )
    return dims[0]


def _linear_same_dim(node: Node, dims: list[Dimension]) -> Dimension:
    a, b = dims
    if a not in NUMERIC or b not in NUMERIC:
        raise _dim_err(node, "operands must be numeric", dims)
    if a != b:
        raise _dim_err(node, "add/sub requires both operands in the same dimension", dims)
    # A sum/difference loses level meaning; must be renormalized before use
    # in comparisons (zscore/rank). This kills the v8 'MACD vs price' class.
    return Dimension.UNSCALED


def _comparison(node: Node, dims: list[Dimension]) -> Dimension:
    a, b = dims
    if a != b:
        raise _dim_err(
            node,
            "comparison requires both sides in the same dimension "
            "(v8 champion compared OSCILLATOR_0_1 against RETURN — rejected here)",
            dims,
        )
    if a not in COMPARABLE:
        raise _dim_err(
            node,
            f"dimension {a.value} is not comparable; normalize first (zscore/rank)",
            dims,
        )
    return Dimension.BOOL


def _all_bool(node: Node, dims: list[Dimension]) -> Dimension:
    if any(d is not Dimension.BOOL for d in dims):
        raise _dim_err(node, "logical operator requires BOOL operands", dims)
    return Dimension.BOOL


def _price_to(target: Dimension) -> Callable[[Node, list[Dimension]], Dimension]:
    def rule(node: Node, dims: list[Dimension]) -> Dimension:
        if dims[0] is not Dimension.PRICE:
            raise _dim_err(node, "operand must be PRICE", dims)
        return target

    return rule


def _stdev_rule(node: Node, dims: list[Dimension]) -> Dimension:
    if dims[0] not in NUMERIC:
        raise _dim_err(node, "operand must be numeric", dims)
    if dims[0] in (Dimension.PRICE, Dimension.RETURN):
        return Dimension.VOLATILITY
    return Dimension.UNSCALED


def _constant_rule(node: Node, dims: list[Dimension]) -> Dimension:
    if node.value is None or node.unit is None:
        raise FeatureTypeError(
            "constant requires 'value' and 'unit'", op=node.op, value=node.value
        )
    try:
        dim = Dimension(node.unit)
    except ValueError:
        raise FeatureTypeError(
            f"constant unit {node.unit!r} is not a known dimension", unit=node.unit
        ) from None
    if dim in (Dimension.BOOL, Dimension.UNSCALED):
        raise FeatureTypeError(
            f"constants of dimension {dim.value} are not allowed "
            "(UNSCALED has no unit; BOOL constants are degenerate rules)",
            unit=node.unit,
        )
    return dim


# --------------------------------------------------------------------- #
# window formulas — the single source of truth shared by batch & stream
# --------------------------------------------------------------------- #


def _w_mean(w: np.ndarray) -> float:
    return float(np.mean(w))


def _w_max(w: np.ndarray) -> float:
    return float(np.max(w))


def _w_min(w: np.ndarray) -> float:
    return float(np.min(w))


def _w_std(w: np.ndarray) -> float:
    return float(np.std(w))


def _w_zscore(w: np.ndarray) -> float:
    s = np.std(w)
    if not np.isfinite(s) or s == 0.0:
        return math.nan
    return float((w[-1] - np.mean(w)) / s)


def _w_rank(w: np.ndarray) -> float:
    if np.isnan(w).any():
        return math.nan
    return float(np.mean(w <= w[-1]))


def _w_delta(w: np.ndarray) -> float:
    return float(w[-1] - w[0])


def _w_ret(w: np.ndarray) -> float:
    if w[0] == 0.0:
        return math.nan
    return float(w[-1] / w[0] - 1.0)


def _w_efficiency_ratio(w: np.ndarray) -> float:
    path = float(np.sum(np.abs(np.diff(w))))
    if not np.isfinite(path) or path == 0.0:
        return math.nan
    return float(abs(w[-1] - w[0]) / path)


def _w_rsi(w: np.ndarray) -> float:
    d = np.diff(w)
    if np.isnan(d).any():
        return math.nan
    up = float(np.mean(np.maximum(d, 0.0)))
    dn = float(np.mean(np.maximum(-d, 0.0)))
    if up + dn == 0.0:
        return 50.0
    return 100.0 * up / (up + dn)


def _chop_value(tr_w: np.ndarray, hi_w: np.ndarray, lo_w: np.ndarray, p: int) -> float:
    rng = float(np.max(hi_w) - np.min(lo_w))
    s = float(np.sum(tr_w))
    if not np.isfinite(rng) or not np.isfinite(s) or rng <= 0.0 or s <= 0.0:
        return math.nan
    return math.log10(s / rng) / math.log10(p)


def _true_range(h: float, l: float, prev_close: float) -> float:
    if math.isnan(prev_close):
        return h - l
    return max(h - l, abs(h - prev_close), abs(l - prev_close))


# --------------------------------------------------------------------- #
# batch helpers
# --------------------------------------------------------------------- #


def _rolling(x: np.ndarray, width: int, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(width - 1, n):
        out[i] = fn(x[i - width + 1 : i + 1])
    return out


def _batch_tr(bars: BarData) -> np.ndarray:
    n = bars.n_bars
    tr = np.empty(n)
    prev_close = math.nan
    for i in range(n):
        tr[i] = _true_range(float(bars.high[i]), float(bars.low[i]), prev_close)
        prev_close = float(bars.close[i])
    return tr


def _batch_ema(x: np.ndarray, p: int) -> np.ndarray:
    alpha = 2.0 / (p + 1.0)
    out = np.full(len(x), np.nan)
    prev = math.nan
    for i, xi in enumerate(x):
        if math.isnan(xi):
            continue  # output stays NaN; recursion state unchanged
        prev = xi if math.isnan(prev) else prev + alpha * (xi - prev)
        out[i] = prev
    return out


# --------------------------------------------------------------------- #
# stream ops
# --------------------------------------------------------------------- #


class _SourceStream(StreamOp):
    def __init__(self, field: str) -> None:
        self.field = field

    def update(self, bar: dict[str, float], child_values: Sequence[Any]) -> float:
        return bar[self.field]


class _ConstStream(StreamOp):
    def __init__(self, value: float) -> None:
        self.value = value

    def update(self, bar: dict[str, float], child_values: Sequence[Any]) -> float:
        return self.value


class _PointwiseStream(StreamOp):
    def __init__(self, fn: Callable[..., Any]) -> None:
        self.fn = fn

    def update(self, bar: dict[str, float], child_values: Sequence[Any]) -> Any:
        return self.fn(*child_values)


class _WindowStream(StreamOp):
    """Buffers child values; applies the shared window formula when full."""

    def __init__(self, width: int, fn: Callable[[np.ndarray], float]) -> None:
        self.buf: deque[float] = deque(maxlen=width)
        self.width = width
        self.fn = fn

    def update(self, bar: dict[str, float], child_values: Sequence[Any]) -> float:
        self.buf.append(float(child_values[0]))
        if len(self.buf) < self.width:
            return math.nan
        return self.fn(np.array(self.buf))


class _EmaStream(StreamOp):
    def __init__(self, period: int) -> None:
        self.alpha = 2.0 / (period + 1.0)
        self.prev = math.nan

    def update(self, bar: dict[str, float], child_values: Sequence[Any]) -> float:
        xi = float(child_values[0])
        if math.isnan(xi):
            return math.nan
        self.prev = xi if math.isnan(self.prev) else self.prev + self.alpha * (xi - self.prev)
        return self.prev


class _CrossStream(StreamOp):
    def __init__(self, above: bool) -> None:
        self.above = above
        self.prev_a = math.nan
        self.prev_b = math.nan

    def update(self, bar: dict[str, float], child_values: Sequence[Any]) -> bool:
        a, b = float(child_values[0]), float(child_values[1])
        pa, pb = self.prev_a, self.prev_b
        self.prev_a, self.prev_b = a, b
        if any(math.isnan(v) for v in (a, b, pa, pb)):
            return False
        if self.above:
            return pa <= pb and a > b
        return pa >= pb and a < b


class _AtrStream(StreamOp):
    def __init__(self, period: int) -> None:
        self.buf: deque[float] = deque(maxlen=period)
        self.period = period
        self.prev_close = math.nan

    def update(self, bar: dict[str, float], child_values: Sequence[Any]) -> float:
        tr = _true_range(bar["high"], bar["low"], self.prev_close)
        self.prev_close = bar["close"]
        self.buf.append(tr)
        if len(self.buf) < self.period:
            return math.nan
        return _w_mean(np.array(self.buf))


class _ChopStream(StreamOp):
    def __init__(self, period: int) -> None:
        self.period = period
        self.tr: deque[float] = deque(maxlen=period)
        self.hi: deque[float] = deque(maxlen=period)
        self.lo: deque[float] = deque(maxlen=period)
        self.prev_close = math.nan

    def update(self, bar: dict[str, float], child_values: Sequence[Any]) -> float:
        self.tr.append(_true_range(bar["high"], bar["low"], self.prev_close))
        self.prev_close = bar["close"]
        self.hi.append(bar["high"])
        self.lo.append(bar["low"])
        if len(self.tr) < self.period:
            return math.nan
        return _chop_value(
            np.array(self.tr), np.array(self.hi), np.array(self.lo), self.period
        )


# --------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------- #


def _src(name: str, field: str, dim: Dimension) -> None:
    def infer(n: Node, d: list[Dimension]) -> Dimension:
        return dim

    def batch(bars: BarData, ch: list[np.ndarray], n: Node) -> np.ndarray:
        return np.asarray(getattr(bars, field), dtype=np.float64)

    def stream(n: Node) -> StreamOp:
        return _SourceStream(field)

    register(
        OpSpec(
            name=name,
            arity=0,
            needs_period=False,
            min_period=1,
            complexity=1,
            infer_dim=infer,
            own_lookback=lambda n: 0,
            batch=batch,
            stream_factory=stream,
        )
    )


_FIELD_MAP = {"close": "close", "open": "open", "high": "high", "low": "low", "volume": "volume"}
for _name, _field in _FIELD_MAP.items():
    _src(_name, _field, Dimension.VOLUME if _name == "volume" else Dimension.PRICE)

register(
    OpSpec(
        name="constant",
        arity=0,
        needs_period=False,
        min_period=1,
        complexity=0,
        infer_dim=_constant_rule,
        own_lookback=lambda n: 0,
        batch=lambda bars, ch, n: np.full(bars.n_bars, float(n.value)),  # type: ignore[arg-type]
        stream_factory=lambda n: _ConstStream(float(n.value)),  # type: ignore[arg-type]
    )
)


def _pointwise(
    name: str,
    arity: int,
    infer: Callable[[Node, list[Dimension]], Dimension],
    array_fn: Callable[..., np.ndarray],
    scalar_fn: Callable[..., Any],
    complexity: int = 1,
) -> None:
    register(
        OpSpec(
            name=name,
            arity=arity,
            needs_period=False,
            min_period=1,
            complexity=complexity,
            infer_dim=infer,
            own_lookback=lambda n: 0,
            batch=lambda bars, ch, n: array_fn(*ch),
            stream_factory=lambda n: _PointwiseStream(scalar_fn),
        )
    )


_pointwise("abs", 1, _signed_passthrough, np.abs, lambda x: abs(x))
_pointwise("neg", 1, _signed_passthrough, np.negative, lambda x: -x)
_pointwise("add", 2, _linear_same_dim, np.add, lambda a, b: a + b)
_pointwise("sub", 2, _linear_same_dim, np.subtract, lambda a, b: a - b)


def _bool_gt(a: Any, b: Any) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    return bool(a > b)


def _bool_lt(a: Any, b: Any) -> bool:
    if math.isnan(a) or math.isnan(b):
        return False
    return bool(a < b)


_pointwise("gt", 2, _comparison, lambda a, b: np.greater(a, b), _bool_gt)
_pointwise("lt", 2, _comparison, lambda a, b: np.less(a, b), _bool_lt)
_pointwise(
    "and",
    -1,
    _all_bool,
    lambda *ch: np.logical_and.reduce(ch),
    lambda *vals: all(bool(v) for v in vals),
)
_pointwise(
    "or",
    -1,
    _all_bool,
    lambda *ch: np.logical_or.reduce(ch),
    lambda *vals: any(bool(v) for v in vals),
)
_pointwise("not", 1, _all_bool, np.logical_not, lambda x: not bool(x))


def _batch_cross(a: np.ndarray, b: np.ndarray, above: bool) -> np.ndarray:
    out = np.zeros(len(a), dtype=bool)
    pa, pb = a[:-1], b[:-1]
    ca, cb = a[1:], b[1:]
    with np.errstate(invalid="ignore"):
        if above:
            out[1:] = (pa <= pb) & (ca > cb)
        else:
            out[1:] = (pa >= pb) & (ca < cb)
    return out


def _register_cross(name: str, above: bool) -> None:
    def batch(bars: BarData, ch: list[np.ndarray], n: Node) -> np.ndarray:
        return _batch_cross(ch[0], ch[1], above)

    def stream(n: Node) -> StreamOp:
        return _CrossStream(above)

    register(
        OpSpec(
            name=name,
            arity=2,
            needs_period=False,
            min_period=1,
            complexity=2,
            infer_dim=_comparison,
            own_lookback=lambda n: 1,
            batch=batch,
            stream_factory=stream,
        )
    )


_register_cross("cross_above", True)
_register_cross("cross_below", False)


def _window_op(
    name: str,
    infer: Callable[[Node, list[Dimension]], Dimension],
    fn: Callable[[np.ndarray], float],
    *,
    width_offset: int = 0,  # window width = period + width_offset
    min_period: int = 1,
    complexity: int = 3,
) -> None:
    register(
        OpSpec(
            name=name,
            arity=1,
            needs_period=True,
            min_period=min_period,
            complexity=complexity,
            infer_dim=infer,
            own_lookback=lambda n: (n.period or 1) + width_offset - 1,
            batch=lambda bars, ch, n: _rolling(ch[0], (n.period or 1) + width_offset, fn),
            stream_factory=lambda n: _WindowStream((n.period or 1) + width_offset, fn),
        )
    )


_window_op("sma", _same_numeric_passthrough, _w_mean)
_window_op("highest", _same_numeric_passthrough, _w_max)
_window_op("lowest", _same_numeric_passthrough, _w_min)
_window_op("stdev", _stdev_rule, _w_std, min_period=2)
_window_op("zscore", _numeric_to(Dimension.ZSCORE), _w_zscore, min_period=2)
_window_op("rank", _numeric_to(Dimension.OSCILLATOR_0_1), _w_rank, min_period=2)
# delta/ret reach back `period` bars => window width = period + 1
_window_op("delta", _numeric_to(Dimension.UNSCALED), _w_delta, width_offset=1)
_window_op("ret", _price_to(Dimension.RETURN), _w_ret, width_offset=1)
_window_op(
    "efficiency_ratio",
    _price_to(Dimension.OSCILLATOR_0_1),
    _w_efficiency_ratio,
    width_offset=1,
    min_period=2,
)
_window_op(
    "rsi",
    _numeric_to(Dimension.OSCILLATOR_0_100),
    _w_rsi,
    width_offset=1,
    min_period=2,
    complexity=4,
)

register(
    OpSpec(
        name="ema",
        arity=1,
        needs_period=True,
        min_period=1,
        complexity=2,
        infer_dim=_same_numeric_passthrough,
        own_lookback=lambda n: 0,  # emits from first finite input (recursive)
        batch=lambda bars, ch, n: _batch_ema(ch[0], n.period or 1),
        stream_factory=lambda n: _EmaStream(n.period or 1),
    )
)


def _batch_atr(bars: BarData, ch: list[np.ndarray], n: Node) -> np.ndarray:
    return _rolling(_batch_tr(bars), n.period or 1, _w_mean)


register(
    OpSpec(
        name="atr",
        arity=0,
        needs_period=True,
        min_period=1,
        complexity=4,
        infer_dim=lambda n, d: Dimension.VOLATILITY,
        own_lookback=lambda n: (n.period or 1) - 1,
        batch=_batch_atr,
        stream_factory=lambda n: _AtrStream(n.period or 1),
    )
)


def _batch_chop(bars: BarData, ch: list[np.ndarray], n: Node) -> np.ndarray:
    p = n.period or 2
    tr = _batch_tr(bars)
    h = np.asarray(bars.high, dtype=np.float64)
    l = np.asarray(bars.low, dtype=np.float64)
    out = np.full(bars.n_bars, np.nan)
    for i in range(p - 1, bars.n_bars):
        sl = slice(i - p + 1, i + 1)
        out[i] = _chop_value(tr[sl], h[sl], l[sl], p)
    return out


register(
    OpSpec(
        name="chop",
        arity=0,
        needs_period=True,
        min_period=2,
        complexity=4,
        infer_dim=lambda n, d: Dimension.OSCILLATOR_0_1,
        own_lookback=lambda n: (n.period or 2) - 1,
        batch=_batch_chop,
        stream_factory=lambda n: _ChopStream(n.period or 2),
    )
)
