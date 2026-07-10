"""Execution cost resolution — metadata-first, fail-closed.

Nothing here invents a number. Every cost the engines apply comes either
from trusted source metadata or from an **explicitly labeled assumption**
supplied by the experimenter; if neither exists the resolution raises.

The v8 system silently applied a constant placeholder spread for nine
years (audit B1). Here the same constant can only enter as
``ExecutionAssumptions(assumed_spread_points=...)`` whose label travels
into every result artifact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from evoquant.data.contracts import InstrumentSpec
from evoquant.data.loaders import BarData
from evoquant.errors import MissingMetadataError

FUNDING_MODES = ("not_applicable_spot", "points_per_day_from_spec")


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Experimenter-stated execution assumptions. All explicit, all labeled."""

    #: Human-readable provenance, e.g. "Binance spot taker, 2026-07 fee tier".
    label: str
    initial_equity: float
    #: Commission per side as a fraction of notional (0.001 = 0.1%).
    commission_pct_notional: float | None = None
    #: Constant spread assumption in points, used ONLY when no trusted series.
    assumed_spread_points: float | None = None
    #: Extra adverse slippage per fill, in points.
    slippage_points: float = 0.0
    #: One of FUNDING_MODES.
    funding_mode: str | None = None
    #: Override when the instrument spec lacks min_notional.
    min_notional: float | None = None
    max_leverage: float = 1.0
    #: Event-engine only: order latency in seconds.
    latency_s: float = 0.0
    #: Event-engine only: probability a market order is rejected.
    rejection_prob: float = 0.0

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise MissingMetadataError(
                "ExecutionAssumptions.label must state provenance of the assumptions"
            )
        if self.initial_equity <= 0:
            raise MissingMetadataError("initial_equity must be > 0")
        if self.max_leverage <= 0:
            raise MissingMetadataError("max_leverage must be > 0")
        if not (0.0 <= self.rejection_prob < 1.0):
            raise MissingMetadataError("rejection_prob must be in [0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedCosts:
    """Everything an engine needs, with provenance flags."""

    label: str
    spread_points: np.ndarray  # per bar, in points
    spread_is_assumed: bool
    commission_pct: float
    slippage_points: float
    funding_mode: str
    #: Convention: POSITIVE points = the holder pays (cost-positive).
    funding_long_points_per_day: float
    funding_short_points_per_day: float
    min_notional: float
    max_leverage: float
    tick_size: float
    lot_size: float

    def half_spread_quote(self, i: int) -> float:
        return float(self.spread_points[i]) * self.tick_size / 2.0

    def slippage_quote(self) -> float:
        return self.slippage_points * self.tick_size

    def provenance(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "spread_is_assumed": self.spread_is_assumed,
            "commission_pct": self.commission_pct,
            "slippage_points": self.slippage_points,
            "funding_mode": self.funding_mode,
            "min_notional": self.min_notional,
            "max_leverage": self.max_leverage,
        }


def resolve_costs(
    spec: InstrumentSpec, bars: BarData, assumptions: ExecutionAssumptions
) -> ResolvedCosts:
    """Combine spec + data + assumptions; raise when anything is unresolved."""
    # --- spread ---------------------------------------------------------
    if spec.has_trusted_spread() and bars.spread_points is not None:
        spread = np.asarray(bars.spread_points, dtype=np.float64)
        spread_is_assumed = False
    elif assumptions.assumed_spread_points is not None:
        if assumptions.assumed_spread_points <= 0:
            raise MissingMetadataError(
                "assumed_spread_points must be > 0", value=assumptions.assumed_spread_points
            )
        spread = np.full(bars.n_bars, float(assumptions.assumed_spread_points))
        spread_is_assumed = True
    else:
        raise MissingMetadataError(
            f"No trusted spread series for {spec.symbol} "
            f"(spread_source={spec.spread_source!r}) and no explicit "
            "assumed_spread_points. Refusing to invent a spread (fail closed).",
            symbol=spec.symbol,
            spread_source=spec.spread_source,
        )

    # --- commission ------------------------------------------------------
    if assumptions.commission_pct_notional is None:
        raise MissingMetadataError(
            "commission_pct_notional must be stated explicitly; the source "
            "header's commission field has ambiguous semantics "
            f"(spec.commission_mode={spec.commission_mode!r}).",
            symbol=spec.symbol,
        )
    if assumptions.commission_pct_notional < 0:
        raise MissingMetadataError("commission_pct_notional must be >= 0")

    # --- funding ----------------------------------------------------------
    if assumptions.funding_mode is None:
        raise MissingMetadataError(
            "funding_mode must be stated: 'not_applicable_spot' for spot, or "
            "'points_per_day_from_spec' with swap metadata. A perpetual "
            "instrument without funding data cannot be researched (fail closed).",
            symbol=spec.symbol,
        )
    if assumptions.funding_mode not in FUNDING_MODES:
        raise MissingMetadataError(
            f"Unknown funding_mode {assumptions.funding_mode!r}",
            allowed=list(FUNDING_MODES),
        )
    if assumptions.funding_mode == "points_per_day_from_spec":
        spec.require("swap_long", "swap_short", "swap_mode")
        funding_long = float(spec.swap_long)  # type: ignore[arg-type]
        funding_short = float(spec.swap_short)  # type: ignore[arg-type]
    else:
        funding_long = 0.0
        funding_short = 0.0

    # --- notional floor ---------------------------------------------------
    min_notional = (
        spec.min_notional if spec.min_notional is not None else assumptions.min_notional
    )
    if min_notional is None:
        raise MissingMetadataError(
            f"min_notional unknown for {spec.symbol}; supply it via "
            "ExecutionAssumptions (exchange rule book) — fail closed.",
            symbol=spec.symbol,
        )

    return ResolvedCosts(
        label=assumptions.label,
        spread_points=spread,
        spread_is_assumed=spread_is_assumed,
        commission_pct=float(assumptions.commission_pct_notional),
        slippage_points=float(assumptions.slippage_points),
        funding_mode=assumptions.funding_mode,
        funding_long_points_per_day=funding_long,
        funding_short_points_per_day=funding_short,
        min_notional=float(min_notional),
        max_leverage=float(assumptions.max_leverage),
        tick_size=spec.tick_size,
        lot_size=spec.lot_size,
    )


@dataclass(frozen=True)
class SizingResult:
    lots: float
    qty_base: float
    notional_quote: float
    reason_no_trade: str | None = None


def size_position(
    equity: float,
    risk_per_trade: float,
    stop_distance_quote: float,
    entry_price: float,
    spec: InstrumentSpec,
    costs: ResolvedCosts,
) -> SizingResult:
    """Risk-based sizing honoring lot grid, min notional, and leverage cap."""
    if stop_distance_quote <= 0 or not np.isfinite(stop_distance_quote):
        return SizingResult(0.0, 0.0, 0.0, reason_no_trade="invalid_stop_distance")
    risk_quote = equity * risk_per_trade
    qty_base = risk_quote / stop_distance_quote
    # leverage cap on notional
    max_notional = equity * costs.max_leverage
    if qty_base * entry_price > max_notional:
        qty_base = max_notional / entry_price
    lots = spec.round_lot(qty_base / spec.lot_size)
    if lots <= 0.0:
        return SizingResult(0.0, 0.0, 0.0, reason_no_trade="below_min_lot")
    qty_base = lots * spec.lot_size
    notional = qty_base * entry_price
    if notional < costs.min_notional:
        return SizingResult(0.0, 0.0, 0.0, reason_no_trade="below_min_notional")
    return SizingResult(lots=lots, qty_base=qty_base, notional_quote=notional)
