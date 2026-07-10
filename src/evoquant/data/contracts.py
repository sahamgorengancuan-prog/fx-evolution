"""Instrument and contract specifications.

Metadata-first, fail-closed: fields that a downstream engine requires but
that are unknown are ``None`` and must be fetched through :meth:`InstrumentSpec.require`,
which raises :class:`MissingMetadataError` instead of silently defaulting.

Source-metadata trust is explicit: the ForexSB header carries a *static*
spread and zero swap for crypto; those are recorded but flagged so no
engine mistakes them for historical cost series.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from evoquant.errors import DataFormatError, MissingMetadataError

#: Values of ``spread_source`` that a research/production cost model may use.
TRUSTED_SPREAD_SOURCES = ("historical_series", "bid_ask_ticks")


@dataclass(frozen=True)
class InstrumentSpec:
    """Contract specification for one tradable instrument."""

    symbol: str
    base_currency: str
    quote_currency: str
    #: Minimum price increment in quote currency (a.k.a. point / tick size).
    tick_size: float
    #: Decimal digits of quotes (redundant with tick_size; both kept for parity checks).
    digits: int
    #: Contract size of 1.0 lot, in base currency units.
    lot_size: float
    min_lot: float
    max_lot: float
    lot_step: float
    #: Commission per side. Interpretation given by commission_mode.
    commission: float | None = None
    #: e.g. "percent_notional", "per_lot", "per_deal"
    commission_mode: str | None = None
    #: Static header spread in points. NOT a historical series.
    static_spread_points: float | None = None
    #: Where spread data for backtesting comes from.
    spread_source: str = "static_header"
    #: Swap/funding per day (long/short). None = unknown => fail closed.
    swap_long: float | None = None
    swap_short: float | None = None
    #: e.g. "none", "points_per_day", "percent_per_day", "funding_schedule"
    swap_mode: str | None = None
    #: Minimum order notional in quote currency. None = unknown.
    min_notional: float | None = None
    #: Broker stop/freeze level in points, 0 = none.
    stop_level_points: float | None = None
    #: Trading sessions. Crypto: "24/7".
    session: str = "24/7"
    #: Free-form provenance (exchange, feed vendor, export tool ...).
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol:
            raise DataFormatError("InstrumentSpec.symbol must be non-empty")
        for name in ("tick_size", "lot_size", "min_lot", "max_lot", "lot_step"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and v > 0):
                raise DataFormatError(
                    f"InstrumentSpec.{name} must be a positive number", field=name, value=v
                )
        if self.digits < 0:
            raise DataFormatError("InstrumentSpec.digits must be >= 0", value=self.digits)
        if self.min_lot > self.max_lot:
            raise DataFormatError(
                "InstrumentSpec.min_lot must not exceed max_lot",
                min_lot=self.min_lot,
                max_lot=self.max_lot,
            )
        expected_tick = 10.0 ** (-self.digits) if self.digits > 0 else None
        if expected_tick is not None and abs(self.tick_size - expected_tick) > 1e-12:
            # digits=1 with tick 0.1 is fine; anything inconsistent is suspicious
            # but some venues use non-decimal ticks, so record, don't reject.
            object.__setattr__(
                self,
                "source_metadata",
                {**self.source_metadata, "tick_digits_inconsistent": True},
            )

    # ------------------------------------------------------------------ #

    def require(self, *fields_: str) -> None:
        """Fail closed: raise if any named field is None."""
        missing = [f for f in fields_ if getattr(self, f) is None]
        if missing:
            raise MissingMetadataError(
                f"Instrument {self.symbol} is missing required metadata: {missing}. "
                "Refusing to substitute defaults (fail-closed policy).",
                symbol=self.symbol,
                missing_fields=missing,
            )

    def has_trusted_spread(self) -> bool:
        return self.spread_source in TRUSTED_SPREAD_SOURCES

    def round_lot(self, lots: float) -> float:
        """Round a raw lot amount down to the venue lot grid; 0.0 if below min."""
        if lots < self.min_lot:
            return 0.0
        stepped = self.min_lot + int((lots - self.min_lot) / self.lot_step) * self.lot_step
        return min(round(stepped, 12), self.max_lot)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------ #

    @classmethod
    def from_fsb_header(cls, header: dict[str, Any]) -> InstrumentSpec:
        """Build a spec from a ForexSB JSON export header.

        The header's ``spread`` is a static placeholder and ``swap*`` fields
        are frequently zero for crypto; both are flagged accordingly.
        """
        required = ("symbol", "point", "digits", "lotSize", "minLot", "maxLot", "lotStep")
        missing = [k for k in required if k not in header]
        if missing:
            raise MissingMetadataError(
                "ForexSB header missing required fields",
                missing_fields=missing,
            )
        base = header.get("baseCurrency") or ""
        quote = header.get("priceIn") or ""
        swap_long = header.get("swapLong")
        swap_short = header.get("swapShort")
        # A zero swap in a ForexSB crypto export means "not modeled", not
        # "funding is zero". Treat as unknown => fail closed downstream.
        if swap_long == 0 and swap_short == 0:
            swap_long = swap_short = None
            swap_mode = None
        else:
            swap_mode = f"fsb_swap_type_{header.get('swapType')}"
        commission = header.get("commission")
        commission_mode = (
            f"fsb_commission_type_{header.get('commissionType')}"
            if commission is not None
            else None
        )
        return cls(
            symbol=str(header["symbol"]),
            base_currency=str(base),
            quote_currency=str(quote),
            tick_size=float(header["point"]),
            digits=int(header["digits"]),
            lot_size=float(header["lotSize"]),
            min_lot=float(header["minLot"]),
            max_lot=float(header["maxLot"]),
            lot_step=float(header["lotStep"]),
            commission=float(commission) if commission is not None else None,
            commission_mode=commission_mode,
            static_spread_points=(
                float(header["spread"]) if header.get("spread") is not None else None
            ),
            spread_source="static_header",
            swap_long=swap_long,
            swap_short=swap_short,
            swap_mode=swap_mode,
            min_notional=None,  # not present in FSB exports; must be sourced separately
            stop_level_points=(
                float(header["stopLevel"]) if header.get("stopLevel") is not None else None
            ),
            session="24/7",
            source_metadata={
                "export_tool": header.get("terminal"),
                "company": header.get("company"),
                "server": header.get("server"),
                "description": header.get("description"),
                "fsb_version": header.get("ver"),
                "point_value": header.get("pointValue"),
                "pip": header.get("pip"),
            },
        )
