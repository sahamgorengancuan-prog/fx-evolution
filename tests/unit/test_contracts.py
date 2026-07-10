from __future__ import annotations

import pytest

from evoquant.data.contracts import InstrumentSpec
from evoquant.errors import DataFormatError, MissingMetadataError
from tests.conftest import make_spec


def test_valid_spec_constructs():
    spec = make_spec()
    assert spec.symbol == "TESTUSDT"
    assert not spec.has_trusted_spread()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tick_size", 0.0),
        ("tick_size", -0.1),
        ("lot_size", 0),
        ("min_lot", -1),
        ("lot_step", 0),
    ],
)
def test_invalid_numeric_fields_rejected(field, value):
    kwargs = dict(
        symbol="X",
        base_currency="X",
        quote_currency="USD",
        tick_size=0.01,
        digits=2,
        lot_size=1000.0,
        min_lot=0.01,
        max_lot=100.0,
        lot_step=0.01,
    )
    kwargs[field] = value
    with pytest.raises(DataFormatError):
        InstrumentSpec(**kwargs)


def test_min_lot_greater_than_max_rejected():
    with pytest.raises(DataFormatError):
        InstrumentSpec(
            symbol="X",
            base_currency="X",
            quote_currency="USD",
            tick_size=0.01,
            digits=2,
            lot_size=1000.0,
            min_lot=10.0,
            max_lot=1.0,
            lot_step=0.01,
        )


def test_require_fails_closed_on_missing_metadata():
    spec = make_spec()  # swap_mode and min_notional are None
    with pytest.raises(MissingMetadataError) as exc:
        spec.require("swap_mode", "min_notional")
    artifact = exc.value.to_artifact()
    assert artifact["error_code"] == "MISSING_METADATA"
    assert set(artifact["details"]["missing_fields"]) == {"swap_mode", "min_notional"}


def test_require_passes_when_present():
    make_spec().require("commission", "static_spread_points")


def test_round_lot():
    spec = make_spec()
    assert spec.round_lot(0.005) == 0.0  # below min lot -> no trade
    assert spec.round_lot(0.014) == pytest.approx(0.01)
    assert spec.round_lot(2.567) == pytest.approx(2.56)
    assert spec.round_lot(10_000.0) == spec.max_lot


def test_from_fsb_header_zero_swap_becomes_unknown():
    header = {
        "symbol": "BNBUSDT",
        "point": 0.1,
        "digits": 1,
        "lotSize": 1000,
        "minLot": 0.01,
        "maxLot": 1000,
        "lotStep": 0.01,
        "spread": 10,
        "stopLevel": 0,
        "swapLong": 0,
        "swapShort": 0,
        "swapType": 0,
        "commissionType": 5,
        "commission": 0.1,
        "baseCurrency": "BNB",
        "priceIn": "USD",
    }
    spec = InstrumentSpec.from_fsb_header(header)
    assert spec.swap_long is None and spec.swap_mode is None  # unknown, not zero
    assert spec.spread_source == "static_header"
    assert not spec.has_trusted_spread()
    with pytest.raises(MissingMetadataError):
        spec.require("swap_mode")


def test_from_fsb_header_missing_fields_fail_closed():
    with pytest.raises(MissingMetadataError):
        InstrumentSpec.from_fsb_header({"symbol": "X", "point": 0.1})
