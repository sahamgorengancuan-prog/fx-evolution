from __future__ import annotations

import dataclasses

import pytest

from evoquant.backtest.costs import (
    ExecutionAssumptions,
    resolve_costs,
    size_position,
)
from evoquant.errors import MissingMetadataError
from tests.conftest import make_spec


def _assumptions(**overrides) -> ExecutionAssumptions:
    base = dict(
        label="unit-test assumptions",
        initial_equity=100_000.0,
        commission_pct_notional=0.001,
        assumed_spread_points=2.0,
        funding_mode="not_applicable_spot",
        min_notional=10.0,
    )
    base.update(overrides)
    return ExecutionAssumptions(**base)


def test_full_resolution_flags_assumed_spread(bars, spec):
    costs = resolve_costs(spec, bars, _assumptions())
    assert costs.spread_is_assumed  # static_header source is untrusted
    assert costs.commission_pct == 0.001
    assert costs.min_notional == 10.0
    assert costs.provenance()["spread_is_assumed"] is True


def test_trusted_spread_series_used_when_available(bars):
    spec = dataclasses.replace(make_spec(), spread_source="historical_series")
    costs = resolve_costs(spec, bars, _assumptions(assumed_spread_points=None))
    assert not costs.spread_is_assumed
    assert costs.spread_points[0] == 10.0  # fixture series value


def test_no_spread_fails_closed(bars, spec):
    with pytest.raises(MissingMetadataError) as exc:
        resolve_costs(spec, bars, _assumptions(assumed_spread_points=None))
    assert "spread" in str(exc.value).lower()


def test_no_commission_fails_closed(bars, spec):
    with pytest.raises(MissingMetadataError):
        resolve_costs(spec, bars, _assumptions(commission_pct_notional=None))


def test_no_funding_decision_fails_closed(bars, spec):
    with pytest.raises(MissingMetadataError):
        resolve_costs(spec, bars, _assumptions(funding_mode=None))
    with pytest.raises(MissingMetadataError):
        resolve_costs(spec, bars, _assumptions(funding_mode="wing_it"))


def test_funding_from_spec_requires_swap_metadata(bars, spec):
    # fixture spec has swap_long/short=None (FSB zero-swap => unknown)
    with pytest.raises(MissingMetadataError):
        resolve_costs(spec, bars, _assumptions(funding_mode="points_per_day_from_spec"))
    funded_spec = dataclasses.replace(
        make_spec(), swap_long=-1.5, swap_short=0.5, swap_mode="points_per_day"
    )
    costs = resolve_costs(funded_spec, bars, _assumptions(funding_mode="points_per_day_from_spec"))
    assert costs.funding_long_points_per_day == -1.5
    assert costs.funding_short_points_per_day == 0.5


def test_no_min_notional_fails_closed(bars, spec):
    with pytest.raises(MissingMetadataError):
        resolve_costs(spec, bars, _assumptions(min_notional=None))


def test_unlabeled_assumptions_rejected():
    with pytest.raises(MissingMetadataError):
        ExecutionAssumptions(label="   ", initial_equity=1000.0)


def test_sizing_hand_computed(bars, spec):
    costs = resolve_costs(spec, bars, _assumptions())
    # equity 100k, risk 1%, stop 2.0 quote => 500 base units; lot_size 1000
    # => 0.5 lots, rounded to lot grid 0.01 => 0.5 exactly
    s = size_position(
        equity=100_000.0,
        risk_per_trade=0.01,
        stop_distance_quote=2.0,
        entry_price=100.0,
        spec=spec,
        costs=costs,
    )
    assert s.lots == pytest.approx(0.5)
    assert s.qty_base == pytest.approx(500.0)
    assert s.notional_quote == pytest.approx(50_000.0)


def test_sizing_below_min_lot_no_trade(bars, spec):
    costs = resolve_costs(spec, bars, _assumptions())
    # tiny risk => qty below min_lot*lot_size = 10 base units
    s = size_position(
        equity=1_000.0,
        risk_per_trade=0.001,
        stop_distance_quote=2.0,  # 0.5 base units
        entry_price=100.0,
        spec=spec,
        costs=costs,
    )
    assert s.lots == 0.0 and s.reason_no_trade == "below_min_lot"


def test_sizing_below_min_notional_no_trade(bars, spec):
    costs = resolve_costs(spec, bars, _assumptions(min_notional=5_000.0))
    s = size_position(
        equity=10_000.0,
        risk_per_trade=0.005,  # 50 risk => 25 base units => notional 2500
        stop_distance_quote=2.0,
        entry_price=100.0,
        spec=spec,
        costs=costs,
    )
    assert s.lots == 0.0 and s.reason_no_trade == "below_min_notional"


def test_sizing_leverage_cap(bars, spec):
    costs = resolve_costs(spec, bars, _assumptions(max_leverage=1.0))
    # risk-based qty would be 10_000 base = 1M notional on 100k equity
    s = size_position(
        equity=100_000.0,
        risk_per_trade=0.1,
        stop_distance_quote=1.0,
        entry_price=100.0,
        spec=spec,
        costs=costs,
    )
    assert s.notional_quote <= 100_000.0 + 1e-9


def test_sizing_invalid_stop_no_trade(bars, spec):
    costs = resolve_costs(spec, bars, _assumptions())
    s = size_position(
        equity=1e5,
        risk_per_trade=0.01,
        stop_distance_quote=0.0,
        entry_price=100.0,
        spec=spec,
        costs=costs,
    )
    assert s.lots == 0.0 and s.reason_no_trade == "invalid_stop_distance"
