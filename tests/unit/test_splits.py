from __future__ import annotations

import pytest

from evoquant.data.splits import (
    IndexRange,
    SplitConfig,
    SplitPlan,
    generate_nested_walk_forward,
    subtract_ranges,
    total_len,
    validate_plan,
)
from evoquant.errors import SplitConfigError, SplitInvariantError

CFG = SplitConfig(
    n_outer_folds=4,
    outer_test_bars=500,
    n_inner_folds=3,
    inner_val_bars=300,
    purge_bars=50,
    embargo_bars=20,
    lockbox_bars=1500,
    min_train_bars=2000,
)
N = 10_000


def test_subtract_ranges():
    base = [IndexRange(0, 100)]
    cuts = [IndexRange(10, 20), IndexRange(90, 200)]
    out = subtract_ranges(base, cuts)
    assert out == [IndexRange(0, 10), IndexRange(20, 90)]
    assert total_len(out) == 80


def test_plan_generation_basic_shape():
    plan = generate_nested_walk_forward(N, CFG)
    assert plan.lockbox == IndexRange(N - 1500, N)
    assert len(plan.outer_folds) == 4
    for f in plan.outer_folds:
        assert len(f.inner_folds) == 3
        assert len(f.test) == 500


def test_train_test_disjoint_and_chronological():
    plan = generate_nested_walk_forward(N, CFG)
    for f in plan.outer_folds:
        for r in f.train:
            assert not r.intersects(f.test)
            assert r.stop + CFG.purge_bars <= f.test.start
        for inner in f.inner_folds:
            for r in inner.train:
                assert not r.intersects(inner.validation)
                assert r.stop + CFG.purge_bars <= inner.validation.start


def test_purge_gap_exact_width():
    plan = generate_nested_walk_forward(N, CFG)
    f0 = plan.outer_folds[0]
    assert max(r.stop for r in f0.train) == f0.test.start - CFG.purge_bars


def test_embargo_excluded_from_later_folds():
    plan = generate_nested_walk_forward(N, CFG)
    # fold 2's train includes fold 0's test period, but not the embargo
    # window immediately after it.
    f0, f2 = plan.outer_folds[0], plan.outer_folds[2]
    train_indices = set()
    for r in f2.train:
        train_indices.update(range(r.start, r.stop))
    assert f0.test.start in train_indices  # earlier test data itself is reusable
    embargo = range(f0.test.stop, f0.test.stop + CFG.embargo_bars)
    assert not train_indices.intersection(embargo), "embargo window leaked into training"


def test_nothing_touches_lockbox_or_its_purge_zone():
    plan = generate_nested_walk_forward(N, CFG)
    limit = plan.lockbox.start - CFG.purge_bars
    for f in plan.outer_folds:
        assert f.test.stop <= limit
        for r in f.train:
            assert r.stop <= limit
        for inner in f.inner_folds:
            assert inner.validation.stop <= limit
            for r in inner.train:
                assert r.stop <= limit


def test_outer_tests_tile_without_overlap():
    plan = generate_nested_walk_forward(N, CFG)
    tests = [f.test for f in plan.outer_folds]
    for a, b in zip(tests, tests[1:], strict=False):
        assert a.stop == b.start


def test_plan_hash_stable_across_roundtrip():
    plan = generate_nested_walk_forward(N, CFG)
    clone = SplitPlan.from_dict(plan.to_dict())
    assert clone.plan_hash == plan.plan_hash
    validate_plan(clone)


def test_plan_deterministic():
    assert (
        generate_nested_walk_forward(N, CFG).plan_hash
        == generate_nested_walk_forward(N, CFG).plan_hash
    )


def test_config_that_does_not_fit_rejected():
    with pytest.raises(SplitConfigError):
        generate_nested_walk_forward(4000, CFG)  # not enough bars for min_train


def test_invalid_config_values_rejected():
    with pytest.raises(SplitConfigError):
        SplitConfig(
            n_outer_folds=0,
            outer_test_bars=500,
            n_inner_folds=3,
            inner_val_bars=300,
            purge_bars=50,
            embargo_bars=20,
            lockbox_bars=1500,
            min_train_bars=2000,
        )
    with pytest.raises(SplitConfigError):
        SplitConfig(
            n_outer_folds=4,
            outer_test_bars=500,
            n_inner_folds=3,
            inner_val_bars=300,
            purge_bars=-1,
            embargo_bars=20,
            lockbox_bars=1500,
            min_train_bars=2000,
        )


def test_validator_catches_hand_built_leakage():
    plan = generate_nested_walk_forward(N, CFG)
    d = plan.to_dict()
    # make fold 0's train overlap its test
    d["outer_folds"][0]["train"] = [
        {"start": 0, "stop": d["outer_folds"][0]["test"]["start"] + 10}
    ]
    with pytest.raises(SplitInvariantError):
        validate_plan(SplitPlan.from_dict(d))


def test_validator_catches_lockbox_intrusion():
    plan = generate_nested_walk_forward(N, CFG)
    d = plan.to_dict()
    d["outer_folds"][-1]["test"] = {"start": N - 1600, "stop": N - 1400}
    with pytest.raises(SplitInvariantError):
        validate_plan(SplitPlan.from_dict(d))


def test_zero_embargo_supported():
    cfg = SplitConfig(
        n_outer_folds=3,
        outer_test_bars=400,
        n_inner_folds=2,
        inner_val_bars=200,
        purge_bars=30,
        embargo_bars=0,
        lockbox_bars=1000,
        min_train_bars=1500,
    )
    plan = generate_nested_walk_forward(8000, cfg)
    validate_plan(plan)
