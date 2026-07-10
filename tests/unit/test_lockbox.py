from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from evoquant.data.manifest import canonical_content_hash
from evoquant.data.splits import SplitConfig, generate_nested_walk_forward
from evoquant.errors import (
    LockboxAccessError,
    LockboxAlreadyOpenError,
    LockboxSealMismatchError,
)
from evoquant.experiment.lockbox import LockboxGuard, research_view
from evoquant.experiment.state import Experiment, ExperimentState
from tests.conftest import make_bars

CFG = SplitConfig(
    n_outer_folds=3,
    outer_test_bars=400,
    n_inner_folds=2,
    inner_val_bars=200,
    purge_bars=30,
    embargo_bars=10,
    lockbox_bars=1000,
    min_train_bars=1500,
)


def _setup(state: ExperimentState = ExperimentState.FROZEN):
    bars = make_bars(8000)
    plan = generate_nested_walk_forward(bars.n_bars, CFG)
    exp = Experiment.create(name="lockbox-test")
    exp.seal(
        data_content_hash=canonical_content_hash(bars),
        split_plan_hash=plan.plan_hash,
        config_hash="c" * 64,
        seed=7,
    )
    path = [
        ExperimentState.SEARCH_ACTIVE,
        ExperimentState.SHORTLISTED,
        ExperimentState.FROZEN,
    ]
    for s in path:
        if exp.state == state:
            break
        exp.transition(s)
    return bars, plan, exp


def test_research_view_excludes_lockbox():
    bars, plan, _ = _setup()
    view = research_view(bars, plan)
    assert view.n_bars == plan.lockbox.start
    assert view.ts_utc_s[-1] < bars.ts_utc_s[plan.lockbox.start]


def test_open_before_frozen_denied():
    bars, plan, exp = _setup(state=ExperimentState.SEARCH_ACTIVE)
    guard = LockboxGuard(exp, plan)
    with pytest.raises(LockboxAccessError):
        guard.open(bars, actor="tester")
    assert exp.state is ExperimentState.SEARCH_ACTIVE  # no side effect


def test_open_exactly_once():
    bars, plan, exp = _setup()
    guard = LockboxGuard(exp, plan)
    view = guard.open(bars, actor="tester", reason="final evaluation")
    assert view.n_bars == len(plan.lockbox)
    assert np.array_equal(view.ts_utc_s, bars.ts_utc_s[plan.lockbox.start :])
    assert exp.state is ExperimentState.DISCLOSED

    with pytest.raises(LockboxAlreadyOpenError):
        guard.open(bars, actor="tester")
    # a fresh guard over the same experiment cannot reopen either
    with pytest.raises(LockboxAlreadyOpenError):
        LockboxGuard(exp, plan).open(bars, actor="tester")
    assert guard.audit()["n_opens"] == 1


def test_open_with_tampered_data_denied():
    bars, plan, exp = _setup()
    close = np.array(bars.close)
    close[-1] += 0.01
    tampered = replace(bars, close=close)
    with pytest.raises(LockboxSealMismatchError):
        LockboxGuard(exp, plan).open(tampered, actor="tester")
    assert exp.state is ExperimentState.FROZEN  # open did not happen


def test_open_with_wrong_plan_denied():
    bars, plan, exp = _setup()
    other_plan = generate_nested_walk_forward(
        bars.n_bars,
        replace(CFG, lockbox_bars=900),
    )
    with pytest.raises(LockboxSealMismatchError):
        LockboxGuard(exp, other_plan).open(bars, actor="tester")


def test_disclosure_event_is_audited():
    bars, plan, exp = _setup()
    LockboxGuard(exp, plan).open(bars, actor="alice", reason="freeze complete")
    event = [e for e in exp.events if e["to"] == "DISCLOSED"][0]
    assert event["actor"] == "lockbox_guard"
    assert event["details"]["opened_by"] == "alice"
    assert event["details"]["lockbox"] == plan.lockbox.to_dict()
