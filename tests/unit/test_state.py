from __future__ import annotations

import json

import pytest

from evoquant.errors import IllegalTransitionError, SealViolationError
from evoquant.experiment.state import Experiment, ExperimentState


def _sealed():
    exp = Experiment.create(name="t")
    exp.seal(
        data_content_hash="d" * 64,
        split_plan_hash="p" * 64,
        config_hash="c" * 64,
        seed=42,
    )
    return exp


def test_lifecycle_happy_path():
    exp = _sealed()
    exp.transition(ExperimentState.SEARCH_ACTIVE)
    exp.transition(ExperimentState.SHORTLISTED)
    exp.transition(ExperimentState.FROZEN)
    # DISCLOSED only via lockbox guard (tested in test_lockbox)
    assert exp.state is ExperimentState.FROZEN
    assert [e["to"] for e in exp.events] == [
        "CREATED",
        "SEALED",
        "SEARCH_ACTIVE",
        "SHORTLISTED",
        "FROZEN",
    ]


def test_no_edge_found_is_legal_terminal():
    exp = _sealed()
    exp.transition(ExperimentState.SEARCH_ACTIVE)
    exp.transition(ExperimentState.CLOSED_NO_EDGE, details={"reason": "budget exhausted"})
    assert exp.state is ExperimentState.CLOSED_NO_EDGE
    with pytest.raises(IllegalTransitionError):
        exp.transition(ExperimentState.SEARCH_ACTIVE)


@pytest.mark.parametrize(
    "target",
    [
        ExperimentState.FROZEN,  # skipping states
        ExperimentState.CLOSED_ACCEPTED,  # closing before disclosure
        ExperimentState.CREATED,  # going backwards
    ],
)
def test_illegal_transitions_rejected(target):
    exp = _sealed()
    exp.transition(ExperimentState.SEARCH_ACTIVE)
    with pytest.raises(IllegalTransitionError):
        exp.transition(target)


def test_disclosed_unreachable_via_public_api():
    exp = _sealed()
    exp.transition(ExperimentState.SEARCH_ACTIVE)
    exp.transition(ExperimentState.SHORTLISTED)
    exp.transition(ExperimentState.FROZEN)
    with pytest.raises(IllegalTransitionError):
        exp.transition(ExperimentState.DISCLOSED)


def test_cannot_seal_twice_or_out_of_state():
    exp = _sealed()
    with pytest.raises(SealViolationError):
        exp.seal(
            data_content_hash="x" * 64,
            split_plan_hash="y" * 64,
            config_hash="z" * 64,
            seed=1,
        )


def test_search_requires_seal():
    exp = Experiment.create()
    with pytest.raises(IllegalTransitionError):
        exp.transition(ExperimentState.SEARCH_ACTIVE)


def test_assert_seal():
    exp = _sealed()
    exp.assert_seal("seed", 42)
    with pytest.raises(SealViolationError):
        exp.assert_seal("seed", 43)
    with pytest.raises(SealViolationError):
        exp.assert_seal("nonexistent", 1)


def test_persistence_roundtrip(tmp_path):
    exp = _sealed()
    exp.transition(ExperimentState.SEARCH_ACTIVE)
    path = exp.save(tmp_path / "exp.json")
    loaded = Experiment.load(path)
    assert loaded.state is ExperimentState.SEARCH_ACTIVE
    assert loaded.seals == exp.seals
    assert loaded.events == exp.events


def test_tampered_persistence_detected(tmp_path):
    exp = _sealed()
    path = exp.save(tmp_path / "exp.json")
    doc = json.loads(path.read_text())
    doc["state"] = "FROZEN"  # skip ahead without events
    path.write_text(json.dumps(doc))
    with pytest.raises(IllegalTransitionError):
        Experiment.load(path)
