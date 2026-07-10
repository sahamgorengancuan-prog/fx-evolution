from __future__ import annotations

import pytest

from evoquant.memory.lessons import LessonError, LessonStore
from evoquant.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "mem.sqlite")


def test_add_and_retrieve_by_context(store):
    lessons = LessonStore(store)
    lessons.add(
        hypothesis="momentum decays after 3 bars",
        observed_outcome="entries later than 3 bars post-signal lose",
        evidence_count=40,
        confidence=0.8,
        applicable_contexts=("hivol_trendy|trend_high_vol",),
        failure_labels=("NO_EDGE_UNDER_REALISTIC_COSTS",),
    )
    lessons.add(
        hypothesis="low-confidence idea",
        observed_outcome="weak evidence",
        evidence_count=2,
        confidence=0.2,
        applicable_contexts=("hivol_trendy|trend_high_vol",),
    )
    hits = lessons.active_for_context("hivol_trendy|trend_high_vol", k=1)
    assert len(hits) == 1
    assert hits[0].confidence == 0.8  # highest confidence first
    assert lessons.active_for_context("other|ctx") == []


def test_supersede_retires_old_lesson(store):
    lessons = LessonStore(store)
    old = lessons.add(
        hypothesis="A",
        observed_outcome="works",
        evidence_count=5,
        confidence=0.5,
        applicable_contexts=("ctx",),
    )
    new = lessons.add(
        hypothesis="A refined",
        observed_outcome="works only in high vol",
        evidence_count=15,
        confidence=0.7,
        applicable_contexts=("ctx",),
        supersedes=old.lesson_id,
    )
    assert lessons.get(old.lesson_id).active is False
    assert lessons.get(old.lesson_id).superseded_by == new.lesson_id
    hits = lessons.active_for_context("ctx")
    assert [le.lesson_id for le in hits] == [new.lesson_id]


def test_lessons_replay_from_store(store, tmp_path):
    lessons = LessonStore(store)
    old = lessons.add(
        hypothesis="A",
        observed_outcome="o",
        evidence_count=3,
        confidence=0.4,
        applicable_contexts=("ctx",),
    )
    lessons.add(
        hypothesis="B",
        observed_outcome="o2",
        evidence_count=6,
        confidence=0.6,
        applicable_contexts=("ctx",),
        supersedes=old.lesson_id,
    )
    # rebuild from the same event log — state must reproduce
    rebuilt = LessonStore(store)
    assert len(rebuilt) == 2
    assert rebuilt.get(old.lesson_id).active is False
    assert len(rebuilt.active_for_context("ctx")) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(hypothesis="", observed_outcome="o", evidence_count=1, confidence=0.5),
        dict(hypothesis="h", observed_outcome="o", evidence_count=0, confidence=0.5),
        dict(hypothesis="h", observed_outcome="o", evidence_count=1, confidence=1.5),
    ],
)
def test_invalid_lessons_rejected(store, kwargs):
    lessons = LessonStore(store)
    with pytest.raises(LessonError):
        lessons.add(applicable_contexts=("ctx",), **kwargs)


def test_supersede_unknown_lesson_rejected(store):
    lessons = LessonStore(store)
    with pytest.raises(LessonError):
        lessons.add(
            hypothesis="h",
            observed_outcome="o",
            evidence_count=1,
            confidence=0.5,
            applicable_contexts=("ctx",),
            supersedes="LSN-doesnotexist",
        )
