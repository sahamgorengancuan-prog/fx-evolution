"""Semantic lesson store.

Lessons are the LLM-readable distillation of empirical evidence: a
hypothesis, what was observed, how much evidence backs it, and where it
applies. Lessons are append-only; corrections supersede rather than
mutate, so contradictory evidence stays visible.
"""
from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from evoquant.errors import EvoquantError
from evoquant.memory.store import MemoryStore


class LessonError(EvoquantError):
    code = "LESSON"


@dataclass(frozen=True)
class Lesson:
    lesson_id: str
    hypothesis: str
    observed_outcome: str
    evidence_count: int
    confidence: float  # 0..1, caller-estimated, never LLM-asserted
    applicable_contexts: tuple[str, ...]  # bandit context keys
    failure_labels: tuple[str, ...] = field(default_factory=tuple)
    active: bool = True
    superseded_by: str | None = None
    created_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LessonStore:
    """In-memory index over lessons, persisted as events in a MemoryStore."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._lessons: dict[str, Lesson] = {}
        for e in store.events("lesson_added"):
            payload = dict(e["lesson"])
            supersedes = payload.pop("supersedes", None)
            lesson = Lesson(**{k: _detuple(k, v) for k, v in payload.items()})
            self._lessons[lesson.lesson_id] = lesson
            if supersedes and supersedes in self._lessons:
                old = self._lessons[supersedes]
                self._lessons[supersedes] = replace(
                    old, active=False, superseded_by=lesson.lesson_id
                )

    def add(
        self,
        hypothesis: str,
        observed_outcome: str,
        evidence_count: int,
        confidence: float,
        applicable_contexts: tuple[str, ...],
        failure_labels: tuple[str, ...] = (),
        supersedes: str | None = None,
    ) -> Lesson:
        if not hypothesis.strip() or not observed_outcome.strip():
            raise LessonError("Lesson requires hypothesis and observed outcome")
        if not (0.0 <= confidence <= 1.0):
            raise LessonError("Lesson confidence must be in [0,1]", value=confidence)
        if evidence_count < 1:
            raise LessonError("Lesson requires evidence_count >= 1")
        if supersedes is not None and supersedes not in self._lessons:
            raise LessonError("Cannot supersede unknown lesson", lesson_id=supersedes)
        lesson = Lesson(
            lesson_id=f"LSN-{uuid.uuid4().hex[:10]}",
            hypothesis=hypothesis.strip(),
            observed_outcome=observed_outcome.strip(),
            evidence_count=evidence_count,
            confidence=confidence,
            applicable_contexts=applicable_contexts,
            failure_labels=failure_labels,
            created_at_utc=_dt.datetime.now(_dt.UTC).isoformat(),
        )
        payload = lesson.to_dict()
        if supersedes:
            payload["supersedes"] = supersedes
        self._store.append("lesson_added", {"lesson": payload})
        self._lessons[lesson.lesson_id] = lesson
        if supersedes:
            old = self._lessons[supersedes]
            self._lessons[supersedes] = replace(
                old, active=False, superseded_by=lesson.lesson_id
            )
        return lesson

    def get(self, lesson_id: str) -> Lesson:
        if lesson_id not in self._lessons:
            raise LessonError("Unknown lesson", lesson_id=lesson_id)
        return self._lessons[lesson_id]

    def active_for_context(self, context: str, k: int = 5) -> list[Lesson]:
        """Highest-confidence active lessons applicable to a context."""
        hits = [
            lesson
            for lesson in self._lessons.values()
            if lesson.active and context in lesson.applicable_contexts
        ]
        hits.sort(key=lambda le: (-le.confidence, -le.evidence_count, le.lesson_id))
        return hits[:k]

    def __len__(self) -> int:
        return len(self._lessons)


def _detuple(key: str, value: Any) -> Any:
    if key in ("applicable_contexts", "failure_labels") and isinstance(value, list):
        return tuple(value)
    return value
