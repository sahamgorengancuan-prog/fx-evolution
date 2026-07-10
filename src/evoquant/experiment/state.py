"""Experiment lifecycle state machine.

States and legal transitions:

```
CREATED ─seal─► SEALED ─► SEARCH_ACTIVE ─► SHORTLISTED ─► FROZEN ─open lockbox─► DISCLOSED
                                │               │                                    │
                                └► CLOSED_NO_EDGE ◄┘             CLOSED_ACCEPTED / CLOSED_REJECTED
```

* Sealing binds the experiment to a data manifest hash, a split-plan hash,
  a config hash, and an RNG seed. Seals are immutable afterwards.
* ``FROZEN → DISCLOSED`` may only be performed by the lockbox guard; the
  transition is recorded in the append-only event log, so a lockbox open is
  permanent and auditable.
* Any redesign after DISCLOSED requires a **new** experiment (new ID, new
  lockbox); the state machine has no path back.
"""
from __future__ import annotations

import datetime as _dt
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from evoquant.errors import IllegalTransitionError, SealViolationError


class ExperimentState(str, Enum):  # noqa: UP042 - StrEnum breaks .value JSON round-trip style used here
    CREATED = "CREATED"
    SEALED = "SEALED"
    SEARCH_ACTIVE = "SEARCH_ACTIVE"
    SHORTLISTED = "SHORTLISTED"
    FROZEN = "FROZEN"
    DISCLOSED = "DISCLOSED"
    CLOSED_ACCEPTED = "CLOSED_ACCEPTED"
    CLOSED_REJECTED = "CLOSED_REJECTED"
    CLOSED_NO_EDGE = "CLOSED_NO_EDGE"


TERMINAL_STATES = {
    ExperimentState.CLOSED_ACCEPTED,
    ExperimentState.CLOSED_REJECTED,
    ExperimentState.CLOSED_NO_EDGE,
}

ALLOWED_TRANSITIONS: dict[ExperimentState, set[ExperimentState]] = {
    ExperimentState.CREATED: {ExperimentState.SEALED},
    ExperimentState.SEALED: {ExperimentState.SEARCH_ACTIVE},
    ExperimentState.SEARCH_ACTIVE: {
        ExperimentState.SHORTLISTED,
        ExperimentState.CLOSED_NO_EDGE,
    },
    ExperimentState.SHORTLISTED: {
        ExperimentState.FROZEN,
        ExperimentState.CLOSED_NO_EDGE,
    },
    ExperimentState.FROZEN: {ExperimentState.DISCLOSED},
    ExperimentState.DISCLOSED: {
        ExperimentState.CLOSED_ACCEPTED,
        ExperimentState.CLOSED_REJECTED,
    },
    ExperimentState.CLOSED_ACCEPTED: set(),
    ExperimentState.CLOSED_REJECTED: set(),
    ExperimentState.CLOSED_NO_EDGE: set(),
}

#: Seal keys required before an experiment may leave CREATED.
REQUIRED_SEALS = ("data_content_hash", "split_plan_hash", "config_hash", "seed")

#: Sentinel actor allowed to perform FROZEN -> DISCLOSED.
LOCKBOX_GUARD_ACTOR = "lockbox_guard"


def _utcnow() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


@dataclass
class Experiment:
    experiment_id: str
    created_at_utc: str
    state: ExperimentState = ExperimentState.CREATED
    seals: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #

    @classmethod
    def create(cls, name: str = "", metadata: dict[str, Any] | None = None) -> Experiment:
        exp = cls(
            experiment_id=f"EXP-{uuid.uuid4().hex[:12]}",
            created_at_utc=_utcnow(),
            metadata={"name": name, **(metadata or {})},
        )
        exp._log_event(None, ExperimentState.CREATED, actor="system", details={"name": name})
        return exp

    def _log_event(
        self,
        from_state: ExperimentState | None,
        to_state: ExperimentState,
        actor: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            {
                "timestamp_utc": _utcnow(),
                "from": from_state.value if from_state else None,
                "to": to_state.value,
                "actor": actor,
                "details": details or {},
            }
        )

    # ------------------------------------------------------------------ #

    def seal(
        self,
        *,
        data_content_hash: str,
        split_plan_hash: str,
        config_hash: str,
        seed: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Bind immutable inputs and transition CREATED -> SEALED."""
        if self.state is not ExperimentState.CREATED:
            raise SealViolationError(
                "Experiment can only be sealed from CREATED",
                experiment_id=self.experiment_id,
                state=self.state.value,
            )
        if self.seals:
            raise SealViolationError(
                "Experiment already carries seals", experiment_id=self.experiment_id
            )
        self.seals = {
            "data_content_hash": data_content_hash,
            "split_plan_hash": split_plan_hash,
            "config_hash": config_hash,
            "seed": seed,
            **(extra or {}),
        }
        self._transition(ExperimentState.SEALED, actor="system", details={"seals": self.seals})

    def assert_seal(self, key: str, value: Any) -> None:
        if key not in self.seals:
            raise SealViolationError(
                f"No seal named {key!r}", experiment_id=self.experiment_id
            )
        if self.seals[key] != value:
            raise SealViolationError(
                f"Seal mismatch for {key!r}",
                experiment_id=self.experiment_id,
                expected=self.seals[key],
                actual=value,
            )

    # ------------------------------------------------------------------ #

    def _transition(
        self, to_state: ExperimentState, actor: str, details: dict[str, Any] | None = None
    ) -> None:
        if to_state not in ALLOWED_TRANSITIONS[self.state]:
            raise IllegalTransitionError(
                f"Illegal transition {self.state.value} -> {to_state.value}",
                experiment_id=self.experiment_id,
                from_state=self.state.value,
                to_state=to_state.value,
            )
        if to_state is ExperimentState.DISCLOSED and actor != LOCKBOX_GUARD_ACTOR:
            raise IllegalTransitionError(
                "Only the lockbox guard may disclose an experiment",
                experiment_id=self.experiment_id,
                actor=actor,
            )
        from_state = self.state
        self.state = to_state
        self._log_event(from_state, to_state, actor=actor, details=details)

    def transition(
        self,
        to_state: ExperimentState,
        actor: str = "system",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Public transition entry point (cannot reach DISCLOSED)."""
        if to_state is ExperimentState.DISCLOSED:
            raise IllegalTransitionError(
                "DISCLOSED is reachable only through the lockbox guard",
                experiment_id=self.experiment_id,
            )
        self._transition(to_state, actor=actor, details=details)

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "created_at_utc": self.created_at_utc,
            "state": self.state.value,
            "seals": self.seals,
            "events": self.events,
            "metadata": self.metadata,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        tmp.rename(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Experiment:
        doc = json.loads(Path(path).read_text())
        exp = cls(
            experiment_id=doc["experiment_id"],
            created_at_utc=doc["created_at_utc"],
            state=ExperimentState(doc["state"]),
            seals=doc["seals"],
            events=doc["events"],
            metadata=doc.get("metadata", {}),
        )
        if exp.events and exp.events[-1]["to"] != exp.state.value:
            raise IllegalTransitionError(
                "Persisted state does not match event log tail (file tampered?)",
                experiment_id=exp.experiment_id,
                state=exp.state.value,
                last_event_to=exp.events[-1]["to"],
            )
        return exp
