"""Event-sourced empirical memory (SQLite, append-only).

Every evaluated candidate, intervention, and generation telemetry record
is an immutable event. There is deliberately no update/delete API —
negative evidence is preserved forever (blueprint truth constraint #7).
Replaying events reconstructs derived state (operator outcome counts),
which is how the Phase-6 bandit and future analytics stay auditable.
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from evoquant.errors import EvoquantError

SCHEMA_VERSION = 1

EVENT_KINDS = (
    "candidate_evaluated",
    "intervention",
    "generation_telemetry",
    "lesson_added",
    "llm_exchange",
)


class MemoryStoreError(EvoquantError):
    code = "MEMORY_STORE"


class MemoryStore:
    """Append-only event log. One SQLite file per experiment."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  ts_utc TEXT NOT NULL,"
            "  kind TEXT NOT NULL,"
            "  payload TEXT NOT NULL)"
        )
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()
        elif int(row[0]) != SCHEMA_VERSION:
            raise MemoryStoreError(
                "Memory store schema version mismatch; run a migration",
                found=int(row[0]),
                expected=SCHEMA_VERSION,
                path=str(self._path),
            )

    # ------------------------------------------------------------------ #

    def append(self, kind: str, payload: dict[str, Any]) -> int:
        if kind not in EVENT_KINDS:
            raise MemoryStoreError(f"Unknown event kind {kind!r}", allowed=EVENT_KINDS)
        cur = self._conn.execute(
            "INSERT INTO events (ts_utc, kind, payload) VALUES (?, ?, ?)",
            (
                _dt.datetime.now(_dt.UTC).isoformat(),
                kind,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid or 0)

    def events(self, kind: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield events in insertion order."""
        if kind is None:
            rows = self._conn.execute(
                "SELECT id, ts_utc, kind, payload FROM events ORDER BY id"
            )
        else:
            rows = self._conn.execute(
                "SELECT id, ts_utc, kind, payload FROM events WHERE kind=? ORDER BY id",
                (kind,),
            )
        for event_id, ts, k, payload in rows:
            yield {"id": event_id, "ts_utc": ts, "kind": k, **json.loads(payload)}

    def count(self, kind: str | None = None) -> int:
        if kind is None:
            row = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE kind=?", (kind,)
            ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ #
    # replay-derived state
    # ------------------------------------------------------------------ #

    def operator_outcomes(self) -> dict[tuple[str, str], dict[str, int]]:
        """Replay candidate events into {(context, operator): {successes, trials}}."""
        out: dict[tuple[str, str], dict[str, int]] = {}
        for e in self.events("candidate_evaluated"):
            op = e.get("operator")
            ctx = e.get("context")
            # only bandit-relevant events: an operator with a decided reward
            # (crossover/random/elite events carry reward=None)
            if not op or ctx is None or e.get("reward") is None:
                continue
            key = (str(ctx), str(op))
            slot = out.setdefault(key, {"successes": 0, "trials": 0})
            slot["trials"] += 1
            if e.get("reward"):
                slot["successes"] += 1
        return out

    def failure_label_counts(self, context: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events("candidate_evaluated"):
            if context is not None and e.get("context") != context:
                continue
            for label in e.get("failure_labels", []):
                counts[label] = counts.get(label, 0) + 1
        return counts
