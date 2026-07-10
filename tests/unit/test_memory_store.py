from __future__ import annotations

import sqlite3

import pytest

from evoquant.memory.failures import failure_labels
from evoquant.memory.store import MemoryStore, MemoryStoreError
from evoquant.search.objectives import CandidateEvaluation, TargetConfig

TARGETS = TargetConfig(
    min_trades_total=10,
    max_drawdown=0.3,
    min_profit_factor=1.0,
    min_median_fold_return=0.0,
    max_fold_dispersion=0.5,
    max_complexity=100,
)


def _event(op="jitter_risk", ctx="hivol_trendy|trend_high_vol", reward=True, labels=()):
    return {
        "genome_hash": "abc",
        "operator": op,
        "context": ctx,
        "reward": reward,
        "failure_labels": list(labels),
    }


class TestStore:
    def test_append_and_replay_roundtrip(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        store.append("candidate_evaluated", _event(reward=True))
        store.append("candidate_evaluated", _event(reward=False))
        store.append("intervention", {"intervention": "NOVELTY_INCREASE"})
        assert store.count() == 3
        assert store.count("candidate_evaluated") == 2
        events = list(store.events("candidate_evaluated"))
        assert [e["reward"] for e in events] == [True, False]

    def test_persists_across_reopen(self, tmp_path):
        path = tmp_path / "mem.sqlite"
        MemoryStore(path).append("candidate_evaluated", _event())
        reopened = MemoryStore(path)
        assert reopened.count("candidate_evaluated") == 1

    def test_unknown_kind_rejected(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        with pytest.raises(MemoryStoreError):
            store.append("delete_everything", {})

    def test_no_update_or_delete_api_exists(self):
        """Append-only by construction: the class exposes no mutators."""
        mutators = [m for m in dir(MemoryStore) if "update" in m or "delete" in m]
        assert mutators == []

    def test_schema_version_mismatch_fails_closed(self, tmp_path):
        path = tmp_path / "mem.sqlite"
        MemoryStore(path).close()
        conn = sqlite3.connect(str(path))
        conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
        conn.commit()
        conn.close()
        with pytest.raises(MemoryStoreError):
            MemoryStore(path)

    def test_operator_outcomes_replay(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        for reward in (True, True, False):
            store.append("candidate_evaluated", _event(op="jitter_risk", reward=reward))
        store.append(
            "candidate_evaluated", _event(op="swap_comparison", reward=False)
        )
        store.append(
            "candidate_evaluated",
            _event(op="jitter_risk", ctx="lovol_choppy|range_low_vol", reward=True),
        )
        out = store.operator_outcomes()
        assert out[("hivol_trendy|trend_high_vol", "jitter_risk")] == {
            "successes": 2,
            "trials": 3,
        }
        assert out[("hivol_trendy|trend_high_vol", "swap_comparison")] == {
            "successes": 0,
            "trials": 1,
        }
        assert out[("lovol_choppy|range_low_vol", "jitter_risk")]["trials"] == 1

    def test_failure_label_counts(self, tmp_path):
        store = MemoryStore(tmp_path / "mem.sqlite")
        store.append("candidate_evaluated", _event(labels=["NO_SIGNAL"]))
        store.append(
            "candidate_evaluated", _event(labels=["NO_SIGNAL", "FOLD_INSTABILITY"])
        )
        counts = store.failure_label_counts()
        assert counts == {"NO_SIGNAL": 2, "FOLD_INSTABILITY": 1}


class TestFailureLabels:
    def _ev(self, **kw) -> CandidateEvaluation:
        base = dict(
            genome_hash="x",
            fold_returns=(0.05, 0.05),
            n_trades_total=50,
            max_drawdown=0.1,
            profit_factor=1.5,
            complexity=20,
            diagnostics={"trade_frequency": 0.02},
        )
        base.update(kw)
        return CandidateEvaluation(**base)

    def test_clean_candidate_gets_no_labels(self):
        assert failure_labels(self._ev(), TARGETS) == []

    def test_no_signal(self):
        assert "NO_SIGNAL" in failure_labels(self._ev(n_trades_total=0), TARGETS)

    def test_insufficient_trades(self):
        labels = failure_labels(self._ev(n_trades_total=3), TARGETS)
        assert "INSUFFICIENT_TRADES" in labels and "NO_SIGNAL" not in labels

    def test_overfire(self):
        ev = self._ev(diagnostics={"trade_frequency": 0.3})
        assert "OVERFIRE" in failure_labels(ev, TARGETS)

    def test_drawdown_and_costs(self):
        ev = self._ev(max_drawdown=0.6, profit_factor=0.5, fold_returns=(-0.05, -0.04))
        labels = failure_labels(ev, TARGETS)
        assert "STOP_GAP_RISK" in labels
        assert "NO_EDGE_UNDER_REALISTIC_COSTS" in labels

    def test_labels_deterministic(self):
        ev = self._ev(max_drawdown=0.6)
        assert failure_labels(ev, TARGETS) == failure_labels(ev, TARGETS)
