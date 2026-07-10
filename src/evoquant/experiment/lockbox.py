"""Final-lockbox access control.

Structural guarantees provided here:

* :func:`research_view` is the only sanctioned way for search code to get
  data — it physically excludes lockbox rows (and their purge zone stays
  inside the research plan, enforced by the split validator).
* :meth:`LockboxGuard.open` is the only code path that returns lockbox
  rows. It requires the experiment to be ``FROZEN``, verifies the sealed
  data-content hash and split-plan hash against what is presented at open
  time, transitions the experiment to ``DISCLOSED`` permanently, and can
  therefore succeed at most once per experiment.

What this module cannot guarantee: Python offers no in-process memory
isolation, so a hostile caller could slice the raw arrays directly. That
residual risk is documented in KNOWN_LIMITATIONS.md; the audit trail
(event log + seals) still makes any legitimate-looking result verifiable.
"""
from __future__ import annotations

from typing import Any

from evoquant.data.loaders import BarData
from evoquant.data.manifest import canonical_content_hash
from evoquant.data.splits import SplitPlan
from evoquant.errors import (
    LockboxAccessError,
    LockboxAlreadyOpenError,
    LockboxSealMismatchError,
)
from evoquant.experiment.state import (
    LOCKBOX_GUARD_ACTOR,
    Experiment,
    ExperimentState,
)


def research_view(bars: BarData, plan: SplitPlan) -> BarData:
    """All data a search/validation process is permitted to see."""
    return bars.slice(0, plan.lockbox.start)


class LockboxGuard:
    """One-shot gate in front of the sealed lockbox segment."""

    def __init__(self, experiment: Experiment, plan: SplitPlan) -> None:
        self._experiment = experiment
        self._plan = plan

    def open(self, bars: BarData, actor: str, reason: str = "") -> BarData:
        """Open the lockbox exactly once. Returns the lockbox bars.

        Requires state FROZEN; verifies seals; transitions to DISCLOSED.
        """
        exp = self._experiment
        if exp.state is ExperimentState.DISCLOSED or any(
            e["to"] == ExperimentState.DISCLOSED.value for e in exp.events
        ):
            raise LockboxAlreadyOpenError(
                "Lockbox has already been opened for this experiment; "
                "a disclosed lockbox can never be reused as unseen data. "
                "Start a new experiment with a new lockbox.",
                experiment_id=exp.experiment_id,
            )
        if exp.state is not ExperimentState.FROZEN:
            raise LockboxAccessError(
                "Lockbox may only be opened when the experiment is FROZEN "
                f"(current state: {exp.state.value}). Freeze the candidate and "
                "thresholds first.",
                experiment_id=exp.experiment_id,
                state=exp.state.value,
            )

        presented_plan_hash = self._plan.plan_hash
        try:
            exp.assert_seal("split_plan_hash", presented_plan_hash)
        except Exception as exc:  # re-brand as lockbox seal failure
            raise LockboxSealMismatchError(
                "Split plan presented at open time does not match the sealed plan",
                experiment_id=exp.experiment_id,
                presented=presented_plan_hash,
            ) from exc

        presented_data_hash = canonical_content_hash(bars)
        try:
            exp.assert_seal("data_content_hash", presented_data_hash)
        except Exception as exc:
            raise LockboxSealMismatchError(
                "Bar data presented at open time does not match the sealed data hash",
                experiment_id=exp.experiment_id,
                presented=presented_data_hash,
            ) from exc

        lb = self._plan.lockbox
        view = bars.slice(lb.start, lb.stop)
        exp._transition(  # noqa: SLF001 - guard is the sole sanctioned caller
            ExperimentState.DISCLOSED,
            actor=LOCKBOX_GUARD_ACTOR,
            details={
                "opened_by": actor,
                "reason": reason,
                "lockbox": lb.to_dict(),
                "lockbox_first_ts_utc_s": int(view.ts_utc_s[0]),
                "lockbox_last_ts_utc_s": int(view.ts_utc_s[-1]),
            },
        )
        return view

    def audit(self) -> dict[str, Any]:
        """Summarize lockbox-related events for reporting."""
        opens = [
            e
            for e in self._experiment.events
            if e["to"] == ExperimentState.DISCLOSED.value
        ]
        return {
            "experiment_id": self._experiment.experiment_id,
            "state": self._experiment.state.value,
            "n_opens": len(opens),
            "open_events": opens,
        }
