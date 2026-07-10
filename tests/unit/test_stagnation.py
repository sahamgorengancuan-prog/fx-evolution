from __future__ import annotations

import numpy as np
import pytest

from evoquant.search.stagnation import (
    LADDER,
    GenerationSignals,
    Intervention,
    StagnationController,
    hypervolume_2d,
)


def sig(gen, hv=1.0, feas=0.5, cov=1, conc=0.5, minviol=0.0) -> GenerationSignals:
    return GenerationSignals(
        generation=gen,
        hypervolume=hv,
        feasibility_rate=feas,
        archive_coverage=cov,
        lineage_concentration=conc,
        min_violation=minviol,
    )


class TestHypervolume:
    def test_hand_computed_2d(self):
        pts = np.array([[0.5, -0.5], [0.0, 0.0]])
        assert hypervolume_2d(pts, ref=(-1.0, -1.0)) == pytest.approx(1.25)

    def test_single_point(self):
        assert hypervolume_2d(np.array([[0.0, 0.0]]), ref=(-1.0, -1.0)) == pytest.approx(1.0)

    def test_dominated_point_adds_nothing(self):
        a = hypervolume_2d(np.array([[0.5, 0.5]]), ref=(-1.0, -1.0))
        b = hypervolume_2d(np.array([[0.5, 0.5], [0.2, 0.2]]), ref=(-1.0, -1.0))
        assert a == pytest.approx(b)

    def test_points_below_ref_ignored(self):
        assert hypervolume_2d(np.array([[-2.0, -2.0]]), ref=(-1.0, -1.0)) == 0.0


class TestEscalationLadder:
    def test_progress_never_triggers_interventions(self):
        ctrl = StagnationController(patience=2)
        for g in range(10):
            assert ctrl.observe(sig(g, hv=float(g))) is None
        assert ctrl.log == []

    def test_ladder_fires_in_order_and_terminates(self):
        ctrl = StagnationController(patience=2)
        ctrl.observe(sig(0, hv=1.0))  # baseline
        fired: list[Intervention] = []
        g = 1
        while not ctrl.terminal and g < 100:
            out = ctrl.observe(sig(g, hv=1.0))  # flat: no progress
            if out is not None:
                fired.append(out)
            g += 1
        assert fired == list(LADDER)
        assert ctrl.terminal

    def test_infeasible_diversity_churn_is_not_progress(self):
        """All-infeasible populations cannot dodge NO_EDGE_FOUND by growing
        hypervolume or archive coverage (v8 D3/D4 regression)."""
        ctrl = StagnationController(patience=1)
        ctrl.observe(sig(0, feas=0.0, minviol=5.0, hv=1.0, cov=1))
        fired: list[Intervention] = []
        for g in range(1, 30):
            if ctrl.terminal:
                break
            out = ctrl.observe(
                sig(g, feas=0.0, minviol=5.0, hv=1.0 + g, cov=1 + g)  # churn "grows"
            )
            if out is not None:
                fired.append(out)
        assert fired == list(LADDER)

    def test_violation_reduction_counts_as_progress_when_infeasible(self):
        ctrl = StagnationController(patience=1)
        ctrl.observe(sig(0, feas=0.0, minviol=5.0))
        assert ctrl.observe(sig(1, feas=0.0, minviol=4.0)) is None  # closer to feasible
        assert ctrl.observe(sig(2, feas=0.0, minviol=3.0)) is None

    def test_every_intervention_is_logged_with_signals(self):
        ctrl = StagnationController(patience=1)
        ctrl.observe(sig(0))
        ctrl.observe(sig(1))
        events = [e for e in ctrl.log if e["event"] == "intervention"]
        assert events and events[0]["intervention"] == "LOCAL_REFINEMENT"
        assert "signals" in events[0] and events[0]["signals"]["generation"] == 1

    def test_improvement_resets_the_rung(self):
        ctrl = StagnationController(patience=1)
        ctrl.observe(sig(0, hv=1.0))
        assert ctrl.observe(sig(1, hv=1.0)) is Intervention.LOCAL_REFINEMENT
        assert ctrl.observe(sig(2, hv=2.0)) is None  # progress resumes
        # next stagnation starts at the bottom rung again
        assert ctrl.observe(sig(3, hv=2.0)) is Intervention.LOCAL_REFINEMENT
        resets = [e for e in ctrl.log if e["event"] == "progress_resumed"]
        assert resets

    def test_archive_coverage_growth_counts_as_progress(self):
        ctrl = StagnationController(patience=1)
        ctrl.observe(sig(0, cov=1))
        assert ctrl.observe(sig(1, cov=2)) is None
        assert ctrl.observe(sig(2, cov=3)) is None
