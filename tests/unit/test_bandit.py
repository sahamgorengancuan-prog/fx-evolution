from __future__ import annotations

import numpy as np

from evoquant.genome.mutations import OPERATOR_NAMES
from evoquant.memory.bandit import ThompsonOperatorSelector, context_key
from evoquant.regimes.fingerprints import PairFingerprint

OPS = tuple(op for op in OPERATOR_NAMES if op != "crossover")


def _fp(atr=0.02, er=0.4) -> PairFingerprint:
    return PairFingerprint(
        symbol="X",
        n_bars=1000,
        atr_pct_median=atr,
        atr_pct_iqr_rel=0.5,
        efficiency_median=er,
        efficiency_p90=0.8,
        abs_ret_median=0.002,
        ret_tail_ratio=5.0,
        gap_over_atr_freq=0.01,
        ret_autocorr_lag1=0.0,
    )


def test_context_key_buckets():
    assert context_key(_fp(atr=0.02, er=0.4), "trend_high_vol") == "hivol_trendy|trend_high_vol"
    assert context_key(_fp(atr=0.001, er=0.1), "range_low_vol") == "lovol_choppy|range_low_vol"
    assert context_key(None, "all") == "anyfp|all"


def test_selection_deterministic_under_seed():
    a = ThompsonOperatorSelector(operators=OPS, seed=42)
    b = ThompsonOperatorSelector(operators=OPS, seed=42)
    assert [a.select("ctx") for _ in range(20)] == [b.select("ctx") for _ in range(20)]


def test_bandit_learns_planted_better_operator():
    """jitter_risk succeeds 80%, everything else 10% — the posterior and the
    selection frequency must both favor it."""
    rng = np.random.default_rng(0)
    sel = ThompsonOperatorSelector(operators=OPS, seed=1)
    ctx = "hivol_trendy|trend_high_vol"
    for _ in range(400):
        op = sel.select(ctx)
        p = 0.8 if op == "jitter_risk" else 0.1
        sel.update(ctx, op, bool(rng.random() < p))

    means = {op: sel.posterior_mean(ctx, op) for op in OPS}
    assert max(means, key=lambda o: means[o]) == "jitter_risk"

    picks = [sel.select(ctx) for _ in range(200)]
    assert picks.count("jitter_risk") > len(picks) * 0.5


def test_contexts_learn_independently():
    sel = ThompsonOperatorSelector(operators=OPS, seed=3)
    for _ in range(50):
        sel.update("ctx_a", "jitter_risk", True)
        sel.update("ctx_b", "jitter_risk", False)
    assert sel.posterior_mean("ctx_a", "jitter_risk") > 0.9
    assert sel.posterior_mean("ctx_b", "jitter_risk") < 0.1


def test_unknown_operator_updates_ignored():
    sel = ThompsonOperatorSelector(operators=OPS, seed=3)
    sel.update("ctx", "quantum_leap", True)
    assert ("ctx", "quantum_leap") not in sel.counts


def test_serialization_roundtrip():
    sel = ThompsonOperatorSelector(operators=OPS, seed=9)
    sel.update("ctx", "jitter_risk", True)
    sel.update("ctx", "swap_comparison", False)
    clone = ThompsonOperatorSelector.from_dict(sel.to_dict())
    assert clone.counts == sel.counts
    assert clone.posterior_mean("ctx", "jitter_risk") == sel.posterior_mean(
        "ctx", "jitter_risk"
    )
