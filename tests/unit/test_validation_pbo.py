from __future__ import annotations

import numpy as np
import pytest

from evoquant.validation.pbo import pbo_cscv
from evoquant.validation.sharpe import ValidationInputError


def test_single_candidate_is_rejected():
    """A per-single-strategy 'PBO' is a category error (v8 audit S1)."""
    with pytest.raises(ValidationInputError) as exc:
        pbo_cscv(np.random.default_rng(0).normal(0, 1, (100, 1)))
    assert "single candidate" in str(exc.value)


def test_planted_overfit_matrix_yields_high_pbo():
    """Each candidate 'wins' only on its own lucky rows: the IS winner is
    reliably an OOS loser — PBO must approach 1."""
    t, n = 64, 16
    m = np.full((t, n), -0.01)
    rng = np.random.default_rng(1)
    for k in range(n):
        lucky_rows = rng.choice(t // 2, size=2, replace=False) * 2  # spread out
        m[lucky_rows, k] = 1.0
    result = pbo_cscv(m, n_partitions=8)
    assert result.pbo > 0.8
    assert result.n_combinations == 70  # C(8,4)


def test_true_edge_yields_low_pbo():
    """One candidate with a genuine mean shift keeps winning OOS."""
    rng = np.random.default_rng(2)
    t, n = 200, 12
    m = rng.normal(0.0, 1.0, (t, n))
    m[:, 3] += 1.0  # persistent true edge
    result = pbo_cscv(m, n_partitions=8)
    assert result.pbo < 0.2


def test_pure_noise_pbo_near_half():
    """With iid noise, IS ranking is uninformative: PBO ≈ 0.5."""
    rng = np.random.default_rng(3)
    m = rng.normal(0.0, 1.0, (240, 20))
    result = pbo_cscv(m, n_partitions=8)
    assert 0.3 < result.pbo < 0.7


def test_pbo_deterministic():
    rng = np.random.default_rng(4)
    m = rng.normal(0, 1, (120, 8))
    assert pbo_cscv(m).pbo == pbo_cscv(m).pbo


@pytest.mark.parametrize("bad_partitions", [3, 0, -2])
def test_odd_or_invalid_partitions_rejected(bad_partitions):
    m = np.random.default_rng(5).normal(0, 1, (100, 4))
    with pytest.raises(ValidationInputError):
        pbo_cscv(m, n_partitions=bad_partitions)


def test_too_few_rows_rejected():
    m = np.random.default_rng(6).normal(0, 1, (10, 4))
    with pytest.raises(ValidationInputError):
        pbo_cscv(m, n_partitions=8)
