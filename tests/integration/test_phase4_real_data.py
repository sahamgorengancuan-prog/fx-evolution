"""Phase-4 acceptance on real BNBUSDT data with the sealed split plan."""
from __future__ import annotations

import json

import numpy as np
import pytest

from evoquant.data.loaders import load_fsb_json
from evoquant.data.splits import SplitConfig, generate_nested_walk_forward
from evoquant.experiment.lockbox import research_view
from evoquant.regimes.fingerprints import compute_fingerprint
from evoquant.regimes.router import QuantileRegimeRouter, Regime
from tests.conftest import REAL_DATA, REPO_ROOT

pytestmark = pytest.mark.skipif(
    not REAL_DATA.exists(), reason="real BNBUSDT data file not present"
)


@pytest.fixture(scope="module")
def setup():
    bars, spec, _ = load_fsb_json(REAL_DATA)
    cfg_doc = json.loads((REPO_ROOT / "configs" / "splits_bnbusdt_h1.json").read_text())
    plan = generate_nested_walk_forward(bars.n_bars, SplitConfig(**cfg_doc))
    return bars, plan


def test_fold_train_fit_infers_one_sided_over_research_view(setup):
    bars, plan = setup
    fold0 = plan.outer_folds[0]
    train_range = fold0.train[0]  # first (contiguous) training range
    train_bars = bars.slice(train_range.start, train_range.stop)

    router = QuantileRegimeRouter(er_period=24, atr_period=14, min_dwell=12)
    params = router.fit(train_bars)
    assert params.params_hash() == router.fit(train_bars).params_hash()

    research = research_view(bars, plan)
    labels = router.infer(research)
    assert len(labels) == research.n_bars

    # one-sided on real data: labels over the train prefix are unchanged
    prefix = router.infer(research.slice(0, train_range.stop))
    assert np.array_equal(labels[: train_range.stop], prefix)

    # all four substantive regimes occur; UNKNOWN exists (warmup at least)
    present = set(int(x) for x in np.unique(labels))
    assert {
        int(Regime.TREND_HIGH_VOL),
        int(Regime.TREND_LOW_VOL),
        int(Regime.RANGE_HIGH_VOL),
        int(Regime.RANGE_LOW_VOL),
    } <= present
    assert int(labels[0]) == int(Regime.UNKNOWN)


def test_fingerprint_on_train_partition_only(setup):
    bars, plan = setup
    fold0 = plan.outer_folds[0]
    r = fold0.train[0]
    fp = compute_fingerprint(bars.slice(r.start, r.stop))
    assert fp.symbol == "BNBUSDT"
    assert fp.n_bars == len(r)
    assert fp.atr_pct_median > 0
    # BNB H1: sanity ranges, not performance claims
    assert 0.0 < fp.efficiency_median < 1.0
    assert fp.ret_tail_ratio > 1.0
