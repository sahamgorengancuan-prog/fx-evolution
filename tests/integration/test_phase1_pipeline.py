"""End-to-end Phase-1 pipeline on the real BNBUSDT H1 export.

This is the acceptance test for Phase 1: load → QA → manifest → split plan
→ sealed experiment → lifecycle → one-shot lockbox.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from evoquant import repro
from evoquant.data.loaders import load_fsb_json
from evoquant.data.manifest import build_manifest, verify_manifest, write_manifest
from evoquant.data.quality import validate_or_raise
from evoquant.data.splits import SplitConfig, generate_nested_walk_forward, validate_plan
from evoquant.errors import LockboxAlreadyOpenError
from evoquant.experiment.lockbox import LockboxGuard, research_view
from evoquant.experiment.state import Experiment, ExperimentState
from tests.conftest import REAL_DATA, REPO_ROOT

pytestmark = pytest.mark.skipif(
    not REAL_DATA.exists(), reason="real BNBUSDT data file not present"
)

SPLIT_CONFIG_PATH = REPO_ROOT / "configs" / "splits_bnbusdt_h1.json"


@pytest.fixture(scope="module")
def loaded():
    return load_fsb_json(REAL_DATA)


def test_real_file_loads_with_expected_shape(loaded):
    bars, spec, header = loaded
    assert bars.symbol == "BNBUSDT"
    assert bars.timeframe_minutes == 60
    assert bars.n_bars == 75_791
    # 2017-11-06 03:00 UTC — matches BNB's Binance listing week
    assert int(bars.ts_utc_s[0]) == 1_509_937_200
    assert spec.tick_size == 0.1 and spec.digits == 1
    assert spec.swap_mode is None  # zero swap in export => unknown, fail closed
    assert header["company"] == "Binance"


def test_real_file_passes_hard_qa_with_expected_warnings(loaded):
    bars, spec, _ = loaded
    report = validate_or_raise(bars, spec)
    assert report.passed
    warn_names = {c.name for c in report.warnings}
    # Known data landmines must be surfaced, not silently absorbed:
    assert "spread_series_informative" in warn_names  # constant synthetic spread
    assert "price_quantization" in warn_names  # 2017-era $0.10 grid on $1.50 prices
    assert "gap_census" in warn_names
    gap = next(c for c in report.checks if c.name == "gap_census")
    assert gap.details["n_gaps"] == 68


def test_full_pipeline(tmp_path, loaded):
    bars, spec, _ = loaded
    report = validate_or_raise(bars, spec)

    # 1. immutable manifest
    manifest = build_manifest(bars, spec, report, REAL_DATA)
    write_manifest(manifest, tmp_path / "data_manifest.json")
    verify_manifest(manifest, bars, source_path=REAL_DATA)

    # 2. split plan from the committed config
    cfg_doc = json.loads(SPLIT_CONFIG_PATH.read_text())
    cfg = SplitConfig(**cfg_doc)
    plan = generate_nested_walk_forward(bars.n_bars, cfg)
    validate_plan(plan)
    assert len(plan.lockbox) == 10_800  # ~15 months of H1

    # 3. sealed experiment + repro record
    exp = Experiment.create(name="BNBUSDT-phase1-acceptance")
    exp.seal(
        data_content_hash=manifest.content_hash,
        split_plan_hash=plan.plan_hash,
        config_hash=repro.config_hash(cfg_doc),
        seed=20260710,
    )
    record = repro.capture(
        seed=20260710,
        config=cfg_doc,
        data_content_hash=manifest.content_hash,
        repo_dir=REPO_ROOT,
    )
    record.write(tmp_path / "repro.json")
    assert record.data_content_hash == manifest.content_hash

    # 4. research view physically excludes the lockbox
    view = research_view(bars, plan)
    assert view.n_bars == bars.n_bars - 10_800
    lockbox_first_ts = bars.ts_utc_s[plan.lockbox.start]
    assert view.ts_utc_s[-1] < lockbox_first_ts

    # 5. lifecycle to FROZEN
    exp.transition(ExperimentState.SEARCH_ACTIVE, actor="pipeline")
    exp.transition(ExperimentState.SHORTLISTED, actor="pipeline")
    exp.transition(ExperimentState.FROZEN, actor="pipeline")

    # 6. one-shot lockbox
    guard = LockboxGuard(exp, plan)
    lockbox_bars = guard.open(bars, actor="pipeline", reason="phase1 acceptance")
    assert lockbox_bars.n_bars == 10_800
    assert int(lockbox_bars.ts_utc_s[0]) == int(lockbox_first_ts)
    assert np.all(np.diff(lockbox_bars.ts_utc_s) > 0)
    with pytest.raises(LockboxAlreadyOpenError):
        guard.open(bars, actor="pipeline")
    assert exp.state is ExperimentState.DISCLOSED
    assert guard.audit()["n_opens"] == 1

    # 7. close and persist; reload must round-trip
    exp.transition(ExperimentState.CLOSED_REJECTED, actor="pipeline")
    path = exp.save(tmp_path / "experiment.json")
    reloaded = Experiment.load(path)
    assert reloaded.state is ExperimentState.CLOSED_REJECTED
    assert reloaded.seals["data_content_hash"] == manifest.content_hash
    disclosed_events = [e for e in reloaded.events if e["to"] == "DISCLOSED"]
    assert len(disclosed_events) == 1


def test_manifest_is_stable_for_committed_file(loaded, tmp_path):
    """Same file, two builds → identical content hash (reproducibility)."""
    bars, spec, _ = loaded
    report = validate_or_raise(bars, spec)
    m1 = build_manifest(bars, spec, report, REAL_DATA)
    bars2, spec2, _ = load_fsb_json(REAL_DATA)
    report2 = validate_or_raise(bars2, spec2)
    m2 = build_manifest(bars2, spec2, report2, REAL_DATA)
    assert m1.content_hash == m2.content_hash
    assert m1.manifest_id == m2.manifest_id
