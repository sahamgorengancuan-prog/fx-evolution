# VALIDATION STATUS

Claim-by-claim status. A claim may only be marked VERIFIED with a test (or
reproducible command) named next to it. Updated at the end of every phase.

Legend: ✅ VERIFIED (test evidence) · 🟡 PARTIAL · ⏳ PENDING · ❌ NOT BUILT

## Phase 1 — data contract & sealed experiment manager

| Claim | Status | Evidence |
|---|---|---|
| ForexSB loader parses BNBUSDT export with explicit UTC timestamps | ✅ | `tests/unit/test_loader.py`, `tests/integration/test_phase1_pipeline.py` |
| QA fails closed on monotonicity/OHLC/NaN/price/volume violations | ✅ | `tests/unit/test_quality.py` |
| QA flags constant synthetic spread + price quantization on real file | ✅ | `tests/integration/test_phase1_pipeline.py` |
| Manifest hash is content-addressed and tamper-evident | ✅ | `tests/unit/test_manifest.py` |
| Manifest is write-once | ✅ | `tests/unit/test_manifest.py` |
| Split plan: train/test disjoint, chronological, purge respected | ✅ | `tests/unit/test_splits.py` |
| Split plan: embargo windows excluded from later training sets | ✅ | `tests/unit/test_splits.py::test_embargo_excluded_from_later_folds` |
| Split plan: nothing intersects lockbox or its purge zone | ✅ | `tests/unit/test_splits.py` |
| Split plan hash stable across serialization round-trip | ✅ | `tests/unit/test_splits.py` |
| State machine rejects illegal transitions; log append-only | ✅ | `tests/unit/test_state.py` |
| Lockbox opens exactly once; second open raises | ✅ | `tests/unit/test_lockbox.py` |
| Lockbox refuses non-FROZEN states and tampered data/plan | ✅ | `tests/unit/test_lockbox.py` |
| Research view physically excludes lockbox rows | ✅ | `tests/unit/test_lockbox.py` |
| Repro record carries data/config/code hashes + seed | ✅ | `tests/unit/test_repro.py` |
| End-to-end pipeline on real BNBUSDT file | ✅ | `tests/integration/test_phase1_pipeline.py` |
| Lockbox isolation is coercion-proof | ❌ | Not claimed — see KNOWN_LIMITATIONS.md #1 |

## Later phases

| Component | Status |
|---|---|
| Typed causal DSL (Phase 2) | ❌ NOT BUILT |
| Backtest engines (Phase 3) | ❌ NOT BUILT |
| Regime router (Phase 4) | ❌ NOT BUILT |
| Search engine (Phase 5) | ❌ NOT BUILT |
| Memory + LLM agents (Phase 6) | ❌ NOT BUILT |
| Statistical validation (Phase 7) | ❌ NOT BUILT |
| MQL5 export/parity (Phase 8) | ❌ NOT BUILT |

No performance claim of any kind is validated. The v8 artifacts' metrics
are documented as invalid in `docs/FORENSIC_AUDIT.md`.
