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

## Phase 2 — typed causal feature DSL

| Claim | Status | Evidence |
|---|---|---|
| v8 champion long entry (`cross_above(rank(...), ret(8))`) rejected at compile | ✅ | `tests/unit/test_features_compiler.py::TestV8ChampionRejection`, `tests/integration/test_phase2_real_data.py` |
| Cross-dimension comparisons rejected; UNSCALED not comparable | ✅ | `tests/unit/test_features_compiler.py::TestDimensionRules` |
| Blueprint example `zscore(sub(ema,ema)) > c` compiles to BOOL | ✅ | `tests/unit/test_features_compiler.py` |
| `abs(abs(x)) → abs(x)`; v8 HOF ranks 1 vs 2 detected as semantic duplicates | ✅ | `tests/unit/test_features_simplifier.py` |
| Commutative/direction canonicalization → equal semantic hashes | ✅ | `tests/unit/test_features_simplifier.py` |
| Prefix invariance for EVERY registered operator | ✅ | `tests/unit/test_features_causality.py::test_prefix_invariance` (30 ops) |
| Future perturbation cannot change past outputs (every op) | ✅ | `tests/unit/test_features_causality.py::test_future_perturbation_cannot_change_the_past` |
| Registry↔probe completeness (unprobed primitive fails CI) | ✅ | `tests/unit/test_features_causality.py::test_every_registered_op_has_a_causality_probe` |
| Incremental == batch to 1e-12 on nested expressions | ✅ | `tests/unit/test_features_incremental.py` |
| Lookback/complexity accounting deterministic | ✅ | `tests/unit/test_features_compiler.py::TestAccounting` |
| Structural validation (unknown op, arity, period bounds, stray fields) | ✅ | `tests/unit/test_features_compiler.py::TestStructuralValidation` |
| Proposal schema fails closed (non-causal, bad types, forbidden formula) | ✅ | `tests/unit/test_features_proposals.py` |
| Determinism + causality + parity on real BNBUSDT bars | ✅ | `tests/integration/test_phase2_real_data.py` |
| Proposal sandbox compilation | ❌ | Phase 6 — see KNOWN_LIMITATIONS.md #15 |
| Search-grade evaluation performance | ❌ | Not claimed — see KNOWN_LIMITATIONS.md #12 |

## Phase 3 — two-tier backtest engines

| Claim | Status | Evidence |
|---|---|---|
| Costs fail closed: no spread source / commission / funding decision / min-notional | ✅ | `tests/unit/test_backtest_costs.py`, `tests/integration/test_phase3_real_data.py::test_costs_fail_closed_on_real_spec` |
| Assumed constant spread is flagged and labeled in every result | ✅ | `tests/unit/test_backtest_costs.py`, integration provenance asserts |
| Sizing: lot grid, min lot, min notional, leverage cap (hand-computed) | ✅ | `tests/unit/test_backtest_costs.py` |
| Exact PnL/costs on hand-computed fixture trade | ✅ | `tests/unit/test_fast_engine.py::TestHandComputedTrade` |
| Mark-to-market equity shows open-position excursion (v8 D9 fix) | ✅ | `tests/unit/test_fast_engine.py::TestMtmExcursion` |
| Gap-through-stop fills at first executable price, never the stop | ✅ | `tests/unit/test_fast_engine.py::TestGapThroughStop`, `tests/unit/test_event_engine.py::TestGapThroughStopOnTicks` |
| SL+TP same bar ⇒ SL taken; Tier A never more optimistic than tick path | ✅ | `tests/unit/test_fast_engine.py::TestWorstCaseOrdering`, `tests/unit/test_event_engine.py::TestConservativeInequality` |
| Signals at bar close execute next bar open (causal) | ✅ | `TestHandComputedTrade` entry-index assert |
| Funding accrual per bar held (cost-positive convention) | ✅ | `tests/unit/test_fast_engine.py::TestFundingAndDeterminism` |
| Deterministic trade hash; Tier B seeded rejection reproducible | ✅ | `TestFundingAndDeterminism`, `tests/unit/test_event_engine.py::TestLatencyAndRejection` |
| Zero-latency Tier B matches Tier A on unambiguous fixtures | ✅ | `tests/unit/test_event_engine.py::TestZeroLatencyParityWithTierA` |
| Latency shifts fills to later ticks | ✅ | `tests/unit/test_event_engine.py::TestLatencyAndRejection` |
| Real-data Tier A run: deterministic, labeled, contract mechanics hold | ✅ | `tests/integration/test_phase3_real_data.py` |
| Tier B against real tick history | ⏳ PENDING | no tick data attached — see KNOWN_LIMITATIONS.md #18 |
| Margin/liquidation model | ❌ | Not built — KNOWN_LIMITATIONS.md #19 |
| Strategy profitability | ❌ | Not claimed, by design |

## Phase 4 — pair fingerprints & regime router

| Claim | Status | Evidence |
|---|---|---|
| Router params are a pure function of the train slice (D7 leakage probe) | ✅ | `tests/unit/test_regimes.py::TestFitDiscipline::test_params_depend_only_on_train_slice` |
| Refit on identical train ⇒ bit-identical params | ✅ | `TestFitDiscipline::test_refit_is_bit_identical` |
| Labels one-sided: prefix invariance + future perturbation | ✅ | `TestOneSidedInference`, real-data variant in `tests/integration/test_phase4_real_data.py` |
| Warmup labeled UNKNOWN; unfitted router refuses inference | ✅ | `TestOneSidedInference::test_warmup_is_unknown`, `TestFitDiscipline` |
| Causal min-dwell smoothing (incl. property test on random sequences) | ✅ | `tests/unit/test_regimes.py::TestDwellSmoothing` |
| Fingerprint deterministic, hashable, slice-local | ✅ | `tests/unit/test_regimes.py::TestFingerprint` |
| Real-data: fold-train fit, one-sided inference over research view, all regimes reachable | ✅ | `tests/integration/test_phase4_real_data.py` |
| HMM / change-point routers | ❌ | Not built — KNOWN_LIMITATIONS.md #23 |
| Calibrated regime probabilities | ❌ | Not built — KNOWN_LIMITATIONS.md #24 |

## Phase 5 — search engine core

| Claim | Status | Evidence |
|---|---|---|
| Generated/mutated genomes always compile (property tests, all operators) | ✅ | `tests/unit/test_genome.py` |
| Semantic duplicates (v8 `abs(abs)` HOF pair) share one genome identity | ✅ | `tests/unit/test_genome.py::test_semantic_duplicates_share_genome_hash` |
| Feasible always outranks infeasible; infeasible ranked by violation | ✅ | `tests/unit/test_nsga.py` |
| Non-dominated sort + crowding on planted fronts | ✅ | `tests/unit/test_nsga.py` |
| PF capped so tiny-sample luck cannot dominate | ✅ | `tests/unit/test_nsga.py::test_pf_cap_prevents_tiny_sample_domination` |
| MAP-Elites cells replaced only under constraint dominance; coverage grows | ✅ | `tests/unit/test_map_elites.py` |
| 2D hypervolume hand-computed; ladder fires in order to NO_EDGE_FOUND | ✅ | `tests/unit/test_stagnation.py` |
| Infeasible diversity churn is not progress (D3/D4 regression) | ✅ | `tests/unit/test_stagnation.py::test_infeasible_diversity_churn_is_not_progress` |
| Every intervention logged with signals; improvement resets rung | ✅ | `tests/unit/test_stagnation.py` |
| Mini search on real BNBUSDT is seed-deterministic (champion, lineage, telemetry) | ✅ | `tests/integration/test_phase5_search.py::test_search_is_seed_deterministic` |
| Every candidate gets a lineage record (hash, parents, operator, objectives) | ✅ | `tests/integration/test_phase5_search.py` |
| Impossible targets ⇒ NO_EDGE_FOUND; a pass is never forced | ✅ | `tests/integration/test_phase5_search.py::test_impossible_targets_terminate_no_edge_found` |
| Contextual operator bandit | ❌ | Phase 6 |
| Pair×regime island hierarchy + migration | ❌ | KNOWN_LIMITATIONS.md #26 |
| CMA-ES local refinement | ❌ | KNOWN_LIMITATIONS.md #28 |
| Search-scale performance | ❌ | KNOWN_LIMITATIONS.md #30 |

## Phase 6 — evolutionary memory & LLM agent contracts

| Claim | Status | Evidence |
|---|---|---|
| Event store is append-only (no mutator API), persists across reopen | ✅ | `tests/unit/test_memory_store.py` |
| Schema-version mismatch fails closed | ✅ | `tests/unit/test_memory_store.py::TestStore::test_schema_version_mismatch_fails_closed` |
| Replay reconstructs per-(context, operator) outcomes exactly | ✅ | `test_memory_store.py`, `test_island_memory_integration.py::test_bandit_replay_reconstructs_selector` |
| Failure labels deterministic, blueprint taxonomy subset | ✅ | `tests/unit/test_memory_store.py::TestFailureLabels` |
| Bandit seed-deterministic; learns planted better operator; contexts independent | ✅ | `tests/unit/test_bandit.py` |
| Lessons: supersede retires, replay reproduces, retrieval ranked+bounded | ✅ | `tests/unit/test_lessons.py` |
| Retrieval context bounded and lockbox-guard clean | ✅ | `tests/unit/test_llm_contracts.py::TestRetrievalIntegration` |
| Lockbox-scoped keys at any depth raise before reaching an LLM | ✅ | `tests/unit/test_llm_contracts.py::TestLockboxGuard` |
| LLM fitness-authority attempts rejected (no fallback) | ✅ | `tests/unit/test_llm_contracts.py::TestResponseParsing` |
| Client fails closed on missing env; redacted keys only; fake-transport round-trip | ✅ | `tests/unit/test_llm_client.py` |
| Island+bandit+store integration deterministic; escalation pool outranks bandit | ✅ | `tests/unit/test_island_memory_integration.py` |
| Live LLM call (OpenAI/OpenRouter) | ⏳ PENDING | never executed from this repo; exercised via dashboard in Phase 9 |
| Proposal sandbox compilation | ❌ | KNOWN_LIMITATIONS.md #33 |

## Phase 7 — statistical validation framework

| Claim | Status | Evidence |
|---|---|---|
| PBO rejects single candidates (v8 S1 regression, API-level) | ✅ | `tests/unit/test_validation_pbo.py::test_single_candidate_is_rejected` |
| PBO: planted overfit → >0.8; true edge → <0.2; noise ≈ 0.5 | ✅ | `tests/unit/test_validation_pbo.py` |
| CPCV: C(S,k) combinations, purge+embargo invariants, deterministic | ✅ | `tests/unit/test_validation_bootstrap_cpcv.py::TestCPCV` |
| Effective trials: identical→1, independent→N, correlated in between | ✅ | `tests/unit/test_validation_sharpe.py::TestEffectiveTrials` |
| PSR: 0.5 at own SR, →1 for strong edge, low for noise vs high benchmark | ✅ | `tests/unit/test_validation_sharpe.py::TestPSR` |
| DSR decreases with trials; effective-N deflates less than raw N (v8 S3 fix) | ✅ | `tests/unit/test_validation_sharpe.py::TestDSR` |
| Block bootstrap seeded/deterministic; Sharpe CI brackets truth | ✅ | `tests/unit/test_validation_bootstrap_cpcv.py::TestBootstrap` |
| Drawdown distribution as a path statistic (legitimate resampling use) | ✅ | `TestBootstrap::test_drawdown_distribution_is_path_statistic` |
| Reality Check: noise → high median p (5 seeds), edge → p<0.05 | ✅ | `TestRealityCheck` |
| Cost stress erodes returns monotonically on pinned-trade fixture | ✅ | `tests/unit/test_validation_stress.py` |
| Parameter/start-offset stress seeded, JSON-safe reports | ✅ | `tests/unit/test_validation_stress.py` |
| Return concentration hand-computed (top-k shares, HHI) | ✅ | `tests/unit/test_validation_bootstrap_cpcv.py::TestConcentration` |
| Terminal-return permutation API absent by construction (v8 S2) | ✅ | `TestForbiddenApis::test_terminal_return_permutation_api_does_not_exist` |
| Hansen SPA | ❌ | KNOWN_LIMITATIONS.md #37 |
| CPCV evaluation harness over real candidate batteries | ⏳ | Phase 9 orchestration |

## Phase 8 — MQL5 export & differential parity

| Claim | Status | Evidence |
|---|---|---|
| Export deterministic (same genome ⇒ byte-identical source) | ✅ | `tests/unit/test_mql5_exporter.py::test_export_is_deterministic` |
| 20 random genomes export; unknown ops raise | ✅ | `test_mql5_exporter.py` |
| Full grammar coverage vs registry | ✅ | `test_mql5_exporter.py::test_grammar_coverage_matches_generator` |
| Genome/risk/contract metadata embedded; MIN_BARS covers lookback | ✅ | `test_mql5_exporter.py` |
| Shared subtrees compile once (semantic-hash naming) | ✅ | `test_mql5_exporter.py::test_subexpressions_are_shared` |
| Tester config pins Model=4 (Every tick based on real ticks) | ✅ | `test_mql5_exporter.py::TestTesterConfig` |
| CSV contracts round-trip | ✅ | `tests/unit/test_mql5_parity.py::TestCsvRoundTrip` |
| Identical EA output ⇒ PASS, zero discrepancies | ✅ | `TestParityComparator` |
| Flipped signal / price / lots / exit-reason / count divergences itemized | ✅ | `TestParityComparator` |
| Missing tester artifacts ⇒ PENDING_REAL_TICK, never PASS | ✅ | `TestParityComparator`, `TestBundle` |
| Bundle carries EA, tester.ini, instructions, expectation dumps, status | ✅ | `TestBundle` |
| `mql5-parity` CLI: PASS→exit 0, FAIL→exit 1, report persisted | ✅ | `TestParityCli` |
| EA compiled and run in a real Strategy Tester | ⏳ PENDING_REAL_TICK | no terminal in this environment — RUN_INSTRUCTIONS.md |
| Python↔EA live parity on real ticks | ⏳ PENDING_REAL_TICK | blocked on the above |

## Later phases

| Component | Status |
|---|---|

No performance claim of any kind is validated. The v8 artifacts' metrics
are documented as invalid in `docs/FORENSIC_AUDIT.md`.
