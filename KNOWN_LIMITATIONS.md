# KNOWN LIMITATIONS

Honest inventory. Anything listed here is *by design not claimed to work*.

## Phase 1 (current)

1. **Lockbox isolation is procedural, not cryptographic/OS-level.** The
   guard makes peeking an API violation and an auditable event, but Python
   cannot stop in-process code from slicing the raw arrays. True isolation
   would need the lockbox segment stored encrypted/exfiltrated from the
   research environment; planned as an option in Phase 7 (separate file +
   key escrow). Until then the event log + sealed hashes make results
   *verifiable*, not *coercion-proof*.
2. **No backtest engine yet.** Nothing in this repo produces performance
   numbers; any strategy claim sourced from v8 artifacts remains invalid
   (see docs/FORENSIC_AUDIT.md).
3. **Funding/swap data absent.** The BNBUSDT export has no funding
   schedule; the spec deliberately reports `swap_mode=None` and engines
   must fail closed. A funding source (exchange history) must be added
   before any perp-style cost modeling.
4. **Spread history absent.** The export's spread series is a constant
   placeholder (flagged by QA). Tier-A backtests may proceed only with an
   explicitly configured spread model labeled as assumption; Tier-B/tick
   validation requires real bid/ask.
5. **H1-bar granularity.** All data in-repo is H1 OHLCV. Nothing here can
   be called tick-accurate; the funnel's tick/MQL5 stages (Phases 3/8) are
   not implemented yet.
6. **Early-era data quality.** 2017–2018 BNB bars are price-quantized
   (tick = $0.10 on a ~$1.50 asset; QA reports the affected span).
   Experiments must exclude or explicitly justify including that era.
7. **Volume units undeclared** in the source export; treated as an opaque
   relative series until a trusted definition is sourced.
8. **`min_notional` unknown** for BNBUSDT from this export; sizing logic in
   Phase 3 will fail closed until it is supplied from exchange specs.
9. **Split plans support expanding walk-forward only.** Sliding windows and
   CPCV come with Phase 7; the plan schema already accommodates them.
10. **The state machine trusts its persistence file's integrity** beyond
    the state/event-tail consistency check; signed event logs are future
    work.
11. **`repro.capture` records the git commit but cannot prove the working
    tree matched it** beyond the dirty flag.

## Phase 2 (current)

12. **Batch feature evaluation is O(n·p) with a Python loop per window.**
    It is the *reference* implementation: exact, causal, parity-tested.
    It is too slow to sit inside a million-candidate search loop; Phase 5
    must add vectorized kernels that are differentially tested against
    this reference before use, and canonical re-evaluation stays on the
    reference path.
13. **The type system is intentionally over-strict.** Some meaningful
    expressions (e.g. comparing an UNSCALED MACD line against zero) are
    rejected and must be expressed via normalizers. This is a documented
    trade-off, not an oversight.
14. **`ema` has no finite warmup**: early outputs depend on the seed value
    and are reported from bar 0. Downstream consumers that need
    converged EMAs must discard an explicit burn-in themselves.
15. **Indicator proposals are validated, never compiled.** There is no
    sandbox yet (Phase 6); the formula hygiene check is a pre-filter, not
    a security boundary.
16. **Grammar coverage is minimal** (~30 ops). Volume-based indicators
    beyond raw volume, session/TIME features, and multi-timeframe
    references are not yet expressible.

## Phase 3 (current)

17. **Tier A remains a bar approximation** no matter how many boxes it
    ticks: intrabar path, real spread dynamics, and queue effects are
    unknowable from OHLC. Its label says so; nothing downstream may drop
    that label.
18. **Tier B has never seen real ticks.** It is validated against
    synthetic tick fixtures only. Latency is a constant, rejection is a
    Bernoulli coin, partial fills and order-book depth are not modeled.
    Real-tick validation is PENDING until tick data is supplied.
19. **No margin/liquidation model.** Leverage is capped at sizing time;
    a position cannot be liquidated mid-trade. Fine at 1x on spot;
    unacceptable for leveraged perp research — must be added before any
    such experiment.
20. **Funding is all-or-nothing**: either spot (zero) or a constant
    points-per-day from spec. Real perp funding (8h schedule, variable
    rate) needs a funding-history source.
21. **Single position, single instrument.** No portfolio interaction,
    no exposure netting across pairs (Phase 9 scope).
22. **Engine throughput is Python-loop bound** (~1M bar-steps/s). Fine
    for validation; the Phase-5 search loop will need batched/vectorized
    evaluation with canonical re-checks on this reference engine.

## Phase 4 (current)

23. **Only the rule-based quantile router exists.** HMM/HSMM and
    change-point routers are interface stubs in the plan, not code. The
    quantile router's regime semantics are crude (median splits) — it
    proves the leakage discipline, not regime-detection quality.
24. **Hard labels only, no transition probabilities.** Low-confidence
    handling is limited to UNKNOWN-on-warmup; there is no calibrated
    uncertainty yet, so "flat when uncertain" currently means "flat
    during warmup".
25. **Fingerprint feature set is small** (8 aggregates) and not yet
    validated as a useful context for operator learning — that evidence
    can only come from Phase 6 outcomes.

## Phase 5 (current)

26. **Single island, single regime.** The hierarchical pair×regime island
    topology, migration, and per-regime policies are not wired yet; the
    island searches one undifferentiated policy per run.
27. **Operator probabilities are uniform** — no contextual bandit yet
    (Phase 6). Search efficiency is accordingly naive.
28. **No CMA-ES.** The local-refinement rung is jitter-based stochastic
    hill-climbing on parameters; covariance-adaptive refinement is
    pending.
29. **Evaluation is inner-validation-block based**, not full nested WFO
    model selection — that arrives with Phase 7's CPCV machinery. Outer
    OOS folds and the lockbox remain untouched by the search.
30. **Search throughput is tiny** (O(n·p) features + Python-loop engine):
    the integration suite runs pop 8 × 3 generations in ~10 s. Real
    experiments need the vectorized kernels flagged in #12/#22 first.
31. **Surrogate ranking absent** (blueprint allows it only as a compute
    scheduler anyway).

## Phase 6 (current)

32. **No agent orchestration yet.** Contracts, client, retrieval, and
    guards exist and are tested; the loop that actually calls a live LLM
    during stagnation (rung 7 of the blueprint ladder) is not wired, and
    no live API call has ever been made from this repo.
33. **Proposal sandbox still absent** — a schema-valid `IndicatorProposal`
    from an LLM still cannot enter the grammar (fail closed); whitelisted
    compilation + causality/parity testing of proposals remains open.
34. **The lockbox-content guard matches key names**, not values. A leak
    laundered through renamed keys would pass; the guard is a tripwire,
    not cryptographic isolation (see #1).
35. **Bandit context buckets are fixed constants** (2×2 fingerprint grid ×
    regime). Whether they carry signal is an empirical question Phase 9's
    experiment must answer; posteriors are replayable either way.
36. **Lesson confidence is caller-asserted.** No calibration mechanism
    yet; treat it as an ordering heuristic, not a probability.

## Phase 7 (current)

37. **Hansen's SPA is not implemented** — White's Reality Check only.
    SPA's studentization and better power matter for large batteries.
38. **Effective-trials estimator is spectrum-based**, one of several
    defensible choices; it has not been benchmarked against clustered
    alternatives on this data.
39. **CPCV generates splits; the CPCV *evaluation harness*** (running a
    candidate battery over all splits to build the PBO matrix from real
    backtests) lands in Phase 9's experiment orchestration.
40. **Stress coverage gaps**: latency perturbation and fill-rejection
    stress exist only in Tier B's parameters, not yet as first-class
    validation reports; regime-sequence stress not implemented.
41. **The DSR default variance approximation** (estimator variance of the
    strategy's own SR) understates cross-trial dispersion when the
    battery is heterogeneous; pass the battery's actual SR variance when
    available.

## Phase 8 (current)

42. **The generated MQL5 has never been compiled or executed** — no
    MetaTrader terminal exists in this environment. Structural tests
    cover determinism, grammar coverage, and metadata embedding; real
    compilation + tester runs are the manual step in RUN_INSTRUCTIONS.md
    and parity stays PENDING_REAL_TICK until then.
43. **EA order management is simpler than Tier A in two knowable ways**:
    equity for sizing is read at fill time (not prior bar close), and
    the tester's own tick path resolves SL/TP ordering. Both are
    expected, bounded divergences the parity report will surface —
    tolerances are explicit parameters, not silent fudge.
44. **The tester report itself (HTML) is not parsed** — parity uses the
    EA's own CSV dumps. Deal-level reconciliation against the tester's
    native report is a future cross-check.
45. **Custom-symbol creation and tick import are documented, not
    automated** (MT5 offers no portable CLI for it).

## Phase 9 (current)

46. **The orchestrator searches one pooled policy per fold**, not the full
    per-regime island hierarchy. Regime labels are computed and recorded,
    but a single policy is evolved against the fold's inner blocks;
    per-regime sub-policies remain future work (KNOWN #26 still applies).
47. **The offline front-ends are single-user and unauthenticated** — the
    Gradio app and the stdlib server both bind to 127.0.0.1 by design.
    They are local control panels, not multi-tenant services; do not
    expose them to a network without adding auth (Gradio's `share=` tunnel
    is deliberately off).
48. **No hosted/serverless deployment.** This build is offline-only by
    decision (see DECISIONS.md, Phase-9 revision): the earlier Cloudflare
    Worker control-plane was removed because the numpy search cannot run
    under Worker CPU limits and a hosted panel added a network surface the
    project does not want. One-click `run_all.bat`/`run_all.sh` bootstrap a
    local venv and open the Gradio app; there is no cloud path.
49. **The front-ends' live LLM "test connection" needs real network**;
    it is fail-closed without a key and exercised offline only via the
    injected-transport unit tests, not against a live provider from CI.
50. **Run scheduling is a background thread, not a job queue.** Both the
    Gradio worker thread and the stdlib server run one heavy job in-process;
    concurrent runs share the process and an in-flight run is not resumed
    across a restart (the stdlib status file survives, the running thread
    does not). Gradio streams progress over its own queue; if the browser
    tab closes mid-run the thread keeps going but its frames are dropped.
51. **The verdict thresholds (DSR≥0.5, target vector) are defaults**, not
    calibrated acceptance bars for live capital. They gate research
    shortlisting only; tick + MQL5 + lockbox stages still stand between
    a SHORTLISTED candidate and any deployment claim.
