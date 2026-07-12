# DECISIONS

Chronological log of engineering decisions. Architecture-level decisions
have full ADRs in `docs/adr/`.

## 2026-07-10 — Phase 1

1. **Repository name/package.** The blueprint suggests `evolution-searcher`;
   the existing GitHub repo is `fx-evolution`. Kept the repo name, used
   `evoquant` as the Python package (blueprint's own package name),
   `src/` layout.
2. **Original notebook not attached.** The audit distinguishes
   `[VERIFIED]` (provable from result JSON / data file) from `[REPORTED]`
   (blueprint's own audit of the notebook). All `[REPORTED]` defects get
   regression tests in the phase that rebuilds the component.
3. **ForexSB time base.** `time` is minutes since 2000-01-01T00:00Z —
   verified: first bar decodes to 2017-11-06 03:00 UTC, matching BNB's
   Binance listing week; deltas are 60 minutes for H1. Loader converts to
   UTC epoch seconds; no naive datetimes anywhere.
4. **FSB `swapLong/Short == 0` treated as *unknown*, not zero.** A crypto
   perp has funding; a spot pair has none — the export can't tell us
   which. Fail-closed: engines requiring funding must call
   `spec.require("swap_mode")` and abort until a real funding source is
   configured.
5. **Static header spread recorded but distrusted.** `spread_source =
   "static_header"` is not in `TRUSTED_SPREAD_SOURCES`; the QA check
   `spread_series_informative` additionally flags the constant series.
   Research-grade cost modeling (Phase 3) requires a trusted source.
6. **Dependencies minimal by design.** Phase 1 uses stdlib + numpy only.
   Pydantic deferred; frozen dataclasses with `__post_init__` validation
   are sufficient and keep the reproducibility surface small. Revisit at
   Phase 2 when schemas get external (LLM) input.
7. **Split layout = expanding walk-forward.** Outer test blocks tile the
   tail of the research region; train = everything before test minus
   purge, minus embargo windows after *earlier* test blocks. Inner
   validation blocks tile the tail of each outer train. Purge also
   separates the research region from the lockbox. Sliding-window and
   CPCV layouts arrive in Phase 7 on the same `SplitPlan` schema.
8. **Purge default 96 bars (H1).** v8 champion held positions up to 48
   bars (`max_bars=48` in hall of fame); 96 = 2x safety margin. Config,
   not constant; must be >= max holding horizon + feature overlap and will
   be re-derived per experiment when the Phase 3 engine knows actual
   horizons.
9. **Lockbox = last 10,800 H1 bars (~15 months)** for the BNBUSDT config —
   inside the blueprint's 12–18 month recommendation.
10. **`DISCLOSED` reachable only through `LockboxGuard`.** The public
    `Experiment.transition()` refuses it; the guard verifies sealed data
    and plan hashes before releasing lockbox rows and the transition is
    permanent. Python cannot enforce memory isolation — documented in
    KNOWN_LIMITATIONS.md.
11. **Committed the 2.9 MB BNBUSDT export under `data/raw/`** so the
    integration suite and the future one-pair demo are reproducible from a
    bare clone. Larger datasets (M1/ticks) will use manifests + external
    storage instead.
12. **Seed policy.** Every experiment seals one integer seed; all future
    stochastic components must derive their streams from it (never
    `random.seed()` global state).

## 2026-07-10 — Phase 2

13. **`UNSCALED` dimension added** beyond the blueprint's nine types. A
    difference of two like-dimensioned series (`sub`, `delta`) loses level
    meaning; making it non-comparable forces normalization (`zscore`/
    `rank`) before any comparison. This is exactly the constraint whose
    absence produced the v8 champion's `rank vs ret` rule.
14. **No `macd`/`keltner` primitives.** They are compositions
    (`sub(ema,ema)`; `abs(sub(close, ema))`) and live in the grammar as
    such — smaller primitive set, same expressive power, honest types.
15. **Deliberately conservative type rules.** `abs`/`neg` only on signed
    dimensions (RETURN/ZSCORE/UNSCALED); `add`/`sub` require matching
    dimensions and always yield UNSCALED. False negatives (rejecting a
    borderline-meaningful rule) are acceptable; false positives were the
    v8 disease.
16. **Indicator variants chosen for window-computability**: Cutler's RSI
    (SMA gains/losses) instead of Wilder's recursive RSI; ATR as simple
    mean of true range; Choppiness normalized to [0,1]. Each is exactly
    reproducible incrementally, which the parity tests require. Wilder
    variants can be added later as distinct ops.
17. **`ema` seeds on the first finite input**; NaN inputs produce NaN
    output without disturbing recursion state. Lookback metadata for ema
    is 0 (recursive); prefix-invariance is what guarantees causality, not
    warmup counting.
18. **Batch engine is a correctness-first O(n·p) reference.** Batch and
    stream share one window-formula implementation, so parity holds to
    1e-12 (bitwise in most cases). Vectorized/GPU kernels come later and
    must re-verify against this reference (ADR-004: fast kernels never
    finalize rankings).
19. **Comparisons involving NaN are False** (no signal), never True —
    warmup periods cannot fire entries.
20. **Proposal schema validates but cannot execute.** Phase 2 ships
    `IndicatorProposal` validation (fail closed, no random fallback);
    whitelisted compilation and sandboxing are Phase 6. Until then no LLM
    proposal can enter the grammar at all.

## 2026-07-10 — Phase 3

21. **Engines execute precomputed signal arrays, not strategy callbacks.**
    `SignalPolicy` carries per-bar causal boolean arrays produced by the
    Phase-2 compiler (whose causality is test-enforced) plus a risk block.
    No user code runs inside the engine loop — determinism and Tier-A/B
    identical semantics come for free.
22. **Timing convention:** signal on bar `i` (close) executes at bar `i+1`
    open (Tier A) / first tick after bar-`i` completion + latency
    (Tier B). Same-bar execution is impossible by construction.
23. **Source prices treated as mid**; spread applied symmetrically (±half
    spread). The spread series itself must come from `resolve_costs` —
    trusted history or an explicitly labeled assumption; the label travels
    into every result artifact.
24. **All execution costs are experimenter-stated**: commission as
    fraction of notional, funding mode (`not_applicable_spot` or
    points-per-day from spec, cost-positive convention), min-notional,
    slippage, leverage cap. Missing any ⇒ `MissingMetadataError`. The FSB
    header's ambiguous `commissionType=5` is deliberately NOT interpreted.
25. **Conservative intrabar rule:** SL dominates TP when both are touched
    in one Tier-A bar; gaps fill at the first executable price (bar open /
    first tick beyond the level). Differential test asserts Tier A is
    never more optimistic than the tick path.
26. **Conflicting simultaneous long+short signals ⇒ no trade** (logged);
    one position at a time; cooldown measured in bars from exit.
27. **Tier B is tested with synthetic ticks only** (unit fixtures, O→H→L→C
    path). No real tick data is attached; real-tick validation is PENDING
    and the engine's README/docstring says so.
28. **Margin/liquidation and partial fills are not modeled** in either
    engine yet (leverage capped at sizing time instead); recorded in
    KNOWN_LIMITATIONS rather than approximated silently.

## 2026-07-10 — Phase 4

29. **Baseline router is rule-based quantile thresholds**, not an HMM.
    Two causal features (trend efficiency, ATR%) with train-median
    thresholds give {trend,range}×{high,low vol} + UNKNOWN. Rationale:
    the leakage discipline (train-only fit, one-sided inference) is the
    hard part and must be provable first; HMM/change-point routers plug
    into the same interface later and must pass the same test battery.
30. **Min-dwell smoothing is a causal challenger/commit automaton**: a
    switch commits only after `min_dwell` consecutive raw bars of the
    same new regime; UNKNOWN feature gaps keep the committed regime
    instead of fabricating switches. The smoother is unit-tested
    directly, including a property test over random sequences.
31. **Fingerprints are aggregates over exactly the slice passed in** —
    the caller (experiment orchestration) is responsible for passing
    train partitions; tests prove slice-locality by perturbing data
    outside the slice.
32. **Regime probabilities deferred.** The baseline router emits hard
    labels + UNKNOWN; calibrated transition uncertainty arrives with the
    probabilistic routers (HMM) and will extend, not replace, the label
    contract.

## 2026-07-10 — Phase 5

33. **Type-directed generation over generate-and-filter.** Random trees
    are grown per dimension so every genome compiles by construction;
    a validate() safety net remains, and a failed mutation returns the
    parent flagged `applied=False` — never a silent random substitute.
34. **Genome identity = semantic hash** of canonicalized ASTs + rounded
    risk block. `abs(abs(x))` and `abs(x)` variants (v8 HOF ranks 1 vs 2)
    collapse to one identity; evaluation caching and lineage use it.
35. **PF objective capped at 10** so a lucky 3-trade candidate cannot
    Pareto-dominate on an unstable ratio.
36. **Stagnation progress semantics:** while a population is entirely
    infeasible, only movement toward feasibility (feasibility rate up or
    min violation down by a ≥0.1% relative step) counts as progress;
    hypervolume/coverage growth among all-infeasible candidates is churn.
    Discovered via a failing integration test where an impossible-target
    island evaded NO_EDGE_FOUND by shuffling infeasible variety.
37. **Budget backstop:** exhausting the compute budget without a single
    feasible candidate ever appearing yields `NO_EDGE_FOUND`, not a
    neutral "ran out of time" — the budget is the evidence threshold.
38. **Operator selection is uniform in Phase 5.** The contextual
    Thompson-sampling bandit is Phase 6; every lineage record already
    carries the operator name so posteriors can be trained on replay.
39. **LOCAL_REFINEMENT rung = jitter-only generation** (parameters of
    stable structures). Full CMA-ES covariance adaptation is deferred and
    listed in KNOWN_LIMITATIONS.

## 2026-07-10 — Phase 6

40. **SQLite over DuckDB** for the event store: stdlib, zero new
    dependencies, append-only by API design (no update/delete methods —
    asserted by a test), versioned via a schema_version row that fails
    closed on mismatch.
41. **Bandit reward is categorical, not scalar**: feasibility flip,
    relative violation reduction, Pareto win over the parent, or a
    MAP-Elites cell win. Crossover/random/elite events carry
    `reward=None` and are excluded from operator outcomes.
42. **Escalation ladder outranks the bandit**: a forced operator pool
    from a stagnation rung bypasses learned preferences for that
    generation (tested).
43. **Lockbox guard for LLM payloads is key-based**
    (`lockbox|holdout|sealed_test|final_test` substrings at any depth).
    It is a tripwire against accidental leaks by compliant code, not a
    semantic censor — consistent with KNOWN_LIMITATIONS #1.
44. **LLM response schema has no fitness channel** and any attempt to
    smuggle one (`fitness`/`score`/`rank`/`gate_override`/`feasible`
    keys) rejects the entire response; unknown mutation operators
    likewise. No repair, no random fallback.
45. **LLM client keys live in env vars only** (`OPENAI_API_KEY` /
    `OPENROUTER_API_KEY`), are never persisted or logged, and only the
    `redacted()` form may enter artifacts. Transport is injectable; unit
    tests never touch the network.
46. **Cloudflare requirement recorded for Phase 9** (user, 2026-07-10):
    Workers dashboard = control plane (config/KV, key testing, run
    monitoring, results download); Python engine = compute plane. A
    Worker cannot and must not run the search or compute fitness; it
    also never stores lockbox data. Details in docs/PHASE_PLAN.md §9.

## 2026-07-10 — Phase 7

47. **PBO rejects single-candidate input by construction** — the v8
    misnomer (a per-strategy "pbo") is now an impossible API call, with
    an error message citing the audit finding.
48. **Effective trials via the participation ratio** of the candidate
    correlation spectrum: (Σλ)²/Σλ². Identical candidates → 1,
    independent → N; simple, deterministic, testable. Cluster-based
    estimators can replace it later behind the same function.
49. **DSR's across-trial SR variance defaults to the estimator variance**
    of the strategy's own SR when the battery's SR variance isn't
    supplied — a documented approximation, not a hidden one.
50. **Statistical tests on noise are themselves noisy**: the Reality
    Check unit test asserts the MEDIAN p-value over five seeded noise
    batteries (a single 5% test rejects noise 5% of the time — asserting
    one draw would be a flake by design).
51. **Stress harnesses re-run the real Tier-A engine** under scaled
    costs / jittered risk / shifted windows — they perturb economics and
    paths, unlike v8's permutation which changed neither. The cost-stress
    fixture pins its trade list (max_bars-only exits) so monotone
    erosion is provable.
52. **No terminal-return permutation API exists** anywhere in
    `evoquant.validation`; a package-scanning test keeps it that way.
53. **White's Reality Check over SPA** for Phase 7: simpler, standard,
    sufficient for the one-pair experiment; Hansen's SPA is a listed
    upgrade, not a silent absence.

## 2026-07-10 — Phase 8

54. **Expression transpilation by semantic hash**: every unique canonical
    subtree becomes exactly one MQL5 function named
    `Expr_<hash12>` — identical subtrees share code and the same genome
    always produces byte-identical `.mq5` source (tested).
55. **The EA re-implements the shared window formulas** (WinMean/WinRank/
    WinZScore/… mirror `features.primitives`) rather than calling MT5
    built-in indicators, so semantics match the Python reference by
    construction instead of by hope. EMA seeds `10 × period` bars back —
    a documented convergence approximation; parity comparison skips the
    declared warmup region.
56. **Model=4 ("Every tick based on real ticks") is pinned in the tester
    config generator** — no other model may be labeled tick validation.
57. **Parity is a three-state machine**: PENDING_REAL_TICK (no tester
    CSVs yet — the honest default), PASS (zero discrepancies), FAIL
    (itemized). Missing artifacts can never yield PASS; the comparator
    itself is proven against simulated EA output with injected
    divergences.
58. **The `mql5-parity` CLI compares CSV-to-CSV** (bundle expectation
    dumps vs tester dumps) so the manual loop needs no Python object
    reconstruction, and it persists `parity_report.json` + flips
    `parity_status.json` in the bundle.

## 2026-07-11 — Phase 9

59. **The orchestrator never names a symbol.** `run_experiment` takes any
    ForexSB JSON + a `RunConfig`; BNBUSDT is only the committed sample. A
    synthetic second-symbol file runs identically (tested). Pair-agnostic
    by construction was the user's explicit requirement.
60. **Outer-OOS is the model-selection surface, not the inner folds.** The
    champion is chosen on inner-validation blocks, then evaluated once on
    the untouched outer test fold; the verdict keys off that OOS result +
    the deflated-Sharpe battery. The lockbox is never opened by a run —
    freezing and the one-shot open stay deliberate human actions.
61. **Three honest verdicts, never a forced pass**: SHORTLISTED (OOS
    feasible and DSR≥0.5), REJECTED (OOS feasible but DSR<0.5), or
    NO_EDGE_FOUND. The offline demo run landed on NO_EDGE_FOUND — the
    correct outcome for a real 6k-bar BNBUSDT slice under honest costs.
62. **Control plane vs compute plane** (user's Cloudflare requirement):
    the numpy search cannot run on a Worker (CPU-time limits), so the
    Worker + offline server are control-only (config, KV, key testing,
    run monitoring, results/download). Neither computes fitness nor
    stores lockbox data. The Worker embeds the *same* dashboard HTML the
    offline server serves — one UI, two hosts.
63. **The dashboard is strictly offline-capable**: no CDN, no external
    fonts/scripts/images, every fetch same-origin, charts hand-drawn on
    canvas. A regex test forbids external resource refs; a real headless
    Chromium run asserts zero external network requests and zero console
    errors end-to-end.
64. **Secrets never leave the server.** API keys are stored 0600 on the
    local server / as `wrangler secret` on the Worker; only presence
    flags (`secrets_present`) are returned to the UI. Verified by a test
    asserting the saved key never appears in any API response.
65. **A real offline browser run is part of acceptance.** Playwright
    drives the dashboard against the committed data: dynamic pair
    discovery → config → start → live telemetry chart render (verified by
    counting non-transparent canvas pixels) → terminal verdict → download
    whose report matches. This is the "coba satu kali run offline dan
    sempurnakan" the user asked for.

## 2026-07-11 — Phase 9 revision (offline-only, Gradio one-click)

66. **User pivot: offline-only, drop Cloudflare.** Removed the Cloudflare
    Worker generator, its CLI command, and its test. The system now has
    zero external-service surface. (Git history retains the removed code
    if ever wanted.)
67. **Gradio is the primary UI**, launched by a one-click `run_all.bat`
    (Windows) / `run_all.sh` (Unix twin) that creates a venv, installs
    `.[gui]`, and opens the browser via `evoquant gui` (Gradio
    `inbrowser=True`). Gradio is an **optional** dependency — the engine
    and the stdlib dashboard never import it; `gradio_app` raises a clear
    install hint if it's missing.
68. **Kept the zero-dependency stdlib dashboard** (`evoquant serve`) as an
    offline fallback for environments without Gradio. Two offline
    front-ends, one orchestrator; neither computes fitness or opens the
    lockbox.
69. **The Gradio run handler streams via a queue+thread**: the
    orchestrator's `on_progress` pushes events onto a `queue.Queue`
    drained by a generator that yields live status + a growing telemetry
    table, then a terminal frame with verdict, outer-OOS metrics, full
    report, and a downloadable `report.json`.
70. **Friendly config-fit errors.** The offline browser drive on a
    deliberately-small data slice surfaced a raw `SplitConfigError` when
    the split didn't fit; the handler now catches it and tells the user
    exactly which knobs to change (max bars / lockbox / fold sizes). Found
    by driving the real UI and fixed — the "sempurnakan" step.
71. **Verified end-to-end in a real headless Gradio session**: pairs
    discovered → Run clicked → 4 generations streamed (incl. a
    LOCAL_REFINEMENT intervention) → verdict `NO_EDGE_FOUND` with honest
    reasons → 12.8 KB report downloadable. Zero external network requests.
72. **mypy stays strict everywhere except the thin `gradio_app` glue**
    (Gradio ships no stubs); one narrow per-module override, documented in
    pyproject.
