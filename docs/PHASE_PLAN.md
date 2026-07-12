# Phased Dependency Plan

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 7 ──► Phase 9
 (data,      (typed      (backtest    (regimes)   (search)     (stats)     (full
  splits,     DSL)        engines)        │           │            │        exp.)
  lockbox)      │            │            └───────────┤            │
                └────────────┴─────► Phase 6 (memory + LLM) ◄──────┘
                                          Phase 8 (MQL5 parity) ◄── Phase 3, 5
```

Rule: a phase may begin only when the phases it depends on have **green
acceptance tests**, recorded in `VALIDATION_STATUS.md`.

---

## Phase 1 — Data contract & sealed experiment manager  *(this commit)*

**Objective.** Immutable, hash-addressed data with fail-closed quality
gates; nested walk-forward split generation with purge/embargo; an
experiment lifecycle state machine whose final lockbox can be opened
exactly once; reproducibility metadata on every artifact.

**Acceptance criteria / tests.**
1. Loader parses the ForexSB BNBUSDT file; timestamps are UTC-explicit,
   strictly increasing; header metadata preserved verbatim.
2. QA fails closed on: non-monotonic time, OHLC invariant violations,
   NaN/inf, non-positive prices, duplicate timestamps; reports (without
   failing) gaps, constant-spread suspicion, quantization regimes.
3. Manifest: same bytes ⇒ same hash; any tampering (one price changed) ⇒
   verification failure; manifest write is write-once.
4. Split plan: for every (outer, inner) fold — train ∩ test = ∅;
   max(train) + purge < min(test); embargo respected; nothing intersects
   the lockbox; folds are chronological; plan serializes to a stable hash.
5. Experiment state machine rejects illegal transitions; every transition
   appended to an event log with timestamps and actor.
6. Lockbox: inaccessible before FROZEN; opens exactly once; second open
   raises; open permanently flips state to DISCLOSED; research data view
   physically excludes lockbox rows.
7. Reproducibility record includes data hash, config hash, code version,
   seed, Python/dependency versions.
8. Integration: full pipeline on the real BNBUSDT file end-to-end.

## Phase 2 — Typed causal feature DSL

**Objective.** Dimension-aware AST (PRICE, RETURN, VOLATILITY, VOLUME,
OSCILLATOR_0_1, OSCILLATOR_0_100, ZSCORE, TIME, BOOL), compiler,
canonicalizer/simplifier, incremental evaluation.
**Acceptance.** Type-invalid trees (e.g. `cross(rank, ret)` — the v8
champion) rejected at compile; `abs(abs(x))` canonicalizes to `abs(x)`;
prefix-invariance: appending future bars never changes past outputs;
incremental == batch to 1e-12; complexity accounting deterministic.

## Phase 3 — Realistic backtest engines

**Objective.** Tier-A bar engine (`BAR_APPROXIMATION` label, mark-to-market,
bid/ask + dynamic spread, fees/funding, gap-through-stop, lot/tick/notional
rounding, deterministic) and Tier-B event/tick engine; differential tests
between them.
**Acceptance.** MTM equity includes open-position excursion; gap fills at
first executable price; no impossible fills under fixture attack cases;
engine refuses to run when required cost metadata is absent (fail closed);
same seed ⇒ identical trade list; Tier-A vs Tier-B agreement bounds
documented per fixture.

## Phase 4 — Pair-regime model

**Objective.** Causal fingerprints and interchangeable regime routers, fit
train-only per fold, one-sided inference, UNKNOWN/FLAT low-confidence state.
**Acceptance.** Prefix-invariance of regime labels; refit on train-only
reproduces bit-identical labels for same fold; no test-period statistic
influences router parameters (leakage probe test).

## Phase 5 — Search engine

**Objective.** Pair/regime islands; constraint-dominance NSGA-II;
MAP-Elites behavioral archive; CMA-ES local refinement gated on structural
stability; contextual Thompson-sampling operator selection; lineage archive;
stagnation detector (hypervolume, feasibility rate, novelty entropy,
lineage concentration, failure clusters) + logged escalation ladder ending
in `NO_EDGE_FOUND`.
**Acceptance.** Feasible dominates infeasible in selection unit tests;
archive coverage grows on synthetic multi-modal fixtures; operator
posteriors update from outcomes; every escalation step emits a structured
log record; `NO_EDGE_FOUND` reachable and terminal.

## Phase 6 — Evolutionary memory & LLM agents

**Objective.** Event-sourced empirical store (every candidate, lineage,
operator, fold metrics, failure labels, hashes); semantic lesson store;
retrieval; hypothesis/diagnosis agents behind strict JSON schemas; sandbox
compilation of proposals; **no fitness authority, no lockbox visibility**.
**Acceptance.** Store is append-only and replayable; schema-invalid LLM
output rejected without fallback-to-random; a probe proving lockbox
metrics never enter any LLM prompt context.

## Phase 7 — Validation framework

**Objective.** Correctly named statistics: CPCV; PBO from the
candidate×split performance matrix; probabilistic & deflated Sharpe with
*effective* trial count; block bootstrap; SPA/White reality check;
parameter & start-date perturbation; cost/latency stress;
return-concentration. Permutation used for path/DD only.
**Acceptance.** PBO unit test on a planted-overfit matrix recovers high
PBO and on an honest matrix low PBO; DSR effective-N < raw N on correlated
trials fixture; terminal-return permutation test is *absent by
construction* (a test asserts the API doesn't exist).

## Phase 8 — MQL5 export & parity

**Objective.** EA code generation, custom-symbol/tick import artifacts,
Strategy Tester run configs (`Every tick based on real ticks`), report
parser, differential parity at signal/regime/order/size/SL-TP/exit-reason/
trade level. Real-terminal execution marked PENDING where no terminal.
**Acceptance.** Round-trip: exported EA on fixture data reproduces Python
signals bit-for-bar; parity report itemizes every discrepancy; missing
terminal ⇒ artifact bundle + exact manual command, never fake success.

## Phase 9 — Full experiment, model card & offline one-click app

**Objective.** One-pair (BNBUSDT) end-to-end: manifest → seal → search →
shortlist → tick validation → (MQL5 parity) → freeze → lockbox once →
model card or rejection report. Generalize to more pairs only after pass.

**Plus (user requirement, revised 2026-07-12): a fully offline one-click
app.** The 2026-07-10 Cloudflare Workers dashboard was removed after the
user changed direction to "offline saja (tanpa cloudflare)"; the
architecture note that killed it still holds — Workers cannot run the
numpy search under CPU-time limits, and a hosted panel adds a network
surface the project does not want. The offline replacement:
- a **Gradio** Blocks app (`evoquant.webui.gradio_app`, primary front-end):
  dynamic pair discovery from the data dir, split/target/search config,
  provider choice + "Test connection", a run streamed live over a
  queue+thread with telemetry/verdict/metrics and a report download;
- a **zero-dependency stdlib** dashboard (`evoquant.webui.server` +
  `dashboard.html`) as the no-install fallback (same governance rules);
- **one-click** `run_all.bat` / `run_all.sh`: locate Python, create a
  venv, install `-e ".[gui]"`, launch `evoquant gui`, open the browser;
- both front-ends bind 127.0.0.1, never compute fitness, never store
  lockbox data, and fail closed without an API key. No cloud path.

**Acceptance.** Experiment replay from manifest reproduces decisions;
lockbox event log shows exactly one open; model card includes negative
evidence; `REJECTED`/`NO_EDGE_FOUND` produce complete reports too. App:
pairs discovered from the data dir; config round-trips; key test returns
provider latency/model or a structured error; a streamed run reaches a
terminal verdict with a verdict-matched downloadable report; a real
headless-Chromium drive of the Gradio app completes an end-to-end run
with zero external network requests.
