# Forensic Audit — v8 "Universal Evolution Searcher"

**Date:** 2026-07-10
**Auditor:** automated forensic pass over the attached artifacts.

## 0. Evidence base and its limits

Artifacts examined:

| Artifact | Contents | Status |
|---|---|---|
| `result_v8_colab202.json` | Final run report: champion genome, hall of fame (10), per-pair verdicts for 5 symbols, 300-generation history, run metadata | Fully examined |
| `BNBUSDT_H11.json` | ForexSB-format export: contract header + 75,791 H1 bars, 2017-11-06 03:00 UTC → 2026-07-07 02:00 UTC | Fully examined, integrity-scanned |
| `Universal_Evolution_Searcher_Fable5_Blueprint.md` | Rebuild blueprint incl. its own audit of the old notebook | Fully examined |
| Original `.ipynb` | **Not attached.** | Not available |

Because the notebook itself was not provided, every finding below is tagged:

- **[VERIFIED]** — directly demonstrable from the result JSON or the data file.
- **[REPORTED]** — asserted by the blueprint's internal audit of the notebook; consistent with the artifacts but not independently re-derivable here. Treated as true for design purposes; each gets a regression test in the rebuild so it can never silently recur.

---

## 1. Architecture map of the existing (v8) system

Reconstructed from the result JSON fields and the blueprint's description:

```
Raw H1 OHLC (ForexSB JSON, 5 crypto pairs)
        │
        ▼
Full-history pair characterization  ── leaks: computed before holdout split [REPORTED]
        │
        ▼
Single evolutionary loop (pop ≈ 400, 300 generations, 118,000 evals, GPU)
  genome = { entry_long AST, entry_short AST, regime_filter AST, risk block }
  fitness = single scalar "universal_score" over ALL pairs jointly [VERIFIED]
        │
        ▼
Walk-forward "folds" (median fold APR reported per pair)
  — but candidates were selected against the union of test slices,
    not re-fit per fold [REPORTED]
        │
        ▼
Holdout evaluation — consulted 6 times during the run
  ("holdout_peeks": 6) [VERIFIED]
        │
        ▼
"Validation" block per pair:
  deflated_sharpe, pbo, mc_dd_p95, mc_return_percentile,
  boot_sharpe_{p05,median} [VERIFIED naming; misnomers — see §3]
        │
        ▼
Verdict: success=false, live_ready=false for all 5 pairs [VERIFIED]
```

Population telemetry per generation: `best` (scalar), `n_passed`, `alive_pct`,
`unique` (~150–270 of ~400), `stagnation` counter, cumulative `evals`.

One genuinely good property [VERIFIED]: the system **reported its own failure
honestly** (`success: false`, per-pair hard-gate breach lists). The gate
*discipline* is worth keeping even though the gate *placement* is wrong.

---

## 2. Verified defects and leakage risks

### D1 — The holdout is not a holdout. [VERIFIED — smoking gun]
`"holdout_peeks": 6` in the run header. The holdout was consulted six times
during a single 92-minute run, and (per blueprint) failures were fed back to
the LLM/search. After the first peek the period stopped measuring
generalization; after the sixth it is effectively a slow validation set.
Consequence: every holdout metric in the report (APR, PF, Sharpe, DD) is
biased in an unknowable direction and the "over-fit signal" gate diagnostics
are self-referential.

### D2 — One genome forced onto all pairs. [VERIFIED]
A single champion genome is scored per pair; `universal_score` and
`n_passed` (max 2 of 5) show the search optimized a cross-pair average.
BNBUSDT/DOGEUSDT verdicts are nearly identical while ETHUSDT produced 17
trades in the holdout — the same structure is simultaneously over-firing on
one instrument and starved on another. The blueprint's core demand (pair- and
regime-specific policies) directly follows.

### D3 — Early scalarization of a multi-objective problem. [VERIFIED]
Fitness is one scalar. History shows the tell: `best` climbs monotonically
from −0.93 to +0.29 over 300 generations while `n_passed` is frozen at 2
from generation 4 onward. The scalar was optimizable without ever improving
feasibility — 296 generations of hill-climbing on a proxy.

### D4 — Diversity collapse / lineage concentration. [VERIFIED]
All 10 hall-of-fame entries are micro-variants of one mechanism: identical
regime filter (`chop(144) < 0.4489`), identical short entry family
(`zscore(abs(keltner(233)),144) > θ`), long entries differing only by
`cross_above`↔`cross_below`↔`lt` and `ret(8)`↔`ret(13)`. `unique` hovered
at ~55–65% of population throughout. The archive preserved one lineage's
neighborhood, not diverse behavior.

### D5 — Type-unsound grammar produces degenerate rules. [VERIFIED]
Champion long entry: `cross_above(rank(delta(macd(3),21),233), ret(8))` —
a cross of a **[0,1] rank** against an **8-bar return** (magnitude ~10⁻²).
The comparison is dimensionally meaningless; it mostly degenerates to "rank
crosses ≈ 0". Long and short entries are structurally unrelated mechanisms
glued together. `abs(abs(keltner(233)))` in the champion shows there is no
AST simplification, canonicalization, or duplicate detection — the search
wastes budget exploring syntactic no-ops.

### D6 — Direction-flip invariance = noise fitting. [VERIFIED]
Hall-of-fame ranks 1 and 2 are the *same* rule with `cross_above` swapped
for `cross_below` (scores 0.2885 vs 0.2796). When inverting the entry logic
barely changes fitness, the fitness surface is noise, not signal.

### D7 — Full-history characterization leaks into search seeding. [REPORTED]
Blueprint §1.2: pair characterization was computed on raw full history
(including holdout) before splitting, then used to bias seeding. Indirect
but real leakage. Rebuild rule: every fitted object (scalers, regime
models, thresholds, operator priors) fits inside each fold's train boundary
only — enforced by construction and by prefix-invariance tests.

### D8 — "Walk-forward" without per-fold refit. [REPORTED]
Blueprint §1.4: genomes were selected on the pooled test slices, making the
outer folds part of the selection set. The reported "median fold APR" is
therefore an in-selection statistic, not OOS. Consistent with D1's holdout
gate then being the only true OOS — which was itself peeked (D1).

### D9 — Non-mark-to-market equity. [REPORTED]
Blueprint §1.7: equity updated mainly on position close, so open-position
adverse excursion is invisible and max-DD is understated. Consistent with
implausibly small "median fold DD" values in the artifact (0.2–1.0%) vs.
holdout DD 5.3% on the same instrument class.

### D10 — Data-quality landmines in the source file. [VERIFIED]
- **Constant synthetic spread:** `spreads[]` is 10 points (= $1.00 at
  point 0.1) for all 75,791 bars, including 2017 bars when BNB traded at
  $1.50 — a 67% notional spread then, 0.17% now. Any backtest using this
  series applies a fictional cost model. The header spread is a
  placeholder, not history.
- **Price quantization:** `digits: 1` ⇒ tick = $0.10. In Nov-2017 bars
  (price ≈ $1.5) that is a ~7% price grid; bar 0 has low 0.5 vs open 1.5
  (a −67% intrabar excursion) — quantization artifacts, not tradable
  prices. Early-era bars are unusable for signal or execution research.
- **68 gaps** (Δt from 120 min up to ~11 h) with no gap policy recorded.
- **Zero swap/funding** (`swapLong/Short: 0`) — perpetual-style funding is
  absent from the cost metadata; a funding-aware engine must fail closed
  or source funding separately.
- Volume units undeclared (base? quote? contracts?). Last bars show volume
  62/28/5 vs 650/8149 early — regime change or unit change; unverifiable.

### D11 — No reproducibility envelope. [VERIFIED by absence]
The result JSON carries no data hash, no code/config hash, no seed, no
dependency lock. The run cannot be reproduced or even re-associated with
the exact data that produced it.

---

## 3. Incorrect statistical assumptions

### S1 — "PBO" that is not PBO. [VERIFIED misnomer]
The artifact reports a per-pair, per-single-strategy `pbo` (e.g. 0.417).
Probability of Backtest Overfitting (Bailey/López de Prado) is defined over
a **candidate-performance matrix** via combinatorially symmetric
cross-validation: it measures how often the IS-best candidate falls in the
OOS bottom half. A single strategy's temporal self-split cannot yield a
PBO; it is at best a temporal-consistency probe. The reported number has no
defined meaning.

### S2 — Terminal-return percentile from permutation. [VERIFIED misnomer]
`mc_return_percentile` (e.g. 0.042) computed by permuting the strategy's
own returns. Compounding is order-invariant: Π(1+rᵢ) is identical under
every permutation. Either the number compares against something other than
what its name says, or it is a comparison of a constant against itself
plus resampling noise. Permutation is legitimate only for **path**
statistics (DD distribution — `mc_dd_p95` is salvageable) — never for
terminal return.

### S3 — Deflated Sharpe with N = every evaluation. [VERIFIED, subtle]
`n_trials: 118000` equals cumulative evolutionary evaluations, and
`deflated_sharpe: 0.0` for every pair. Two compounding errors:
(a) evolutionary trials are highly correlated (D4: one lineage) — the
*effective* number of independent trials is orders of magnitude smaller;
using raw N over-deflates and guarantees DSR≈0 for any achievable Sharpe,
making the test unable to ever pass — a gate that always fails is not a
test. (b) Simultaneously the sign was correct here (the strategy is indeed
worthless), which can breed false confidence in the metric. The rebuild
must estimate effective trials (correlation-clustered) and report both.

### S4 — Selection-then-testing on the same data. [VERIFIED consequence of D1/D8]
All "validation" statistics were computed on data that had already
participated in selection (peeked holdout, pooled folds). Bootstrap CIs
and DSR are only meaningful on data untouched by selection; here nothing
qualifies.

### S5 — No multiple-testing control across the gate battery. [VERIFIED by absence]
Five pairs × ~6 hard gates × soft gates evaluated with no family-wise
framing; conversely nothing tracks that 118k candidates were sequentially
screened against the same slices (data snooping). White Reality Check /
SPA is absent.

---

## 4. Unrealistic backtesting assumptions

| # | Assumption in v8 | Reality requirement (rebuild) |
|---|---|---|
| B1 | Constant 10-point spread for 9 years [VERIFIED, D10] | Historical/bid-ask spread series; fail closed when absent |
| B2 | H1 OHLC treated as sufficient for fills [REPORTED + structural] | Bar engine labeled `BAR_APPROXIMATION`, conservative intrabar ambiguity (worst-case ordering); tick/event engine for shortlist; MQL5 real-tick as arbiter |
| B3 | Equity marked on close only [REPORTED, D9] | Mark-to-market every bar/event incl. open positions |
| B4 | No funding/swap for perpetual-style crypto [VERIFIED: swap=0 in metadata, no funding source] | Funding schedule modeled or instrument flagged spot-only |
| B5 | No gap-through-stop handling evidenced; 68 gaps in data [VERIFIED gaps] | Stops fill at first executable price after gap, never at the stop price inside a gap |
| B6 | No latency, rejection, partial-fill, session model [REPORTED] | Event engine models latency dist., rejects, min-notional, lot-step rounding |
| B7 | Sizing "risk/trade = 0.300%" with no evidence of lot-step / min-notional rounding [VERIFIED text only] | Size from stop distance + contract spec, rounded to lot step, min-notional enforced |
| B8 | Quantized 2017 prices treated as tradable [VERIFIED, D10] | QA flags quantization regimes; research window must exclude or down-weight them explicitly |

---

## 5. Components worth retaining (as concepts)

1. **Typed expression-tree genome** — right idea, wrong type system. Keep
   the AST genome; add dimensional types + canonicalization (Phase 2).
2. **Hard/soft gate verdict structure** with human-readable breach reasons —
   keep the reporting shape; move gates to genuinely unseen data.
3. **Honest failure reporting** (`success:false`, per-pair reasons) — keep.
4. **Stagnation counter + telemetry per generation** (`alive_pct`, `unique`,
   `evals`, wall-clock) — keep and extend to hypervolume/novelty/lineage.
5. **Hall-of-fame archive** — keep, but as a behavioral-diversity archive
   (MAP-Elites), not a top-N-by-scalar list.
6. **Risk block as first-class genome part** (SL/TP/trail/max_bars/cooldown)
   — keep, extend to full execution policy.
7. **`mc_dd_p95`** — the one Monte-Carlo output whose math survives; keep
   under block-resampling.
8. **ForexSB data header** — usable seed for the instrument spec (lot step,
   digits, commission), with the spread/swap fields explicitly distrusted.

## 6. Components that must be rewritten (not patched)

| Component | Reason | Fate |
|---|---|---|
| Split/holdout management | D1, D7, D8 | **Rewrite** → sealed nested WFO + one-shot lockbox (Phase 1, this repo) |
| Fitness (scalar universal score) | D2, D3 | **Rewrite** → constraint-dominance NSGA + QD archive (Phase 5) |
| Grammar/type system | D5, D6 | **Rewrite** → dimension-typed DSL + canonicalizer (Phase 2) |
| Backtester | B1–B8 | **Rewrite** → two-tier engine (Phase 3) |
| "Validation" statistics | S1–S5 | **Rewrite** → correctly named battery (Phase 7) |
| Cost model | B1, B4 | **Rewrite** → metadata-first, fail-closed (Phases 1+3) |
| Characterization/seeding | D7 | **Rewrite** → fold-scoped, train-only (Phase 4) |
| LLM feedback loop | D1 (peeks fed to LLM) | **Rewrite** → schema-gated proposals, no fitness authority, no lockbox visibility (Phase 6) |
| Reproducibility | D11 | **New** (Phase 1) |

## 7. Architecture decision records

See `docs/adr/ADR-001` … `ADR-007`. Summary:

- ADR-001 Pair-specific policies over shared genome
- ADR-002 Sealed nested walk-forward + one-shot lockbox
- ADR-003 Dimension-typed feature DSL
- ADR-004 Hybrid search: GP structure + constraint-dominance NSGA + MAP-Elites + CMA-ES refinement + contextual bandit operators
- ADR-005 Event-sourced evolutionary memory
- ADR-006 Two-tier backtesting (bar approximation → tick/event → MQL5 real tick)
- ADR-007 MQL5 export + differential parity as release gate

## 8. Phased dependency plan

See `docs/PHASE_PLAN.md`.

## 9. Acceptance tests per phase

Stated at the top of each phase in `docs/PHASE_PLAN.md`; Phase 1's are
implemented in `tests/` in this commit. `VALIDATION_STATUS.md` tracks which
claims currently have test evidence.
