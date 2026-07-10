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
