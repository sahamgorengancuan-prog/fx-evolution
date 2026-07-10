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
