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
