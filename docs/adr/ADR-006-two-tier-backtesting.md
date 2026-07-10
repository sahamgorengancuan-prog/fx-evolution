# ADR-006: Two-tier backtesting — bar approximation, then tick/event

**Status:** Accepted, 2026-07-10 (implementation: Phase 3)

## Context
H1 OHLC cannot reproduce intrabar tick order, real spreads, gap fills, or
latency (audit B1–B8). v8 applied a constant synthetic spread over 9 years
and marked equity on close only.

## Decision
- **Tier A — fast engine**, permanently labeled `BAR_APPROXIMATION`:
  causal, mark-to-market per bar, bid/ask aware with dynamic spread series,
  fee/funding aware, gap-through-stop fills at first executable price,
  tick-size/lot-step/min-notional rounding, conservative worst-case
  ordering for intrabar SL+TP ambiguity, deterministic under seed.
- **Tier B — event/tick engine** for shortlisted candidates only:
  chronological ticks/events, historical spread, latency distribution,
  order-type rules, stop/freeze levels, rejection/partial-fill or explicit
  conservative fallback, margin/liquidation, session/outage handling.
- Cost model is **metadata-first and fail-closed**: a missing spread
  series, funding schedule, or contract field required by the instrument
  class aborts the run with a structured error — no silent defaults, no
  synthetic fallback data in research/production modes.
- Differential tests bound Tier-A vs Tier-B divergence per fixture; the
  funnel is Tier A → Tier B → MQL5 real ticks (ADR-007).
