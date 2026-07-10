# ADR-007: MQL5 export and differential parity as a release gate

**Status:** Accepted, 2026-07-10 (implementation: Phase 8)

## Context
Research-Python and execution-MQL5 divergence is a classic silent killer;
final-equity comparison hides sign-flipped signals that happen to net out.

## Decision
Every frozen candidate exports to an MQL5 EA / policy interpreter.
Validation uses MetaTrader 5 Strategy Tester, model **"Every tick based on
real ticks"** whenever real tick history exists, with actual/custom symbol
specifications and imported ticks for instruments brokers don't carry.
Parity is differential at the **signal, regime state, order request,
position size (post-rounding), SL/TP level, exit reason, and trade** level;
final-equity-only comparison is insufficient by policy. Where no terminal
is available in CI, the exporter emits the exact tester config + manual
command and the parity item is marked `PENDING`, never faked.

## Consequences
A candidate without a parity report cannot reach `FROZEN → DISCLOSED`
lockbox evaluation. Discrepancies are release blockers, surfaced in the
model card.
