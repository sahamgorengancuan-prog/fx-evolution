# ADR-002: Sealed nested walk-forward + one-shot final lockbox

**Status:** Accepted, 2026-07-10

## Context
v8's run header records `holdout_peeks: 6` (audit D1); characterization ran
on full history before splitting (D7); fold selection pooled test slices
(D8). Every reported OOS number was contaminated.

## Decision
Four-tier split, generated once, hashed, and sealed at experiment creation:

1. **Inner train/validation** folds — all search, fitting, characterization.
2. **Outer walk-forward OOS** folds — model selection only.
3. **Final lockbox** (most recent segment) — opened exactly once, after
   candidate + thresholds are frozen; opening permanently transitions the
   experiment to `DISCLOSED`. Redesign afterwards ⇒ new experiment ID and a
   new/independent lockbox.
4. Purge (≥ max information overlap / holding horizon) and embargo bars
   separate every train/test boundary.

Access control is structural: the default research data view physically
excludes lockbox rows; lockbox rows are only reachable through
`LockboxGuard.open()`, which is single-shot and event-logged.

## Consequences
- Peeking is an API impossibility for compliant code, and an audit-log
  event for non-compliant code (Python cannot fully sandbox itself —
  recorded in KNOWN_LIMITATIONS).
- Fewer effective data points for search; accepted as the price of a
  meaningful final number.
