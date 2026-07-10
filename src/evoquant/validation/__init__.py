"""Statistical validation framework (Phase 7).

Correctly named tests only: CPCV, candidate-matrix PBO (CSCV),
probabilistic/deflated Sharpe with effective trial counts, block
bootstrap, White's Reality Check, execution/parameter stress, and
return concentration.

Deliberately absent — and kept absent by a test: any "terminal return
percentile from permuted returns". Compounding is order-invariant, so
that quantity is undefined (v8 audit S2). Resampling here is used for
path statistics (drawdown) and estimator confidence only.
"""
