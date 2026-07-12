"""Offline UI for evoquant (Phase 9). No external services.

Two fully-offline front-ends over the same pair-agnostic orchestrator:

* ``gradio_app`` — the one-click app (``evoquant gui`` / run_all.bat|sh)
  that opens in the browser. Primary UI.
* ``server`` + ``dashboard.html`` — a zero-dependency stdlib fallback
  (``evoquant serve``) for environments without Gradio.

Neither computes fitness, and neither opens the sealed lockbox.
"""
