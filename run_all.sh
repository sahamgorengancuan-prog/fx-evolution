#!/usr/bin/env bash
# ============================================================
#  evoquant — one-click OFFLINE run (macOS / Linux twin of run_all.bat).
#  Creates a local venv, installs the package + Gradio, then launches
#  the Gradio app which opens in your browser. No Cloudflare, no
#  external services.  Usage:  ./run_all.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")"

echo
echo "=== evoquant one-click (offline) ==="
echo

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "[ERROR] Python 3.11+ not found on PATH. Install it and retry." >&2
  exit 1
fi

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv
fi
VPY=".venv/bin/python"

if [ ! -f ".venv/.evoquant_installed" ]; then
  echo "Installing evoquant + Gradio (first run only, may take a minute) ..."
  "$VPY" -m pip install --upgrade pip
  "$VPY" -m pip install -e ".[gui]"
  echo installed > ".venv/.evoquant_installed"
fi

echo
echo "Launching the evoquant dashboard ... your browser will open shortly."
echo "Press Ctrl+C to stop."
echo
exec "$VPY" -m evoquant.cli gui --data-dir "data/raw" --out-dir "experiments/gui"
