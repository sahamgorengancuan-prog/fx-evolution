@echo off
REM ============================================================
REM  evoquant — one-click OFFLINE run.
REM  Creates a local venv, installs the package + Gradio, then
REM  launches the Gradio app which opens in your browser.
REM  No Cloudflare, no external services. Double-click to run.
REM ============================================================
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo(
echo === evoquant one-click (offline) ===
echo(

REM --- 1. locate Python -------------------------------------------------
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo [ERROR] Python 3.11+ was not found on PATH.
  echo         Install it from https://www.python.org/downloads/ and retry.
  pause
  exit /b 1
)

REM --- 2. create the virtual environment once ---------------------------
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 ( echo [ERROR] venv creation failed. & pause & exit /b 1 )
)
set "VPY=.venv\Scripts\python.exe"

REM --- 3. install the package + Gradio (idempotent; marker-guarded) -----
if not exist ".venv\.evoquant_installed" (
  echo Installing evoquant + Gradio  ^(first run only, may take a minute^) ...
  "%VPY%" -m pip install --upgrade pip
  "%VPY%" -m pip install -e ".[gui]"
  if errorlevel 1 ( echo [ERROR] install failed. & pause & exit /b 1 )
  echo installed> ".venv\.evoquant_installed"
)

REM --- 4. launch the offline Gradio app (opens the browser) ------------
echo(
echo Launching the evoquant dashboard ... your browser will open shortly.
echo Close this window to stop the app.
echo(
"%VPY%" -m evoquant.cli gui --data-dir "data\raw" --out-dir "experiments\gui"

pause
endlocal
