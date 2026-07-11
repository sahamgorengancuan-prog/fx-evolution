"""Local control-plane server (stdlib only — fully offline).

Serves the self-contained dashboard and a JSON API:

    GET  /                      dashboard (no external resources)
    GET  /api/pairs             dynamic pair discovery from the data dir
    GET  /api/config            saved settings (secrets redacted)
    PUT  /api/config            save settings (+ optional API keys)
    POST /api/llm-test          OpenAI/OpenRouter connectivity check
    POST /api/runs              start an experiment run (background thread)
    GET  /api/runs              list runs
    GET  /api/runs/{id}         status + progress events
    GET  /api/runs/{id}/report  full report JSON
    GET  /api/runs/{id}/download  report as attachment

The same dashboard HTML is embedded into the Cloudflare Worker by
`cloudflare/build_worker.py`; this server is the offline twin. Secrets are
stored in `<state>/secrets.json` with 0600 permissions and are never
echoed back; only presence flags leave the server.
"""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from evoquant.errors import EvoquantError

_DASHBOARD = Path(__file__).with_name("dashboard.html")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class WebUIState:
    """Filesystem-backed state shared across request threads."""

    def __init__(self, state_dir: str | Path, data_dir: str | Path) -> None:
        self.state_dir = Path(state_dir)
        self.data_dir = Path(data_dir)
        self.runs_dir = self.state_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.threads: dict[str, threading.Thread] = {}

    # ---- config / secrets ------------------------------------------- #

    @property
    def config_path(self) -> Path:
        return self.state_dir / "config.json"

    @property
    def secrets_path(self) -> Path:
        return self.state_dir / "secrets.json"

    def load_config(self) -> dict[str, Any]:
        if self.config_path.exists():
            return dict(json.loads(self.config_path.read_text()))
        return {"provider": "openrouter", "model": "", "run_defaults": {}}

    def save_config(self, doc: dict[str, Any]) -> None:
        secrets = {
            k: v
            for k, v in doc.pop("secrets", {}).items()
            if k in ("openai_api_key", "openrouter_api_key") and isinstance(v, str) and v
        }
        if secrets:
            existing = self.load_secrets()
            existing.update(secrets)
            self.secrets_path.write_text(json.dumps(existing))
            self.secrets_path.chmod(0o600)
        self.config_path.write_text(json.dumps(doc, indent=2, sort_keys=True))

    def load_secrets(self) -> dict[str, str]:
        if self.secrets_path.exists():
            return dict(json.loads(self.secrets_path.read_text()))
        return {}

    def public_config(self) -> dict[str, Any]:
        cfg = self.load_config()
        secrets = self.load_secrets()
        cfg["secrets_present"] = {
            "openai_api_key": bool(secrets.get("openai_api_key")),
            "openrouter_api_key": bool(secrets.get("openrouter_api_key")),
        }
        cfg["data_dir"] = str(self.data_dir)
        return cfg

    # ---- pair discovery ---------------------------------------------- #

    def discover_pairs(self) -> list[dict[str, Any]]:
        """Any ForexSB JSON dropped into the data dir becomes a pair."""
        out: list[dict[str, Any]] = []
        for path in sorted(self.data_dir.glob("*.json")):
            try:
                doc = json.loads(path.read_text())
                if not all(k in doc for k in ("symbol", "period", "time", "close")):
                    continue
                out.append(
                    {
                        "file": str(path),
                        "name": path.name,
                        "symbol": str(doc["symbol"]),
                        "period_minutes": int(doc["period"]),
                        "n_bars": len(doc["time"]),
                        "description": doc.get("description", ""),
                    }
                )
            except (json.JSONDecodeError, OSError, ValueError, TypeError):
                continue
        return out

    # ---- runs ---------------------------------------------------------- #

    def run_dir(self, run_id: str) -> Path:
        if not _RUN_ID_RE.match(run_id):
            raise EvoquantError("invalid run id", run_id=run_id)
        return self.runs_dir / run_id

    def start_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        from evoquant.experiment.orchestrator import RunConfig, run_experiment

        data_file = str(payload.get("data_file", ""))
        if not Path(data_file).is_file():
            raise EvoquantError("data_file does not exist", data_file=data_file)
        run_id = f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        rdir = self.run_dir(run_id)
        rdir.mkdir(parents=True)

        allowed = set(RunConfig.__dataclass_fields__) - {"data_file", "out_dir"}
        overrides = {k: v for k, v in payload.items() if k in allowed}
        config = RunConfig(data_file=data_file, out_dir=str(rdir / "artifacts"), **overrides)

        status: dict[str, Any] = {
            "run_id": run_id,
            "state": "RUNNING",
            "data_file": data_file,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config": config.to_dict(),
        }
        self._write_status(rdir, status)

        events_path = rdir / "events.jsonl"

        def on_progress(event: dict[str, Any]) -> None:
            with self.lock:
                with open(events_path, "a") as fh:
                    fh.write(json.dumps(event) + "\n")

        def worker() -> None:
            try:
                report = run_experiment(config, on_progress=on_progress)
                status.update(state="FINISHED", verdict=report.get("verdict"),
                              symbol=report.get("symbol"))
            except EvoquantError as exc:
                status.update(state="ERROR", error=exc.to_artifact())
            except Exception as exc:  # surfaced, never swallowed
                status.update(state="ERROR", error={"message": repr(exc)})
            finally:
                status["finished_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                )
                self._write_status(rdir, status)

        thread = threading.Thread(target=worker, name=run_id, daemon=True)
        self.threads[run_id] = thread
        thread.start()
        return status

    def _write_status(self, rdir: Path, status: dict[str, Any]) -> None:
        with self.lock:
            tmp = rdir / "status.json.tmp"
            tmp.write_text(json.dumps(status, indent=2, sort_keys=True))
            tmp.rename(rdir / "status.json")

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for status_file in sorted(self.runs_dir.glob("*/status.json"), reverse=True):
            try:
                runs.append(json.loads(status_file.read_text()))
            except (json.JSONDecodeError, OSError):
                continue
        return runs

    def run_detail(self, run_id: str, tail: int = 200) -> dict[str, Any]:
        rdir = self.run_dir(run_id)
        status: dict[str, Any] = dict(json.loads((rdir / "status.json").read_text()))
        events: list[dict[str, Any]] = []
        events_path = rdir / "events.jsonl"
        if events_path.exists():
            lines = events_path.read_text().splitlines()[-tail:]
            events = [json.loads(line) for line in lines if line.strip()]
        status["events"] = events
        return status

    def run_report(self, run_id: str) -> dict[str, Any]:
        path = self.run_dir(run_id) / "artifacts" / "report.json"
        return dict(json.loads(path.read_text()))


def _llm_test(state: WebUIState, payload: dict[str, Any]) -> dict[str, Any]:
    from evoquant.llm.client import ChatClient, LLMConfig

    provider = str(payload.get("provider", "")).strip().lower()
    model = str(payload.get("model", "")).strip()
    api_key = str(payload.get("api_key", "")).strip()
    if not api_key:
        secrets = state.load_secrets()
        api_key = secrets.get(
            "openai_api_key" if provider == "openai" else "openrouter_api_key", ""
        )
    config = LLMConfig(provider=provider, api_key=api_key, model=model, timeout_s=20.0)
    return ChatClient(config).test_connection()


class _Handler(BaseHTTPRequestHandler):
    state: WebUIState  # injected by make_server

    # ---- plumbing ----------------------------------------------------- #

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        pass

    def _json(self, code: int, doc: Any) -> None:
        body = json.dumps(doc).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        return dict(json.loads(self.rfile.read(length)))

    # ---- routes --------------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        try:
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                html = _DASHBOARD.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
            elif path == "/api/pairs":
                self._json(200, {"pairs": self.state.discover_pairs()})
            elif path == "/api/config":
                self._json(200, self.state.public_config())
            elif path == "/api/runs":
                self._json(200, {"runs": self.state.list_runs()})
            elif path.startswith("/api/runs/"):
                parts = path.split("/")
                run_id = parts[3]
                if len(parts) == 4:
                    self._json(200, self.state.run_detail(run_id))
                elif parts[4] == "report":
                    self._json(200, self.state.run_report(run_id))
                elif parts[4] == "download":
                    body = json.dumps(
                        self.state.run_report(run_id), indent=2, sort_keys=True
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header(
                        "Content-Disposition",
                        f'attachment; filename="evoquant_{run_id}_report.json"',
                    )
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self._json(404, {"error": "unknown run resource"})
            else:
                self._json(404, {"error": "not found"})
        except FileNotFoundError:
            self._json(404, {"error": "not found"})
        except EvoquantError as exc:
            self._json(400, exc.to_artifact())
        except Exception as exc:  # pragma: no cover - defensive
            self._json(500, {"error": repr(exc)})

    def do_PUT(self) -> None:  # noqa: N802
        try:
            if self.path.split("?")[0] == "/api/config":
                self.state.save_config(self._body())
                self._json(200, self.state.public_config())
            else:
                self._json(404, {"error": "not found"})
        except EvoquantError as exc:
            self._json(400, exc.to_artifact())
        except Exception as exc:  # pragma: no cover
            self._json(500, {"error": repr(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            path = self.path.split("?")[0]
            if path == "/api/llm-test":
                self._json(200, _llm_test(self.state, self._body()))
            elif path == "/api/runs":
                self._json(200, self.state.start_run(self._body()))
            else:
                self._json(404, {"error": "not found"})
        except EvoquantError as exc:
            self._json(502, {"ok": False, **exc.to_artifact()})
        except Exception as exc:  # pragma: no cover
            self._json(500, {"ok": False, "error": repr(exc)})


def make_server(
    state_dir: str | Path,
    data_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8787,
) -> tuple[ThreadingHTTPServer, WebUIState]:
    state = WebUIState(state_dir, data_dir)
    handler = type("BoundHandler", (_Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    return server, state


def serve(state_dir: str, data_dir: str, host: str, port: int) -> None:  # pragma: no cover
    server, _ = make_server(state_dir, data_dir, host, port)
    print(f"evoquant dashboard: http://{host}:{server.server_address[1]}/")
    print(f"data dir: {data_dir}   state dir: {state_dir}")
    server.serve_forever()
