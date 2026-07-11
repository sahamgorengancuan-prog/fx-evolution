"""Phase-9 acceptance: pair-agnostic end-to-end run on the real file, the
offline control-plane API, and the Cloudflare worker + offline-safety."""
from __future__ import annotations

import json
import re
import threading
import urllib.request
from pathlib import Path

import pytest

from evoquant.experiment.orchestrator import RunConfig, run_experiment
from evoquant.webui.cloudflare import build_worker_bundle
from evoquant.webui.server import make_server
from tests.conftest import REAL_DATA, REPO_ROOT

pytestmark = pytest.mark.skipif(
    not REAL_DATA.exists(), reason="real BNBUSDT data file not present"
)

# small but real budget so the suite stays fast
_FAST = dict(
    max_bars=6000,
    lockbox_bars=1500,
    n_outer_folds=3,
    outer_test_bars=700,
    n_inner_folds=3,
    inner_val_bars=350,
    min_train_bars=1500,
    purge_bars=48,
    embargo_bars=12,
    population_size=6,
    n_generations=2,
    outer_folds_to_search=1,
    inner_blocks_per_fold=2,
    eval_warmup_bars=250,
)


class TestOrchestrator:
    def test_end_to_end_run_is_honest_and_complete(self, tmp_path):
        cfg = RunConfig(data_file=str(REAL_DATA), out_dir=str(tmp_path / "run"), **_FAST)
        events: list[dict] = []
        report = run_experiment(cfg, on_progress=events.append)

        # verdict is one of the three honest outcomes
        assert report["verdict"] in ("SHORTLISTED", "NO_EDGE_FOUND", "REJECTED")
        assert report["symbol"] == "BNBUSDT"
        # lockbox never opened by a run
        assert "not been opened" in report["lockbox_note"].lower()
        assert report["state"] in ("SHORTLISTED", "CLOSED_NO_EDGE")
        # disclaimer present and the engine is labeled approximate
        assert "BAR_APPROXIMATION" in report["engine_disclaimer"]
        # artifacts on disk
        art = tmp_path / "run"
        for name in ("report.json", "MODEL_CARD.md", "experiment.json",
                     "data_manifest.json", "split_plan.json", "memory.sqlite"):
            assert (art / name).exists(), name
        assert (art / "mql5_bundle" / "parity_status.json").exists()
        # progress telemetry includes generation events with the chart fields
        gens = [e for e in events if e["stage"] == "search_generation"]
        assert gens and "hypervolume" in gens[0] and "feasibility_rate" in gens[0]

    def test_run_is_pair_agnostic_reproducible(self, tmp_path):
        """Same file + seed => identical champion + verdict (any pair)."""
        a = run_experiment(RunConfig(str(REAL_DATA), str(tmp_path / "a"), **_FAST))
        b = run_experiment(RunConfig(str(REAL_DATA), str(tmp_path / "b"), **_FAST))
        assert a["verdict"] == b["verdict"]
        if a.get("champion") and b.get("champion"):
            assert a["champion"]["genome_hash"] == b["champion"]["genome_hash"]

    def test_synthetic_second_pair_file_also_runs(self, tmp_path):
        """Prove pair-agnosticism: a *different* symbol file runs identically.
        Build a valid ForexSB doc for a fake symbol from the real arrays."""
        real = json.loads(REAL_DATA.read_text())
        n = 6000
        fake = dict(real)
        fake["symbol"] = "XYZUSDT"
        fake["description"] = "synthetic second pair (structure only)"
        for k in ("time", "open", "high", "low", "close", "volume", "spreads"):
            fake[k] = real[k][-n:]
        fake["bars"] = n
        pair_file = tmp_path / "XYZUSDT_H1.json"
        pair_file.write_text(json.dumps(fake))

        report = run_experiment(
            RunConfig(str(pair_file), str(tmp_path / "xyz"),
                      **{**_FAST, "max_bars": None})
        )
        assert report["symbol"] == "XYZUSDT"
        assert report["verdict"] in ("SHORTLISTED", "NO_EDGE_FOUND", "REJECTED")


class TestDashboardOfflineSafety:
    def test_no_external_network_references(self):
        html = (REPO_ROOT / "src" / "evoquant" / "webui" / "dashboard.html").read_text()
        # no protocol-based external resource loads anywhere in the page
        offenders = re.findall(r'(?:src|href)\s*=\s*["\']https?://', html, re.I)
        assert offenders == [], f"external resource refs found: {offenders}"
        # no external hosts loaded (cdn domains, protocol-relative urls)
        assert not re.search(r'(?:https?:)?//[a-z0-9.-]*cdn', html, re.I)
        assert "<script src" not in html.lower()  # no external scripts
        # only same-origin fetches
        for m in re.findall(r'fetch\(\s*["\']([^"\']+)', html):
            assert m.startswith("/"), f"non-relative fetch: {m}"


class TestControlPlaneApi:
    def _serve(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # a discoverable pair file
        real = json.loads(REAL_DATA.read_text())
        small = dict(real)
        for k in ("time", "open", "high", "low", "close", "volume", "spreads"):
            small[k] = real[k][:500]
        small["bars"] = 500
        (data_dir / "BNBUSDT_H1.json").write_text(json.dumps(small))
        server, state = make_server(tmp_path / "state", data_dir, "127.0.0.1", 0)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server, port

    def _get(self, port, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as r:
            return json.loads(r.read())

    def _req(self, port, path, method, body):
        import urllib.error

        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}", data=data, method=method,
            headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:  # read structured error bodies too
            return json.loads(exc.read())

    def test_dashboard_and_dynamic_pairs_and_config(self, tmp_path):
        server, port = self._serve(tmp_path)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5) as r:
                assert r.status == 200 and b"evoquant" in r.read()
            # dynamic pair discovery
            pairs = self._get(port, "/api/pairs")["pairs"]
            assert len(pairs) == 1 and pairs[0]["symbol"] == "BNBUSDT"
            # config save round-trips; secret is stored, never echoed
            self._req(port, "/api/config", "PUT",
                      {"provider": "openrouter", "model": "free-model",
                       "secrets": {"openrouter_api_key": "sk-or-secret"}})
            cfg = self._get(port, "/api/config")
            assert cfg["model"] == "free-model"
            assert cfg["secrets_present"]["openrouter_api_key"] is True
            assert "sk-or-secret" not in json.dumps(cfg)
        finally:
            server.shutdown()

    def test_run_lifecycle_and_download(self, tmp_path):
        server, port = self._serve(tmp_path)
        try:
            pairs = self._get(port, "/api/pairs")["pairs"]
            # a run with a config that cannot fit -> ERROR state, surfaced
            st = self._req(port, "/api/runs", "POST",
                           {"data_file": pairs[0]["file"], "lockbox_bars": 999999})
            run_id = st["run_id"]
            import time
            for _ in range(50):
                d = self._get(port, f"/api/runs/{run_id}")
                if d["state"] in ("FINISHED", "ERROR"):
                    break
                time.sleep(0.1)
            assert d["state"] == "ERROR"  # config doesn't fit -> honest error
            assert "error" in d
        finally:
            server.shutdown()

    def test_llm_test_with_injected_transport_not_needed(self, tmp_path):
        """The live llm-test path requires a network; here we only assert it
        fails closed without a key (never silently 'ok')."""
        server, port = self._serve(tmp_path)
        try:
            r = self._req(port, "/api/llm-test", "POST",
                          {"provider": "openai", "model": "gpt-4o-mini"})
            assert r.get("ok") is not True  # no key => not ok
        finally:
            server.shutdown()


class TestCloudflareWorker:
    def test_worker_bundle_embeds_same_dashboard(self, tmp_path):
        paths = build_worker_bundle(tmp_path / "worker")
        worker = Path(paths["worker"]).read_text()
        toml = Path(paths["wrangler_toml"]).read_text()
        readme = Path(paths["readme"]).read_text()
        # the exact dashboard html is embedded
        dashboard = (REPO_ROOT / "src" / "evoquant" / "webui" / "dashboard.html").read_text()
        assert json.dumps(dashboard) in worker
        # KV bindings + real-tick governance + runner token gate
        assert "EVOQUANT_KV" in toml and "EVOQUANT_RUNS" in toml
        assert "wrangler secret put OPENAI_API_KEY" in readme
        assert "EVOQUANT_RUNNER_TOKEN" in worker  # runner auth enforced
        assert "never computes fitness" in worker
        # deploy is documented, not executed
        assert "wrangler deploy" in readme
