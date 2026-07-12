"""Phase-9 Gradio one-click UI: pair discovery, app construction, and a
real streamed end-to-end run driven through the app's run handler."""
from __future__ import annotations

import json

import pytest

from tests.conftest import REAL_DATA

pytest.importorskip("gradio")

pytestmark = pytest.mark.skipif(
    not REAL_DATA.exists(), reason="real BNBUSDT data file not present"
)

from evoquant.webui.gradio_app import (  # noqa: E402  (after importorskip)
    _pairs_table,
    _run_streaming,
    build_app,
)

_FAST = dict(
    seed=20260710, max_bars=6000, generations=2, population=6,
    outer_folds=3, outer_test_bars=700, inner_folds=3, inner_val_bars=350,
    min_train_bars=1500, lockbox_bars=1500, commission=0.001,
    spread_points=5.0, initial_equity=1_000_000,
)


def _data_dir(tmp_path):
    """A data dir containing a small slice of the real file (fast run)."""
    doc = json.loads(REAL_DATA.read_text())
    n = 6000
    small = dict(doc)
    for k in ("time", "open", "high", "low", "close", "volume", "spreads"):
        small[k] = doc[k][-n:]
    small["bars"] = n
    d = tmp_path / "data"
    d.mkdir()
    (d / "BNBUSDT_H1.json").write_text(json.dumps(small))
    return d


def test_pairs_table_discovers_any_json(tmp_path):
    d = _data_dir(tmp_path)
    rows, mapping = _pairs_table(str(d))
    assert len(rows) == 1 and rows[0][0] == "BNBUSDT"
    assert len(mapping) == 1
    label = next(iter(mapping))
    assert mapping[label].endswith("BNBUSDT_H1.json")


def test_app_builds_offline(tmp_path):
    d = _data_dir(tmp_path)
    app = build_app(str(d), str(tmp_path / "out"))
    # constructed without launching and without touching the network
    assert app is not None


def test_streamed_run_reaches_terminal_verdict(tmp_path):
    d = _data_dir(tmp_path)
    _, mapping = _pairs_table(str(d))
    label = next(iter(mapping))
    out_root = str(tmp_path / "out")

    updates = list(_run_streaming(label, mapping, out_root, **_FAST))
    assert updates, "handler yielded nothing"
    # progress streamed then a terminal frame
    final = updates[-1]
    status, telemetry, verdict_md, metrics, report, download = final
    assert "finished" in status
    assert any(v in verdict_md for v in ("SHORTLISTED", "NO_EDGE_FOUND", "REJECTED"))
    assert report["verdict"] in ("SHORTLISTED", "NO_EDGE_FOUND", "REJECTED")
    # telemetry table has EVERY searched generation (not just the first)
    assert len(telemetry) >= 2, "telemetry must stream multiple generations"
    assert all(len(row) == 6 for row in telemetry)  # fold,gen,hv,feas,archive,interv
    assert any(row[0] == "lockbox" for row in metrics)
    assert download is not None and download.endswith("report.json")
    # the download file really exists and matches the shown report
    from pathlib import Path
    assert Path(download).exists()
    assert json.loads(Path(download).read_text())["verdict"] == report["verdict"]


def test_run_handler_requires_a_pair(tmp_path):
    updates = list(_run_streaming("", {}, str(tmp_path / "o"), **_FAST))
    assert len(updates) == 1
    assert "Select a pair" in updates[0][0]
