from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from evoquant.data.loaders import FSB_EPOCH_UNIX_S, load_fsb_json
from evoquant.errors import DataFormatError
from tests.conftest import make_fsb_doc


def test_load_roundtrip(fsb_file):
    bars, spec, header = load_fsb_json(fsb_file)
    assert bars.symbol == "TESTUSDT"
    assert bars.timeframe_minutes == 60
    assert bars.n_bars == 200
    assert bars.ts_utc_s.dtype == np.int64
    assert spec.tick_size == 0.01
    assert header["company"] == "test"
    # arrays are immutable
    with pytest.raises(ValueError):
        bars.close[0] = 1.0


def test_fsb_time_base_is_minutes_since_2000():
    assert FSB_EPOCH_UNIX_S == int(
        dt.datetime(2000, 1, 1, tzinfo=dt.UTC).timestamp()
    )


def test_declared_bar_count_mismatch_rejected(tmp_path):
    doc = make_fsb_doc()
    doc["bars"] = 999
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(DataFormatError):
        load_fsb_json(p)


def test_missing_arrays_rejected(tmp_path):
    doc = make_fsb_doc()
    del doc["close"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(DataFormatError) as exc:
        load_fsb_json(p)
    assert "close" in exc.value.details["missing"]


def test_missing_file_raises_not_synthesizes(tmp_path):
    with pytest.raises(DataFormatError):
        load_fsb_json(tmp_path / "does_not_exist.json")


def test_invalid_json_rejected(tmp_path):
    p = tmp_path / "garbage.json"
    p.write_text("{not json")
    with pytest.raises(DataFormatError):
        load_fsb_json(p)


def test_slice_bounds_and_content(fsb_file):
    bars, _, _ = load_fsb_json(fsb_file)
    view = bars.slice(10, 20)
    assert view.n_bars == 10
    assert view.ts_utc_s[0] == bars.ts_utc_s[10]
    with pytest.raises(DataFormatError):
        bars.slice(-1, 10)
    with pytest.raises(DataFormatError):
        bars.slice(0, bars.n_bars + 1)
