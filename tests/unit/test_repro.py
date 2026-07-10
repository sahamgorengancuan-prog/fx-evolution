from __future__ import annotations

import json

from evoquant import repro
from tests.conftest import REPO_ROOT


def test_capture_contains_required_fields(tmp_path):
    record = repro.capture(
        seed=123,
        config={"a": 1, "b": [2, 3]},
        data_content_hash="d" * 64,
        repo_dir=REPO_ROOT,
    )
    d = record.to_dict()
    for key in (
        "created_at_utc",
        "evoquant_version",
        "python_version",
        "platform",
        "packages",
        "git",
        "seed",
        "config_hash",
        "data_content_hash",
    ):
        assert key in d
    assert d["seed"] == 123
    assert d["packages"]["numpy"] != "not-installed"

    path = record.write(tmp_path / "repro.json")
    assert json.loads(path.read_text())["data_content_hash"] == "d" * 64


def test_config_hash_stable_and_order_independent():
    a = repro.config_hash({"x": 1, "y": {"z": 2}})
    b = repro.config_hash({"y": {"z": 2}, "x": 1})
    c = repro.config_hash({"x": 1, "y": {"z": 3}})
    assert a == b
    assert a != c
