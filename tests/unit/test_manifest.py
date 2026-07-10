from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from evoquant.data.loaders import load_fsb_json
from evoquant.data.manifest import (
    build_manifest,
    canonical_content_hash,
    load_manifest,
    sha256_file,
    verify_manifest,
    write_manifest,
)
from evoquant.data.quality import validate_or_raise
from evoquant.errors import ManifestImmutabilityError, ManifestVerificationError
from tests.conftest import make_bars


def _manifest_for(fsb_file):
    bars, spec, _ = load_fsb_json(fsb_file)
    report = validate_or_raise(bars, spec)
    return bars, build_manifest(bars, spec, report, fsb_file)


def test_content_hash_deterministic():
    a = make_bars(seed=1)
    b = make_bars(seed=1)
    c = make_bars(seed=2)
    assert canonical_content_hash(a) == canonical_content_hash(b)
    assert canonical_content_hash(a) != canonical_content_hash(c)


def test_tampering_one_price_changes_hash():
    a = make_bars()
    close = np.array(a.close)
    close[123] += 1e-9
    b = replace(a, close=close)
    assert canonical_content_hash(a) != canonical_content_hash(b)


def test_sha256_file_matches_hashlib(fsb_file):
    assert sha256_file(fsb_file) == hashlib.sha256(fsb_file.read_bytes()).hexdigest()


def test_write_once_idempotent_and_immutable(fsb_file, tmp_path):
    bars, manifest = _manifest_for(fsb_file)
    path = tmp_path / "m.json"
    write_manifest(manifest, path)
    write_manifest(manifest, path)  # identical: idempotent no-op

    tampered = replace(manifest, content_hash="0" * 64)
    with pytest.raises(ManifestImmutabilityError):
        write_manifest(tampered, path)


def test_load_roundtrip_preserves_identity(fsb_file, tmp_path):
    _, manifest = _manifest_for(fsb_file)
    path = write_manifest(manifest, tmp_path / "m.json")
    loaded = load_manifest(path)
    assert loaded.semantic_dict() == manifest.semantic_dict()
    assert loaded.manifest_id == manifest.manifest_id


def test_verify_detects_data_tampering(fsb_file):
    bars, manifest = _manifest_for(fsb_file)
    verify_manifest(manifest, bars, source_path=fsb_file)  # clean passes

    close = np.array(bars.close)
    close[0] += 0.0001
    tampered = replace(bars, close=close)
    with pytest.raises(ManifestVerificationError):
        verify_manifest(manifest, tampered)


def test_verify_detects_source_file_tampering(fsb_file):
    bars, manifest = _manifest_for(fsb_file)
    fsb_file.write_text(fsb_file.read_text() + " ")
    with pytest.raises(ManifestVerificationError):
        verify_manifest(manifest, bars, source_path=fsb_file)
