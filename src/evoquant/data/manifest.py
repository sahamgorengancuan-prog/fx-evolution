"""Immutable, hash-addressed data manifests.

A manifest binds together: the raw source file (sha256 of bytes), the
canonical parsed content (sha256 over deterministically serialized arrays),
the instrument spec, and the QA summary. Experiments seal the manifest's
content hash; any later drift in the data is detectable and fatal.

Write-once: an existing manifest file may only ever be re-written with
byte-identical semantic content.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evoquant import __version__ as _evoquant_version
from evoquant.data.contracts import InstrumentSpec
from evoquant.data.loaders import BarData
from evoquant.data.quality import QAReport
from evoquant.errors import ManifestImmutabilityError, ManifestVerificationError

MANIFEST_SCHEMA_VERSION = "1.0"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(obj: Any) -> str:
    """Deterministic JSON used everywhere a hash is derived from a dict."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_of_dict(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def canonical_content_hash(bars: BarData) -> str:
    """Hash of the parsed bar content, independent of source formatting."""
    digest = hashlib.sha256()
    digest.update(f"symbol={bars.symbol};tf={bars.timeframe_minutes};".encode())
    for name in ("ts_utc_s", "open", "high", "low", "close", "volume"):
        arr = np.ascontiguousarray(getattr(bars, name))
        digest.update(f"{name}:{arr.dtype.str}:{arr.shape};".encode())
        digest.update(arr.tobytes())
    if bars.spread_points is not None:
        arr = np.ascontiguousarray(bars.spread_points)
        digest.update(f"spread_points:{arr.dtype.str}:{arr.shape};".encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class DataManifest:
    schema_version: str
    symbol: str
    timeframe_minutes: int
    n_bars: int
    first_ts_utc: str
    last_ts_utc: str
    source_file_name: str
    source_sha256: str
    content_hash: str
    instrument_spec: dict[str, Any]
    qa_summary: dict[str, Any]
    created_at_utc: str
    generator: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def manifest_id(self) -> str:
        return f"dm_{self.content_hash[:16]}"

    def semantic_dict(self) -> dict[str, Any]:
        """Everything identity-relevant (excludes creation timestamp)."""
        d = asdict(self)
        d.pop("created_at_utc")
        return d

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["manifest_id"] = self.manifest_id
        return d


def build_manifest(
    bars: BarData,
    spec: InstrumentSpec,
    qa_report: QAReport,
    source_path: str | Path,
    extra: dict[str, Any] | None = None,
) -> DataManifest:
    source_path = Path(source_path)
    def iso(s: int | float) -> str:
        return _dt.datetime.fromtimestamp(int(s), _dt.UTC).isoformat()
    return DataManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        symbol=bars.symbol,
        timeframe_minutes=bars.timeframe_minutes,
        n_bars=bars.n_bars,
        first_ts_utc=iso(bars.ts_utc_s[0]),
        last_ts_utc=iso(bars.ts_utc_s[-1]),
        source_file_name=source_path.name,
        source_sha256=sha256_file(source_path),
        content_hash=canonical_content_hash(bars),
        instrument_spec=spec.to_dict(),
        qa_summary=qa_report.summary(),
        created_at_utc=_dt.datetime.now(_dt.UTC).isoformat(),
        generator=f"evoquant {_evoquant_version}",
        extra=extra or {},
    )


def write_manifest(manifest: DataManifest, path: str | Path) -> Path:
    """Write-once persistence.

    Re-writing an identical manifest is an idempotent no-op; any semantic
    difference raises :class:`ManifestImmutabilityError`.
    """
    path = Path(path)
    if path.exists():
        existing = load_manifest(path)
        if existing.semantic_dict() != manifest.semantic_dict():
            raise ManifestImmutabilityError(
                "Manifest already exists with different content; manifests are immutable.",
                path=str(path),
                existing_content_hash=existing.content_hash,
                new_content_hash=manifest.content_hash,
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    tmp.rename(path)
    return path


def load_manifest(path: str | Path) -> DataManifest:
    doc = json.loads(Path(path).read_text())
    doc.pop("manifest_id", None)
    return DataManifest(**doc)


def verify_manifest(
    manifest: DataManifest,
    bars: BarData,
    source_path: str | Path | None = None,
) -> None:
    """Raise :class:`ManifestVerificationError` if data no longer matches."""
    actual = canonical_content_hash(bars)
    if actual != manifest.content_hash:
        raise ManifestVerificationError(
            "Parsed bar content does not match manifest content hash "
            "(data tampered with, corrupted, or wrong file).",
            expected=manifest.content_hash,
            actual=actual,
            symbol=bars.symbol,
        )
    if bars.n_bars != manifest.n_bars:
        raise ManifestVerificationError(
            "Bar count mismatch", expected=manifest.n_bars, actual=bars.n_bars
        )
    if source_path is not None:
        actual_file = sha256_file(source_path)
        if actual_file != manifest.source_sha256:
            raise ManifestVerificationError(
                "Source file sha256 does not match manifest",
                expected=manifest.source_sha256,
                actual=actual_file,
                path=str(source_path),
            )
