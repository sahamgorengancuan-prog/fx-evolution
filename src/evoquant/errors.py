"""Structured errors.

Every error that aborts a pipeline stage must be representable as a
structured artifact (JSON) so failures are auditable, not just printed.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any


class EvoquantError(Exception):
    """Base error. Carries a machine-readable payload."""

    code = "EVOQUANT_ERROR"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_artifact(self) -> dict[str, Any]:
        return {
            "error_code": self.code,
            "message": self.message,
            "details": self.details,
            "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        }

    def write_artifact(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%S%f")
        path = directory / f"error_{self.code}_{ts}.json"
        path.write_text(json.dumps(self.to_artifact(), indent=2, sort_keys=True))
        return path


class MissingMetadataError(EvoquantError):
    """A required contract/cost metadata field is absent. Fail closed."""

    code = "MISSING_METADATA"


class DataFormatError(EvoquantError):
    """Source file cannot be parsed into the expected schema."""

    code = "DATA_FORMAT"


class DataQualityError(EvoquantError):
    """Hard data-quality check failed."""

    code = "DATA_QUALITY"


class ManifestError(EvoquantError):
    code = "MANIFEST"


class ManifestImmutabilityError(ManifestError):
    """Attempt to overwrite an existing manifest with different content."""

    code = "MANIFEST_IMMUTABLE"


class ManifestVerificationError(ManifestError):
    """Data on disk no longer matches its manifest (tampering/corruption)."""

    code = "MANIFEST_VERIFY"


class SplitConfigError(EvoquantError):
    code = "SPLIT_CONFIG"


class SplitInvariantError(EvoquantError):
    """A generated split plan violates a leakage invariant."""

    code = "SPLIT_INVARIANT"


class IllegalTransitionError(EvoquantError):
    code = "ILLEGAL_TRANSITION"


class SealViolationError(EvoquantError):
    """Attempt to alter sealed experiment inputs."""

    code = "SEAL_VIOLATION"


class LockboxError(EvoquantError):
    code = "LOCKBOX"


class LockboxAccessError(LockboxError):
    """Lockbox data requested in a state that does not permit it."""

    code = "LOCKBOX_ACCESS"


class LockboxAlreadyOpenError(LockboxError):
    """The one-shot lockbox open was attempted a second time."""

    code = "LOCKBOX_ALREADY_OPEN"


class LockboxSealMismatchError(LockboxError):
    """Sealed hashes do not match the data/plan presented at open time."""

    code = "LOCKBOX_SEAL_MISMATCH"
