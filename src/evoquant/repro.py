"""Reproducibility metadata.

Every run artifact must be traceable to: data content hash, config hash,
code version (git commit + dirty flag), RNG seed, Python and dependency
versions. The v8 result JSON carried none of these (audit D11).
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from importlib import metadata as _im
from pathlib import Path
from typing import Any

from evoquant import __version__ as _evoquant_version
from evoquant.data.manifest import sha256_of_dict

_TRACKED_PACKAGES = ("numpy",)


def _git_info(cwd: str | Path | None = None) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
    }


@dataclass(frozen=True)
class ReproRecord:
    created_at_utc: str
    evoquant_version: str
    python_version: str
    platform: str
    packages: dict[str, str]
    git: dict[str, Any]
    seed: int
    config_hash: str
    data_content_hash: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path


def config_hash(config: dict[str, Any]) -> str:
    """Stable hash of a JSON-serializable configuration dict."""
    return sha256_of_dict(config)


def capture(
    *,
    seed: int,
    config: dict[str, Any],
    data_content_hash: str,
    repo_dir: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> ReproRecord:
    packages: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            packages[pkg] = _im.version(pkg)
        except _im.PackageNotFoundError:
            packages[pkg] = "not-installed"
    return ReproRecord(
        created_at_utc=_dt.datetime.now(_dt.UTC).isoformat(),
        evoquant_version=_evoquant_version,
        python_version=sys.version,
        platform=platform.platform(),
        packages=packages,
        git=_git_info(repo_dir),
        seed=seed,
        config_hash=config_hash(config),
        data_content_hash=data_content_hash,
        extra=extra or {},
    )
