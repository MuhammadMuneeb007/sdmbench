r"""Cache layout and path resolution.

sdmbench keeps everything it downloads under a single cache root so that a
benchmark can be re-run, audited, or deleted wholesale::

    ~/.cache/sdmbench/
        datasets/disdat/<version>/...      standardised Parquet + manifest
        datasets/disdat/raw/...            untouched source payloads
        models/tabpfn_sdm/...              Hugging Face checkpoints
        upstream/tabpfn_sdm/...            cloned upstream repositories
        runs/<run_id>/...                  results, manifests, checkpoints

The root can be overridden with ``$SDMBENCH_CACHE``; on Windows the platform
default is ``%LOCALAPPDATA%\sdmbench\Cache``.

Source data is never overwritten. Raw payloads land in ``raw/`` and are treated
as immutable; every derived artefact is written beside them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = [
    "cache_root",
    "datasets_dir",
    "models_dir",
    "upstream_dir",
    "runs_dir",
    "ensure_dir",
]

_ENV_VAR = "SDMBENCH_CACHE"


def cache_root() -> Path:
    """Return the sdmbench cache root, honouring ``$SDMBENCH_CACHE``."""
    override = os.environ.get(_ENV_VAR)
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "sdmbench" / "Cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "sdmbench"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base_dir = Path(xdg) if xdg else Path.home() / ".cache"
    return base_dir / "sdmbench"


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def datasets_dir(name: str | None = None) -> Path:
    """Directory holding standardised datasets."""
    p = cache_root() / "datasets"
    return p / name if name else p


def models_dir(name: str | None = None) -> Path:
    """Directory holding downloaded model checkpoints."""
    p = cache_root() / "models"
    return p / name if name else p


def upstream_dir(name: str | None = None) -> Path:
    """Directory holding cloned upstream repositories."""
    p = cache_root() / "upstream"
    return p / name if name else p


def runs_dir(run_id: str | None = None) -> Path:
    """Directory holding benchmark run outputs."""
    p = cache_root() / "runs"
    return p / run_id if run_id else p
