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
import shutil
import sys
from pathlib import Path

__all__ = [
    "cache_root",
    "datasets_dir",
    "models_dir",
    "upstream_dir",
    "runs_dir",
    "ensure_dir",
    "check_cache_writable",
    "free_space_mb",
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


#: ``errno`` values that mean "this filesystem will not take your data".
#: EDQUOT (122) is the one that bites on HPC clusters, where ``$HOME`` is a
#: small quota-limited volume and the real space is on a scratch filesystem.
_OUT_OF_SPACE = {28, 122}  # ENOSPC, EDQUOT


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and parents) if needed and return it.

    Raises :class:`~sdmbench.exceptions.CacheError` with an actionable message
    when the filesystem is full or over quota, rather than letting a bare
    ``OSError: [Errno 122] Disk quota exceeded`` surface from deep inside a
    fetch.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno in _OUT_OF_SPACE:
            raise _cache_space_error(path, exc) from exc
        raise
    return path


def _cache_space_error(path: Path, exc: OSError):
    """Build the 'your cache volume is full' error, with the fix."""
    from sdmbench.exceptions import CacheError

    reason = "disk quota exceeded" if exc.errno == 122 else "no space left on device"
    root = cache_root()
    return CacheError(
        f"cannot write to the sdmbench cache: {reason}\n"
        f"  path: {path}\n"
        f"  cache root: {root}\n"
        "\n"
        "  The cache holds datasets, model checkpoints and run outputs, so it needs\n"
        "  real space -- not a quota-limited home directory. Point it somewhere with\n"
        "  room:\n"
        "\n"
        "      export SDMBENCH_CACHE=/path/to/scratch/sdmbench     # bash/zsh\n"
        "      setx SDMBENCH_CACHE C:\\scratch\\sdmbench            # Windows\n"
        "\n"
        "  On an HPC cluster this is usually your project or scratch filesystem\n"
        "  (/scratch/$USER, /data/<group>/$USER, $TMPDIR), not $HOME.\n"
        "  Add the export to your ~/.bashrc so it survives new sessions.\n"
        "\n"
        "  Check the current setting with:  sdmbench env check"
    )


def check_cache_writable(path: Path | None = None) -> tuple[bool, str]:
    """Report whether the cache root can actually be written to.

    Used by ``sdmbench env check`` so a full or unwritable cache is diagnosed
    up front rather than several minutes into a fetch.
    """
    root = Path(path) if path else cache_root()
    probe = root / ".sdmbench-write-probe"
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"ok")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        reason = {
            28: "no space left on device",
            122: "disk quota exceeded",
            13: "permission denied",
            30: "read-only filesystem",
        }.get(exc.errno, exc.strerror or str(exc))
        return False, reason
    return True, ""


def free_space_mb(path: Path | None = None) -> float | None:
    """Free space at the cache root in MB, or ``None`` if it cannot be read."""
    root = Path(path) if path else cache_root()
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free / 1e6
    except OSError:
        return None


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
