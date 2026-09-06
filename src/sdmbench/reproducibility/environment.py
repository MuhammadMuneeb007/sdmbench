"""Capture of the computational environment.

Recorded once per run into ``run_manifest.json`` and, in condensed form, into
every result row. The point is that a number produced today can be interpreted
in five years: which package versions, which CUDA, which R, which commit.

Nothing here installs, upgrades, or modifies anything -- it only observes.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from sdmbench.optional import package_version, try_import

__all__ = [
    "capture_environment",
    "python_packages",
    "gpu_info",
    "r_info",
    "git_commit",
    "PACKAGES_OF_INTEREST",
]

#: Packages whose versions can change a benchmark number and are therefore
#: always recorded, installed or not.
PACKAGES_OF_INTEREST = (
    "sdmbench",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "pyarrow",
    "xgboost",
    "lightgbm",
    "catboost",
    "torch",
    "torch-geometric",
    "tabpfn",
    "huggingface-hub",
    "autogluon.tabular",
    "h2o",
    "flaml",
    "rasterio",
    "rioxarray",
    "xarray",
    "geopandas",
    "shapely",
    "pyproj",
    "pyyaml",
)


def python_packages() -> dict[str, str]:
    """Version of each package of interest (``not-installed`` when absent)."""
    return {name: package_version(name) for name in PACKAGES_OF_INTEREST}


def gpu_info() -> dict[str, Any]:
    """Describe available accelerators without requiring torch."""
    info: dict[str, Any] = {
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
        "mps_available": False,
    }
    torch = try_import("torch")
    if torch is not None:
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
            info["cuda_version"] = getattr(torch.version, "cuda", None)
            if info["cuda_available"]:
                info["device_count"] = int(torch.cuda.device_count())
                info["devices"] = [
                    torch.cuda.get_device_name(i) for i in range(info["device_count"])
                ]
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            info["mps_available"] = bool(mps.is_available()) if mps is not None else False
        except Exception as exc:  # noqa: BLE001 - never let probing break a run
            info["error"] = str(exc)
        return info

    # No torch: fall back to nvidia-smi so CPU-only installs still record a GPU.
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if out.returncode == 0 and out.stdout.strip():
                devices = [ln.strip() for ln in out.stdout.strip().splitlines() if ln.strip()]
                info["devices"] = devices
                info["device_count"] = len(devices)
                info["cuda_available"] = True
                info["source"] = "nvidia-smi"
        except Exception:  # noqa: BLE001, S110 - probing is best-effort
            pass
    return info


def r_info(r_home: str | None = None) -> dict[str, Any]:
    """Locate R and report its version, without installing anything."""
    rscript = r_home or os.environ.get("SDMBENCH_RSCRIPT") or shutil.which("Rscript")
    info: dict[str, Any] = {"available": False, "rscript": rscript, "version": None}
    if not rscript:
        return info
    try:
        out = subprocess.run(
            [rscript, "-e", "cat(paste(R.version$major, R.version$minor, sep='.'))"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if out.returncode == 0:
            info["available"] = True
            info["version"] = out.stdout.strip()
        else:
            info["error"] = (out.stderr or "").strip()[:500]
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    return info


def git_commit(path: str | Path | None = None) -> str:
    """Return the current git commit of ``path``, or an empty string."""
    cwd = Path(path) if path else Path(__file__).resolve().parents[3]
    git = shutil.which("git")
    if not git:
        return ""
    try:
        out = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if out.returncode == 0:
            commit = out.stdout.strip()
            dirty = subprocess.run(
                [git, "status", "--porcelain"],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if dirty.returncode == 0 and dirty.stdout.strip():
                commit += "-dirty"
            return commit
    except Exception:  # noqa: BLE001, S110
        pass
    return ""


def capture_environment(*, include_r: bool = True) -> dict[str, Any]:
    """Full environment snapshot for ``run_manifest.json``."""
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "packages": python_packages(),
        "gpu": gpu_info(),
        "git_commit": git_commit(),
        "env_vars": {
            k: os.environ.get(k)
            for k in ("SDMBENCH_CACHE", "SDMBENCH_RSCRIPT", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")
            if os.environ.get(k) is not None
        },
    }
    if include_r:
        env["r"] = r_info()
    return env
