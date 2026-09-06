"""Optional dependency handling.

sdmbench's core (schema, splits, metrics, aggregation, reporting) depends only
on numpy/pandas/scipy/scikit-learn/pyarrow. Everything heavier -- torch,
torch-geometric, tabpfn, xgboost, the AutoML frameworks, rasterio, R -- is an
optional extra.

The rule enforced here: *never* let a missing optional dependency raise a bare
``ImportError`` from deep inside an adapter. Adapters call :func:`require` and
get a :class:`~sdmbench.exceptions.MissingDependencyError`, which the run
engine converts into a ``SKIPPED_DEPENDENCY`` result row so the rest of the
benchmark continues.
"""

from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
from types import ModuleType
from typing import Any

from sdmbench.exceptions import MissingDependencyError, NoGpuError

__all__ = [
    "require",
    "try_import",
    "have",
    "package_version",
    "torch_device",
    "resolve_device",
]

#: Maps an importable module name onto the ``pip install "sdmbench[extra]"``
#: extra that provides it, and any extra guidance worth printing.
_EXTRA_FOR_MODULE: dict[str, tuple[str, str | None]] = {
    "torch": ("deep", None),
    "torch_geometric": ("graph", "torch-geometric also needs a matching torch build."),
    "tabpfn": ("tabpfn", "TabPFN is distributed under the Prior Labs License v1.1."),
    "huggingface_hub": ("tabpfn", None),
    "xgboost": ("boosting", None),
    "lightgbm": ("boosting", None),
    "catboost": ("boosting", None),
    "autogluon": ("automl", None),
    "h2o": ("automl", None),
    "flaml": ("automl", None),
    "rasterio": ("raster", None),
    "rioxarray": ("raster", None),
    "xarray": ("raster", None),
    "geopandas": ("raster", None),
    "shapely": ("raster", None),
    "pyproj": ("raster", None),
    "matplotlib": ("plots", None),
    "matplotlib.pyplot": ("plots", None),
}


def try_import(module: str) -> ModuleType | None:
    """Import ``module``, returning ``None`` if it is unavailable."""
    try:
        return importlib.import_module(module)
    except ImportError:
        return None


def have(module: str) -> bool:
    """Return whether ``module`` can be imported."""
    return try_import(module) is not None


def require(module: str, *, hint: str | None = None) -> ModuleType:
    """Import ``module`` or raise a :class:`MissingDependencyError`.

    Parameters
    ----------
    module:
        Importable module name, e.g. ``"xgboost"``.
    hint:
        Extra guidance appended to the error message.
    """
    mod = try_import(module)
    if mod is not None:
        return mod
    extra, default_hint = _EXTRA_FOR_MODULE.get(module, (None, None))
    raise MissingDependencyError(module, extra=extra, hint=hint or default_hint)


def package_version(name: str, default: str = "not-installed") -> str:
    """Return the installed version of a distribution, or ``default``."""
    try:
        return importlib_metadata.version(name)
    except Exception:  # noqa: BLE001 - metadata lookup can fail many ways
        mod = try_import(name)
        return getattr(mod, "__version__", default) if mod else default


def torch_device(device: str = "auto") -> Any:
    """Resolve a torch device object, requiring torch.

    ``"auto"`` picks CUDA when available, otherwise MPS, otherwise CPU.
    """
    torch = require("torch")
    return torch.device(resolve_device(device))


def resolve_device(device: str = "auto", *, require_gpu: bool = False) -> str:
    """Resolve a device string without necessarily importing torch.

    Parameters
    ----------
    device:
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"cuda:N"``, ``"mps"``.
    require_gpu:
        If True and no accelerator is available, raise
        :class:`~sdmbench.exceptions.NoGpuError` so the caller becomes a
        ``SKIPPED_NO_GPU`` row rather than silently running 100x slower.

    Returns
    -------
    str
        A concrete device string.
    """
    torch = try_import("torch")
    if device != "auto":
        if device.startswith("cuda"):
            if torch is None or not torch.cuda.is_available():
                if require_gpu:
                    raise NoGpuError(f"device={device!r} requested but CUDA is not available")
                return "cpu"
        return device

    if torch is not None:
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    if require_gpu:
        raise NoGpuError("no CUDA or MPS device is available on this machine")
    return "cpu"
