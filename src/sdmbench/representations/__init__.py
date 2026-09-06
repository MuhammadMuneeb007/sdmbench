"""Representation encoders.

Importing this package registers the encoders that need no optional
dependency, and attempts the rest. A missing backend leaves that encoder
unregistered rather than breaking the import -- and the planner reports it as
unavailable rather than failing mid-run.
"""

from sdmbench.representations.base import (
    REPRESENTATIONS,
    BaseEncoder,
    EncodedModality,
    encoder_spec,
)
from sdmbench.representations.raw import (  # noqa: F401 - registration side effects
    CategoricalEncoder,
    ICAEncoder,
    IdentityEmbeddingEncoder,
    PCAEncoder,
    RawTabularEncoder,
    StandardizedTabularEncoder,
)
from sdmbench.representations.raster import (  # noqa: F401
    PatchStatisticsEncoder,
)

__all__ = [
    "REPRESENTATIONS",
    "BaseEncoder",
    "EncodedModality",
    "encoder_spec",
    "RawTabularEncoder",
    "StandardizedTabularEncoder",
    "CategoricalEncoder",
    "PCAEncoder",
    "ICAEncoder",
    "IdentityEmbeddingEncoder",
    "PatchStatisticsEncoder",
    "FOUNDATION_PROVIDERS",
]


def _register_optional() -> None:
    """Register encoders whose backends may be absent."""
    import importlib

    for module in (
        "sdmbench.representations.deep",
        "sdmbench.representations.foundation",
    ):
        try:
            importlib.import_module(module)
        except ImportError:  # pragma: no cover - optional backends
            continue
    # The raster torch encoders live alongside the always-available statistics
    # encoder, so importing the module again is cheap and idempotent.
    try:
        importlib.import_module("sdmbench.representations.raster")
    except ImportError:  # pragma: no cover
        pass


_register_optional()


def __getattr__(name: str):  # pragma: no cover - lazy accessor
    if name == "FOUNDATION_PROVIDERS":
        from sdmbench.representations.foundation import FOUNDATION_PROVIDERS

        return FOUNDATION_PROVIDERS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
