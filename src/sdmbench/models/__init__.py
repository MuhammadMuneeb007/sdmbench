"""Model adapters.

Importing this package only pulls in the interface and the registry; individual
backends are imported lazily by :func:`~sdmbench.models.base.get_model` so that
``import sdmbench`` never requires torch, R, or an AutoML framework.
"""

from sdmbench.models.base import (
    MODEL_REGISTRY,
    MODEL_SETS,
    FitResult,
    ModelAdapter,
    ModelStatus,
    SklearnAdapter,
    get_model,
    list_models,
    model_info,
    register_model,
    resolve_model_names,
)
from sdmbench.models.ensemble import PresenceBackgroundEnsemble

__all__ = [
    "MODEL_REGISTRY",
    "MODEL_SETS",
    "FitResult",
    "ModelAdapter",
    "ModelStatus",
    "SklearnAdapter",
    "PresenceBackgroundEnsemble",
    "get_model",
    "list_models",
    "model_info",
    "register_model",
    "resolve_model_names",
]
