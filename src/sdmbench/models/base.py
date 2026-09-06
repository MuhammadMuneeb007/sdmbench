"""The model adapter interface and registry.

Every algorithm sdmbench evaluates -- a scikit-learn classifier, an R MaxEnt, a
finetuned TabPFN, a graph network, an AutoML system -- is wrapped in a
:class:`ModelAdapter`. The adapter's job is narrow on purpose:

    take a :class:`~sdmbench.data.base.PreparedSplit`, fit on the training part,
    return a probability of presence for each test row.

It does **not** choose the split, the preprocessing, the metrics, or the test
set. The benchmark owns those. That separation is what makes the leaderboard a
comparison rather than a collection of anecdotes.

Adapters must never touch ``split.y_test``. The one legitimate use of test
*features* is transductive inference (some graph methods), which must be
declared via :attr:`ModelAdapter.uses_test_features` so the auditor can check
the graph accordingly.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from sdmbench.data.base import PreparedSplit
from sdmbench.exceptions import SdmbenchError

__all__ = [
    "ModelStatus",
    "ModelAdapter",
    "SklearnAdapter",
    "FitResult",
    "register_model",
    "get_model",
    "list_models",
    "model_info",
    "MODEL_REGISTRY",
    "MODEL_SETS",
]


class ModelStatus(str, enum.Enum):
    """Outcome of one model job.

    Non-``OK`` statuses still produce a result row. A benchmark that quietly
    omitted every model it could not run would overstate the coverage of the
    comparison.
    """

    OK = "OK"
    FAILED = "FAILED"
    SKIPPED_DEPENDENCY = "SKIPPED_DEPENDENCY"
    SKIPPED_NO_GPU = "SKIPPED_NO_GPU"
    SKIPPED_LICENSE = "SKIPPED_LICENSE"
    SKIPPED_DATA = "SKIPPED_DATA"
    SKIPPED_EXISTING = "SKIPPED_EXISTING"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass
class FitResult:
    """Predictions and cost from one fit/predict cycle."""

    probabilities: np.ndarray
    fit_seconds: float = 0.0
    predict_seconds: float = 0.0
    peak_memory_mb: float | None = None
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_seconds(self) -> float:
        return self.fit_seconds + self.predict_seconds


class ModelAdapter:
    """Base class for every model.

    Subclasses implement :meth:`fit` and :meth:`predict_proba`, or override
    :meth:`run` when the underlying library couples fitting and prediction (as
    TabPFN's in-context learning does).

    Class attributes
    ----------------
    name:
        Registry key, e.g. ``"random-forest"``.
    family:
        Grouping used in reports: ``published``, ``classical``, ``boosting``,
        ``foundation``, ``graph``, ``neural``, ``automl``.
    requires_gpu:
        When True and no accelerator is present, the job becomes
        ``SKIPPED_NO_GPU``.
    supports_categorical:
        Whether the adapter consumes native categorical columns. When False,
        the runner passes integer-coded arrays instead.
    uses_test_features:
        Declares transductive access to test *features* (never labels).
    """

    name: str = "model"
    family: str = "classical"
    requires_gpu: bool = False
    supports_categorical: bool = False
    uses_test_features: bool = False
    #: Set by subclasses that perform internal model selection, so the leakage
    #: auditor can confirm it happened on training data only.
    tunes_hyperparameters: bool = False

    def __init__(self, **hyperparameters: Any) -> None:
        self.hyperparameters = dict(hyperparameters)
        self.seed = int(self.hyperparameters.get("seed", 32639))
        self._fitted: Any = None

    # ---------------------------------------------------------------- fitting --
    def fit(self, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> ModelAdapter:
        raise NotImplementedError(f"{type(self).__name__} must implement fit()")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return the probability of presence for each row of ``X``.

        Implementations may return either an ``(n,)`` vector or an ``(n, 2)``
        matrix; :meth:`run` normalises to a vector.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement predict_proba()")

    def run(self, split: PreparedSplit) -> FitResult:
        """Fit on the training part of ``split`` and predict its test part.

        The default implementation times :meth:`fit` and :meth:`predict_proba`
        separately, which is what populates ``fit_time_seconds`` and
        ``predict_time_seconds`` in the results table.
        """
        X_train, X_test = self.prepare_features(split)
        y_train = np.asarray(split.y_train, dtype=int)

        t0 = time.perf_counter()
        self.fit(X_train, y_train, split=split)
        fit_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        probabilities = self.predict_proba(X_test)
        predict_seconds = time.perf_counter() - t1

        return FitResult(
            probabilities=_as_presence_probability(probabilities, n_expected=len(X_test)),
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            device=self.device_used(),
        )

    def prepare_features(self, split: PreparedSplit) -> tuple[Any, Any]:
        """Convert a split into whatever this adapter's backend expects.

        Adapters that handle native categoricals get the DataFrames; the rest
        get numeric arrays with categoricals integer-coded on training levels.
        """
        if self.supports_categorical:
            return split.X_train, split.X_test
        X_train, _, X_test, _ = split.as_arrays()
        return X_train, X_test

    def device_used(self) -> str:
        return "cpu"

    # --------------------------------------------------------------- metadata --
    def get_metadata(self) -> dict[str, Any]:
        """Everything needed to identify this model configuration in results."""
        return {
            "model_name": self.name,
            "model_family": self.family,
            "model_version": self.version(),
            "hyperparameters": self.effective_hyperparameters(),
            "requires_gpu": self.requires_gpu,
            "supports_categorical": self.supports_categorical,
            "uses_test_features": self.uses_test_features,
            "tunes_hyperparameters": self.tunes_hyperparameters,
        }

    def effective_hyperparameters(self) -> dict[str, Any]:
        """Hyperparameters actually used, including resolved defaults."""
        return dict(self.hyperparameters)

    def version(self) -> str:
        """Version of the underlying implementation."""
        return ""

    def check_available(self) -> None:
        """Raise a skippable error if this adapter cannot run here.

        Called before the job starts so that an unavailable model costs nothing
        and produces a clear ``SKIPPED_*`` row.
        """
        if self.requires_gpu:
            from sdmbench.optional import resolve_device

            resolve_device(self.hyperparameters.get("device", "auto"), require_gpu=True)


class SklearnAdapter(ModelAdapter):
    """Adapter for any scikit-learn-compatible estimator.

    sdmbench never reimplements a learning algorithm. This class exists so that
    "our KNN" is literally ``sklearn.neighbors.KNeighborsClassifier`` and a
    reviewer can verify that by reading twenty lines.
    """

    #: Subclasses set this to a zero-argument factory returning the estimator.
    estimator_factory: Callable[..., Any] | None = None
    #: Distribution name used for the recorded ``model_version``.
    version_package: str = "scikit-learn"

    def build_estimator(self) -> Any:
        if self.estimator_factory is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set estimator_factory or override build_estimator()"
            )
        return self.estimator_factory(**self.estimator_kwargs())

    def estimator_kwargs(self) -> dict[str, Any]:
        """Keyword arguments passed to the estimator constructor."""
        return {k: v for k, v in self.hyperparameters.items() if k not in {"seed", "device"}}

    def fit(self, X, y, **kwargs: Any) -> SklearnAdapter:
        self._fitted = self.build_estimator()
        self._fitted.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._fitted is None:
            raise SdmbenchError(f"{self.name} was not fitted before predict_proba()")
        if hasattr(self._fitted, "predict_proba"):
            proba = self._fitted.predict_proba(X)
            classes = list(getattr(self._fitted, "classes_", [0, 1]))
            if proba.ndim == 2 and 1 in classes:
                return proba[:, classes.index(1)]
            return proba
        if hasattr(self._fitted, "decision_function"):
            # Map an unbounded margin onto (0, 1). Rank-based metrics such as
            # ROC-AUC are unaffected; calibration metrics obviously are.
            scores = np.asarray(self._fitted.decision_function(X), dtype=float)
            return 1.0 / (1.0 + np.exp(-scores))
        raise SdmbenchError(f"{self.name}: estimator exposes neither predict_proba nor decision_function")

    def effective_hyperparameters(self) -> dict[str, Any]:
        params = dict(self.hyperparameters)
        if self._fitted is not None and hasattr(self._fitted, "get_params"):
            try:
                params = {**self._fitted.get_params(deep=False), **params}
            except Exception:  # noqa: BLE001
                pass
        return {k: _jsonable(v) for k, v in params.items()}

    def version(self) -> str:
        from sdmbench.optional import package_version

        return package_version(self.version_package)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type[ModelAdapter]] = {}

#: Convenience aliases usable wherever a model list is accepted.
MODEL_SETS: dict[str, tuple[str, ...]] = {
    "published": ("maxnet", "random-forest", "brt", "gam", "tabpfn-sdm"),
    "standard": ("logistic-regression", "knn", "random-forest-sklearn", "xgboost"),
    "classical": (
        "logistic-regression",
        "knn",
        "decision-tree",
        "random-forest-sklearn",
        "extra-trees",
        "svm",
        "gaussian-nb",
        "gradient-boosting",
        "adaboost",
    ),
    "boosting": ("xgboost", "lightgbm", "catboost"),
    "tabpfn": (
        "tabpfn-default",
        "tabpfn-balanced",
        "tabpfn-ss",
        "tabpfn-sdm",
        "tabpfn-sdm-spatial",
    ),
    "graph": ("gcn", "graphsage", "gatv2", "graph-transformer"),
    "neural": ("mlp", "residual-mlp", "wide-and-deep", "ft-transformer"),
    "automl": ("autogluon", "h2o-automl", "flaml"),
}


def register_model(
    name: str, *aliases: str
) -> Callable[[type[ModelAdapter]], type[ModelAdapter]]:
    """Class decorator registering an adapter under ``name`` and any aliases."""

    def decorator(cls: type[ModelAdapter]) -> type[ModelAdapter]:
        cls.name = name
        for key in (name, *aliases):
            if key in MODEL_REGISTRY and MODEL_REGISTRY[key] is not cls:
                raise ValueError(f"model name {key!r} is already registered")
            MODEL_REGISTRY[key] = cls
        return cls

    return decorator


def _load_all_adapters() -> None:
    """Import every adapter module so the registry is populated.

    Import failures are swallowed on purpose: a module whose *optional* backend
    is missing must not prevent the rest of the registry from loading. Adapters
    themselves raise a skippable error at ``check_available()`` time.
    """
    import importlib

    for module in (
        "sdmbench.models.sklearn_models",
        "sdmbench.models.boosting",
        "sdmbench.models.tabpfn",
        "sdmbench.models.r_models",
        "sdmbench.models.graph",
        "sdmbench.models.neural",
        "sdmbench.models.automl",
    ):
        try:
            importlib.import_module(module)
        except ImportError:  # pragma: no cover - optional backends
            continue


def get_model(name: str, **hyperparameters: Any) -> ModelAdapter:
    """Instantiate a registered adapter by name."""
    if not MODEL_REGISTRY:
        _load_all_adapters()
    key = name.strip().lower()
    cls = MODEL_REGISTRY.get(key) or MODEL_REGISTRY.get(key.replace("_", "-"))
    if cls is None:
        raise KeyError(
            f"unknown model {name!r}. Available: {', '.join(list_models())}\n"
            f"  Model set aliases: {', '.join(sorted(MODEL_SETS))}"
        )
    return cls(**hyperparameters)


def list_models() -> list[str]:
    """Every registered model name, sorted."""
    if not MODEL_REGISTRY:
        _load_all_adapters()
    return sorted(MODEL_REGISTRY)


def model_info() -> dict[str, dict[str, Any]]:
    """Family and requirements of every registered model, for ``sdmbench models``."""
    if not MODEL_REGISTRY:
        _load_all_adapters()
    seen: dict[str, dict[str, Any]] = {}
    for key, cls in MODEL_REGISTRY.items():
        seen[key] = {
            "family": cls.family,
            "requires_gpu": cls.requires_gpu,
            "supports_categorical": cls.supports_categorical,
            "class": f"{cls.__module__}.{cls.__qualname__}",
        }
    return dict(sorted(seen.items()))


def resolve_model_names(names: Iterable[str]) -> list[str]:
    """Expand model-set aliases into concrete model names, preserving order."""
    out: list[str] = []
    for raw in names:
        key = raw.strip().lower()
        if key in MODEL_SETS:
            out.extend(MODEL_SETS[key])
        elif key == "all":
            out.extend(list_models())
        else:
            out.append(key)
    seen: set[str] = set()
    return [n for n in out if not (n in seen or seen.add(n))]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_presence_probability(proba: np.ndarray, *, n_expected: int) -> np.ndarray:
    """Normalise a prediction array to a 1-D probability-of-presence vector."""
    arr = np.asarray(proba, dtype=float)
    if arr.ndim == 2:
        if arr.shape[1] == 2:
            arr = arr[:, 1]
        elif arr.shape[1] == 1:
            arr = arr[:, 0]
        else:
            raise SdmbenchError(f"expected 1 or 2 prediction columns, got {arr.shape[1]}")
    arr = arr.ravel()
    if len(arr) != n_expected:
        raise SdmbenchError(f"model returned {len(arr)} predictions for {n_expected} test rows")
    return arr


def _jsonable(value: Any) -> Any:
    """Coerce a hyperparameter into something JSON can hold."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)
