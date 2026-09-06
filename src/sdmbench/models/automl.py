"""AutoML adapters (optional, ``sdmbench[automl]``).

The rule these adapters must obey
---------------------------------
AutoML systems normally want to own the whole experiment: they split the data,
tune, ensemble, and report a score. In a benchmark that is unacceptable -- a
system that carves its own test set out of the benchmark's training data and
reports performance on it is not comparable with anything, and one that touches
the benchmark's evaluation data has invalidated it.

So the contract here is:

* The **benchmark** owns the train/test definition. Adapters receive a
  :class:`~sdmbench.data.base.PreparedSplit` and never see ``y_test``.
* AutoML may do whatever internal model selection it likes **on the training
  partition** -- that is its whole value proposition and is legitimate.
* Compute budgets are explicit and recorded in every result row. An AutoML
  number without its time limit is uninterpretable, since the system will
  happily spend more time to score better.

Each adapter sets ``tunes_hyperparameters = True`` so the leakage auditor
records that internal selection occurred and confirms it used training data
only.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from sdmbench.data.base import PreparedSplit
from sdmbench.exceptions import SdmbenchError
from sdmbench.models.base import FitResult, ModelAdapter, register_model
from sdmbench.optional import package_version, require

__all__ = ["AutoMLAdapter", "AutoGluonAdapter", "H2OAutoMLAdapter", "FLAMLAdapter"]

TARGET = "__sdmbench_target__"


class AutoMLAdapter(ModelAdapter):
    """Base class carrying the compute-budget contract."""

    family = "automl"
    tunes_hyperparameters = True
    supports_categorical = True

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.time_limit_seconds = int(hyperparameters.get("time_limit_seconds", 3600))
        self.cpu_limit = int(hyperparameters.get("cpu_limit", 8))
        self.gpu_limit = int(hyperparameters.get("gpu_limit", 0))
        self.memory_limit_gb = hyperparameters.get("memory_limit_gb")

    def compute_budget(self) -> dict[str, Any]:
        """Budget recorded in the result row."""
        return {
            "time_limit_seconds": self.time_limit_seconds,
            "cpu_limit": self.cpu_limit,
            "gpu_limit": self.gpu_limit,
            "memory_limit_gb": self.memory_limit_gb,
        }

    def effective_hyperparameters(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in self.hyperparameters.items() if k != "seed"},
            **self.compute_budget(),
            "seed": self.seed,
        }

    @staticmethod
    def _training_frame(split: PreparedSplit) -> pd.DataFrame:
        """Training design matrix plus the target column.

        Only training rows and training labels. The test labels are not
        available to this class by construction.
        """
        frame = split.X_train.copy()
        frame[TARGET] = np.asarray(split.y_train, dtype=int)
        return frame


@register_model("autogluon", "autogluon-tabular")
class AutoGluonAdapter(AutoMLAdapter):
    """AutoGluon Tabular.

    ``TabularPredictor.fit`` performs its own internal validation split of the
    data it is given -- which here is the benchmark's training partition only,
    so that split is internal model selection, not benchmark leakage.
    """

    def check_available(self) -> None:
        require("autogluon.tabular", hint="pip install autogluon.tabular")
        super().check_available()

    def run(self, split: PreparedSplit) -> FitResult:
        import tempfile

        tabular = require("autogluon.tabular")
        train = self._training_frame(split)

        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="sdmbench-autogluon-") as tmpdir:
            predictor = tabular.TabularPredictor(
                label=TARGET,
                problem_type="binary",
                eval_metric=self.hyperparameters.get("eval_metric", "roc_auc"),
                path=tmpdir,
                verbosity=0,
            )
            predictor.fit(
                train,
                time_limit=self.time_limit_seconds,
                presets=self.hyperparameters.get("presets", "medium_quality"),
                num_cpus=self.cpu_limit,
                num_gpus=self.gpu_limit,
            )
            fit_seconds = time.perf_counter() - started

            t1 = time.perf_counter()
            proba = predictor.predict_proba(split.X_test)
            predict_seconds = time.perf_counter() - t1
            leaderboard_size = len(predictor.model_names())

        probabilities = proba[1].to_numpy() if 1 in proba.columns else proba.iloc[:, -1].to_numpy()
        return FitResult(
            probabilities=np.asarray(probabilities, dtype=float),
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            extra={"n_models_trained": leaderboard_size, "budget": self.compute_budget()},
        )

    def fit(self, X, y, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("AutoGluon fits inside run()")

    def predict_proba(self, X):  # pragma: no cover
        raise NotImplementedError("AutoGluon predicts inside run()")

    def version(self) -> str:
        return package_version("autogluon.tabular")


@register_model("h2o-automl", "h2o")
class H2OAutoMLAdapter(AutoMLAdapter):
    """H2O AutoML.

    H2O runs a JVM cluster. The adapter starts one, uses it, and shuts it down,
    so a long benchmark does not leak JVM processes across hundreds of species.
    """

    def check_available(self) -> None:
        require("h2o", hint="H2O also needs a Java runtime (Java 8+).")
        super().check_available()

    def run(self, split: PreparedSplit) -> FitResult:
        h2o = require("h2o")
        from h2o.automl import H2OAutoML

        train = self._training_frame(split)
        started = time.perf_counter()
        h2o.init(
            nthreads=self.cpu_limit,
            max_mem_size=f"{int(self.memory_limit_gb)}G" if self.memory_limit_gb else None,
            verbose=False,
        )
        try:
            h2o_train = h2o.H2OFrame(train)
            h2o_train[TARGET] = h2o_train[TARGET].asfactor()
            features = [c for c in train.columns if c != TARGET]

            automl = H2OAutoML(
                max_runtime_secs=self.time_limit_seconds,
                seed=self.seed,
                sort_metric="AUC",
                nfolds=int(self.hyperparameters.get("nfolds", 5)),
            )
            automl.train(x=features, y=TARGET, training_frame=h2o_train)
            fit_seconds = time.perf_counter() - started

            t1 = time.perf_counter()
            h2o_test = h2o.H2OFrame(split.X_test)
            prediction = automl.leader.predict(h2o_test).as_data_frame()
            predict_seconds = time.perf_counter() - t1

            column = "p1" if "p1" in prediction.columns else prediction.columns[-1]
            probabilities = prediction[column].to_numpy()
            leader = str(automl.leader.model_id)
        finally:
            try:
                h2o.cluster().shutdown()
            except Exception:  # noqa: BLE001, S110 - shutdown is best-effort
                pass

        return FitResult(
            probabilities=np.asarray(probabilities, dtype=float),
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            extra={"leader_model": leader, "budget": self.compute_budget()},
        )

    def fit(self, X, y, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("H2O fits inside run()")

    def predict_proba(self, X):  # pragma: no cover
        raise NotImplementedError("H2O predicts inside run()")

    def version(self) -> str:
        return package_version("h2o")


@register_model("flaml")
class FLAMLAdapter(AutoMLAdapter):
    """FLAML -- fast, cheap AutoML with an explicit time budget."""

    supports_categorical = False

    def check_available(self) -> None:
        require("flaml")
        super().check_available()

    def run(self, split: PreparedSplit) -> FitResult:
        flaml = require("flaml")
        X_train, y_train, X_test, _ = split.as_arrays()

        started = time.perf_counter()
        automl = flaml.AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="classification",
            metric=self.hyperparameters.get("metric", "roc_auc"),
            time_budget=self.time_limit_seconds,
            n_jobs=self.cpu_limit,
            seed=self.seed,
            verbose=0,
        )
        fit_seconds = time.perf_counter() - started

        t1 = time.perf_counter()
        proba = automl.predict_proba(X_test)
        predict_seconds = time.perf_counter() - t1
        probabilities = proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba

        return FitResult(
            probabilities=np.asarray(probabilities, dtype=float),
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            extra={
                "best_estimator": str(automl.best_estimator),
                "best_config": dict(automl.best_config or {}),
                "budget": self.compute_budget(),
            },
        )

    def fit(self, X, y, **kwargs: Any):  # pragma: no cover
        raise NotImplementedError("FLAML fits inside run()")

    def predict_proba(self, X):  # pragma: no cover
        raise NotImplementedError("FLAML predicts inside run()")

    def version(self) -> str:
        return package_version("flaml")
