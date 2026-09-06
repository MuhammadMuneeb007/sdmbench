"""Adapters for the published baselines, executed in R.

Dinnage & Warren (2026) sec. 2.2.1 states that for each method they "followed
the best-practice configurations specified in Valavi et al. (2022), ensuring
that our comparison reflects each algorithm's optimized rather than naive
performance". Those configurations are properties of specific R packages:

===============  ==========================================================
``maxnet``       MaxEnt via elastic-net GLM, cloglog output
``brt``          ``dismo::gbm.step()`` with CV-selected tree count
``random-forest``  ``randomForest`` with per-class down-sampling
``gam``          ``mgcv`` penalised thin-plate splines, REML
===============  ==========================================================

Reproducing these in Python would change the numbers. In strict reproduction
mode sdmbench therefore calls the actual R implementations through
:mod:`sdmbench.rbridge`. Where R is unavailable the job becomes
``SKIPPED_DEPENDENCY`` -- it is never silently swapped for a scikit-learn
look-alike.

The category-handling contract also comes from the paper (sec. 2.1.2):
"MaxNet, BRT, Random Forest, and GAM all received factor variables as native R
factors without conversion", so these adapters pass DataFrames with the
categorical columns intact.
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from sdmbench.data.base import PreparedSplit
from sdmbench.exceptions import MissingDependencyError, SdmbenchError
from sdmbench.models.base import FitResult, ModelAdapter, register_model
from sdmbench.rbridge.runner import RRunner

__all__ = [
    "RModelAdapter",
    "MaxnetAdapter",
    "BRTAdapter",
    "RandomForestAdapter",
    "GAMAdapter",
]


class RModelAdapter(ModelAdapter):
    """Base class for a model whose fit and predict happen inside R.

    R does the whole cycle in one invocation because keeping a fitted R object
    alive between two Python calls would mean an embedded interpreter and a far
    heavier dependency. :meth:`run` is therefore the entry point, and the R
    script reports its own fit/predict timings so the results table separates
    them honestly.
    """

    family = "published"
    supports_categorical = True

    #: R script under ``rbridge/scripts/``.
    script: ClassVar[str] = ""
    #: R packages the script needs.
    r_packages: ClassVar[tuple[str, ...]] = ()

    def __init__(self, **hyperparameters: Any) -> None:
        super().__init__(**hyperparameters)
        self.runner = RRunner(
            hyperparameters.get("rscript"),
            timeout=int(hyperparameters.get("timeout", 3600)),
        )
        self._r_versions: dict[str, str | None] = {}

    def check_available(self) -> None:
        self.runner.require_available()
        self.runner.require_packages(self.r_packages)
        super().check_available()

    def payload(self, split: PreparedSplit) -> dict[str, Any]:
        """Parameters sent to the R script; subclasses extend this."""
        return {
            "predictors": list(split.feature_names),
            "categorical": list(split.categorical_features),
            "seed": self.seed,
            **{k: v for k, v in self.hyperparameters.items() if k not in {"rscript", "timeout"}},
        }

    def run(self, split: PreparedSplit) -> FitResult:
        if not self.script:
            raise SdmbenchError(f"{type(self).__name__} must set `script`")

        train = split.X_train.copy()
        train["occ"] = np.asarray(split.y_train, dtype=int)
        test = split.X_test.copy()
        # The R side never receives test labels -- only the covariates it needs
        # in order to predict.

        started = time.perf_counter()
        result = self.runner.run_script(
            self.script,
            self.payload(split),
            frames={"train": train, "test": test},
        )
        wall_seconds = time.perf_counter() - started
        data = result.require()

        predictions = np.asarray(data.get("predictions", []), dtype=float)
        if len(predictions) != len(test):
            raise SdmbenchError(
                f"{self.name}: R returned {len(predictions)} predictions for {len(test)} test rows"
            )

        self._r_versions[self.name] = data.get("model_version")
        self._resolved_hyperparameters = data.get("hyperparameters", {})

        fit_seconds = float(data.get("fit_seconds", 0.0) or 0.0)
        predict_seconds = float(data.get("predict_seconds", 0.0) or 0.0)
        return FitResult(
            probabilities=predictions,
            fit_seconds=fit_seconds,
            predict_seconds=predict_seconds,
            device="cpu",
            extra={
                # Wall time exceeds fit+predict by the R start-up and CSV
                # exchange cost; recording both keeps the runtime column honest.
                "r_wall_seconds": wall_seconds,
                "r_overhead_seconds": max(0.0, wall_seconds - fit_seconds - predict_seconds),
            },
        )

    def fit(self, X, y, **kwargs: Any):  # pragma: no cover - run() is the entry point
        raise NotImplementedError(f"{self.name} fits inside run() via the R bridge")

    def predict_proba(self, X):  # pragma: no cover
        raise NotImplementedError(f"{self.name} predicts inside run() via the R bridge")

    def effective_hyperparameters(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in self.hyperparameters.items() if k not in {"rscript", "timeout"}},
            **getattr(self, "_resolved_hyperparameters", {}),
        }

    def version(self) -> str:
        return str(self._r_versions.get(self.name) or "")


@register_model("maxnet", "maxent")
class MaxnetAdapter(RModelAdapter):
    """MaxEnt via the ``maxnet`` R package.

    regmult = 1, feature classes ``lqpht``, cloglog output (paper sec. 2.2.1).
    """

    script = "maxnet.R"
    r_packages = ("maxnet", "jsonlite")


@register_model("brt", "boosted-regression-trees", "gbm-step")
class BRTAdapter(RModelAdapter):
    """Boosted regression trees via ``dismo::gbm.step()``.

    Tree complexity is chosen adaptively from the presence count (1 below 50
    presences, 5 above), learning rate 0.001, bag fraction 0.75, 5-fold CV to
    pick the tree count up to 10,000, with background points down-weighted by
    the presence:background ratio.
    """

    script = "brt.R"
    r_packages = ("dismo", "gbm", "jsonlite")


@register_model("random-forest", "rf-downsampled")
class RandomForestAdapter(RModelAdapter):
    """Random forest with per-class down-sampling (R ``randomForest``).

    The published configuration -- 1,000 trees, both classes sampled to the
    size of the minority class. Distinct from ``random-forest-sklearn``.
    """

    script = "randomforest.R"
    r_packages = ("randomForest", "jsonlite")


@register_model("gam", "mgcv-gam")
class GAMAdapter(RModelAdapter):
    """Generalised additive model via ``mgcv``.

    Penalised thin-plate regression splines for continuous predictors, factor
    terms for categorical ones, binomial/logit, REML, presence-background
    weighting.

    The exact per-region formulas are deferred by the paper to Valavi et al.
    (2022) and the authors' repository. Until that is public, the formula is
    reconstructed from the paper's description and flagged UNVERIFIED; pass
    ``formula=`` to supply the authoritative one.
    """

    script = "gam.R"
    r_packages = ("mgcv", "jsonlite")


@register_model("maxent-java")
class MaxentJavaAdapter(RModelAdapter):
    """The original Java MaxEnt accessed through ``dismo``.

    Paper sec. 2.2.1: "MaxEnt (Java) is the original Java-based MaxEnt
    implementation accessed via the dismo package ... We used automatic feature
    selection and cloglog output format for comparability."

    Requires ``maxent.jar`` to be present in ``dismo``'s ``java`` directory and
    a working Java runtime; both are checked by the R script rather than
    assumed.
    """

    script = "maxent_java.R"
    r_packages = ("dismo", "rJava", "jsonlite")

    def check_available(self) -> None:
        try:
            super().check_available()
        except MissingDependencyError as exc:
            raise MissingDependencyError(
                "MaxEnt (Java) via dismo/rJava",
                extra="r",
                hint=(
                    "MaxEnt (Java) additionally needs a Java runtime and maxent.jar placed in "
                    "system.file('java', package='dismo'). See "
                    "https://biodiversityinformatics.amnh.org/open_source/maxent/"
                ),
            ) from exc
