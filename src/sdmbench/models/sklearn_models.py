"""scikit-learn model adapters.

Every algorithm here is the stock scikit-learn implementation. sdmbench
supplies the benchmark, not the learner -- a reviewer should be able to satisfy
themselves that "our KNN" is exactly ``KNeighborsClassifier`` by reading the
factory line.

Note on the two random forests
------------------------------
``random-forest`` (in :mod:`sdmbench.models.r_models`) is the *published*
model: R's ``randomForest`` with per-class down-sampling, as used by Dinnage &
Warren (2026) and Valavi et al. (2022). ``random-forest-sklearn`` here is a
different model and is reported under a different name. Silently swapping one
for the other would misattribute a published number.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sdmbench.models.base import ModelAdapter, SklearnAdapter, register_model
from sdmbench.splits.random import subsample_background

__all__ = [
    "LogisticRegressionAdapter",
    "KNNAdapter",
    "DecisionTreeAdapter",
    "RandomForestSklearnAdapter",
    "ExtraTreesAdapter",
    "SVMAdapter",
    "GaussianNBAdapter",
    "GradientBoostingAdapter",
    "AdaBoostAdapter",
    "KNN_GRID",
]


@register_model("logistic-regression", "logreg", "logistic_regression")
class LogisticRegressionAdapter(SklearnAdapter):
    """Regularised logistic regression -- the linear reference point."""

    family = "classical"

    def build_estimator(self) -> Any:
        from sklearn.linear_model import LogisticRegression

        params = {
            "max_iter": 2000,
            "class_weight": None,
            "random_state": self.seed,
            **self.estimator_kwargs(),
        }
        return LogisticRegression(**params)


@register_model("knn", "k-nearest-neighbours", "k-nearest-neighbors")
class KNNAdapter(SklearnAdapter):
    """k-nearest neighbours.

    The default configuration (``metric="manhattan"``, ``n_neighbors=5``, with
    standardised predictors) is the one that performed well in our earlier
    leopard analysis. That is the *reason it is included*, not evidence that it
    is good here: it is a candidate discovered on a different dataset with a
    different species, and it is evaluated on ``disdat`` under exactly the same
    rules as every other model. It is not a state-of-the-art method and must
    not be described as one.

    Standardisation is supplied by the benchmark's preprocessing recipe (which
    normalises to mean 0 / unit SD), so no scaler is added here -- doing so
    would double-scale the features.
    """

    family = "classical"

    def build_estimator(self) -> Any:
        from sklearn.neighbors import KNeighborsClassifier

        params = {
            "n_neighbors": 5,
            "metric": "manhattan",
            "weights": "uniform",
            "n_jobs": 1,
            **self.estimator_kwargs(),
        }
        return KNeighborsClassifier(**params)

    def fit(self, X, y, **kwargs: Any) -> KNNAdapter:
        # k cannot exceed the number of training rows.
        n_train = len(X)
        requested = int(self.hyperparameters.get("n_neighbors", 5))
        if requested > n_train:
            self.hyperparameters["n_neighbors"] = max(1, n_train)
        return super().fit(X, y, **kwargs)


#: A controlled grid for KNN. Selection must use training data only -- see
#: :class:`~sdmbench.models.tuning.InnerCvTuner`, which searches with grouped
#: cross-validation on the training partition and never sees test labels.
KNN_GRID = {
    "n_neighbors": [3, 5, 7, 9, 15],
    "metric": ["euclidean", "manhattan"],
    "weights": ["uniform", "distance"],
}


@register_model("decision-tree", "decision_tree", "cart")
class DecisionTreeAdapter(SklearnAdapter):
    family = "classical"

    def build_estimator(self) -> Any:
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(random_state=self.seed, **self.estimator_kwargs())


@register_model("random-forest-sklearn", "rf-sklearn", "random_forest_sklearn")
class RandomForestSklearnAdapter(SklearnAdapter):
    """scikit-learn random forest.

    Distinct from the published ``random-forest`` (R, down-sampled). Reported
    separately so the two are never confused.
    """

    family = "classical"

    def build_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestClassifier

        params = {
            "n_estimators": 1000,
            "n_jobs": 1,
            "random_state": self.seed,
            **self.estimator_kwargs(),
        }
        return RandomForestClassifier(**params)


@register_model("extra-trees", "extratrees", "extra_trees")
class ExtraTreesAdapter(SklearnAdapter):
    family = "classical"

    def build_estimator(self) -> Any:
        from sklearn.ensemble import ExtraTreesClassifier

        params = {
            "n_estimators": 1000,
            "n_jobs": 1,
            "random_state": self.seed,
            **self.estimator_kwargs(),
        }
        return ExtraTreesClassifier(**params)


@register_model("svm", "svc", "support-vector-machine")
class SVMAdapter(SklearnAdapter):
    """Support vector classifier with probability estimates.

    ``probability=True`` fits Platt scaling by internal cross-validation on the
    *training* data, which is legitimate. It is also slow and scales poorly, so
    the adapter subsamples background points beyond ``max_train_n`` rather than
    stalling a 226-species run on a 10,000-row background sample.
    """

    family = "classical"

    def build_estimator(self) -> Any:
        from sklearn.svm import SVC

        params = {
            "kernel": "rbf",
            "probability": True,
            "random_state": self.seed,
            **{k: v for k, v in self.estimator_kwargs().items() if k != "max_train_n"},
        }
        return SVC(**params)

    def fit(self, X, y, **kwargs: Any) -> SVMAdapter:
        max_train_n = int(self.hyperparameters.get("max_train_n", 20000))
        y = np.asarray(y, dtype=int)
        if len(y) > max_train_n:
            n_pres = int((y == 1).sum())
            keep = subsample_background(
                y, n_background=max(max_train_n - n_pres, n_pres), seed=self.seed
            )
            X = X[keep] if isinstance(X, np.ndarray) else X.iloc[keep]
            y = y[keep]
            self.hyperparameters["subsampled_to"] = int(len(y))
        return super().fit(X, y, **kwargs)


@register_model("gaussian-nb", "naive-bayes", "gaussian_nb")
class GaussianNBAdapter(SklearnAdapter):
    family = "classical"

    def build_estimator(self) -> Any:
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB(**self.estimator_kwargs())


@register_model("gradient-boosting", "gbm-sklearn", "gradient_boosting")
class GradientBoostingAdapter(SklearnAdapter):
    family = "classical"

    def build_estimator(self) -> Any:
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(random_state=self.seed, **self.estimator_kwargs())


@register_model("adaboost", "ada-boost")
class AdaBoostAdapter(SklearnAdapter):
    family = "classical"

    def build_estimator(self) -> Any:
        from sklearn.ensemble import AdaBoostClassifier

        return AdaBoostClassifier(random_state=self.seed, **self.estimator_kwargs())


@register_model("dummy", "baseline-prevalence")
class DummyAdapter(SklearnAdapter):
    """Predicts the training prevalence for every row.

    A sanity floor: any model that cannot beat this on ROC-AUC (which it scores
    at exactly 0.5) has learned nothing. Useful for validating that a pipeline
    is wired correctly before spending GPU hours.
    """

    family = "baseline"

    def build_estimator(self) -> Any:
        from sklearn.dummy import DummyClassifier

        params = {"strategy": "prior", **self.estimator_kwargs()}
        return DummyClassifier(**params)
