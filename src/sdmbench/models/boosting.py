"""Gradient boosting adapters: XGBoost, LightGBM, CatBoost.

All three are optional (``pip install "sdmbench[boosting]"``). A missing
library produces a ``SKIPPED_DEPENDENCY`` row, never an import error at
collection time.

Class imbalance
---------------
Presence-background data is imbalanced by construction -- often 1:100. Each
library has its own idiomatic control (``scale_pos_weight``, ``is_unbalance``,
``auto_class_weights``); the default here applies the same
presence-to-background ratio weighting that the published BRT and GAM use, so
the boosting models are treated consistently with the published baselines
rather than each being tuned differently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sdmbench.models.base import ModelAdapter, register_model
from sdmbench.optional import package_version, require
from sdmbench.exceptions import SdmbenchError

__all__ = ["XGBoostAdapter", "LightGBMAdapter", "CatBoostAdapter"]


def _scale_pos_weight(y: np.ndarray) -> float:
    """Background-to-presence ratio, the standard imbalance correction."""
    y = np.asarray(y, dtype=int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return float(n_neg / n_pos) if n_pos else 1.0


@register_model("xgboost", "xgb")
class XGBoostAdapter(ModelAdapter):
    """XGBoost gradient boosting (``xgboost.XGBClassifier``)."""

    family = "boosting"
    supports_categorical = False

    def check_available(self) -> None:
        require("xgboost")
        super().check_available()

    def fit(self, X, y, **kwargs: Any) -> XGBoostAdapter:
        xgboost = require("xgboost")
        y = np.asarray(y, dtype=int)
        params: dict[str, Any] = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": 1,
            "random_state": self.seed,
        }
        params.update({k: v for k, v in self.hyperparameters.items() if k not in {"seed", "device"}})
        params.setdefault("scale_pos_weight", _scale_pos_weight(y))
        self._resolved = params
        self._fitted = xgboost.XGBClassifier(**params)
        self._fitted.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._fitted is None:
            raise SdmbenchError("xgboost adapter was not fitted")
        return self._fitted.predict_proba(X)[:, 1]

    def effective_hyperparameters(self) -> dict[str, Any]:
        return dict(getattr(self, "_resolved", self.hyperparameters))

    def version(self) -> str:
        return package_version("xgboost")


@register_model("lightgbm", "lgbm", "lgb")
class LightGBMAdapter(ModelAdapter):
    """LightGBM gradient boosting (``lightgbm.LGBMClassifier``).

    Consumes native pandas ``category`` columns, so the benchmark's categorical
    predictors reach it as categories rather than as integer codes that would
    imply a spurious ordering.
    """

    family = "boosting"
    supports_categorical = True

    def check_available(self) -> None:
        require("lightgbm")
        super().check_available()

    def fit(self, X, y, **kwargs: Any) -> LightGBMAdapter:
        lightgbm = require("lightgbm")
        y = np.asarray(y, dtype=int)
        params: dict[str, Any] = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": 1,
            "verbose": -1,
            "random_state": self.seed,
        }
        params.update({k: v for k, v in self.hyperparameters.items() if k not in {"seed", "device"}})
        params.setdefault("scale_pos_weight", _scale_pos_weight(y))
        self._resolved = params
        self._fitted = lightgbm.LGBMClassifier(**params)
        self._fitted.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._fitted is None:
            raise SdmbenchError("lightgbm adapter was not fitted")
        return self._fitted.predict_proba(X)[:, 1]

    def effective_hyperparameters(self) -> dict[str, Any]:
        return dict(getattr(self, "_resolved", self.hyperparameters))

    def version(self) -> str:
        return package_version("lightgbm")


@register_model("catboost", "cat-boost")
class CatBoostAdapter(ModelAdapter):
    """CatBoost gradient boosting (``catboost.CatBoostClassifier``).

    CatBoost requires categorical columns to be non-null strings or integers,
    so unseen/missing categories are filled with an explicit sentinel level
    rather than being dropped.
    """

    family = "boosting"
    supports_categorical = True

    _MISSING_CATEGORY = "__sdmbench_missing__"

    def check_available(self) -> None:
        require("catboost")
        super().check_available()

    def prepare_features(self, split):
        X_train, X_test = split.X_train.copy(), split.X_test.copy()
        self._cat_features = [c for c in split.categorical_features if c in X_train.columns]
        for col in self._cat_features:
            for frame in (X_train, X_test):
                frame[col] = (
                    frame[col].astype("object").fillna(self._MISSING_CATEGORY).astype(str)
                )
        return X_train, X_test

    def fit(self, X, y, **kwargs: Any) -> CatBoostAdapter:
        catboost = require("catboost")
        y = np.asarray(y, dtype=int)
        params: dict[str, Any] = {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 1,
            "random_seed": self.seed,
        }
        params.update({k: v for k, v in self.hyperparameters.items() if k not in {"seed", "device"}})
        params.setdefault("auto_class_weights", "Balanced")
        self._resolved = params
        self._fitted = catboost.CatBoostClassifier(**params)
        self._fitted.fit(X, y, cat_features=getattr(self, "_cat_features", None))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._fitted is None:
            raise SdmbenchError("catboost adapter was not fitted")
        return self._fitted.predict_proba(X)[:, 1]

    def effective_hyperparameters(self) -> dict[str, Any]:
        return dict(getattr(self, "_resolved", self.hyperparameters))

    def version(self) -> str:
        return package_version("catboost")
