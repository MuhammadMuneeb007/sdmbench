"""The preprocessing recipe.

Dinnage & Warren (2026) sec. 2.1.2 specifies exactly three sequential steps,
implemented in R with ``tidymodels`` recipes:

    "The preprocessing pipeline ... consisted of three sequential steps:
     removal of zero-variance numeric predictors, Yeo-Johnson power
     transformation of all numeric predictors to approximate normality while
     accommodating zero and negative values, and normalization to mean zero and
     unit standard deviation. The recipe was fitted on training data alone; the
     same transformation parameters were then applied to test data to prevent
     information leakage."

and, for categorical handling:

    "Categorical variables were explicitly converted to factors before
     preprocessing. ... MaxNet, BRT, Random Forest, and GAM all received factor
     variables as native R factors without conversion. Only the local TabPFN
     implementation required explicit specification of categorical feature
     indices."

Two consequences encoded here:

* The numeric steps apply to **numeric predictors only**. Categorical
  predictors pass through untouched and are handed to models as categories.
* ``fit`` sees training rows only. :class:`SdmRecipe` records ``fitted_on`` so
  :class:`~sdmbench.splits.leakage.LeakageAuditor` can verify this rather than
  assume it.

Fidelity note
-------------
``recipes::step_YeoJohnson`` estimates lambda by maximum likelihood, bounded to
``[-5, 5]`` by default; scikit-learn's :class:`~sklearn.preprocessing.PowerTransformer`
does the same by MLE but without those bounds. The bound is applied here so the
two agree. Combined Yeo-Johnson + standardisation is exactly
``PowerTransformer(standardize=True)``, matching the paper's step order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from sdmbench.exceptions import DataError

__all__ = ["SdmRecipe", "RecipeState"]

#: Bounds ``recipes::step_YeoJohnson`` places on the estimated lambda.
YEO_JOHNSON_LAMBDA_LIMITS = (-5.0, 5.0)


@dataclass
class RecipeState:
    """Parameters estimated from the training data."""

    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    dropped_zero_variance: list[str] = field(default_factory=list)
    lambdas: dict[str, float] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    categorical_levels: dict[str, list[str]] = field(default_factory=dict)
    fitted_on: str = "train"

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted_on": self.fitted_on,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "dropped_zero_variance": self.dropped_zero_variance,
            "lambdas": self.lambdas,
            "means": self.means,
            "stds": self.stds,
            "categorical_levels": {k: list(v) for k, v in self.categorical_levels.items()},
        }


class SdmRecipe:
    """Zero-variance removal -> Yeo-Johnson -> standardisation.

    Parameters
    ----------
    numeric_predictors, categorical_predictors:
        Column names. Categorical columns bypass every numeric step.
    yeo_johnson, normalize, remove_zero_variance:
        Individual steps can be disabled for benchmarks with a different
        recipe; the TabPFN-SDM 2026 recipe enables all three.
    variance_threshold:
        A predictor whose training variance is at or below this is dropped as
        zero-variance (``recipes::step_zv`` drops predictors with a single
        unique value).
    """

    def __init__(
        self,
        numeric_predictors: Sequence[str],
        categorical_predictors: Sequence[str] = (),
        *,
        yeo_johnson: bool = True,
        normalize: bool = True,
        remove_zero_variance: bool = True,
        variance_threshold: float = 0.0,
    ) -> None:
        self.numeric_predictors = list(numeric_predictors)
        self.categorical_predictors = list(categorical_predictors)
        self.yeo_johnson = yeo_johnson
        self.normalize = normalize
        self.remove_zero_variance = remove_zero_variance
        self.variance_threshold = variance_threshold
        self.state: RecipeState | None = None

    # -------------------------------------------------------------------- fit --
    def fit(self, train: pd.DataFrame) -> SdmRecipe:
        """Estimate all parameters from ``train``. Never call this on test data."""
        missing = [
            c
            for c in (*self.numeric_predictors, *self.categorical_predictors)
            if c not in train.columns
        ]
        if missing:
            raise DataError(f"recipe cannot fit: missing column(s) {missing}")

        state = RecipeState(
            categorical_columns=list(self.categorical_predictors), fitted_on="train"
        )

        kept: list[str] = []
        for col in self.numeric_predictors:
            values = pd.to_numeric(train[col], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if self.remove_zero_variance and (
                len(finite) == 0
                or len(np.unique(finite)) <= 1
                or float(np.var(finite)) <= self.variance_threshold
            ):
                state.dropped_zero_variance.append(col)
                continue
            kept.append(col)

            if self.yeo_johnson:
                lam = _estimate_yeo_johnson_lambda(finite)
                state.lambdas[col] = lam
                transformed = _yeo_johnson(values, lam)
            else:
                transformed = values

            if self.normalize:
                finite_t = transformed[np.isfinite(transformed)]
                mean = float(np.mean(finite_t)) if len(finite_t) else 0.0
                std = float(np.std(finite_t, ddof=1)) if len(finite_t) > 1 else 0.0
                # A constant column after transformation would divide by zero;
                # leaving it centred-only keeps the column harmless.
                state.means[col] = mean
                state.stds[col] = std if std > 0 else 1.0

        state.numeric_columns = kept
        for col in self.categorical_predictors:
            levels = pd.unique(train[col].astype("object").dropna())
            state.categorical_levels[col] = [str(v) for v in levels]

        self.state = state
        return self

    # -------------------------------------------------------------- transform --
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted transformation. Safe to call on test data."""
        if self.state is None:
            raise DataError("recipe must be fitted before transform()")
        state = self.state

        out = pd.DataFrame(index=data.index)
        for col in state.numeric_columns:
            if col not in data.columns:
                raise DataError(f"column {col!r} missing at transform time")
            values = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
            if self.yeo_johnson and col in state.lambdas:
                values = _yeo_johnson(values, state.lambdas[col])
            if self.normalize and col in state.means:
                values = (values - state.means[col]) / state.stds[col]
            out[col] = values

        for col in state.categorical_columns:
            if col not in data.columns:
                raise DataError(f"categorical column {col!r} missing at transform time")
            levels = state.categorical_levels.get(col, [])
            # Levels come from training only. An unseen test level becomes NaN
            # rather than silently extending the encoding.
            out[col] = pd.Categorical(data[col].astype("object").astype("string"),
                                      categories=levels)
        return out

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(train).transform(train)

    @property
    def output_columns(self) -> list[str]:
        if self.state is None:
            raise DataError("recipe must be fitted first")
        return [*self.state.numeric_columns, *self.state.categorical_columns]

    def describe(self) -> dict[str, Any]:
        return {
            "steps": [
                *(["remove_zero_variance"] if self.remove_zero_variance else []),
                *(["yeo_johnson"] if self.yeo_johnson else []),
                *(["normalize"] if self.normalize else []),
            ],
            "source": "Dinnage & Warren 2026 sec. 2.1.2",
            "state": self.state.to_dict() if self.state else None,
        }


# ---------------------------------------------------------------------------
# Yeo-Johnson
# ---------------------------------------------------------------------------


def _yeo_johnson(x: np.ndarray, lam: float) -> np.ndarray:
    """Yeo-Johnson transform, defined for negative, zero and positive values."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if abs(lam) < 1e-8:
            out[pos] = np.log1p(x[pos])
        else:
            out[pos] = (np.power(x[pos] + 1.0, lam) - 1.0) / lam
        if abs(lam - 2.0) < 1e-8:
            out[neg] = -np.log1p(-x[neg])
        else:
            out[neg] = -(np.power(-x[neg] + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    # NaNs in, NaNs out: missingness is a model's problem, not the recipe's.
    out[~np.isfinite(x)] = np.nan
    return out


def _estimate_yeo_johnson_lambda(x: np.ndarray) -> float:
    """MLE for the Yeo-Johnson lambda, bounded like ``recipes::step_YeoJohnson``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2 or len(np.unique(x)) < 2:
        return 1.0
    lo, hi = YEO_JOHNSON_LAMBDA_LIMITS
    try:
        from scipy import optimize, stats

        def neg_llf(lam: float) -> float:
            return -stats.yeojohnson_llf(lam, x)

        result = optimize.minimize_scalar(neg_llf, bounds=(lo, hi), method="bounded")
        lam = float(result.x) if result.success else 1.0
    except Exception:  # noqa: BLE001 - degrade to identity rather than fail a run
        return 1.0
    return float(np.clip(lam, lo, hi))
