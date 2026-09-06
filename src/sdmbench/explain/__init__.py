"""Explainability.

Explanation has to operate at the level of the *methodology*, not just the
model. The questions worth answering are ecological:

* Which **modality** mattered? -> :func:`modality_ablation`
* Which **variable** mattered? -> :func:`permutation_importance`, SHAP
* Which **scale** mattered?    -> :func:`scale_ablation`
* Did **graph structure** help? -> edge/node attribution
* Where does the model **fail geographically**? -> :func:`spatial_residuals`

Ablation is the primary tool here rather than an afterthought. A SHAP value
tells you what the fitted model used; an ablation tells you what the benchmark
would have lost without it -- and only the second answers "should we have
collected this data?".

Attribution methods are optional (``sdmbench[explain]``) and degrade to a clear
message when absent.

On attention
------------
Attention weights from the fusion and graph models are reported, but attention
is *not* an explanation on its own -- a well-documented caveat in the
interpretability literature. They are labelled ``attention_weights``, never
``importance``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from sdmbench.optional import require, try_import

__all__ = [
    "AblationResult",
    "modality_ablation",
    "scale_ablation",
    "permutation_importance",
    "shap_values",
    "spatial_residuals",
]


@dataclass
class AblationResult:
    """Performance with and without a component."""

    component: str
    baseline_score: float
    ablated_score: float
    metric: str = "roc_auc"
    n_species: int = 0
    per_species: dict[str, float] = field(default_factory=dict)

    @property
    def contribution(self) -> float:
        """How much performance is lost by removing the component."""
        return float(self.baseline_score - self.ablated_score)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "metric": self.metric,
            "baseline_score": self.baseline_score,
            "ablated_score": self.ablated_score,
            "contribution": self.contribution,
            "n_species": self.n_species,
        }


def modality_ablation(
    results: pd.DataFrame,
    *,
    baseline_model: str,
    metric: str = "roc_auc",
    scenario: str | None = None,
) -> pd.DataFrame:
    """Contribution of each modality, from a completed ablation run.

    Expects results whose ``model_name`` follows the planner's naming --
    ``<base>``, ``<base>-minus-<modality>``, ``only-<modality>`` -- and reports
    both marginal value (alone) and unique contribution (leave-one-out).

    The two can disagree sharply, and the disagreement is the finding: a
    modality that scores well alone but adds nothing leave-one-out is redundant
    with the others, which is an argument against collecting it.
    """
    from sdmbench.evaluation.aggregate import per_species_table

    table = per_species_table(results, metric=metric, scenario=scenario)
    if table.empty:
        return pd.DataFrame()

    baseline = table[baseline_model] if baseline_model in table.columns else None
    rows: list[dict[str, Any]] = []

    for column in table.columns:
        if "-minus-" in column:
            modality = column.split("-minus-", 1)[1]
            kind = "leave_one_out"
        elif column.startswith("only-"):
            modality = column.split("only-", 1)[1]
            kind = "alone"
        else:
            continue
        paired = (
            pd.concat([baseline, table[column]], axis=1).dropna()
            if baseline is not None
            else None
        )
        if paired is None or paired.empty:
            continue
        difference = paired.iloc[:, 0] - paired.iloc[:, 1]
        rows.append(
            {
                "modality": modality,
                "comparison": kind,
                "baseline_mean": float(paired.iloc[:, 0].mean()),
                "variant_mean": float(paired.iloc[:, 1].mean()),
                # For leave-one-out this is the unique contribution; for
                # "alone" it is the gap to the full model.
                "difference": float(difference.mean()),
                "difference_sd": float(difference.std(ddof=1)) if len(difference) > 1 else np.nan,
                "n_species": int(len(paired)),
                "wins": int((difference > 0).sum()),
                "losses": int((difference < 0).sum()),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("difference", ascending=False).reset_index(drop=True) if not frame.empty else frame


def scale_ablation(
    results: pd.DataFrame,
    *,
    modality: str,
    metric: str = "roc_auc",
    scenario: str | None = None,
) -> pd.DataFrame:
    """Performance by spatial scale for one modality.

    Reads model names of the form ``<base>-<modality>@<scale>`` produced by
    :meth:`~sdmbench.core.planner.StagedPlanner.plan_scale`.
    """
    from sdmbench.evaluation.aggregate import per_species_table, summarise_metric

    table = per_species_table(results, metric=metric, scenario=scenario)
    if table.empty:
        return pd.DataFrame()

    marker = f"-{modality}@"
    rows = []
    for column in table.columns:
        if marker not in column:
            continue
        scale = column.split(marker, 1)[1]
        stats = summarise_metric(table[column].dropna())
        rows.append(
            {
                "modality": modality,
                "scale": scale,
                "mean": stats["mean"],
                "median": stats["median"],
                "sd": stats["sd"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "n_species": stats["n"],
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("mean", ascending=False).reset_index(drop=True) if not frame.empty else frame


def permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    metric: str = "roc_auc",
    n_repeats: int = 10,
    seed: int = 32639,
    feature_groups: dict[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    """Permutation importance, optionally grouped by modality.

    ``feature_groups`` permutes a whole modality's columns together, which is
    the right unit when features within a modality are correlated -- permuting
    them one at a time understates the modality's importance because the others
    still carry the signal.

    Model-agnostic: needs only ``predict_proba``.
    """
    from sdmbench.metrics import METRIC_FUNCTIONS

    score_fn = METRIC_FUNCTIONS[metric]
    rng = np.random.default_rng(seed)

    def score(frame: pd.DataFrame) -> float:
        proba = model.predict_proba(frame)
        proba = np.asarray(proba, dtype=float)
        prob = proba[:, 1] if proba.ndim == 2 else proba.ravel()
        return float(score_fn(y, prob))

    baseline = score(X)
    groups = feature_groups or {c: [c] for c in X.columns}

    rows = []
    for name, columns in groups.items():
        present = [c for c in columns if c in X.columns]
        if not present:
            continue
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            order = rng.permutation(len(shuffled))
            # Permute the whole group with ONE ordering, preserving the
            # within-group correlation structure while destroying its
            # relationship to the target.
            for column in present:
                shuffled[column] = shuffled[column].to_numpy()[order]
            drops.append(baseline - score(shuffled))
        rows.append(
            {
                "feature": name,
                "importance_mean": float(np.mean(drops)),
                "importance_sd": float(np.std(drops, ddof=1)) if n_repeats > 1 else np.nan,
                "baseline_score": baseline,
                "n_columns": len(present),
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def shap_values(
    model: Any,
    X: pd.DataFrame,
    *,
    max_samples: int = 500,
    seed: int = 32639,
) -> pd.DataFrame:
    """SHAP values for a fitted model (optional, needs ``shap``).

    Subsampled by default: exact SHAP is exponential in features and the
    sampling approximations still cost minutes on a full background sample.
    """
    shap = try_import("shap")
    if shap is None:
        from sdmbench.exceptions import MissingDependencyError

        raise MissingDependencyError("shap", extra="explain")

    rng = np.random.default_rng(seed)
    sample = (
        X.iloc[rng.choice(len(X), size=max_samples, replace=False)]
        if len(X) > max_samples
        else X
    )
    try:
        explainer = shap.Explainer(model.predict_proba, sample)
        values = explainer(sample)
        array = values.values
        if array.ndim == 3:  # (n, features, classes) -> presence class
            array = array[:, :, 1]
    except Exception:  # noqa: BLE001 - fall back to the model-agnostic explainer
        explainer = shap.KernelExplainer(
            lambda data: model.predict_proba(pd.DataFrame(data, columns=X.columns))[:, 1],
            shap.sample(sample, min(100, len(sample)), random_state=seed),
        )
        array = np.asarray(explainer.shap_values(sample))

    return pd.DataFrame(
        {
            "feature": list(X.columns),
            "mean_abs_shap": np.abs(array).mean(axis=0),
            "mean_shap": array.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def spatial_residuals(
    coords: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Where the model errs, geographically.

    Bins observations on a spatial grid and reports mean residual per cell.
    Strong spatial structure in the residuals means the model is missing
    something that varies over space -- an unmeasured variable, a dispersal
    limit, or sampling bias.
    """
    coords = np.asarray(coords, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    residual = y_true - y_prob

    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    extent = np.maximum(maxs - mins, np.finfo(float).eps)
    cell = np.floor((coords - mins) / extent * n_bins).astype(int)
    cell = np.clip(cell, 0, n_bins - 1)

    frame = pd.DataFrame(
        {
            "cell_x": cell[:, 0],
            "cell_y": cell[:, 1],
            "residual": residual,
            "y_true": y_true,
            "y_prob": y_prob,
        }
    )
    grouped = (
        frame.groupby(["cell_x", "cell_y"])
        .agg(
            n=("residual", "size"),
            mean_residual=("residual", "mean"),
            mean_observed=("y_true", "mean"),
            mean_predicted=("y_prob", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values("mean_residual", key=np.abs, ascending=False).reset_index(drop=True)
