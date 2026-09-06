"""Metric registry.

The three metrics of the published protocol -- ROC-AUC, PR-AUC and Miller's
calibration slope (Dinnage & Warren 2026 sec. 2.6) -- are the defaults. The
optional ecological metrics are available to any benchmark but are never
substituted for the published set in strict reproduction mode.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from sdmbench.metrics.calibration import (
    DEFAULT_EPS,
    brier_score,
    calibration_intercept,
    calibration_slope,
    log_loss_score,
    miller_calibration,
)
from sdmbench.metrics.discrimination import average_precision, pr_auc, prevalence, roc_auc
from sdmbench.metrics.ecological import (
    balanced_accuracy,
    boyce_index,
    continuous_boyce_index,
    max_tss_threshold,
    sensitivity_specificity,
)

__all__ = [
    "METRIC_FUNCTIONS",
    "PUBLISHED_METRICS",
    "compute_metrics",
    "available_metrics",
    "roc_auc",
    "pr_auc",
    "average_precision",
    "prevalence",
    "miller_calibration",
    "calibration_slope",
    "calibration_intercept",
    "brier_score",
    "log_loss_score",
    "boyce_index",
    "continuous_boyce_index",
    "sensitivity_specificity",
    "balanced_accuracy",
    "max_tss_threshold",
    "DEFAULT_EPS",
]

#: The metric set used by the TabPFN-SDM 2026 reproduction.
PUBLISHED_METRICS = ("roc_auc", "pr_auc", "calibration_slope")

#: Name -> callable ``(y_true, y_prob) -> float``.
METRIC_FUNCTIONS: dict[str, Callable[..., float]] = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "average_precision": average_precision,
    "calibration_slope": calibration_slope,
    "calibration_intercept": calibration_intercept,
    "boyce": continuous_boyce_index,
    "boyce_index": continuous_boyce_index,
    "brier": brier_score,
    "log_loss": log_loss_score,
    "balanced_accuracy": balanced_accuracy,
    "prevalence": prevalence,
}


def available_metrics() -> list[str]:
    """Every metric name accepted in a configuration file."""
    return sorted({*METRIC_FUNCTIONS, "sensitivity", "specificity"})


def compute_metrics(
    y_true,
    y_prob,
    metrics: Sequence[str] | None = None,
    *,
    threshold: float | None = None,
    strict: bool = False,
) -> dict[str, float]:
    """Compute a set of metrics for one set of predictions.

    Parameters
    ----------
    metrics:
        Metric names. Defaults to :data:`PUBLISHED_METRICS`.
    threshold:
        Threshold for sensitivity/specificity/balanced accuracy. ``None``
        selects the MaxTSS threshold.
    strict:
        Re-raise metric errors instead of recording ``nan``. A single
        pathological species should not abort a 226-species run, so the
        default records ``nan`` and carries on.

    Returns
    -------
    dict
        Metric name -> value. Undefined metrics are ``nan``, never a
        placeholder like 0.5 that would distort an aggregate.
    """
    names = list(metrics) if metrics else list(PUBLISHED_METRICS)
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    out: dict[str, float] = {}

    # sensitivity/specificity share one threshold search; compute them once.
    wants_threshold_metrics = any(n in {"sensitivity", "specificity"} for n in names)
    if wants_threshold_metrics:
        try:
            parts = sensitivity_specificity(y_true, y_prob, threshold=threshold)
        except Exception:  # noqa: BLE001
            if strict:
                raise
            parts = {"sensitivity": float("nan"), "specificity": float("nan")}
        for name in ("sensitivity", "specificity"):
            if name in names:
                out[name] = float(parts[name])

    for name in names:
        if name in out:
            continue
        func = METRIC_FUNCTIONS.get(name)
        if func is None:
            if strict:
                raise KeyError(
                    f"unknown metric {name!r}; available: {', '.join(available_metrics())}"
                )
            continue
        try:
            if name == "balanced_accuracy":
                out[name] = float(func(y_true, y_prob, threshold=threshold))
            else:
                out[name] = float(func(y_true, y_prob))
        except Exception:  # noqa: BLE001
            if strict:
                raise
            out[name] = float("nan")

    # The slope and intercept come from one regression; if the slope was asked
    # for, the intercept is free and worth recording.
    if "calibration_slope" in out and "calibration_intercept" not in out:
        try:
            out["calibration_intercept"] = float(
                miller_calibration(y_true, y_prob)["calibration_intercept"]
            )
        except Exception:  # noqa: BLE001
            out["calibration_intercept"] = float("nan")
    return out


def validate_metric_names(names: Iterable[str]) -> list[str]:
    """Raise on unknown metric names, returning the validated list."""
    names = list(names)
    known = set(available_metrics())
    unknown = [n for n in names if n not in known]
    if unknown:
        raise KeyError(f"unknown metric(s): {unknown}; available: {sorted(known)}")
    return names
