"""Discrimination metrics: ROC-AUC and PR-AUC.

Dinnage & Warren (2026) sec. 2.6 computed these with R's ``yardstick``.
sdmbench computes them with scikit-learn, which is dependency-light and fast.
The two agree for ROC-AUC (both use the trapezoidal rule with proper tie
handling) but *can* differ for PR-AUC, because the two ecosystems disagree on
how to interpolate a precision-recall curve:

* ``sklearn.metrics.average_precision_score`` uses the step-wise estimator
  ``sum((R_n - R_{n-1}) * P_n)`` -- no interpolation.
* ``yardstick::pr_auc`` integrates the curve trapezoidally.

Trapezoidal interpolation of a PR curve is known to be optimistically biased,
so ``average_precision`` is the default here. :func:`pr_auc` exposes both via
``method=`` and the R parity test (``tests/test_metric_parity_r.py``) pins the
size of the difference rather than pretending there is none.
"""

from __future__ import annotations

import numpy as np

__all__ = ["roc_auc", "pr_auc", "average_precision", "prevalence"]


def _validate(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(y_true) != len(y_prob):
        raise ValueError(f"length mismatch: y_true={len(y_true)}, y_prob={len(y_prob)}")
    if len(y_true) == 0:
        raise ValueError("cannot compute a metric on an empty vector")
    return y_true, y_prob


def _single_class(y_true: np.ndarray) -> bool:
    return len(np.unique(y_true)) < 2


def roc_auc(y_true, y_prob) -> float:
    """Area under the ROC curve.

    Returns ``nan`` for a single-class evaluation set, where the metric is
    undefined -- never 0.5, which would silently pull an aggregate mean toward
    "random" and misrepresent a species that could not be evaluated at all.
    """
    y_true, y_prob = _validate(y_true, y_prob)
    if _single_class(y_true):
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_prob))


def average_precision(y_true, y_prob) -> float:
    """Average precision -- the step-wise PR-AUC estimator."""
    y_true, y_prob = _validate(y_true, y_prob)
    if _single_class(y_true):
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y_true, y_prob))


def pr_auc(y_true, y_prob, *, method: str = "average_precision") -> float:
    """Area under the precision-recall curve.

    Parameters
    ----------
    method:
        ``"average_precision"`` (default, step-wise) or ``"trapezoid"``
        (matches ``yardstick::pr_auc``).
    """
    if method == "average_precision":
        return average_precision(y_true, y_prob)
    if method != "trapezoid":
        raise ValueError(f"unknown pr_auc method: {method!r}")

    y_true, y_prob = _validate(y_true, y_prob)
    if _single_class(y_true):
        return float("nan")
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    order = np.argsort(recall)
    return float(np.trapezoid(precision[order], recall[order]))


def prevalence(y_true) -> float:
    """Proportion of positives -- the PR-AUC baseline for a random model."""
    y_true = np.asarray(y_true).astype(int).ravel()
    return float(np.mean(y_true == 1)) if len(y_true) else float("nan")
