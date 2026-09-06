"""Ecological metrics.

The Continuous Boyce Index (CBI)
-------------------------------
Boyce et al. (2002), continuous form by Hirzel et al. (2006). It is the
presence-only metric of choice because it needs no absences: it asks whether
predicted-suitable bins contain *more* presences than expected from their
availability. Values range from -1 to 1; a model no better than random scores
about 0.

This implementation follows ``ecospat::ecospat.boyce`` defaults exactly, which
is what ``tidysdm::boyce_cont()`` wraps:

* moving window of width ``(max(fit) - min(fit)) / 10``
* ``res = 100`` windows, whose lower bounds are the correlation's x-values
* ``F = P/E`` where ``P`` is the fraction of *presences* in the window and
  ``E`` the fraction of *all* evaluation points in the window
* consecutive duplicate ``F`` values dropped (``rm.duplicate = TRUE``)
* Spearman correlation between the retained ``F`` values and window lower bounds

Two details are easy to get wrong and are called out because they change the
number: the correlation uses the window's **lower bound**, not its midpoint;
and ``E`` is computed over *all* evaluation points, not the absences only.

Cross-language validation against ``tidysdm::boyce_cont()`` is specified in
``tests/test_metric_parity_r.py``. Those tests are written but, per this
session's instructions, have not been executed.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "boyce_index",
    "continuous_boyce_index",
    "sensitivity_specificity",
    "balanced_accuracy",
    "max_tss_threshold",
]


def continuous_boyce_index(
    y_true,
    y_prob,
    *,
    n_bins: int = 0,
    window_width: float | None = None,
    resolution: int = 100,
    remove_duplicates: bool = True,
) -> float:
    """Continuous Boyce Index.

    Parameters
    ----------
    y_true:
        Binary outcomes; ``1`` marks a presence.
    y_prob:
        Predicted suitability at every evaluation point.
    n_bins:
        ``0`` (default) uses the moving-window estimator. A positive value uses
        that many fixed, non-overlapping classes instead.
    window_width:
        Window width. Defaults to one tenth of the prediction range.
    resolution:
        Number of moving windows.

    Returns
    -------
    float
        The index, or ``nan`` when it is undefined (no presences, constant
        predictions, or fewer than two usable windows).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    fit = np.asarray(y_prob, dtype=float).ravel()
    if len(y_true) != len(fit):
        raise ValueError(f"length mismatch: y_true={len(y_true)}, y_prob={len(fit)}")

    finite = np.isfinite(fit)
    fit = fit[finite]
    y_true = y_true[finite]
    obs = fit[y_true == 1]

    if len(obs) == 0 or len(fit) < 2:
        return float("nan")
    lo, hi = float(np.min(fit)), float(np.max(fit))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    if n_bins and n_bins > 0:
        edges = np.linspace(lo, hi, int(n_bins) + 1)
        intervals = np.column_stack([edges[:-1], edges[1:]])
        x_values = intervals[:, 0]
    else:
        width = float(window_width) if window_width else (hi - lo) / 10.0
        if width <= 0:
            return float("nan")
        step = (hi - lo - width) / resolution
        if step <= 0:
            return float("nan")
        lower = lo + step * np.arange(resolution + 1)
        # ecospat nudges the final window past the maximum so the largest
        # prediction is included; reproduced here for exact parity.
        upper = lower + width
        upper[-1] = max(upper[-1], hi) + 1e-12
        intervals = np.column_stack([lower, upper])
        x_values = lower

    n_obs = len(obs)
    n_fit = len(fit)
    f_values = np.empty(len(intervals), dtype=float)
    for i, (a, b) in enumerate(intervals):
        p_i = np.count_nonzero((obs >= a) & (obs <= b)) / n_obs
        e_i = np.count_nonzero((fit >= a) & (fit <= b)) / n_fit
        f_values[i] = p_i / e_i if e_i > 0 else np.nan

    keep = np.isfinite(f_values)
    f_kept = f_values[keep]
    x_kept = x_values[keep]
    if len(f_kept) < 2:
        return float("nan")

    if remove_duplicates:
        # Drop each value equal to its successor, matching ecospat's
        # `f != c(f[-1], TRUE)` idiom (the final element is always kept).
        keep_mask = np.ones(len(f_kept), dtype=bool)
        keep_mask[:-1] = f_kept[:-1] != f_kept[1:]
        f_kept = f_kept[keep_mask]
        x_kept = x_kept[keep_mask]
    if len(f_kept) < 2 or np.ptp(f_kept) == 0 or np.ptp(x_kept) == 0:
        return float("nan")

    from scipy.stats import spearmanr

    rho = spearmanr(f_kept, x_kept).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


#: ``boyce_index`` is the short alias used in configuration files.
boyce_index = continuous_boyce_index


def max_tss_threshold(y_true, y_prob) -> float:
    """Threshold maximising the True Skill Statistic (sensitivity + specificity - 1).

    Threshold-dependent metrics need a threshold, and 0.5 is arbitrary for
    presence-background models whose outputs are not calibrated probabilities
    of occurrence. MaxTSS is the standard ecological choice.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    tss = tpr - fpr
    best = int(np.argmax(tss))
    return float(thresholds[best])


def sensitivity_specificity(y_true, y_prob, *, threshold: float | None = None) -> dict[str, float]:
    """Sensitivity, specificity and the threshold used.

    ``threshold=None`` selects the MaxTSS threshold.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    nan = {"sensitivity": float("nan"), "specificity": float("nan"), "threshold": float("nan")}
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return nan

    thr = max_tss_threshold(y_true, y_prob) if threshold is None else float(threshold)
    if not np.isfinite(thr):
        return nan
    predicted = (y_prob >= thr).astype(int)
    tp = int(np.sum((predicted == 1) & (y_true == 1)))
    fn = int(np.sum((predicted == 0) & (y_true == 1)))
    tn = int(np.sum((predicted == 0) & (y_true == 0)))
    fp = int(np.sum((predicted == 1) & (y_true == 0)))
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "threshold": thr,
    }


def balanced_accuracy(y_true, y_prob, *, threshold: float | None = None) -> float:
    """Mean of sensitivity and specificity."""
    parts = sensitivity_specificity(y_true, y_prob, threshold=threshold)
    sens, spec = parts["sensitivity"], parts["specificity"]
    if not (np.isfinite(sens) and np.isfinite(spec)):
        return float("nan")
    return float((sens + spec) / 2.0)
