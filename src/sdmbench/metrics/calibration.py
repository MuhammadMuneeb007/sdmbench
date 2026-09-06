"""Calibration metrics.

Miller's calibration slope
--------------------------
This is *not* a generic calibration slope, and the difference matters. Dinnage
& Warren (2026) sec. 2.6 defines it precisely:

    "Miller's calibration slope and intercept are obtained by regressing
     observed outcomes on the logit-transformed predicted probabilities. A
     slope of 1.0 indicates that predicted probability ratios correspond
     exactly to observed odds ratios -- that is, the model's relative
     confidence is well-scaled."

So the estimator is a **logistic** regression of the binary outcome on
``logit(p)``::

    glm(y ~ logit(p), family = binomial)

with the slope being the coefficient on ``logit(p)``. Some literature instead
regresses outcomes on probabilities with ordinary least squares, or bins
predictions and fits a line through bin means. Those give different numbers.
Only the definition above reproduces the paper's reported slope of 1.110 for
finetuned TabPFN.

The paper also explains why only the *slope* is interpretable here:

    "The intercept captures overall predicted prevalence, though we note that
     no presence-only model can achieve absolute calibration (intercept = 0)
     because the baseline prevalence is confounded with the intercept term. We
     therefore focus on the slope as a measure of ratio calibration."

The intercept is still returned, but the leaderboard ranks on the slope.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "miller_calibration",
    "calibration_slope",
    "calibration_intercept",
    "brier_score",
    "log_loss_score",
    "DEFAULT_EPS",
]

#: Clipping applied before the logit so that a prediction of exactly 0 or 1
#: does not produce an infinite covariate. 1e-6 bounds |logit| at ~13.8.
DEFAULT_EPS = 1e-6


def _logit(p: np.ndarray, eps: float = DEFAULT_EPS) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def miller_calibration(
    y_true,
    y_prob,
    *,
    eps: float = DEFAULT_EPS,
) -> dict[str, float]:
    """Miller's calibration slope and intercept.

    Returns
    -------
    dict
        ``{"calibration_slope": float, "calibration_intercept": float}``.
        Both are ``nan`` when the outcome is single-class or the logits are
        constant, in which case the regression is not identified.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(y_true) != len(y_prob):
        raise ValueError(f"length mismatch: y_true={len(y_true)}, y_prob={len(y_prob)}")

    nan = {"calibration_slope": float("nan"), "calibration_intercept": float("nan")}
    if len(y_true) < 3 or len(np.unique(y_true)) < 2:
        return nan

    lp = _logit(y_prob, eps=eps)
    if not np.all(np.isfinite(lp)) or np.ptp(lp) == 0:
        # A constant predictor carries no information about slope.
        return nan

    slope, intercept = _fit_logistic_1d(lp, y_true)
    return {"calibration_slope": float(slope), "calibration_intercept": float(intercept)}


def _fit_logistic_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Unpenalised logistic regression of ``y`` on a single covariate ``x``.

    Uses scikit-learn with the penalty disabled so the fit is the maximum
    likelihood estimate that R's ``glm`` would produce. Falls back to
    Newton-Raphson (IRLS) if the installed scikit-learn predates
    ``penalty=None``, or if lbfgs fails to converge on a separable problem.
    """
    X = x.reshape(-1, 1)
    try:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(X, y)
        slope = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])
        if np.isfinite(slope) and np.isfinite(intercept):
            return slope, intercept
    except Exception:  # noqa: BLE001 - fall through to IRLS
        pass
    return _irls_logistic(x, y)


def _irls_logistic(
    x: np.ndarray, y: np.ndarray, *, max_iter: int = 100, tol: float = 1e-10
) -> tuple[float, float]:
    """Newton-Raphson / IRLS fit, mirroring how ``stats::glm`` solves this."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
        z = eta + (y - mu) / w
        XtW = X.T * w
        try:
            beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan")
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return float(beta[1]), float(beta[0])


def calibration_slope(y_true, y_prob, *, eps: float = DEFAULT_EPS) -> float:
    """Miller's calibration slope alone."""
    return miller_calibration(y_true, y_prob, eps=eps)["calibration_slope"]


def calibration_intercept(y_true, y_prob, *, eps: float = DEFAULT_EPS) -> float:
    """Miller's calibration intercept alone."""
    return miller_calibration(y_true, y_prob, eps=eps)["calibration_intercept"]


def brier_score(y_true, y_prob) -> float:
    """Mean squared error of the predicted probabilities."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss_score(y_true, y_prob, *, eps: float = DEFAULT_EPS) -> float:
    """Binary cross-entropy, clipped to keep it finite."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), eps, 1.0 - eps)
    if len(y_true) == 0:
        return float("nan")
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))
