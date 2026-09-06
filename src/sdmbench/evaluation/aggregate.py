"""Aggregating per-species results into a leaderboard.

The rule that governs this module: **a model is ranked by its performance
across species, never by its best species.** Picking the best species is how a
weak method is made to look strong, and it is the single easiest way to produce
a misleading SDM comparison.

So :func:`build_leaderboard` aggregates over every species a model was
evaluated on, reports dispersion (SD, SE, 95% CI) alongside the mean, and
states ``n_species`` and ``n_regions`` explicitly -- because a mean over 40
species is not comparable with a mean over 226.

Coverage
--------
Skipped and failed jobs are excluded from the metric statistics (they have no
metrics) but are counted in ``n_failed``/``n_skipped``. A model that only
succeeded on the easy half of the benchmark will show it in those columns, and
:func:`common_species_leaderboard` builds the strictly like-for-like table on
the species every model completed.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from sdmbench.results import METRIC_COLUMNS

__all__ = [
    "build_leaderboard",
    "summarise_metric",
    "per_species_table",
    "common_species_leaderboard",
    "coverage_report",
    "RANKABLE_METRICS",
]

#: Metrics the leaderboard can be sorted by, and the direction of "better".
#: Calibration is special: 1.0 is perfect, so it ranks by distance from 1.
RANKABLE_METRICS: dict[str, str] = {
    "roc_auc": "higher",
    "pr_auc": "higher",
    "boyce": "higher",
    "brier": "lower",
    "log_loss": "lower",
    "balanced_accuracy": "higher",
    "calibration": "closest_to_one",
    "calibration_slope": "closest_to_one",
    "runtime": "lower",
}


def _successful(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["status"] == "OK"] if "status" in frame.columns else frame


def summarise_metric(values: Iterable[float], *, confidence: float = 0.95) -> dict[str, float]:
    """Mean, median, SD, SE and a confidence interval for one metric.

    The CI uses the t distribution, which matters at the small species counts
    that arise when a benchmark is filtered to one region.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {
            "mean": np.nan, "median": np.nan, "sd": np.nan, "se": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "n": 0,
        }
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    if n == 1:
        return {
            "mean": mean, "median": median, "sd": np.nan, "se": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "n": 1,
        }
    sd = float(np.std(arr, ddof=1))
    se = sd / np.sqrt(n)
    try:
        from scipy import stats

        critical = float(stats.t.ppf(0.5 + confidence / 2.0, df=n - 1))
    except ImportError:  # pragma: no cover - scipy is a core dependency
        critical = 1.96
    return {
        "mean": mean,
        "median": median,
        "sd": sd,
        "se": se,
        "ci_low": mean - critical * se,
        "ci_high": mean + critical * se,
        "n": n,
    }


def build_leaderboard(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    scenario: str | None = None,
    metrics: Sequence[str] | None = None,
    ascending: bool | None = None,
    min_species: int = 1,
) -> pd.DataFrame:
    """Aggregate results into one row per model.

    Parameters
    ----------
    metric:
        Metric to rank by. See :data:`RANKABLE_METRICS`.
    scenario:
        Restrict to ``"nonspatial"`` or ``"spatial"``. Mixing the two would
        average over experiments with different species counts, so it must be
        chosen deliberately.
    metrics:
        Metrics to summarise. Defaults to every metric present.
    min_species:
        Drop models evaluated on fewer species than this.

    Returns
    -------
    pandas.DataFrame
        One row per model, ordered best-first by ``metric``.
    """
    if results.empty:
        return pd.DataFrame(
            columns=["model_name", "model_family", "n_species", "n_regions", f"{metric}_mean"]
        )

    frame = results.copy()
    if scenario is not None:
        frame = frame[frame["evaluation_scenario"] == scenario]
    if frame.empty:
        return pd.DataFrame(columns=["model_name", "model_family", "n_species", "n_regions"])

    ok = _successful(frame)
    chosen = list(metrics) if metrics else [
        m for m in METRIC_COLUMNS if m in ok.columns and ok[m].notna().any()
    ]

    rows: list[dict[str, Any]] = []
    for model_name, group in frame.groupby("model_name", sort=False):
        successes = _successful(group)
        row: dict[str, Any] = {
            "model_name": model_name,
            "model_family": (
                group["model_family"].dropna().iloc[0] if group["model_family"].notna().any() else ""
            ),
            "n_species": int(successes["species_id"].nunique()),
            "n_regions": int(successes["region"].nunique()),
            "n_jobs": int(len(group)),
            "n_ok": int(len(successes)),
            "n_failed": int((group["status"] == "FAILED").sum()),
            "n_skipped": int(group["status"].astype(str).str.startswith("SKIPPED").sum()),
        }
        if row["n_skipped"]:
            reasons = (
                group.loc[group["status"].astype(str).str.startswith("SKIPPED"), "status"]
                .value_counts()
                .to_dict()
            )
            row["skip_reasons"] = "; ".join(f"{k}={v}" for k, v in reasons.items())
        else:
            row["skip_reasons"] = ""

        for name in chosen:
            stats_ = summarise_metric(successes[name]) if name in successes.columns else summarise_metric([])
            row[f"{name}_mean"] = stats_["mean"]
            row[f"{name}_median"] = stats_["median"]
            row[f"{name}_sd"] = stats_["sd"]
            row[f"{name}_se"] = stats_["se"]
            row[f"{name}_ci_low"] = stats_["ci_low"]
            row[f"{name}_ci_high"] = stats_["ci_high"]
            row[f"{name}_n"] = stats_["n"]

        for column, label in (
            ("total_time_seconds", "runtime_mean"),
            ("fit_time_seconds", "fit_time_mean"),
        ):
            if column in successes.columns and successes[column].notna().any():
                row[label] = float(np.nanmean(successes[column].astype(float)))
            else:
                row[label] = np.nan
        row["runtime_total"] = (
            float(np.nansum(successes["total_time_seconds"].astype(float)))
            if "total_time_seconds" in successes.columns
            else np.nan
        )
        row["device"] = (
            successes["device"].dropna().iloc[0] if successes["device"].notna().any() else ""
        )
        rows.append(row)

    board = pd.DataFrame(rows)
    if min_species > 1:
        board = board[board["n_species"] >= min_species]
    return _sort_leaderboard(board, metric=metric, ascending=ascending)


def _sort_leaderboard(
    board: pd.DataFrame, *, metric: str, ascending: bool | None
) -> pd.DataFrame:
    """Order a leaderboard, respecting each metric's notion of 'better'."""
    if board.empty:
        return board
    direction = RANKABLE_METRICS.get(metric, "higher")

    if metric == "runtime":
        key_column, default_ascending = "runtime_mean", True
    elif direction == "closest_to_one":
        # Calibration: rank by |slope - 1|, so 0.9 and 1.1 tie.
        source = "calibration_slope_mean"
        if source not in board.columns:
            return board
        board = board.copy()
        board["_calibration_distance"] = (board[source] - 1.0).abs()
        key_column, default_ascending = "_calibration_distance", True
    else:
        key_column, default_ascending = f"{metric}_mean", direction == "lower"

    if key_column not in board.columns:
        return board
    order = default_ascending if ascending is None else ascending
    out = board.sort_values(key_column, ascending=order, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.drop(columns=[c for c in ("_calibration_distance",) if c in out.columns])


def per_species_table(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    scenario: str | None = None,
) -> pd.DataFrame:
    """Wide table of ``species x model`` values for one metric.

    This is the input to every paired comparison: because each species is
    evaluated by every model under identical rules, a row is a matched set.
    """
    frame = results.copy()
    if scenario is not None:
        frame = frame[frame["evaluation_scenario"] == scenario]
    frame = _successful(frame)
    if frame.empty or metric not in frame.columns:
        return pd.DataFrame()
    return frame.pivot_table(
        index=["region", "species_id"],
        columns="model_name",
        values=metric,
        aggfunc="mean",
    )


def common_species_leaderboard(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    scenario: str | None = None,
) -> pd.DataFrame:
    """Leaderboard restricted to species every model completed.

    The strictly like-for-like view. If one model failed on a hard subset, the
    full leaderboard flatters it; this one does not.
    """
    table = per_species_table(results, metric=metric, scenario=scenario)
    if table.empty:
        return pd.DataFrame()
    complete = table.dropna(axis=0, how="any")
    rows = []
    for model in complete.columns:
        stats_ = summarise_metric(complete[model])
        rows.append(
            {
                "model_name": model,
                f"{metric}_mean": stats_["mean"],
                f"{metric}_median": stats_["median"],
                f"{metric}_sd": stats_["sd"],
                f"{metric}_se": stats_["se"],
                f"{metric}_ci_low": stats_["ci_low"],
                f"{metric}_ci_high": stats_["ci_high"],
                "n_species": stats_["n"],
            }
        )
    board = pd.DataFrame(rows)
    return _sort_leaderboard(board, metric=metric, ascending=None)


def coverage_report(results: pd.DataFrame) -> pd.DataFrame:
    """Per-model, per-scenario job outcome counts."""
    if results.empty:
        return pd.DataFrame()
    grouped = (
        results.groupby(["model_name", "evaluation_scenario", "status"])
        .size()
        .reset_index(name="n")
    )
    return grouped.pivot_table(
        index=["model_name", "evaluation_scenario"],
        columns="status",
        values="n",
        fill_value=0,
    ).reset_index()
