"""Plots (optional, ``sdmbench[plots]``).

matplotlib is an optional dependency: a benchmark run must never fail because
a plotting library is missing. Every function here calls
:func:`sdmbench.optional.require` and returns a ``Figure``, leaving the caller
to save or display it.

Design choices that are scientific rather than cosmetic:

* Leaderboards are drawn with error bars. A bar chart of means alone implies a
  precision the data does not have.
* Paired comparisons are drawn per species, not as two summary bars, because
  the pairing is the evidence.
* The compute-performance frontier is plotted on a log time axis, because the
  models being compared span sub-second inference to hours.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from sdmbench.optional import require

__all__ = [
    "plot_leaderboard",
    "plot_paired_comparison",
    "plot_metric_distribution",
    "plot_compute_frontier",
    "save_all",
]


def plot_leaderboard(
    leaderboard: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    title: str | None = None,
    reference_lines: dict[str, float] | None = None,
) -> Any:
    """Horizontal bar chart of model means with 95% confidence intervals.

    ``reference_lines`` draws published values as vertical markers, clearly
    labelled so they are never mistaken for measured results.
    """
    plt = require("matplotlib.pyplot")
    if leaderboard.empty:
        raise ValueError("cannot plot an empty leaderboard")

    mean_col = f"{metric}_mean"
    if mean_col not in leaderboard.columns:
        raise KeyError(f"leaderboard has no column {mean_col!r}")

    board = leaderboard.sort_values(mean_col, ascending=True)
    names = board["model_name"].tolist()
    means = board[mean_col].to_numpy(dtype=float)

    lo_col, hi_col = f"{metric}_ci_low", f"{metric}_ci_high"
    if lo_col in board.columns and hi_col in board.columns:
        lo = board[lo_col].to_numpy(dtype=float)
        hi = board[hi_col].to_numpy(dtype=float)
        errors = np.vstack([means - lo, hi - means])
        errors = np.nan_to_num(errors, nan=0.0)
    else:
        errors = None

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.42 * len(names) + 1.5)))
    ax.barh(names, means, xerr=errors, capsize=3, color="#4C72B0", alpha=0.85)
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title(title or f"Model performance ({metric}), mean +/- 95% CI across species")

    if metric == "roc_auc":
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=1)
        ax.annotate("random", xy=(0.5, -0.6), color="grey", fontsize=8)

    for label, value in (reference_lines or {}).items():
        ax.axvline(value, color="#C44E52", linestyle="--", linewidth=1)
        ax.annotate(
            f"published: {label} ({value:.3f})",
            xy=(value, len(names) - 0.5),
            fontsize=8,
            color="#C44E52",
            rotation=90,
            va="top",
        )

    if "n_species" in board.columns:
        for i, n in enumerate(board["n_species"].tolist()):
            ax.annotate(f"n={int(n)}", xy=(means[i], i), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=7, color="#333333")

    fig.tight_layout()
    return fig


def plot_paired_comparison(
    per_species: pd.DataFrame,
    model_a: str,
    model_b: str,
    *,
    metric: str = "roc_auc",
) -> Any:
    """Scatter of per-species values for two models, with the identity line.

    Points above the diagonal are species where ``model_a`` wins. The visual
    question is not "is the cloud higher" but "is it consistently on one side".
    """
    plt = require("matplotlib.pyplot")
    if model_a not in per_species.columns or model_b not in per_species.columns:
        raise KeyError(f"per-species table lacks {model_a!r} or {model_b!r}")

    paired = per_species.loc[:, [model_a, model_b]].dropna()
    a = paired[model_a].to_numpy(dtype=float)
    b = paired[model_b].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(b, a, alpha=0.6, s=26, color="#4C72B0", edgecolor="none")
    lo = float(min(a.min(), b.min())) if len(a) else 0.0
    hi = float(max(a.max(), b.max())) if len(a) else 1.0
    pad = 0.02 * (hi - lo or 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"{model_b} ({metric})")
    ax.set_ylabel(f"{model_a} ({metric})")

    wins = int(np.sum(a > b))
    losses = int(np.sum(a < b))
    ties = len(a) - wins - losses
    ax.set_title(
        f"{model_a} vs {model_b}\n"
        f"{wins} wins / {ties} ties / {losses} losses across {len(a)} species"
    )
    fig.tight_layout()
    return fig


def plot_metric_distribution(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    scenario: str | None = None,
    models: Sequence[str] | None = None,
) -> Any:
    """Box plot of the per-species metric distribution for each model.

    Shows what a leaderboard mean hides: the spread, and the species where a
    model fails outright.
    """
    plt = require("matplotlib.pyplot")
    frame = results[results["status"] == "OK"].copy()
    if scenario:
        frame = frame[frame["evaluation_scenario"] == scenario]
    if models:
        frame = frame[frame["model_name"].isin(list(models))]
    if frame.empty or metric not in frame.columns:
        raise ValueError(f"no {metric!r} values to plot")

    order = (
        frame.groupby("model_name")[metric].median().sort_values(ascending=True).index.tolist()
    )
    data = [frame.loc[frame["model_name"] == m, metric].dropna().to_numpy() for m in order]

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.42 * len(order) + 1.5)))
    ax.boxplot(data, vert=False, labels=order, showfliers=True, widths=0.6)
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title(
        f"Per-species {metric} distribution"
        + (f" ({scenario})" if scenario else "")
    )
    if metric == "roc_auc":
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=1)
    fig.tight_layout()
    return fig


def plot_compute_frontier(
    leaderboard: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    time_column: str = "runtime_mean",
) -> Any:
    """Performance against compute cost, with the Pareto frontier marked.

    The question this answers: does the expensive model earn its cost? A
    gradient-boosted tree at 0.72 in 2 seconds and a transformer at 0.73 in an
    hour are not obviously ranked.
    """
    plt = require("matplotlib.pyplot")
    mean_col = f"{metric}_mean"
    if mean_col not in leaderboard.columns or time_column not in leaderboard.columns:
        raise KeyError(f"leaderboard needs {mean_col!r} and {time_column!r}")

    board = leaderboard.dropna(subset=[mean_col, time_column]).copy()
    if board.empty:
        raise ValueError("no rows with both a metric and a runtime")

    times = board[time_column].to_numpy(dtype=float)
    scores = board[mean_col].to_numpy(dtype=float)

    # Pareto front: no other model is both faster and better.
    on_front = np.ones(len(board), dtype=bool)
    for i in range(len(board)):
        on_front[i] = not np.any((times < times[i]) & (scores > scores[i]))

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(times[~on_front], scores[~on_front], s=40, color="#999999", label="dominated")
    ax.scatter(times[on_front], scores[on_front], s=70, color="#C44E52", label="Pareto frontier")
    for name, t, s in zip(board["model_name"], times, scores):
        ax.annotate(name, xy=(t, s), xytext=(4, 3), textcoords="offset points", fontsize=7)

    ax.set_xscale("log")
    ax.set_xlabel("mean runtime per species (s, log scale)")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.set_title("Performance vs compute cost")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def save_all(figures: dict[str, Any], directory: str, *, dpi: int = 150) -> list[str]:
    """Save a mapping of ``name -> Figure`` as PNGs, returning the paths."""
    from pathlib import Path

    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name, figure in figures.items():
        path = out_dir / f"{name}.png"
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(str(path))
    return paths
