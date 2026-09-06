"""Leaderboard rendering.

Formatting only -- every number here is computed in
:mod:`sdmbench.evaluation.aggregate`. The presentation rules that matter
scientifically are enforced upstream: aggregation is across species, dispersion
travels with every mean, and coverage (``n_species``, skips, failures) is always
visible so two rows built from different species counts cannot be silently
compared.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from sdmbench.evaluation.aggregate import RANKABLE_METRICS, build_leaderboard

__all__ = ["render_leaderboard", "render_dataframe", "leaderboard_markdown"]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def render_dataframe(
    frame: pd.DataFrame,
    columns: Sequence[tuple[str, str]],
    *,
    max_width: int = 24,
) -> str:
    """Render selected columns as a fixed-width table.

    ``columns`` is a sequence of ``(column_name, header)`` pairs.
    """
    if frame.empty:
        return "(no results)"
    present = [(c, h) for c, h in columns if c in frame.columns]
    if not present:
        return "(no matching columns)"

    rows: list[list[str]] = []
    for _, record in frame.iterrows():
        rows.append([_fmt(record[c]) for c, _ in present])

    headers = [h for _, h in present]
    widths = [
        min(max_width, max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i]))
        for i in range(len(headers))
    ]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            cell[:w].ljust(w) if i == 0 else cell[:w].rjust(w)
            for i, (cell, w) in enumerate(zip(cells, widths))
        )

    out = [line(headers), "  ".join("-" * w for w in widths)]
    out.extend(line(r) for r in rows)
    return "\n".join(out)


def render_leaderboard(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    scenario: str | None = None,
    show_ci: bool = True,
    title: str | None = None,
) -> str:
    """Render a ranked leaderboard as text."""
    if metric not in RANKABLE_METRICS:
        raise KeyError(
            f"cannot rank by {metric!r}; available: {sorted(RANKABLE_METRICS)}"
        )
    board = build_leaderboard(results, metric=metric, scenario=scenario)
    if board.empty:
        return "(no results to rank)"

    metric_column = "calibration_slope" if metric.startswith("calibration") else metric
    columns: list[tuple[str, str]] = [
        ("rank", "#"),
        ("model_name", "Model"),
        ("model_family", "Family"),
    ]
    for name, header in (
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("calibration_slope", "Calib"),
        ("boyce", "Boyce"),
    ):
        if f"{name}_mean" in board.columns and board[f"{name}_mean"].notna().any():
            columns.append((f"{name}_mean", header))
    if show_ci and f"{metric_column}_ci_low" in board.columns:
        columns.append((f"{metric_column}_ci_low", "CI low"))
        columns.append((f"{metric_column}_ci_high", "CI high"))
        columns.append((f"{metric_column}_sd", "SD"))
    columns.extend(
        [
            ("n_species", "Species"),
            ("n_regions", "Regions"),
            ("runtime_mean", "Runtime s"),
        ]
    )

    heading = title or (
        f"Leaderboard -- ranked by {metric}"
        + (f" ({scenario})" if scenario else " (all scenarios pooled)")
    )
    parts = [heading, "=" * len(heading), "", render_dataframe(board, columns)]

    skipped = board[board["n_skipped"] > 0] if "n_skipped" in board.columns else pd.DataFrame()
    failed = board[board["n_failed"] > 0] if "n_failed" in board.columns else pd.DataFrame()
    if not skipped.empty or not failed.empty:
        parts.append("")
        parts.append("Coverage notes:")
        for _, row in skipped.iterrows():
            parts.append(
                f"  {row['model_name']}: {int(row['n_skipped'])} job(s) skipped "
                f"({row.get('skip_reasons', '')})"
            )
        for _, row in failed.iterrows():
            parts.append(f"  {row['model_name']}: {int(row['n_failed'])} job(s) failed")

    counts = board["n_species"].dropna().unique() if "n_species" in board.columns else []
    if len(counts) > 1:
        parts.append("")
        parts.append(
            "WARNING: models were evaluated on different numbers of species "
            f"({sorted(int(c) for c in counts)}). Means are not directly comparable; "
            "use `sdmbench leaderboard --common-species` for the like-for-like table."
        )
    if scenario is None:
        parts.append("")
        parts.append(
            "NOTE: scenarios are pooled. The non-spatial and spatial experiments cover "
            "different species sets, so prefer --scenario nonspatial / --scenario spatial."
        )
    return "\n".join(parts)


def leaderboard_markdown(
    results: pd.DataFrame,
    *,
    metric: str = "roc_auc",
    scenario: str | None = None,
) -> str:
    """Render a leaderboard as a Markdown table for reports."""
    board = build_leaderboard(results, metric=metric, scenario=scenario)
    if board.empty:
        return "_No results._"
    columns = [
        ("rank", "#"),
        ("model_name", "Model"),
        ("roc_auc_mean", "ROC-AUC"),
        ("pr_auc_mean", "PR-AUC"),
        ("calibration_slope_mean", "Calibration"),
        ("n_species", "Species"),
    ]
    present = [(c, h) for c, h in columns if c in board.columns]
    header = "| " + " | ".join(h for _, h in present) + " |"
    divider = "| " + " | ".join("---" for _ in present) + " |"
    rows = [
        "| " + " | ".join(_fmt(record[c]) for c, _ in present) + " |"
        for _, record in board.iterrows()
    ]
    return "\n".join([header, divider, *rows])
