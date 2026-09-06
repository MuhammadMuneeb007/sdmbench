"""Rendering results: leaderboards, reproduction reports, plots."""

from sdmbench.reporting.leaderboard import (
    leaderboard_markdown,
    render_dataframe,
    render_leaderboard,
)
from sdmbench.reporting.reproduction import (
    CLOSE_TOLERANCE,
    MATCH_TOLERANCE,
    ReproductionReport,
    ReproductionRow,
    build_reproduction_report,
    extra_models_table,
)

__all__ = [
    "leaderboard_markdown",
    "render_dataframe",
    "render_leaderboard",
    "CLOSE_TOLERANCE",
    "MATCH_TOLERANCE",
    "ReproductionReport",
    "ReproductionRow",
    "build_reproduction_report",
    "extra_models_table",
]
