"""The evaluation engine: running, storing, aggregating and comparing."""

from sdmbench.evaluation.aggregate import (
    RANKABLE_METRICS,
    build_leaderboard,
    common_species_leaderboard,
    coverage_report,
    per_species_table,
    summarise_metric,
)
from sdmbench.evaluation.compare import ComparisonResult, compare_all_pairs, compare_models
from sdmbench.evaluation.runner import Benchmark, BenchmarkResults, BenchmarkRunner
from sdmbench.evaluation.store import ResultStore, job_id_for

__all__ = [
    "RANKABLE_METRICS",
    "build_leaderboard",
    "common_species_leaderboard",
    "coverage_report",
    "per_species_table",
    "summarise_metric",
    "ComparisonResult",
    "compare_all_pairs",
    "compare_models",
    "Benchmark",
    "BenchmarkResults",
    "BenchmarkRunner",
    "ResultStore",
    "job_id_for",
]
