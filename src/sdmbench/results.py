"""The result record -- the single unit of scientific output.

Every ``(species x model x scenario x split x repeat)`` job produces exactly one
:class:`RunResult`. These are written independently so a benchmark spanning
thousands of jobs is resumable, and are later concatenated into a Parquet
table for aggregation.

Design notes
------------
* Metrics live in a ``metrics`` dict rather than fixed columns so that adding a
  metric does not change the schema; :meth:`RunResult.to_row` flattens the
  well-known ones into stable top-level columns for convenience.
* ``published_reference`` never appears here. Computed numbers and published
  numbers are kept in separate namespaces so a reproduction report can never
  accidentally echo the paper's value back as if it had been measured.
* Failed and skipped jobs still produce a row. A benchmark where LightGBM was
  not installed must say so, not silently omit the model.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RunResult",
    "RESULT_COLUMNS",
    "METRIC_COLUMNS",
    "results_to_dataframe",
]

#: Metrics promoted to top-level columns in the Parquet table. Any other metric
#: computed by a benchmark remains available in the ``metrics_json`` column.
METRIC_COLUMNS = (
    "roc_auc",
    "pr_auc",
    "calibration_slope",
    "calibration_intercept",
    "boyce",
    "brier",
    "log_loss",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
)

#: Canonical column order of the results table.
RESULT_COLUMNS = (
    # identity of the experiment
    "benchmark_id",
    "benchmark_version",
    "run_id",
    "job_id",
    # identity of the biological unit
    "species_id",
    "region",
    "group",
    # identity of the evaluation
    "evaluation_scenario",
    "split_id",
    "repeat_id",
    "seed",
    # identity of the model
    "model_family",
    "model_name",
    "model_version",
    "hyperparameters_json",
    # data volumes
    "train_n",
    "train_presence_n",
    "train_background_n",
    "test_n",
    "test_presence_n",
    "test_absence_n",
    # outcome
    "status",
    "status_detail",
    *METRIC_COLUMNS,
    "metrics_json",
    # cost
    "fit_time_seconds",
    "predict_time_seconds",
    "total_time_seconds",
    "peak_memory_mb",
    "device",
    "compute_budget_json",
    # reproducibility
    "dataset_hash",
    "split_hash",
    "model_config_hash",
    "leakage_audit",
    "package_versions_json",
    "git_commit",
    "upstream_commit",
    "timestamp",
)


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunResult:
    """One evaluated ``(species, model, scenario, split)`` job.

    Notes
    -----
    ``status`` uses the values of :class:`sdmbench.models.base.ModelStatus`.
    A row with a non-``OK`` status carries no metrics but still documents that
    the combination was attempted, which is what makes a leaderboard honest.
    """

    # --- experiment identity -------------------------------------------------
    benchmark_id: str
    benchmark_version: str = "0"
    run_id: str = ""
    job_id: str = ""

    # --- biological unit -----------------------------------------------------
    species_id: str = ""
    region: str = ""
    group: str | None = None

    # --- evaluation ----------------------------------------------------------
    evaluation_scenario: str = "nonspatial"
    split_id: str = "default"
    repeat_id: int = 0
    seed: int | None = None

    # --- model ---------------------------------------------------------------
    model_family: str = ""
    model_name: str = ""
    model_version: str = ""
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # --- data volumes --------------------------------------------------------
    train_n: int = 0
    train_presence_n: int = 0
    train_background_n: int = 0
    test_n: int = 0
    test_presence_n: int = 0
    test_absence_n: int = 0

    # --- outcome -------------------------------------------------------------
    status: str = "OK"
    status_detail: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    # --- cost ----------------------------------------------------------------
    fit_time_seconds: float | None = None
    predict_time_seconds: float | None = None
    total_time_seconds: float | None = None
    peak_memory_mb: float | None = None
    device: str = "cpu"
    compute_budget: dict[str, Any] = field(default_factory=dict)

    # --- reproducibility -----------------------------------------------------
    dataset_hash: str = ""
    split_hash: str = ""
    model_config_hash: str = ""
    leakage_audit: str = "NOT_RUN"
    package_versions: dict[str, str] = field(default_factory=dict)
    git_commit: str = ""
    upstream_commit: str = ""
    timestamp: str = field(default_factory=_utcnow)

    # ------------------------------------------------------------------ API --
    @property
    def ok(self) -> bool:
        """Whether this job produced usable metrics."""
        return self.status == "OK"

    def to_row(self) -> dict[str, Any]:
        """Flatten to a single tabular row using :data:`RESULT_COLUMNS`."""
        row: dict[str, Any] = {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "run_id": self.run_id,
            "job_id": self.job_id,
            "species_id": self.species_id,
            "region": self.region,
            "group": self.group,
            "evaluation_scenario": self.evaluation_scenario,
            "split_id": self.split_id,
            "repeat_id": self.repeat_id,
            "seed": self.seed,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "hyperparameters_json": json.dumps(self.hyperparameters, sort_keys=True, default=str),
            "train_n": self.train_n,
            "train_presence_n": self.train_presence_n,
            "train_background_n": self.train_background_n,
            "test_n": self.test_n,
            "test_presence_n": self.test_presence_n,
            "test_absence_n": self.test_absence_n,
            "status": self.status,
            "status_detail": self.status_detail,
            "metrics_json": json.dumps(self.metrics, sort_keys=True, default=str),
            "fit_time_seconds": self.fit_time_seconds,
            "predict_time_seconds": self.predict_time_seconds,
            "total_time_seconds": self.total_time_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            "device": self.device,
            "compute_budget_json": json.dumps(self.compute_budget, sort_keys=True, default=str),
            "dataset_hash": self.dataset_hash,
            "split_hash": self.split_hash,
            "model_config_hash": self.model_config_hash,
            "leakage_audit": self.leakage_audit,
            "package_versions_json": json.dumps(self.package_versions, sort_keys=True),
            "git_commit": self.git_commit,
            "upstream_commit": self.upstream_commit,
            "timestamp": self.timestamp,
        }
        for metric in METRIC_COLUMNS:
            row[metric] = self.metrics.get(metric)
        return {col: row.get(col) for col in RESULT_COLUMNS}

    def to_dict(self) -> dict[str, Any]:
        """Full nested representation (used for the per-job JSON sidecar)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunResult:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def write_json(self, path) -> None:
        """Write this result to ``path`` atomically."""
        from pathlib import Path

        from sdmbench.paths import ensure_dir

        path = Path(path)
        ensure_dir(path.parent)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def read_json(cls, path) -> RunResult:
        from pathlib import Path

        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def results_to_dataframe(results):
    """Build a pandas DataFrame with the canonical column order.

    An empty input still yields a correctly-typed empty frame, so downstream
    aggregation code never needs to special-case "no results yet".
    """
    import pandas as pd

    rows = [r.to_row() for r in results]
    if not rows:
        return pd.DataFrame(columns=list(RESULT_COLUMNS))
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))
