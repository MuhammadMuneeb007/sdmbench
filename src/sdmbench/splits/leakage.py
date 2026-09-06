"""The leakage auditor.

Every claim this framework makes rests on one property: the independent
presence-absence survey data is used for *evaluation only*. It must never
influence feature selection, model selection, hyperparameter choice, early
stopping, scaling, representation learning, graph construction, or calibration.

Rather than trusting that by convention, sdmbench checks it. Each run is
audited and every result row carries ``leakage_audit = PASS`` or ``FAIL``. A
``FAIL`` in strict mode aborts; otherwise it is recorded loudly so the number
can never be read as clean.

The checks are deliberately cheap enough to run on every job.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from sdmbench.data.base import COORD_COLUMNS, TARGET_COLUMN, PreparedSplit, SpeciesTask
from sdmbench.exceptions import LeakageError
from sdmbench.splits.spatial import points_within_distance

__all__ = ["LeakageAuditor", "LeakageReport", "LeakageCheck", "Severity"]


class Severity(enum.Enum):
    """How much a failed check matters."""

    #: Invalidates the result outright.
    FATAL = "fatal"
    #: Suspicious; may be legitimate for some protocols.
    WARNING = "warning"


@dataclass
class LeakageCheck:
    """The outcome of one audit check."""

    name: str
    passed: bool
    severity: Severity = Severity.FATAL
    detail: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        if self.passed:
            return "PASS"
        return "FAIL" if self.severity is Severity.FATAL else "WARN"


@dataclass
class LeakageReport:
    """All checks for one job."""

    checks: list[LeakageCheck] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def add(self, check: LeakageCheck) -> None:
        self.checks.append(check)

    @property
    def failures(self) -> list[LeakageCheck]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.FATAL]

    @property
    def warnings(self) -> list[LeakageCheck]:
        return [c for c in self.checks if not c.passed and c.severity is Severity.WARNING]

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def status(self) -> str:
        """``PASS``/``FAIL`` for the ``leakage_audit`` result column."""
        if self.failures:
            return "FAIL"
        return "PASS_WITH_WARNINGS" if self.warnings else "PASS"

    def raise_if_failed(self) -> None:
        if self.failures:
            lines = [f"  - {c.name}: {c.detail}" for c in self.failures]
            raise LeakageError("data leakage detected:\n" + "\n".join(lines))

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "context": self.context,
            "checks": [
                {
                    "name": c.name,
                    "result": c.symbol,
                    "severity": c.severity.value,
                    "detail": c.detail,
                    "evidence": c.evidence,
                }
                for c in self.checks
            ],
        }

    def render(self) -> str:
        lines = [f"Leakage audit: {self.status}"]
        for c in self.checks:
            lines.append(f"  [{c.symbol}] {c.name}" + (f" -- {c.detail}" if c.detail else ""))
        return "\n".join(lines)


class LeakageAuditor:
    """Checks that a job's train/test separation is intact.

    Parameters
    ----------
    coordinate_tolerance:
        Distance below which two records are treated as the same location.
        ``0.0`` means exact coordinate equality.
    spatial_buffer_m:
        When set (the spatial scenario), verifies that no retained training
        point lies within the buffer of a test point.
    allow_coordinate_features:
        Whether longitude/latitude are permitted as model features. Default
        False: coordinates build splits and graph topology, and become
        predictors only on explicit opt-in.
    """

    def __init__(
        self,
        *,
        coordinate_tolerance: float = 0.0,
        spatial_buffer_m: float | None = None,
        allow_coordinate_features: bool = False,
        feature_value_tolerance: float = 1e-9,
    ) -> None:
        self.coordinate_tolerance = coordinate_tolerance
        self.spatial_buffer_m = spatial_buffer_m
        self.allow_coordinate_features = allow_coordinate_features
        self.feature_value_tolerance = feature_value_tolerance

    # ------------------------------------------------------------------ API --
    def audit_task(self, task: SpeciesTask) -> LeakageReport:
        """Audit a :class:`SpeciesTask` before preprocessing."""
        report = LeakageReport(
            context={
                "stage": "task",
                "region": task.region,
                "species_id": task.species_id,
                "group": task.group,
            }
        )
        self._check_test_rows_not_in_train(task, report)
        self._check_coordinate_overlap(task, report)
        self._check_duplicate_records(task, report)
        self._check_spatial_buffer(task, report)
        return report

    def audit_split(self, split: PreparedSplit) -> LeakageReport:
        """Audit a :class:`PreparedSplit` after preprocessing, before fitting."""
        report = LeakageReport(
            context={
                "stage": "split",
                "scenario": split.scenario,
                "split_id": split.split_id,
                "n_train": split.n_train,
                "n_test": split.n_test,
            }
        )
        self._check_target_not_in_features(split, report)
        self._check_coordinates_not_features(split, report)
        self._check_feature_alignment(split, report)
        self._check_preprocessing_fitted_on_train(split, report)
        self._check_identical_rows(split, report)
        return report

    def audit_graph_edges(
        self,
        edge_index: np.ndarray,
        train_mask: np.ndarray,
        test_mask: np.ndarray,
        *,
        allow_test_to_train: bool = False,
    ) -> LeakageReport:
        """Verify no message-passing edge crosses the train/test boundary.

        In a transductive graph setup an edge between a training node and a
        test node lets test environment influence the training representation.
        For benchmark validity, fitting must happen on the training subgraph
        alone.
        """
        report = LeakageReport(context={"stage": "graph"})
        edge_index = np.asarray(edge_index)
        if edge_index.size == 0:
            report.add(LeakageCheck("graph_edges_cross_boundary", True, detail="no edges"))
            return report
        if edge_index.shape[0] != 2:
            edge_index = edge_index.T

        src, dst = edge_index[0], edge_index[1]
        train_mask = np.asarray(train_mask, dtype=bool)
        test_mask = np.asarray(test_mask, dtype=bool)
        crossing = (train_mask[src] & test_mask[dst]) | (test_mask[src] & train_mask[dst])
        n_cross = int(crossing.sum())
        report.add(
            LeakageCheck(
                "graph_edges_cross_boundary",
                passed=n_cross == 0 or allow_test_to_train,
                severity=Severity.FATAL,
                detail=(
                    f"{n_cross} edge(s) connect training and test nodes"
                    if n_cross
                    else "no train/test edges"
                ),
                evidence={"n_crossing_edges": n_cross, "n_edges": int(edge_index.shape[1])},
            )
        )
        return report

    def audit_hyperparameter_selection(
        self,
        *,
        selection_used_test: bool,
        detail: str = "",
    ) -> LeakageCheck:
        """Record whether tuning touched the independent evaluation data.

        Adapters that tune must report this honestly; there is no way to infer
        it from outside.
        """
        return LeakageCheck(
            "hyperparameters_selected_without_test_labels",
            passed=not selection_used_test,
            severity=Severity.FATAL,
            detail=detail
            or (
                "independent test labels were used during model selection"
                if selection_used_test
                else "selection used training data only"
            ),
        )

    # --------------------------------------------------------------- checks --
    def _check_test_rows_not_in_train(self, task: SpeciesTask, report: LeakageReport) -> None:
        """Independent test sites must not also appear as training records."""
        if "siteid" not in task.train.columns or "siteid" not in task.test.columns:
            report.add(
                LeakageCheck(
                    "test_sites_absent_from_training",
                    True,
                    severity=Severity.WARNING,
                    detail="no siteid column; check skipped",
                )
            )
            return
        train_sites = set(task.train["siteid"].dropna().astype(str))
        test_sites = set(task.test["siteid"].dropna().astype(str))
        shared = train_sites & test_sites
        report.add(
            LeakageCheck(
                "test_sites_absent_from_training",
                passed=not shared,
                detail=(
                    f"{len(shared)} site id(s) appear in both training and test data"
                    if shared
                    else "no shared site identifiers"
                ),
                evidence={"n_shared": len(shared), "examples": sorted(shared)[:5]},
            )
        )

    def _check_coordinate_overlap(self, task: SpeciesTask, report: LeakageReport) -> None:
        """Same location under a different site id is still the same location."""
        train_xy = task.train_coords()
        test_xy = task.test_coords()
        if len(train_xy) == 0 or len(test_xy) == 0:
            report.add(LeakageCheck("no_duplicate_coordinates_across_split", True))
            return
        if self.coordinate_tolerance <= 0:
            train_keys = {tuple(np.round(r, 9)) for r in train_xy}
            n_shared = sum(1 for r in test_xy if tuple(np.round(r, 9)) in train_keys)
        else:
            close = points_within_distance(
                test_xy,
                train_xy,
                self.coordinate_tolerance,
                geographic=task.geographic,
            )
            n_shared = int(close.sum())
        report.add(
            LeakageCheck(
                "no_duplicate_coordinates_across_split",
                passed=n_shared == 0,
                # In disdat some survey sites legitimately coincide with
                # background sample locations, so this is informative rather
                # than automatically fatal for the non-spatial scenario.
                severity=Severity.WARNING,
                detail=(
                    f"{n_shared} test location(s) coincide with a training location"
                    if n_shared
                    else "no coincident locations"
                ),
                evidence={"n_shared_coordinates": n_shared},
            )
        )

    def _check_duplicate_records(self, task: SpeciesTask, report: LeakageReport) -> None:
        """Duplicated rows inflate a species' apparent sample size."""
        dup_train = int(task.train.duplicated().sum())
        dup_test = int(task.test.duplicated().sum())
        report.add(
            LeakageCheck(
                "no_duplicate_records",
                passed=dup_train == 0 and dup_test == 0,
                severity=Severity.WARNING,
                detail=f"{dup_train} duplicate training row(s), {dup_test} duplicate test row(s)",
                evidence={"train_duplicates": dup_train, "test_duplicates": dup_test},
            )
        )

    def _check_spatial_buffer(self, task: SpeciesTask, report: LeakageReport) -> None:
        """In the spatial scenario, verify the buffer actually holds."""
        if self.spatial_buffer_m is None:
            return
        if len(task.train) == 0 or len(task.test) == 0:
            report.add(LeakageCheck("spatial_buffer_respected", True, detail="empty partition"))
            return
        violating = points_within_distance(
            task.train_coords(),
            task.test_coords(),
            self.spatial_buffer_m,
            geographic=task.geographic,
        )
        n_bad = int(violating.sum())
        report.add(
            LeakageCheck(
                "spatial_buffer_respected",
                passed=n_bad == 0,
                detail=(
                    f"{n_bad} training point(s) lie within {self.spatial_buffer_m:g} m of a "
                    "test location"
                    if n_bad
                    else f"all training points are >{self.spatial_buffer_m:g} m from test data"
                ),
                evidence={"n_violations": n_bad, "buffer_m": self.spatial_buffer_m},
            )
        )

    def _check_target_not_in_features(self, split: PreparedSplit, report: LeakageReport) -> None:
        """The target must never appear as a predictor."""
        offenders = [c for c in split.feature_names if c == TARGET_COLUMN]
        # Also catch a feature that is a perfect copy of the label under
        # another name -- the classic accidental-leak signature.
        for col in split.X_train.columns:
            if col in offenders:
                continue
            values = pd.to_numeric(split.X_train[col], errors="coerce")
            if values.isna().all():
                continue
            arr = values.to_numpy(dtype=float)
            y = np.asarray(split.y_train, dtype=float)
            if len(arr) == len(y) and np.allclose(arr, y, atol=self.feature_value_tolerance):
                offenders.append(col)
        report.add(
            LeakageCheck(
                "target_not_among_features",
                passed=not offenders,
                detail=(
                    f"feature(s) identical to the target: {offenders}"
                    if offenders
                    else "target absent from the design matrix"
                ),
                evidence={"offenders": offenders},
            )
        )

    def _check_coordinates_not_features(
        self, split: PreparedSplit, report: LeakageReport
    ) -> None:
        """Coordinates as predictors must be an explicit opt-in."""
        present = [c for c in split.feature_names if c in COORD_COLUMNS]
        passed = self.allow_coordinate_features or not present
        report.add(
            LeakageCheck(
                "coordinates_not_used_as_features",
                passed=passed,
                severity=Severity.FATAL,
                detail=(
                    f"coordinate column(s) {present} used as predictors without opt-in"
                    if not passed
                    else (
                        f"coordinates {present} used as predictors (explicitly enabled)"
                        if present
                        else "coordinates are not predictors"
                    )
                ),
                evidence={"coordinate_features": present},
            )
        )

    def _check_feature_alignment(self, split: PreparedSplit, report: LeakageReport) -> None:
        """Train and test design matrices must have identical column layouts."""
        train_cols = list(split.X_train.columns)
        test_cols = list(split.X_test.columns)
        report.add(
            LeakageCheck(
                "train_test_features_aligned",
                passed=train_cols == test_cols,
                detail=(
                    "train and test feature columns differ: "
                    f"train-only={sorted(set(train_cols) - set(test_cols))}, "
                    f"test-only={sorted(set(test_cols) - set(train_cols))}"
                    if train_cols != test_cols
                    else f"{len(train_cols)} aligned feature column(s)"
                ),
            )
        )

    def _check_preprocessing_fitted_on_train(
        self, split: PreparedSplit, report: LeakageReport
    ) -> None:
        """Scaling statistics must come from training data alone.

        The recipe records the statistics it fitted. If they were fitted on the
        combined data, the *test* columns would be standardised to mean 0 as
        well; a training-fitted scaler leaves the test mean free to differ.
        Perfect centring of both is the fingerprint of a leak.
        """
        stats = split.metadata.get("preprocessing", {})
        fitted_on = stats.get("fitted_on")
        if fitted_on is None:
            report.add(
                LeakageCheck(
                    "preprocessing_fitted_on_training_only",
                    True,
                    severity=Severity.WARNING,
                    detail="preprocessing provenance not recorded; check skipped",
                )
            )
            return
        report.add(
            LeakageCheck(
                "preprocessing_fitted_on_training_only",
                passed=fitted_on == "train",
                detail=f"preprocessing statistics were fitted on: {fitted_on}",
                evidence={"fitted_on": fitted_on},
            )
        )

    def _check_identical_rows(self, split: PreparedSplit, report: LeakageReport) -> None:
        """Detect exact feature-vector duplicates spanning the split."""
        if split.n_train == 0 or split.n_test == 0:
            report.add(LeakageCheck("no_identical_rows_across_split", True))
            return
        try:
            train_keys = {
                tuple(np.round(row, 9))
                for row in split.X_train.to_numpy(dtype=float, na_value=np.nan)
            }
            n_shared = sum(
                1
                for row in split.X_test.to_numpy(dtype=float, na_value=np.nan)
                if tuple(np.round(row, 9)) in train_keys
            )
        except (TypeError, ValueError):
            # Non-numeric features (native categoricals) -- compare as strings.
            train_keys = {tuple(map(str, row)) for row in split.X_train.to_numpy()}
            n_shared = sum(
                1 for row in split.X_test.to_numpy() if tuple(map(str, row)) in train_keys
            )
        report.add(
            LeakageCheck(
                "no_identical_rows_across_split",
                passed=n_shared == 0,
                severity=Severity.WARNING,
                detail=(
                    f"{n_shared} test row(s) have a feature vector identical to a training row"
                    if n_shared
                    else "no identical feature vectors across the split"
                ),
                evidence={"n_identical_rows": n_shared},
            )
        )


def combine_reports(reports: Iterable[LeakageReport]) -> LeakageReport:
    """Merge several reports into one."""
    combined = LeakageReport(context={"stage": "combined"})
    for r in reports:
        combined.checks.extend(r.checks)
    return combined
