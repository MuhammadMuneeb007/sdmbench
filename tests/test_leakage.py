"""Leakage auditor tests.

Every one of these constructs a *deliberate* leak and asserts the auditor
catches it. A leakage detector that has never been shown a leak is not
evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdmbench.data.base import PreparedSplit
from sdmbench.exceptions import LeakageError
from sdmbench.splits.leakage import LeakageAuditor, LeakageReport, Severity


def make_split(**overrides) -> PreparedSplit:
    """A clean split, with individual fields overridable to inject a leak."""
    rng = np.random.default_rng(0)
    n_train, n_test = 100, 40
    X_train = pd.DataFrame(
        {"bio01": rng.normal(size=n_train), "bio12": rng.normal(size=n_train)}
    )
    X_test = pd.DataFrame(
        {"bio01": rng.normal(size=n_test), "bio12": rng.normal(size=n_test)}
    )
    defaults = dict(
        X_train=X_train,
        y_train=rng.integers(0, 2, n_train),
        X_test=X_test,
        y_test=rng.integers(0, 2, n_test),
        coords_train=rng.uniform(0, 100_000, (n_train, 2)),
        coords_test=rng.uniform(0, 100_000, (n_test, 2)),
        feature_names=["bio01", "bio12"],
        categorical_features=[],
        scenario="nonspatial",
        metadata={"preprocessing": {"fitted_on": "train"}},
    )
    defaults.update(overrides)
    return PreparedSplit(**defaults)


class TestCleanSplitPasses:
    def test_clean_split_passes(self):
        report = LeakageAuditor().audit_split(make_split())
        assert report.passed
        assert report.status in {"PASS", "PASS_WITH_WARNINGS"}

    def test_clean_task_passes(self, species_task):
        assert LeakageAuditor().audit_task(species_task).passed


class TestTargetLeakage:
    def test_detects_a_feature_identical_to_the_target(self):
        """The classic accidental leak: the label copied under another name."""
        split = make_split()
        split.X_train["sneaky"] = split.y_train.astype(float)
        split.X_test["sneaky"] = split.y_test.astype(float)
        split.feature_names = [*split.feature_names, "sneaky"]

        report = LeakageAuditor().audit_split(split)
        assert not report.passed
        assert any(c.name == "target_not_among_features" for c in report.failures)

    def test_detects_the_target_column_by_name(self):
        split = make_split()
        split.X_train["occ"] = 0.0
        split.X_test["occ"] = 0.0
        split.feature_names = [*split.feature_names, "occ"]
        assert not LeakageAuditor().audit_split(split).passed


class TestCoordinateFeatures:
    def test_coordinates_as_features_fail_by_default(self):
        """Coordinate features must be an explicit opt-in, not a default.

        This is the easiest way to produce an impressive-looking SDM that has
        learned nothing but where the surveys happened.
        """
        split = make_split()
        split.X_train["x"] = split.coords_train[:, 0]
        split.X_test["x"] = split.coords_test[:, 0]
        split.feature_names = [*split.feature_names, "x"]

        report = LeakageAuditor().audit_split(split)
        assert not report.passed
        assert any(c.name == "coordinates_not_used_as_features" for c in report.failures)

    def test_coordinates_pass_when_explicitly_allowed(self):
        split = make_split()
        split.X_train["x"] = split.coords_train[:, 0]
        split.X_test["x"] = split.coords_test[:, 0]
        split.feature_names = [*split.feature_names, "x"]

        report = LeakageAuditor(allow_coordinate_features=True).audit_split(split)
        assert report.passed


class TestPreprocessingLeakage:
    def test_detects_preprocessing_fitted_on_pooled_data(self):
        split = make_split(metadata={"preprocessing": {"fitted_on": "train+test"}})
        report = LeakageAuditor().audit_split(split)
        assert not report.passed
        assert any(
            c.name == "preprocessing_fitted_on_training_only" for c in report.failures
        )

    def test_missing_provenance_warns_rather_than_fails(self):
        """An unrecorded fit is suspicious, but not proof of a leak."""
        report = LeakageAuditor().audit_split(make_split(metadata={}))
        assert report.passed
        assert any(
            c.name == "preprocessing_fitted_on_training_only" for c in report.warnings
        )


class TestFeatureAlignment:
    def test_detects_mismatched_columns(self):
        split = make_split()
        split.X_test = split.X_test.drop(columns=["bio12"])
        report = LeakageAuditor().audit_split(split)
        assert not report.passed
        assert any(c.name == "train_test_features_aligned" for c in report.failures)


class TestSpatialBufferAudit:
    def test_detects_a_violated_buffer(self, species_task):
        """Unfiltered training data must fail an audit that expects a buffer."""
        auditor = LeakageAuditor(spatial_buffer_m=1e9)
        report = auditor.audit_task(species_task)
        assert not report.passed
        assert any(c.name == "spatial_buffer_respected" for c in report.failures)

    def test_passes_after_the_buffer_is_applied(self, species_task):
        from sdmbench.splits.spatial import apply_spatial_buffer

        filtered, _ = apply_spatial_buffer(species_task, buffer_m=10_000.0)
        if len(filtered.train) == 0:
            pytest.skip("buffer removed all training data for this fixture")
        auditor = LeakageAuditor(spatial_buffer_m=10_000.0)
        report = auditor.audit_task(filtered)
        assert not any(c.name == "spatial_buffer_respected" for c in report.failures)


class TestSharedSites:
    def test_detects_a_test_site_in_the_training_data(self, species_task):
        contaminated = species_task.train.copy()
        contaminated.loc[0, "siteid"] = species_task.test.loc[0, "siteid"]
        task = species_task.with_train(contaminated)
        report = LeakageAuditor().audit_task(task)
        assert not report.passed
        assert any(c.name == "test_sites_absent_from_training" for c in report.failures)


class TestGraphLeakage:
    def test_detects_an_edge_crossing_the_boundary(self):
        """A train-test edge lets test environment shape the fitted representation."""
        train_mask = np.array([True, True, False, False])
        test_mask = ~train_mask
        edge_index = np.array([[0, 1], [1, 2]])  # 1 -> 2 crosses
        report = LeakageAuditor().audit_graph_edges(edge_index, train_mask, test_mask)
        assert not report.passed

    def test_accepts_a_purely_within_partition_graph(self):
        train_mask = np.array([True, True, False, False])
        test_mask = ~train_mask
        edge_index = np.array([[0, 2], [1, 3]])  # 0-1 and 2-3
        assert LeakageAuditor().audit_graph_edges(edge_index, train_mask, test_mask).passed

    def test_transductive_is_allowed_when_declared(self):
        train_mask = np.array([True, True, False, False])
        test_mask = ~train_mask
        edge_index = np.array([[0, 1], [1, 2]])
        report = LeakageAuditor().audit_graph_edges(
            edge_index, train_mask, test_mask, allow_test_to_train=True
        )
        assert report.passed


class TestHyperparameterSelection:
    def test_flags_selection_that_used_test_labels(self):
        check = LeakageAuditor().audit_hyperparameter_selection(selection_used_test=True)
        assert not check.passed
        assert check.severity is Severity.FATAL

    def test_accepts_training_only_selection(self):
        assert LeakageAuditor().audit_hyperparameter_selection(selection_used_test=False).passed


class TestReport:
    def test_raise_if_failed(self):
        split = make_split(metadata={"preprocessing": {"fitted_on": "train+test"}})
        with pytest.raises(LeakageError, match="leakage"):
            LeakageAuditor().audit_split(split).raise_if_failed()

    def test_serialises_to_a_dict(self):
        payload = LeakageAuditor().audit_split(make_split()).to_dict()
        assert "status" in payload
        assert isinstance(payload["checks"], list)

    def test_renders_human_readable_text(self):
        assert "Leakage audit" in LeakageAuditor().audit_split(make_split()).render()

    def test_empty_report_passes(self):
        assert LeakageReport().passed
