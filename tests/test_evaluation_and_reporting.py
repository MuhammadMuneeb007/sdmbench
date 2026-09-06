"""Aggregation, paired comparison, result storage and reporting tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sdmbench.evaluation.aggregate import (
    build_leaderboard,
    common_species_leaderboard,
    coverage_report,
    per_species_table,
    summarise_metric,
)
from sdmbench.evaluation.compare import bootstrap_ci, compare_all_pairs, compare_models
from sdmbench.evaluation.store import ResultStore, job_id_for
from sdmbench.reporting.leaderboard import render_leaderboard
from sdmbench.reporting.reproduction import build_reproduction_report
from sdmbench.results import RESULT_COLUMNS, RunResult, results_to_dataframe


class TestSummariseMetric:
    def test_reports_dispersion_alongside_the_mean(self):
        stats = summarise_metric([0.7, 0.8, 0.9])
        assert stats["mean"] == pytest.approx(0.8)
        assert stats["median"] == pytest.approx(0.8)
        assert stats["sd"] > 0
        assert stats["ci_low"] < stats["mean"] < stats["ci_high"]
        assert stats["n"] == 3

    def test_ignores_nan_values(self):
        assert summarise_metric([0.5, np.nan, 0.7])["n"] == 2

    def test_empty_input_is_all_nan(self):
        stats = summarise_metric([])
        assert stats["n"] == 0 and np.isnan(stats["mean"])

    def test_single_value_has_no_interval(self):
        stats = summarise_metric([0.75])
        assert stats["mean"] == 0.75 and np.isnan(stats["ci_low"])


class TestLeaderboard:
    def test_aggregates_across_species_not_the_best_one(self, results_frame):
        """The central anti-cherry-picking rule."""
        board = build_leaderboard(results_frame, metric="roc_auc", scenario="nonspatial")
        best_single = results_frame[
            (results_frame["model_name"] == "model-a")
            & (results_frame["evaluation_scenario"] == "nonspatial")
        ]["roc_auc"].max()
        row = board[board["model_name"] == "model-a"].iloc[0]
        assert row["roc_auc_mean"] < best_single

    def test_ranks_the_better_model_first(self, results_frame):
        board = build_leaderboard(results_frame, metric="roc_auc", scenario="nonspatial")
        assert board.iloc[0]["model_name"] == "model-a"

    def test_reports_species_and_region_counts(self, results_frame):
        board = build_leaderboard(results_frame, scenario="nonspatial")
        assert (board["n_species"] == 15).all()
        assert (board["n_regions"] == 2).all()

    def test_calibration_ranks_by_distance_from_one(self):
        """A slope of 0.9 and 1.1 are equally miscalibrated."""
        frame = pd.DataFrame(
            [
                {"model_name": "under", "model_family": "c", "species_id": "s1",
                 "region": "r", "evaluation_scenario": "nonspatial", "status": "OK",
                 "calibration_slope": 0.5, "roc_auc": 0.8, "total_time_seconds": 1.0,
                 "device": "cpu"},
                {"model_name": "good", "model_family": "c", "species_id": "s1",
                 "region": "r", "evaluation_scenario": "nonspatial", "status": "OK",
                 "calibration_slope": 1.02, "roc_auc": 0.8, "total_time_seconds": 1.0,
                 "device": "cpu"},
                {"model_name": "over", "model_family": "c", "species_id": "s1",
                 "region": "r", "evaluation_scenario": "nonspatial", "status": "OK",
                 "calibration_slope": 1.6, "roc_auc": 0.8, "total_time_seconds": 1.0,
                 "device": "cpu"},
            ]
        )
        board = build_leaderboard(frame, metric="calibration_slope")
        assert board.iloc[0]["model_name"] == "good"

    def test_runtime_ranks_ascending(self, results_frame):
        board = build_leaderboard(results_frame, metric="runtime", scenario="nonspatial")
        assert board["runtime_mean"].is_monotonic_increasing

    def test_counts_skipped_and_failed_jobs(self, results_frame):
        frame = results_frame.copy()
        frame.loc[frame["model_name"] == "model-c", "status"] = "SKIPPED_DEPENDENCY"
        board = build_leaderboard(frame, scenario="nonspatial")
        row = board[board["model_name"] == "model-c"].iloc[0]
        assert row["n_skipped"] > 0
        assert row["n_species"] == 0

    def test_empty_input_returns_an_empty_frame(self):
        assert build_leaderboard(pd.DataFrame()).empty

    def test_common_species_leaderboard_uses_only_shared_species(self, results_frame):
        frame = results_frame.copy()
        drop = (frame["model_name"] == "model-b") & (frame["species_id"] == "AWT_sp00")
        frame = frame[~drop]
        board = common_species_leaderboard(frame, scenario="nonspatial")
        assert (board["n_species"] == board["n_species"].iloc[0]).all()

    def test_render_warns_when_species_counts_differ(self, results_frame):
        frame = results_frame.copy()
        frame.loc[frame["model_name"] == "model-c", "status"] = "FAILED"
        text = render_leaderboard(frame, scenario="nonspatial")
        assert "different numbers of species" in text

    def test_render_warns_when_scenarios_are_pooled(self, results_frame):
        assert "scenarios are pooled" in render_leaderboard(results_frame)

    def test_coverage_report_tabulates_statuses(self, results_frame):
        assert not coverage_report(results_frame).empty


class TestPairedComparison:
    def test_detects_a_consistent_difference(self, results_frame):
        result = compare_models(
            results_frame, "model-a", "model-b", metric="roc_auc", scenario="nonspatial"
        )
        assert result.n_species == 15
        assert result.mean_difference > 0
        assert result.wins > result.losses
        assert result.verdict() == "model-a_BETTER"

    def test_pairs_only_species_both_models_completed(self, results_frame):
        frame = results_frame.copy()
        drop = (frame["model_name"] == "model-b") & (frame["species_id"] == "AWT_sp00")
        frame = frame[~drop]
        result = compare_models(frame, "model-a", "model-b", scenario="nonspatial")
        assert result.n_species == 14
        assert any("dropped" in n for n in result.notes)

    def test_refuses_to_call_a_winner_on_a_mean_alone(self):
        """A larger mean with wide overlap must NOT be reported as better."""
        rng = np.random.default_rng(0)
        rows = []
        for i in range(20):
            for model, offset in (("a", 0.005), ("b", 0.0)):
                rows.append(
                    {
                        "model_name": f"model-{model}",
                        "species_id": f"s{i}",
                        "region": "r",
                        "evaluation_scenario": "nonspatial",
                        "status": "OK",
                        "roc_auc": 0.7 + offset + rng.normal(0, 0.15),
                    }
                )
        result = compare_models(pd.DataFrame(rows), "model-a", "model-b")
        assert result.verdict() == "NO_RELIABLE_DIFFERENCE"

    def test_reports_bootstrap_interval_and_tests(self, results_frame):
        result = compare_models(results_frame, "model-a", "model-b", scenario="nonspatial")
        assert np.isfinite(result.ci_low) and np.isfinite(result.ci_high)
        assert np.isfinite(result.wilcoxon_p)
        assert np.isfinite(result.ttest_p)
        assert np.isfinite(result.effect_size)

    def test_win_tie_loss_counts_sum_to_n(self, results_frame):
        result = compare_models(results_frame, "model-a", "model-c", scenario="nonspatial")
        assert result.wins + result.ties + result.losses == result.n_species

    def test_unknown_model_raises(self, results_frame):
        with pytest.raises(KeyError, match="available models"):
            compare_models(results_frame, "model-a", "nonexistent")

    def test_all_pairs_applies_multiple_comparison_correction(self, results_frame):
        frame = compare_all_pairs(results_frame, scenario="nonspatial", correction="holm")
        assert "wilcoxon_p_adjusted" in frame.columns
        assert (frame["wilcoxon_p_adjusted"] >= frame["wilcoxon_p"]).all()

    def test_reference_mode_compares_against_one_model(self, results_frame):
        frame = compare_all_pairs(
            results_frame, scenario="nonspatial", reference="model-a"
        )
        assert (frame["model_a"] == "model-a").all()
        assert len(frame) == 2

    def test_bootstrap_ci_brackets_the_mean(self):
        values = np.random.default_rng(0).normal(0.05, 0.02, 200)
        low, high = bootstrap_ci(values, n_boot=2000, seed=1)
        assert low < values.mean() < high

    def test_bootstrap_ci_is_reproducible(self):
        values = np.random.default_rng(0).normal(size=50)
        assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


class TestPerSpeciesTable:
    def test_produces_a_species_by_model_matrix(self, results_frame):
        table = per_species_table(results_frame, scenario="nonspatial")
        assert table.shape == (30, 3)


class TestRunResult:
    def test_row_uses_the_canonical_column_order(self):
        row = RunResult(benchmark_id="b", metrics={"roc_auc": 0.8}).to_row()
        assert list(row) == list(RESULT_COLUMNS)
        assert row["roc_auc"] == 0.8

    def test_metrics_are_available_flattened_and_as_json(self):
        row = RunResult(benchmark_id="b", metrics={"roc_auc": 0.8, "custom": 1.5}).to_row()
        assert row["roc_auc"] == 0.8
        assert json.loads(row["metrics_json"])["custom"] == 1.5

    def test_round_trips_through_json(self, tmp_path):
        original = RunResult(
            benchmark_id="b", species_id="sp", metrics={"roc_auc": 0.75},
            hyperparameters={"k": 5}
        )
        path = tmp_path / "r.json"
        original.write_json(path)
        restored = RunResult.read_json(path)
        assert restored.species_id == "sp"
        assert restored.metrics["roc_auc"] == 0.75
        assert restored.hyperparameters["k"] == 5

    def test_ok_property_reflects_status(self):
        assert RunResult(benchmark_id="b", status="OK").ok
        assert not RunResult(benchmark_id="b", status="FAILED").ok

    def test_empty_result_list_yields_a_typed_empty_frame(self):
        frame = results_to_dataframe([])
        assert frame.empty and list(frame.columns) == list(RESULT_COLUMNS)


class TestResultStore:
    def test_job_ids_are_deterministic(self):
        kwargs = dict(
            benchmark_id="b", scenario="nonspatial", region="AWT",
            species_id="sp1", model_name="knn"
        )
        assert job_id_for(**kwargs) == job_id_for(**kwargs)

    def test_job_ids_are_filesystem_safe(self):
        job = job_id_for(
            benchmark_id="b/x", scenario="s", region="R",
            species_id="sp:1", model_name="a b"
        )
        assert not set(job) & set('/\\:*?"<>|')

    def test_writes_and_reloads_results(self, tmp_path):
        store = ResultStore(tmp_path)
        store.ensure()
        store.write(
            RunResult(
                benchmark_id="b", species_id="sp1", region="AWT",
                model_name="knn", job_id="j1", metrics={"roc_auc": 0.8}
            )
        )
        loaded = store.load_all()
        assert len(loaded) == 1 and loaded[0].species_id == "sp1"

    def test_completed_job_ids_drive_resumption(self, tmp_path):
        store = ResultStore(tmp_path)
        store.ensure()
        store.write(RunResult(benchmark_id="b", job_id="done", model_name="m"))
        assert "done" in store.completed_job_ids()

    def test_failed_jobs_can_be_excluded_for_retry(self, tmp_path):
        store = ResultStore(tmp_path)
        store.ensure()
        store.write(
            RunResult(benchmark_id="b", job_id="bad", model_name="m", status="FAILED")
        )
        assert "bad" in store.completed_job_ids(include_failed=True)
        assert "bad" not in store.completed_job_ids(include_failed=False)

    def test_consolidates_to_parquet(self, tmp_path):
        store = ResultStore(tmp_path)
        store.ensure()
        for i in range(3):
            store.write(
                RunResult(
                    benchmark_id="b", job_id=f"j{i}", species_id=f"s{i}",
                    model_name="knn", metrics={"roc_auc": 0.7 + i * 0.05}
                )
            )
        path = store.write_parquet()
        assert path.is_file()
        assert len(ResultStore.read_parquet(path)) == 3


class TestReproductionReport:
    def test_published_and_reproduced_stay_separate(self):
        """The critical guard: a published value is never echoed as measured."""
        from sdmbench.benchmarks import get_benchmark

        report = build_reproduction_report(get_benchmark("tabpfn-sdm-2026"), None)
        assert report.overall_status == "NOT RUN"
        for row in report.rows:
            assert row.reproduced is None
            assert row.published > 0
            assert row.status == "NOT RUN"

    def test_published_values_match_the_paper(self):
        from sdmbench.benchmarks import get_benchmark

        references = get_benchmark("tabpfn-sdm-2026").published_results()
        lookup = {
            (e.model, e.scenario, e.metric): e.value for e in references.entries
        }
        assert lookup[("tabpfn-sdm", "nonspatial", "roc_auc")] == 0.762
        assert lookup[("maxnet", "nonspatial", "roc_auc")] == 0.732
        assert lookup[("random-forest", "nonspatial", "roc_auc")] == 0.727
        assert lookup[("brt", "nonspatial", "roc_auc")] == 0.724
        assert lookup[("gam", "nonspatial", "roc_auc")] == 0.717
        assert lookup[("tabpfn-sdm", "spatial", "roc_auc")] == 0.699

    def test_notes_separate_checkpoint_metrics_from_benchmark_results(self):
        """The model card's 0.747 is NOT the paper's 0.762 and must not be confused."""
        from sdmbench.benchmarks import get_benchmark

        notes = " ".join(get_benchmark("tabpfn-sdm-2026").published_results().notes)
        assert "0.747" in notes and "CHECKPOINT-VALIDATION" in notes

    def test_status_bands_are_applied(self):
        from sdmbench.reporting.reproduction import _status_for

        assert _status_for(0.005, match=0.010, close=0.025) == "MATCH"
        assert _status_for(0.020, match=0.010, close=0.025) == "CLOSE"
        assert _status_for(0.060, match=0.010, close=0.025) == "MISMATCH"
        assert _status_for(None, match=0.010, close=0.025) == "NOT RUN"

    def test_tolerances_are_documented_in_the_output(self):
        from sdmbench.benchmarks import get_benchmark

        payload = build_reproduction_report(get_benchmark("tabpfn-sdm-2026"), None).to_dict()
        assert "justification" in payload["tolerances"]

    def test_renders_provenance_warnings(self):
        from sdmbench.benchmarks import get_benchmark

        text = build_reproduction_report(get_benchmark("tabpfn-sdm-2026"), None).render()
        assert "UNVERIFIED" in text
        assert "PAPER" in text
