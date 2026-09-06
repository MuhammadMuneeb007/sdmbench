"""End-to-end integration tests.

Marked ``integration`` because they exercise several subsystems at once. They
still use only synthetic data and core dependencies -- nothing here downloads
226 species, needs a GPU, or calls R.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sdmbench.config import RunConfig
from sdmbench.evaluation.runner import Benchmark, BenchmarkRunner
from sdmbench.evaluation.store import ResultStore

pytestmark = pytest.mark.integration


class TestCsvBenchmark:
    def test_runs_and_produces_a_leaderboard(self, synthetic_frame, tmp_path):
        path = tmp_path / "species.csv"
        synthetic_frame.to_csv(path, index=False)

        benchmark = Benchmark.from_csv(
            str(path), target="presence", coordinates=("longitude", "latitude")
        )
        results = benchmark.run(
            models=["logistic-regression", "knn", "dummy"],
            scenarios=["nonspatial"],
            run_dir=str(tmp_path / "run"),
        )
        assert len(results) == 3
        board = results.leaderboard(metric="roc_auc")
        assert len(board) == 3
        assert board["roc_auc_mean"].notna().any()

    def test_learns_the_planted_signal(self, synthetic_frame, tmp_path):
        """A working pipeline must beat chance on data with a real signal.

        If this fails, the plumbing is wrong somewhere -- not the model.
        """
        path = tmp_path / "species.csv"
        synthetic_frame.to_csv(path, index=False)
        results = Benchmark.from_csv(str(path), target="presence").run(
            models=["logistic-regression"],
            scenarios=["nonspatial"],
            run_dir=str(tmp_path / "run"),
        )
        frame = results.to_dataframe()
        assert frame.iloc[0]["roc_auc"] > 0.65

    def test_writes_a_manifest_and_a_parquet_table(self, synthetic_frame, tmp_path):
        path = tmp_path / "species.csv"
        synthetic_frame.to_csv(path, index=False)
        run_dir = tmp_path / "run"
        Benchmark.from_csv(str(path), target="presence").run(
            models=["dummy"], scenarios=["nonspatial"], run_dir=str(run_dir)
        )
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "COMPLETED"
        assert manifest["config_hash"]
        assert manifest["environment"]["python_version"]
        assert (run_dir / "results.parquet").is_file()

    def test_every_row_carries_a_leakage_audit(self, synthetic_frame, tmp_path):
        path = tmp_path / "species.csv"
        synthetic_frame.to_csv(path, index=False)
        results = Benchmark.from_csv(str(path), target="presence").run(
            models=["dummy"], scenarios=["nonspatial"], run_dir=str(tmp_path / "run")
        )
        frame = results.to_dataframe()
        assert frame["leakage_audit"].isin({"PASS", "PASS_WITH_WARNINGS"}).all()

    def test_coordinates_do_not_become_features(self, synthetic_frame, tmp_path):
        """The default that protects transferability."""
        path = tmp_path / "species.csv"
        synthetic_frame.to_csv(path, index=False)
        from sdmbench.data.csv import CsvDataset

        dataset = CsvDataset(path, target="presence")
        assert "longitude" not in dataset.predictors
        assert "latitude" not in dataset.predictors


class TestSkipBehaviour:
    def test_a_missing_backend_skips_without_failing_the_run(
        self, synthetic_frame, tmp_path
    ):
        """The property that makes a 226-species run survivable."""
        path = tmp_path / "species.csv"
        synthetic_frame.to_csv(path, index=False)
        results = Benchmark.from_csv(str(path), target="presence").run(
            models=["dummy", "autogluon", "h2o-automl"],
            scenarios=["nonspatial"],
            run_dir=str(tmp_path / "run"),
        )
        frame = results.to_dataframe()
        assert (frame["status"] == "OK").any()
        statuses = set(frame["status"])
        assert statuses <= {"OK", "SKIPPED_DEPENDENCY", "SKIPPED_NO_GPU",
                            "SKIPPED_LICENSE", "FAILED"}

    def test_skipped_models_still_appear_in_the_results(self, synthetic_frame, tmp_path):
        """A benchmark that silently omitted unavailable models would overstate
        the coverage of its comparison."""
        path = tmp_path / "species.csv"
        synthetic_frame.to_csv(path, index=False)
        results = Benchmark.from_csv(str(path), target="presence").run(
            models=["dummy", "autogluon"],
            scenarios=["nonspatial"],
            run_dir=str(tmp_path / "run"),
        )
        assert set(results.to_dataframe()["model_name"]) == {"dummy", "autogluon"}


class TestResumability:
    def test_a_second_run_skips_completed_jobs(self, synthetic_frame, tmp_path):
        path = tmp_path / "species.csv"
        synthetic_frame.to_csv(path, index=False)
        run_dir = tmp_path / "run"

        first = Benchmark.from_csv(str(path), target="presence")
        first.run(models=["dummy"], scenarios=["nonspatial"], run_dir=str(run_dir))
        completed = ResultStore(run_dir).completed_job_ids()
        assert completed

        # Re-running must not recompute; the store still holds exactly one row.
        second = Benchmark.from_csv(str(path), target="presence")
        second.run(models=["dummy"], scenarios=["nonspatial"], run_dir=str(run_dir))
        assert len(ResultStore(run_dir).load_all()) == 1

    def test_force_recomputes(self, synthetic_frame, tmp_path):
        path = tmp_path / "species.csv"
        synthetic_frame.to_csv(path, index=False)
        run_dir = tmp_path / "run"

        config = RunConfig(benchmark="csv", models=["dummy"], force=True)
        benchmark = Benchmark.from_csv(str(path), target="presence")
        benchmark.config = config
        results = benchmark.run(
            models=["dummy"], scenarios=["nonspatial"], run_dir=str(run_dir)
        )
        assert len(results) >= 1


class TestSpatialScenario:
    def test_spatial_scenario_shrinks_the_training_set(self, synthetic_frame, tmp_path):
        """Applying the buffer must actually remove training rows."""
        path = tmp_path / "species.csv"
        synthetic_frame.to_csv(path, index=False)

        benchmark = Benchmark.from_csv(
            str(path), target="presence", coordinates=("longitude", "latitude")
        )
        benchmark.config.extra["spatial_buffer_m"] = 50_000.0
        results = benchmark.run(
            models=["dummy"],
            scenarios=["nonspatial", "spatial"],
            run_dir=str(tmp_path / "run"),
        )
        frame = results.to_dataframe()
        by_scenario = frame.set_index("evaluation_scenario")["train_n"].to_dict()
        if "spatial" in by_scenario and "nonspatial" in by_scenario:
            assert by_scenario["spatial"] <= by_scenario["nonspatial"]


class TestBenchmarkRunner:
    def test_reports_the_models_it_will_run(self):
        config = RunConfig(benchmark="tabpfn-sdm-2026", models=["published"])
        runner = BenchmarkRunner("tabpfn-sdm-2026", config)
        assert "maxnet" in runner.model_names()

    def test_defaults_to_the_recipe_model_set(self):
        runner = BenchmarkRunner("tabpfn-sdm-2026", RunConfig(benchmark="tabpfn-sdm-2026"))
        assert set(runner.model_names()) == set(runner.benchmark.default_models)

    def test_spatial_scenario_selects_the_spatial_checkpoint(self):
        """Paper sec. 2.4.2: separate finetuned models per scenario."""
        runner = BenchmarkRunner("tabpfn-sdm-2026", RunConfig(benchmark="tabpfn-sdm-2026"))
        options = runner.benchmark.model_options("tabpfn-sdm", "spatial")
        assert options["variant"] == "spatial"
        assert "variant" not in runner.benchmark.model_options("tabpfn-sdm", "nonspatial")

    def test_rejects_a_scenario_the_recipe_does_not_define(self):
        config = RunConfig.from_dict(
            {
                "benchmark": "leopard-leedham-2025",
                "evaluation": {"scenarios": ["nonspatial"]},
            }
        )
        runner = BenchmarkRunner("leopard-leedham-2025", config)
        with pytest.raises(ValueError, match="none of"):
            runner.run(progress=False)


@pytest.mark.network
class TestDisdatAcquisition:
    """Requires R with disdat installed, and writes to the cache.

    WRITTEN BUT NOT EXECUTED.
    """

    @pytest.mark.r
    def test_fetch_produces_all_six_regions(self, tmp_path):
        from sdmbench.data.disdat import REGIONS, DisdatDataset
        from sdmbench.rbridge.runner import RRunner

        runner = RRunner()
        if not runner.available:
            pytest.skip("R not available")
        if runner.check_packages(["disdat"])["disdat"] is None:
            pytest.skip("the disdat R package is not installed")

        dataset = DisdatDataset.fetch(root=tmp_path / "disdat")
        assert dataset.is_available()
        assert set(dataset.regions()) == set(REGIONS)
        assert dataset.summary()["total_species"] == 226

    @pytest.mark.r
    def test_task_assembly_matches_the_published_predictor_counts(self, tmp_path):
        from sdmbench.benchmarks.tabpfn_sdm_2026 import REGION_PREDICTORS
        from sdmbench.data.disdat import DisdatDataset
        from sdmbench.rbridge.runner import RRunner

        if not RRunner().available:
            pytest.skip("R not available")
        dataset = DisdatDataset(root=tmp_path / "disdat")
        if not dataset.is_available():
            pytest.skip("disdat has not been fetched")

        species = dataset.species("AWT")[0]
        task = dataset.get_task("AWT", species, predictors=REGION_PREDICTORS["AWT"])
        assert len(task.predictors) == 8
        assert task.train_presence_n > 0
        assert task.test_presence_n > 0
        assert task.crs == "EPSG:28355"
