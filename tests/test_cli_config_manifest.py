"""CLI, configuration, reproducibility and R-bridge tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdmbench.cli import build_parser, main
from sdmbench.config import RunConfig, load_config
from sdmbench.exceptions import ConfigurationError
from sdmbench.provenance import Provenance, ProvenanceReport, Sourced
from sdmbench.rbridge.runner import RRunner, find_rscript
from sdmbench.reproducibility.hashes import (
    hash_dataframe,
    hash_json,
    hash_split,
)
from sdmbench.reproducibility.manifest import RunManifest, new_run_id

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestConfig:
    def test_flattens_grouped_model_lists(self):
        config = RunConfig.from_dict(
            {
                "benchmark": {"name": "tabpfn-sdm-2026"},
                "models": {"published": ["maxnet", "gam"], "classical": ["knn"]},
            }
        )
        assert config.models == ["maxnet", "gam", "knn"]

    def test_deduplicates_models_preserving_order(self):
        config = RunConfig.from_dict(
            {"benchmark": "b", "models": {"a": ["knn", "gam"], "b": ["knn"]}}
        )
        assert config.models == ["knn", "gam"]

    def test_parses_a_comma_separated_string(self):
        config = RunConfig.from_dict({"benchmark": "b", "models": "knn, gam,brt"})
        assert config.models == ["knn", "gam", "brt"]

    def test_requires_a_benchmark_name(self):
        with pytest.raises(ConfigurationError, match="benchmark.name"):
            RunConfig.from_dict({})

    def test_rejects_an_empty_scenario_list(self):
        with pytest.raises(ConfigurationError, match="at least one"):
            RunConfig.from_dict({"benchmark": "b", "evaluation": {"scenarios": []}})

    def test_accepts_a_recipe_specific_scenario_name(self):
        """Which scenario NAMES are valid belongs to the benchmark, not the config.

        `leopard-leedham-2025` defines `spatial_cv`, so the configuration format
        must not hard-code {nonspatial, spatial}. The runner rejects a name the
        chosen recipe does not define -- see
        `TestBenchmarkRunner::test_rejects_a_scenario_the_recipe_does_not_define`.
        """
        config = RunConfig.from_dict(
            {"benchmark": "leopard-leedham-2025", "evaluation": {"scenarios": ["spatial_cv"]}}
        )
        assert config.evaluation.scenarios == ["spatial_cv"]

    def test_defaults_to_the_published_seed(self):
        assert RunConfig(benchmark="b").seed == 32639

    def test_canonical_json_is_order_independent(self):
        a = RunConfig.from_dict({"benchmark": "b", "models": ["x"], "seed": 1})
        b = RunConfig.from_dict({"seed": 1, "models": ["x"], "benchmark": "b"})
        assert a.canonical_json() == b.canonical_json()

    def test_overrides_ignore_none(self):
        config = load_config(None, overrides={"benchmark": "b", "seed": None})
        assert config.seed == 32639

    @pytest.mark.parametrize(
        "path",
        [
            "configs/tabpfn_sdm_2026.yaml",
            "configs/extended_benchmark.yaml",
            "configs/leopard_leedham_2025.yaml",
            "configs/examples/minimal_csv.yaml",
        ],
    )
    def test_shipped_configs_parse(self, path):
        config = RunConfig.from_yaml(REPO_ROOT / path)
        assert config.benchmark
        assert config.seed == 32639

    def test_shipped_tabpfn_config_names_the_published_models(self):
        config = RunConfig.from_yaml(REPO_ROOT / "configs/tabpfn_sdm_2026.yaml")
        for model in ("maxnet", "random-forest", "brt", "gam", "tabpfn-sdm"):
            assert model in config.models
        assert config.evaluation.scenarios == ["nonspatial", "spatial"]
        assert config.metrics == ["roc_auc", "pr_auc", "calibration_slope"]


class TestProvenance:
    def test_sourced_requires_a_citation(self):
        with pytest.raises(ValueError, match="citation"):
            Sourced(1.0, Provenance.PAPER, "")

    def test_unverified_values_are_flagged(self):
        report = ProvenanceReport("b")
        report.add("verified", Sourced(1, Provenance.PAPER, "sec. 1"))
        report.add("guessed", Sourced(2, Provenance.UNVERIFIED, "no source"))
        assert not report.is_fully_verified
        assert set(report.unverified) == {"guessed"}

    def test_render_marks_unverified_entries(self):
        report = ProvenanceReport("b")
        report.add("guessed", Sourced(2, Provenance.UNVERIFIED, "no source"))
        text = report.render()
        assert "!" in text and "UNVERIFIED" in text

    def test_tabpfn_recipe_declares_its_gaps(self):
        """The reproduction must be honest about what it could not verify."""
        from sdmbench.benchmarks import get_benchmark

        report = get_benchmark("tabpfn-sdm-2026").provenance()
        assert not report.is_fully_verified
        assert "upstream_repository" in report.unverified
        assert "gam_region_formulas" in report.unverified
        # And the things it CAN verify are marked as such.
        assert report.entries["spatial_buffer_m"].provenance is Provenance.PAPER
        assert report.entries["seed"].provenance is Provenance.PAPER


class TestHashes:
    def test_dataframe_hash_is_column_order_independent(self):
        a = pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]})
        assert hash_dataframe(a) == hash_dataframe(a[["y", "x"]])

    def test_dataframe_hash_changes_with_values(self):
        a = pd.DataFrame({"x": [1, 2]})
        b = pd.DataFrame({"x": [1, 3]})
        assert hash_dataframe(a) != hash_dataframe(b)

    def test_dataframe_hash_tolerates_float_noise(self):
        a = pd.DataFrame({"x": [1.0, 2.0]})
        b = pd.DataFrame({"x": [1.0 + 1e-15, 2.0]})
        assert hash_dataframe(a) == hash_dataframe(b)

    def test_json_hash_is_key_order_independent(self):
        assert hash_json({"a": 1, "b": 2}) == hash_json({"b": 2, "a": 1})

    def test_split_hash_describes_membership_not_order(self):
        assert hash_split([1, 2, 3], [4, 5]) == hash_split([3, 2, 1], [5, 4])

    def test_split_hash_changes_with_membership(self):
        assert hash_split([1, 2, 3], [4, 5]) != hash_split([1, 2], [3, 4, 5])

    def test_split_hash_includes_extras(self):
        assert hash_split([1], [2], extra={"buffer_m": 10_000}) != hash_split(
            [1], [2], extra={"buffer_m": 5_000}
        )


class TestManifest:
    def test_run_ids_are_unique_and_time_ordered(self):
        ids = [new_run_id("t") for _ in range(5)]
        assert len(set(ids)) == 5
        assert all(i.startswith("t-") for i in ids)

    def test_captures_the_environment(self):
        manifest = RunManifest.start(benchmark_id="b", include_r=False)
        assert manifest.environment["python_version"]
        assert "packages" in manifest.environment
        assert manifest.status == "RUNNING"

    def test_round_trips_through_disk(self, tmp_path):
        manifest = RunManifest.start(benchmark_id="b", config={"x": 1}, include_r=False)
        manifest.note("a caveat worth recording")
        manifest.finish("COMPLETED")
        manifest.write(tmp_path)

        restored = RunManifest.read(tmp_path)
        assert restored.benchmark_id == "b"
        assert restored.status == "COMPLETED"
        assert "a caveat worth recording" in restored.notes

    def test_config_hash_is_stable(self, tmp_path):
        a = RunManifest.start(benchmark_id="b", config={"x": 1, "y": 2}, include_r=False)
        b = RunManifest.start(benchmark_id="b", config={"y": 2, "x": 1}, include_r=False)
        assert a.config_hash == b.config_hash

    def test_records_checkpoint_provenance(self, tmp_path):
        manifest = RunManifest.start(benchmark_id="b", include_r=False)
        manifest.record_checkpoint(
            "tabpfn-sdm", {"repo_id": "r", "sha256": "abc", "revision": "def"}
        )
        manifest.write(tmp_path)
        assert RunManifest.read(tmp_path).model_checkpoints["tabpfn-sdm"]["sha256"] == "abc"


class TestCliParser:
    def test_builds_without_error(self):
        assert build_parser() is not None

    @pytest.mark.parametrize(
        "argv",
        [
            ["data", "fetch", "disdat"],
            ["data", "info"],
            ["data", "list"],
            ["inspect", "f.csv"],
            ["benchmark", "--benchmark", "tabpfn-sdm-2026"],
            ["benchmark-csv", "f.csv", "--target", "presence"],
            ["reproduce", "tabpfn-sdm-2026"],
            ["report", "tabpfn-sdm-2026"],
            ["leaderboard", "--metric", "roc_auc"],
            ["compare", "a", "b"],
            ["models"],
            ["benchmarks"],
            ["provenance", "tabpfn-sdm-2026"],
            ["upstream", "check", "tabpfn-sdm-2026"],
            ["env", "check"],
        ],
    )
    def test_documented_commands_parse(self, argv):
        args = build_parser().parse_args(argv)
        assert hasattr(args, "func")

    def test_csv_lists_are_split(self):
        args = build_parser().parse_args(
            ["benchmark", "--models", "knn,xgboost, gam"]
        )
        assert args.models == ["knn", "xgboost", "gam"]

    def test_include_coordinates_is_off_by_default(self):
        args = build_parser().parse_args(["benchmark-csv", "f.csv"])
        assert args.include_coordinates is False


class TestCliCommands:
    def test_models_command_runs(self, capsys):
        assert main(["models"]) == 0
        assert "knn" in capsys.readouterr().out

    def test_benchmarks_command_lists_recipes(self, capsys):
        assert main(["benchmarks"]) == 0
        assert "tabpfn-sdm-2026" in capsys.readouterr().out

    def test_provenance_command_exits_nonzero_when_unverified(self, capsys):
        """An unverified recipe must not report a clean exit code."""
        assert main(["provenance", "tabpfn-sdm-2026"]) == 3
        assert "UNVERIFIED" in capsys.readouterr().out

    def test_report_without_results_says_not_run(self, capsys):
        assert main(["report", "tabpfn-sdm-2026", "--run-dir", "/nonexistent"]) == 0
        assert "NOT RUN" in capsys.readouterr().out

    def test_data_info_fails_cleanly_when_not_fetched(self, capsys, monkeypatch, tmp_path):
        monkeypatch.setenv("SDMBENCH_CACHE", str(tmp_path))
        assert main(["data", "info"]) == 1
        assert "sdmbench data fetch disdat" in capsys.readouterr().out

    @pytest.mark.integration
    def test_console_entry_point_is_installed(self):
        result = subprocess.run(
            [sys.executable, "-m", "sdmbench.cli", "--version"],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0


class TestRBridge:
    def test_find_rscript_returns_a_path_or_none(self):
        found = find_rscript()
        assert found is None or Path(found).exists()

    def test_runner_reports_availability(self):
        assert isinstance(RRunner().available, bool)

    def test_missing_r_raises_an_actionable_error(self):
        runner = RRunner()
        if runner.available:
            pytest.skip("R is installed; cannot exercise the missing path")
        from sdmbench.exceptions import MissingDependencyError

        with pytest.raises(MissingDependencyError, match="SDMBENCH_RSCRIPT"):
            runner.require_available()

    def test_every_referenced_r_script_exists(self):
        """A missing script would only surface mid-run without this."""
        from sdmbench.rbridge.runner import SCRIPTS_DIR

        for script in (
            "common.R", "fetch_disdat.R", "maxnet.R", "brt.R", "gam.R",
            "randomforest.R", "maxent_java.R", "metrics.R", "spatial_split.R",
        ):
            assert (SCRIPTS_DIR / script).is_file(), f"missing R script: {script}"

    @pytest.mark.r
    def test_check_packages_reports_versions(self):
        runner = RRunner()
        if not runner.available:
            pytest.skip("R not available")
        versions = runner.check_packages(["jsonlite", "definitely-not-a-package"])
        assert versions["definitely-not-a-package"] is None

    @pytest.mark.r
    def test_metrics_script_agrees_with_python(self):
        """Cross-language parity for ROC-AUC and Miller's calibration slope.

        WRITTEN BUT NOT EXECUTED in the session that created this file.
        """
        runner = RRunner()
        if not runner.available:
            pytest.skip("R not available")
        runner.require_packages(["jsonlite", "yardstick"])

        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300).tolist()
        p = rng.uniform(0.01, 0.99, 300).tolist()

        payload = runner.run_script(
            "metrics.R", {"truth": y, "prob": p, "metrics": ["roc_auc", "calibration"]}
        ).require()

        from sdmbench.metrics import roc_auc
        from sdmbench.metrics.calibration import miller_calibration

        assert payload["roc_auc"] == pytest.approx(roc_auc(y, p), abs=1e-6)
        assert payload["calibration_slope"] == pytest.approx(
            miller_calibration(y, p)["calibration_slope"], abs=1e-4
        )

    @pytest.mark.r
    def test_spatial_filter_agrees_with_sf(self):
        """The Python buffer must select the same rows as sf::st_is_within_distance.

        WRITTEN BUT NOT EXECUTED.
        """
        runner = RRunner()
        if not runner.available:
            pytest.skip("R not available")
        runner.require_packages(["jsonlite", "sf"])

        rng = np.random.default_rng(1)
        train = pd.DataFrame(
            {"x": rng.uniform(0, 200_000, 200), "y": rng.uniform(0, 200_000, 200)}
        )
        test = pd.DataFrame(
            {"x": rng.uniform(0, 200_000, 30), "y": rng.uniform(0, 200_000, 30)}
        )
        payload = runner.run_script(
            "spatial_split.R",
            {"crs": "EPSG:28355", "buffer_m": 10_000},
            frames={"train": train, "test": test},
        ).require()

        from sdmbench.splits.spatial import spatial_buffer_mask

        keep = spatial_buffer_mask(
            train.to_numpy(), test.to_numpy(), buffer_m=10_000.0, geographic=False
        )
        # R returns 1-based indices.
        assert sorted(np.flatnonzero(keep) + 1) == sorted(payload["retained_indices"])

    @pytest.mark.r
    def test_boyce_agrees_with_tidysdm(self):
        """Cross-language validation of the Continuous Boyce Index.

        WRITTEN BUT NOT EXECUTED.
        """
        runner = RRunner()
        if not runner.available:
            pytest.skip("R not available")
        runner.require_packages(["jsonlite", "tidysdm"])

        rng = np.random.default_rng(2)
        p = rng.uniform(size=1_000)
        y = (rng.uniform(size=1_000) < p).astype(int)
        payload = runner.run_script(
            "metrics.R", {"truth": y.tolist(), "prob": p.tolist(), "metrics": ["boyce"]}
        ).require()

        from sdmbench.metrics.ecological import continuous_boyce_index

        if payload.get("boyce") is not None:
            assert payload["boyce"] == pytest.approx(
                continuous_boyce_index(y, p), abs=0.05
            )
