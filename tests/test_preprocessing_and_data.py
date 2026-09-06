"""Preprocessing recipe, data schema and dataset-inspection tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdmbench.data.base import PreparedSplit, SpeciesTask, validate_occurrence_frame
from sdmbench.data.csv import CsvDataset, validate_table
from sdmbench.data.disdat import CATEGORICAL_VARIABLES, REGION_CRS, REGIONS
from sdmbench.data.inspector import ColumnRole, DatasetInspector
from sdmbench.exceptions import DataError, InsufficientDataError
from sdmbench.preprocessing import SdmRecipe, _estimate_yeo_johnson_lambda, _yeo_johnson


class TestSdmRecipe:
    def test_applies_the_three_published_steps(self):
        """zero-variance removal -> Yeo-Johnson -> normalise (sec. 2.1.2)."""
        recipe = SdmRecipe(["a", "b", "constant"], [])
        train = pd.DataFrame(
            {
                "a": np.random.default_rng(0).lognormal(size=200),
                "b": np.random.default_rng(1).normal(size=200),
                "constant": np.ones(200),
            }
        )
        out = recipe.fit_transform(train)
        assert "constant" not in out.columns
        assert recipe.state.dropped_zero_variance == ["constant"]
        assert out["a"].mean() == pytest.approx(0.0, abs=1e-8)
        assert out["a"].std(ddof=1) == pytest.approx(1.0, abs=1e-6)

    def test_is_fitted_on_training_data_only(self):
        """The whole point: test statistics must NOT be centred to zero."""
        rng = np.random.default_rng(2)
        recipe = SdmRecipe(["a"], [])
        train = pd.DataFrame({"a": rng.normal(10, 1, 300)})
        test = pd.DataFrame({"a": rng.normal(20, 1, 100)})  # deliberately shifted

        recipe.fit(train)
        transformed_test = recipe.transform(test)
        # If the recipe had been fitted on the pooled data, this mean would be
        # near zero. It must not be.
        assert abs(transformed_test["a"].mean()) > 1.0
        assert recipe.state.fitted_on == "train"

    def test_categorical_columns_bypass_numeric_steps(self):
        recipe = SdmRecipe(["a"], ["veg"])
        train = pd.DataFrame(
            {"a": np.random.default_rng(3).normal(size=50), "veg": ["x", "y"] * 25}
        )
        out = recipe.fit_transform(train)
        assert set(out["veg"].dropna().unique()) <= {"x", "y"}
        assert "veg" not in recipe.state.lambdas

    def test_unseen_test_category_becomes_nan_not_a_new_level(self):
        """Levels come from training only; adding one at transform time would
        change the feature space between fit and predict."""
        recipe = SdmRecipe([], ["veg"])
        recipe.fit(pd.DataFrame({"veg": ["x", "y"] * 10}))
        out = recipe.transform(pd.DataFrame({"veg": ["x", "z"]}))
        assert out["veg"].iloc[0] == "x"
        assert pd.isna(out["veg"].iloc[1])

    def test_transform_before_fit_raises(self):
        with pytest.raises(DataError, match="fitted"):
            SdmRecipe(["a"], []).transform(pd.DataFrame({"a": [1.0]}))

    def test_reports_its_own_provenance(self):
        recipe = SdmRecipe(["a"], [])
        recipe.fit(pd.DataFrame({"a": np.random.default_rng(4).normal(size=30)}))
        described = recipe.describe()
        assert described["steps"] == ["remove_zero_variance", "yeo_johnson", "normalize"]
        assert described["state"]["fitted_on"] == "train"


class TestYeoJohnson:
    def test_handles_negative_zero_and_positive_values(self):
        x = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        out = _yeo_johnson(x, 0.5)
        assert np.all(np.isfinite(out))
        assert np.all(np.diff(out) > 0)  # strictly monotone

    def test_lambda_one_is_a_shift_for_positive_values(self):
        x = np.array([0.0, 1.0, 2.0])
        assert np.allclose(_yeo_johnson(x, 1.0), x)

    def test_lambda_zero_is_log1p_for_positive_values(self):
        x = np.array([0.0, 1.0, 9.0])
        assert np.allclose(_yeo_johnson(x, 0.0), np.log1p(x))

    def test_estimated_lambda_is_bounded_like_recipes(self):
        """recipes::step_YeoJohnson bounds lambda to [-5, 5]."""
        lam = _estimate_yeo_johnson_lambda(
            np.random.default_rng(0).lognormal(size=500)
        )
        assert -5.0 <= lam <= 5.0

    def test_degenerate_input_falls_back_to_identity(self):
        assert _estimate_yeo_johnson_lambda(np.ones(10)) == 1.0

    def test_nan_passes_through(self):
        out = _yeo_johnson(np.array([1.0, np.nan, 3.0]), 0.5)
        assert np.isnan(out[1])


class TestSpeciesTask:
    def test_counts_presences_and_background(self, species_task):
        assert species_task.train_presence_n + species_task.train_background_n == len(
            species_task.train
        )

    def test_rejects_a_missing_predictor(self, species_task):
        with pytest.raises(DataError, match="missing predictor"):
            SpeciesTask(
                region="TEST",
                species_id="sp",
                train=species_task.train,
                test=species_task.test,
                predictors=["nonexistent"],
            )

    def test_rejects_a_non_binary_target(self, species_task):
        broken = species_task.train.copy()
        broken.loc[0, "occ"] = 7
        with pytest.raises(DataError, match="binary"):
            SpeciesTask(
                region="TEST",
                species_id="sp",
                train=broken,
                test=species_task.test,
                predictors=["bio01"],
            )

    def test_single_class_training_data_raises_skippable(self, species_task):
        single = species_task.train.copy()
        single["occ"] = 1
        with pytest.raises(InsufficientDataError):
            species_task.with_train(single).require_both_classes()

    def test_with_train_preserves_metadata(self, species_task):
        replaced = species_task.with_train(species_task.train.head(10))
        assert replaced.crs == species_task.crs
        assert replaced.predictors == species_task.predictors
        assert len(replaced.train) == 10


class TestPreparedSplit:
    def test_categorical_indices_are_positional(self):
        split = PreparedSplit(
            X_train=pd.DataFrame({"a": [1.0], "veg": ["x"], "b": [2.0]}),
            y_train=np.array([1]),
            X_test=pd.DataFrame({"a": [1.0], "veg": ["x"], "b": [2.0]}),
            y_test=np.array([0]),
            coords_train=np.zeros((1, 2)),
            coords_test=np.zeros((1, 2)),
            feature_names=["a", "veg", "b"],
            categorical_features=["veg"],
        )
        assert split.categorical_indices == [1]

    def test_as_arrays_encodes_categoricals_on_training_levels(self):
        split = PreparedSplit(
            X_train=pd.DataFrame({"veg": ["x", "y"]}),
            y_train=np.array([0, 1]),
            X_test=pd.DataFrame({"veg": ["y", "unseen"]}),
            y_test=np.array([1, 0]),
            coords_train=np.zeros((2, 2)),
            coords_test=np.zeros((2, 2)),
            feature_names=["veg"],
            categorical_features=["veg"],
        )
        _, _, X_test, _ = split.as_arrays()
        assert X_test[1, 0] == -1  # unseen level, not a new code

    def test_rejects_length_mismatch(self):
        with pytest.raises(DataError, match="length mismatch"):
            PreparedSplit(
                X_train=pd.DataFrame({"a": [1.0, 2.0]}),
                y_train=np.array([1]),
                X_test=pd.DataFrame({"a": [1.0]}),
                y_test=np.array([0]),
                coords_train=np.zeros((2, 2)),
                coords_test=np.zeros((1, 2)),
                feature_names=["a"],
            )


class TestDisdatConstants:
    def test_six_regions(self):
        assert len(REGIONS) == 6
        assert set(REGIONS) == {"AWT", "CAN", "NSW", "NZ", "SA", "SWI"}

    def test_five_categorical_variables(self):
        """Paper sec. 2.1.1 and the disdat vignette agree on exactly these."""
        assert set(CATEGORICAL_VARIABLES) == {"ontveg", "vegsys", "toxicats", "age", "calc"}

    def test_projected_and_geographic_regions_are_distinguished(self):
        """The distinction the 10 km buffer depends on."""
        assert not REGION_CRS["AWT"].geographic  # UTM 55S, metres
        assert not REGION_CRS["NZ"].geographic   # NZMG, metres
        assert not REGION_CRS["SWI"].geographic  # CH1903, metres
        assert REGION_CRS["CAN"].geographic      # longlat
        assert REGION_CRS["NSW"].geographic      # WGS84
        assert REGION_CRS["SA"].geographic       # WGS84

    def test_paper_predictor_lists_are_subsets_not_the_full_disdat_set(self):
        """The paper uses FEWER predictors than disPredictors() returns.

        AWT: 8 of 13, NSW: 12 of 13. Using the full set would change every
        number, so this pins the published selection.
        """
        from sdmbench.benchmarks.tabpfn_sdm_2026 import REGION_PREDICTORS

        assert len(REGION_PREDICTORS["AWT"]) == 8
        assert len(REGION_PREDICTORS["NSW"]) == 12
        assert len(REGION_PREDICTORS["CAN"]) == 7
        assert len(REGION_PREDICTORS["NZ"]) == 11
        assert len(REGION_PREDICTORS["SA"]) == 8
        assert len(REGION_PREDICTORS["SWI"]) == 12
        assert "tempmin" not in REGION_PREDICTORS["NSW"]  # dropped by the paper
        assert "bc01" not in REGION_PREDICTORS["AWT"]


class TestCsvValidation:
    def test_accepts_a_well_formed_table(self, synthetic_frame):
        report = validate_table(
            synthetic_frame, target="presence", longitude="longitude", latitude="latitude"
        )
        assert report.ok

    def test_rejects_a_non_binary_target(self, synthetic_frame):
        frame = synthetic_frame.copy()
        frame["presence"] = np.arange(len(frame))
        assert not validate_table(frame, target="presence").ok

    def test_rejects_a_missing_target_column(self, synthetic_frame):
        assert not validate_table(synthetic_frame, target="nope").ok

    def test_warns_about_coordinate_like_predictors(self, synthetic_frame):
        report = validate_table(
            synthetic_frame,
            target="presence",
            predictors=["longitude", "latitude", "bio01"],
        )
        assert any("coordinate" in w.lower() for w in report.warnings)

    def test_warns_about_extreme_imbalance(self):
        frame = pd.DataFrame(
            {"presence": [1] * 5 + [0] * 5000, "bio01": np.random.default_rng(0).normal(size=5005)}
        )
        report = validate_table(frame, target="presence")
        assert any("imbalance" in w.lower() for w in report.warnings)

    def test_csv_dataset_excludes_coordinates_from_predictors_by_default(
        self, synthetic_frame, tmp_path
    ):
        path = tmp_path / "s.csv"
        synthetic_frame.to_csv(path, index=False)
        dataset = CsvDataset(path, target="presence")
        assert "longitude" not in dataset.predictors
        assert "latitude" not in dataset.predictors

    def test_coordinates_included_only_on_explicit_opt_in(self, synthetic_frame, tmp_path):
        path = tmp_path / "s.csv"
        synthetic_frame.to_csv(path, index=False)
        dataset = CsvDataset(path, target="presence", include_coordinates=True)
        assert "longitude" in dataset.predictors

    def test_holdout_provenance_is_recorded(self, synthetic_frame, tmp_path):
        """A holdout from the same table is NOT an independent survey."""
        path = tmp_path / "s.csv"
        synthetic_frame.to_csv(path, index=False)
        task = CsvDataset(path, target="presence").to_task()
        assert task.metadata["test_data"] == "holdout"
        assert "not an independent survey" in task.metadata["note"]

    def test_maps_yes_no_labels(self, tmp_path):
        frame = pd.DataFrame(
            {
                "presence": ["yes", "no"] * 20,
                "longitude": np.linspace(0, 1, 40),
                "latitude": np.linspace(0, 1, 40),
                "bio01": np.random.default_rng(0).normal(size=40),
            }
        )
        path = tmp_path / "s.csv"
        frame.to_csv(path, index=False)
        task = CsvDataset(path, target="presence").to_task()
        assert set(task.train["occ"].unique()) <= {0, 1}


class TestDatasetInspector:
    def test_identifies_the_target(self, synthetic_frame):
        report = DatasetInspector(synthetic_frame).inspect()
        best = report.best(ColumnRole.TARGET)
        assert best is not None and best.column == "presence"

    def test_identifies_coordinates(self, synthetic_frame):
        report = DatasetInspector(synthetic_frame).inspect()
        assert report.best(ColumnRole.LONGITUDE).column == "longitude"
        assert report.best(ColumnRole.LATITUDE).column == "latitude"

    def test_hints_at_the_climate_modality(self, synthetic_frame):
        report = DatasetInspector(synthetic_frame).inspect()
        bio01 = next(g for g in report.guesses if g.column == "bio01")
        assert bio01.modality_hint == "climate"

    def test_treats_a_string_column_as_categorical(self, synthetic_frame):
        report = DatasetInspector(synthetic_frame).inspect()
        veg = next(g for g in report.guesses if g.column == "vegetation")
        assert veg.role is ColumnRole.CATEGORICAL_PREDICTOR

    def test_flags_an_all_unique_column_as_an_identifier(self, synthetic_frame):
        frame = synthetic_frame.copy()
        frame["record_id"] = [f"r{i}" for i in range(len(frame))]
        report = DatasetInspector(frame).inspect()
        guess = next(g for g in report.guesses if g.column == "record_id")
        assert guess.role is ColumnRole.IDENTIFIER

    def test_produces_an_editable_draft_schema(self, synthetic_frame):
        schema = DatasetInspector(synthetic_frame).inspect().draft_schema()
        assert schema["target"] == "presence"
        assert schema["coordinates"]["longitude"] == "longitude"
        assert "modalities" in schema

    def test_suggestions_carry_confidence_and_a_reason(self, synthetic_frame):
        for guess in DatasetInspector(synthetic_frame).inspect().guesses:
            assert 0.0 <= guess.confidence <= 1.0
            assert guess.reason


class TestValidateOccurrenceFrame:
    def test_requires_coordinate_columns(self):
        with pytest.raises(DataError, match="coordinate"):
            validate_occurrence_frame(
                pd.DataFrame({"occ": [1], "a": [1.0]}), predictors=["a"]
            )

    def test_requires_the_target_unless_waived(self):
        frame = pd.DataFrame({"x": [1.0], "y": [1.0], "a": [1.0]})
        with pytest.raises(DataError, match="target"):
            validate_occurrence_frame(frame, predictors=["a"])
        validate_occurrence_frame(frame, predictors=["a"], require_target=False)
