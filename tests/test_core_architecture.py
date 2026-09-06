"""Tests for the v2 architecture: registries, modalities, methodology, planner, DAG."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdmbench.core.experiment import (
    ExperimentGraph,
    ExperimentNode,
    NodeKind,
    RepresentationCache,
    build_experiment_graph,
)
from sdmbench.core.methodology import Methodology, ModalitySpec
from sdmbench.core.modality import (
    STANDARD_MODALITIES,
    EcoDataset,
    ModalityData,
    ModalityKind,
    ModalityMetadata,
)
from sdmbench.core.planner import Stage, StagedPlanner
from sdmbench.core.registry import Registry, Spec
from sdmbench.exceptions import ConfigurationError, DataError
from sdmbench.fusion import FUSION_STRATEGIES
from sdmbench.graphs import GRAPH_BUILDERS
from sdmbench.objectives import OBJECTIVES
from sdmbench.representations import REPRESENTATIONS


class TestRegistry:
    def test_registers_and_resolves(self):
        registry: Registry = Registry("test_kind_a")

        @registry.register("thing", "thing-alias", description="a thing")
        class Thing:
            pass

        assert registry.get("thing") is Thing
        assert registry.get("thing-alias") is Thing
        assert "thing" in registry

    def test_normalises_underscores_and_case(self):
        registry: Registry = Registry("test_kind_b")

        @registry.register("my_thing")
        class MyThing:
            pass

        assert registry.get("my-thing") is MyThing
        assert registry.get("MY_THING") is MyThing

    def test_duplicate_registration_raises(self):
        """Silent shadowing would make a config look right and behave wrong."""
        registry: Registry = Registry("test_kind_c")

        @registry.register("dup")
        class A:
            pass

        with pytest.raises(ConfigurationError, match="already registered"):

            @registry.register("dup")
            class B:
                pass

    def test_replace_is_allowed_when_explicit(self):
        registry: Registry = Registry("test_kind_d")

        @registry.register("x")
        class A:
            pass

        class B:
            pass

        registry.add("x", B, replace=True)
        assert registry.get("x") is B

    def test_unknown_name_lists_alternatives(self):
        registry: Registry = Registry("test_kind_e")
        registry.add("known", type("K", (), {}))
        with pytest.raises(ConfigurationError, match="known"):
            registry.get("unknown")

    def test_spec_records_accepted_kinds(self):
        registry: Registry = Registry("test_kind_f")

        @registry.register("r", spec=Spec(accepts=("raster",)))
        class R:
            pass

        assert registry.spec("r").accepts_kind("raster")
        assert not registry.spec("r").accepts_kind("tabular")
        assert registry.accepting("raster") == ["r"]

    def test_availability_reflects_missing_dependencies(self):
        registry: Registry = Registry("test_kind_g")

        @registry.register("needs", spec=Spec(requires=("definitely_not_a_module",)))
        class Needs:
            pass

        available, detail = registry.resolve("needs").available()
        assert not available and "missing" in detail


class TestBuiltinRegistries:
    def test_representations_are_populated(self):
        for name in ("raw-tabular", "standardized-tabular", "pca", "categorical",
                     "patch-statistics", "temporal-summary"):
            assert name in REPRESENTATIONS

    def test_fusion_strategies_are_populated(self):
        for name in ("concat", "early", "late", "weighted", "gated",
                     "cross-attention", "moe"):
            assert name in FUSION_STRATEGIES

    def test_objectives_cover_both_families(self):
        for name in ("bce", "weighted-bce", "focal", "pairwise-ranking"):
            assert name in OBJECTIVES
        for name in ("maxent", "poisson", "deepmaxent"):
            assert name in OBJECTIVES

    def test_graph_builders_are_populated(self):
        for name in ("geographic-knn", "geographic-radius", "environmental-knn",
                     "hybrid", "patch-adjacency", "user"):
            assert name in GRAPH_BUILDERS

    def test_point_process_objectives_declare_their_assumption(self):
        """The distinction that makes the objective axis meaningful."""
        assert "quadrature" in OBJECTIVES.create("poisson").info().treats_background_as
        assert "absence" in OBJECTIVES.create("bce").info().treats_background_as

    def test_deepmaxent_declares_it_is_unverified(self):
        info = OBJECTIVES.create("deepmaxent").info()
        assert "verified_against_upstream=False" in info.reference


class TestModality:
    def test_standard_modalities_cover_the_documented_set(self):
        for name in ("climate", "terrain", "soil", "vegetation", "landcover",
                     "remote_sensing", "hydrology", "human_pressure",
                     "species_traits", "temporal_climate", "earth_observation"):
            assert name in STANDARD_MODALITIES

    def test_terrain_defaults_to_multiscale(self):
        assert STANDARD_MODALITIES["terrain"].kind is ModalityKind.RASTER
        assert STANDARD_MODALITIES["terrain"].default_representation == "multiscale-cnn"

    def test_modality_data_keeps_provenance(self):
        modality = ModalityData(
            metadata=ModalityMetadata.now(
                name="climate", source="worldclim", license="CC-BY-4.0", checksum="abc"
            ),
            values=pd.DataFrame({"bio01": [1.0, 2.0]}),
        )
        described = modality.describe()
        assert described["license"] == "CC-BY-4.0"
        assert described["checksum"] == "abc"
        assert described["downloaded_at"]

    def test_raster_modality_has_no_tabular_view(self):
        modality = ModalityData(
            metadata=ModalityMetadata(name="terrain", kind=ModalityKind.RASTER),
            values=np.zeros((4, 2, 8, 8)),
        )
        with pytest.raises(DataError, match="no tabular view"):
            modality.as_frame()


class TestEcoDataset:
    def _dataset(self) -> EcoDataset:
        n = 20
        return EcoDataset(
            observations=pd.DataFrame({"id": range(n)}),
            target=np.array([1, 0] * (n // 2)),
            coordinates=np.random.default_rng(0).uniform(0, 100, (n, 2)),
            modalities={
                "climate": ModalityData(
                    ModalityMetadata(name="climate"),
                    pd.DataFrame({"bio01": np.arange(n, dtype=float)}),
                ),
                "soil": ModalityData(
                    ModalityMetadata(name="soil"),
                    pd.DataFrame({"ph": np.arange(n, dtype=float)}),
                ),
            },
        )

    def test_ablation_removes_one_modality(self):
        ablated = self._dataset().without("soil")
        assert ablated.modality_names == ["climate"]
        assert ablated.metadata["ablated_modalities"] == ["soil"]

    def test_ablating_everything_is_rejected(self):
        dataset = self._dataset()
        with pytest.raises(DataError):
            dataset.without("climate", "soil", "nonexistent")

    def test_subset_applies_across_all_modalities(self):
        subset = self._dataset().subset([0, 1, 2])
        assert subset.n_samples == 3
        for modality in subset.modalities.values():
            assert modality.n_samples == 3

    def test_flattening_prefixes_columns_with_the_modality(self):
        frame = self._dataset().concat_tabular()
        assert "climate__bio01" in frame.columns
        assert "soil__ph" in frame.columns

    def test_row_count_mismatch_is_rejected(self):
        with pytest.raises(DataError, match="rows"):
            EcoDataset(
                observations=pd.DataFrame({"id": range(5)}),
                target=np.zeros(5),
                coordinates=np.zeros((5, 2)),
                modalities={
                    "climate": ModalityData(
                        ModalityMetadata(name="climate"), pd.DataFrame({"a": [1.0]})
                    )
                },
            )


class TestMethodology:
    def _methodology(self) -> Methodology:
        return Methodology(
            name="base",
            modalities=[
                ModalitySpec("climate", "raw-tabular"),
                ModalitySpec("terrain", "patch-statistics", scales=("1km", "5km")),
            ],
            fusion="concat",
            model="random-forest-sklearn",
        )

    def test_hash_ignores_the_display_name(self):
        """Renaming must not invalidate cached results."""
        from dataclasses import replace

        base = self._methodology()
        renamed = replace(base, name="something-else")
        assert base.methodology_hash == renamed.methodology_hash

    def test_hash_changes_with_a_substantive_change(self):
        base = self._methodology()
        assert base.methodology_hash != base.with_fusion("gated").methodology_hash
        assert base.methodology_hash != base.with_model("xgboost").methodology_hash

    def test_ablation_helpers(self):
        base = self._methodology()
        assert base.ablate_modality("terrain").modality_names == ["climate"]
        assert base.with_modalities(["climate"]).modality_names == ["climate"]

    def test_ablating_the_last_modality_is_rejected(self):
        single = Methodology(name="m", modalities=[ModalitySpec("climate")])
        with pytest.raises(ConfigurationError, match="no modalities"):
            single.ablate_modality("climate")

    def test_validation_accepts_a_coherent_methodology(self):
        assert self._methodology().validate().is_valid

    def test_validation_rejects_an_incompatible_encoder(self):
        """A CNN cannot be applied to a tabular modality."""
        methodology = Methodology(
            name="bad",
            modalities=[ModalitySpec("climate", "cnn")],  # climate is tabular
            model="random-forest-sklearn",
        )
        validation = methodology.validate()
        assert not validation.is_valid
        assert any("accepts" in issue.message for issue in validation.errors)

    def test_validation_rejects_unknown_components(self):
        for kwargs in (
            {"fusion": "nonexistent"},
            {"objective": "nonexistent"},
            {"model": "nonexistent"},
            {"graph": "nonexistent"},
        ):
            methodology = Methodology(
                name="m", modalities=[ModalitySpec("climate")], **kwargs
            )
            assert not methodology.validate().is_valid

    def test_warns_when_an_objective_cannot_apply_to_the_model(self):
        """A random forest has a fixed internal objective."""
        methodology = Methodology(
            name="m",
            modalities=[ModalitySpec("climate")],
            objective="poisson",
            model="random-forest-sklearn",
        )
        validation = methodology.validate()
        assert any("fixed internal objective" in w.message for w in validation.warnings)

    def test_warns_when_coordinates_are_enabled(self):
        methodology = Methodology(
            name="m", modalities=[ModalitySpec("climate")], include_coordinates=True
        )
        assert any("coordinates" in w.component for w in methodology.validate().warnings)

    def test_round_trips_through_a_dict(self):
        base = self._methodology()
        assert Methodology.from_dict(base.to_dict()).methodology_hash == base.methodology_hash

    def test_parses_the_yaml_mapping_form(self):
        methodology = Methodology.from_dict(
            {
                "name": "y",
                "modalities": {
                    "climate": {"representation": "raw-tabular"},
                    "terrain": {"representation": "patch-statistics", "scales": ["1km"]},
                },
                "fusion": {"method": "gated", "hidden_dim": 32},
                "model": {"model": "xgboost", "n_estimators": 100},
            }
        )
        assert set(methodology.modality_names) == {"climate", "terrain"}
        assert methodology.fusion == "gated"
        assert methodology.fusion_options["hidden_dim"] == 32
        assert methodology.model == "xgboost"


class TestStagedPlanner:
    def _base(self) -> Methodology:
        return Methodology(
            name="base",
            modalities=[
                ModalitySpec("climate", "raw-tabular"),
                ModalitySpec("terrain", "patch-statistics", scales=("1km",)),
                ModalitySpec("soil", "raw-tabular"),
            ],
            model="xgboost",
        )

    def test_information_stage_covers_singles_and_leave_one_out(self):
        plan = StagedPlanner(self._base()).plan_information(
            ["climate", "terrain", "soil"]
        )
        names = [m.name for m in plan.methodologies]
        assert any(n.startswith("only-") for n in names)
        assert any("minus" in n for n in names)
        assert plan.stage is Stage.INFORMATION

    def test_scale_stage_applies_only_to_raster_modalities(self):
        planner = StagedPlanner(self._base())
        plan = planner.plan_scale(self._base(), scales=("1km", "5km"))
        assert plan.n_configurations > 0
        assert all("terrain@" in m.name for m in plan.methodologies)

    def test_scale_stage_is_skipped_without_raster_modalities(self):
        tabular_only = Methodology(
            name="t", modalities=[ModalitySpec("climate", "raw-tabular")]
        )
        plan = StagedPlanner(tabular_only).plan_scale(tabular_only)
        assert plan.n_configurations == 0
        assert any("no raster" in n for n in plan.notes)

    def test_fusion_stage_is_skipped_for_a_single_modality(self):
        single = Methodology(name="s", modalities=[ModalitySpec("climate")])
        plan = StagedPlanner(single).plan_fusion(single)
        assert plan.n_configurations == 0
        assert any("nothing to fuse" in n for n in plan.notes)

    def test_spatial_stage_records_the_confounding_caveat(self):
        """Graph arms swap the model too; the plan must say so."""
        plan = StagedPlanner(self._base()).plan_spatial(self._base())
        assert any("confounds" in n for n in plan.notes)

    def test_objective_stage_warns_about_intensity_outputs(self):
        plan = StagedPlanner(self._base()).plan_objective(self._base())
        assert any("intensity" in n for n in plan.notes)

    def test_full_plan_is_far_smaller_than_the_factorial(self):
        """The staged design's entire justification."""
        plan = StagedPlanner(self._base(), n_species=226, n_scenarios=2).plan_all()
        assert plan.total_configurations < 200
        assert plan.estimated_jobs() == plan.total_configurations * 226 * 2

    def test_plan_records_its_own_statistical_caveats(self):
        plan = StagedPlanner(self._base()).plan_all()
        joined = " ".join(plan.notes)
        assert "greedy" in joined.lower()
        assert "biased" in joined.lower()

    def test_stages_are_ordered_with_model_last(self):
        """Algorithm choice comes after methodology, deliberately."""
        assert Stage.INFORMATION.order == 0
        assert Stage.MODEL.order == 6
        assert Stage.INFORMATION.order < Stage.REPRESENTATION.order < Stage.MODEL.order


class TestExperimentGraph:
    def test_topological_order_respects_dependencies(self):
        graph = ExperimentGraph()
        graph.add(ExperimentNode("a", NodeKind.DATASET))
        graph.add(ExperimentNode("b", NodeKind.REPRESENTATION, inputs=["a"]))
        graph.add(ExperimentNode("c", NodeKind.MODEL, inputs=["b"]))
        assert [n.node_id for n in graph.topological_order()] == ["a", "b", "c"]

    def test_cycles_are_detected(self):
        graph = ExperimentGraph()
        graph.add(ExperimentNode("a", NodeKind.DATASET))
        graph.add(ExperimentNode("b", NodeKind.REPRESENTATION, inputs=["a"]))
        graph.nodes["a"].inputs = ["b"]
        with pytest.raises(ValueError, match="cycle"):
            graph.topological_order()

    def test_unknown_input_is_rejected(self):
        graph = ExperimentGraph()
        with pytest.raises(ValueError, match="unknown inputs"):
            graph.add(ExperimentNode("a", NodeKind.MODEL, inputs=["missing"]))

    def test_identical_subtrees_share_a_key(self):
        """The property that makes caching correct, not just fast."""
        methodology = Methodology(
            name="m", modalities=[ModalitySpec("climate", "raw-tabular")]
        )
        a = build_experiment_graph(methodology, dataset_hash="d1", split_hash="s1")
        b = build_experiment_graph(methodology, dataset_hash="d1", split_hash="s1")
        assert a.keys()["encode:climate:raw-tabular"] == b.keys()["encode:climate:raw-tabular"]

    def test_a_different_split_changes_the_key(self):
        """A PCA basis fitted on one split must never be reused for another."""
        methodology = Methodology(
            name="m", modalities=[ModalitySpec("climate", "pca")]
        )
        a = build_experiment_graph(methodology, dataset_hash="d1", split_hash="s1")
        b = build_experiment_graph(methodology, dataset_hash="d1", split_hash="s2")
        assert a.keys()["encode:climate:pca"] != b.keys()["encode:climate:pca"]

    def test_a_different_dataset_changes_the_key(self):
        methodology = Methodology(name="m", modalities=[ModalitySpec("climate")])
        a = build_experiment_graph(methodology, dataset_hash="d1", split_hash="s1")
        b = build_experiment_graph(methodology, dataset_hash="d2", split_hash="s1")
        assert a.keys()["encode:climate:raw-tabular"] != b.keys()["encode:climate:raw-tabular"]

    def test_savings_report_counts_unique_computations(self):
        methodology = Methodology(
            name="m",
            modalities=[ModalitySpec("climate"), ModalitySpec("soil")],
        )
        graph = build_experiment_graph(methodology, dataset_hash="d", split_hash="s")
        savings = graph.savings_report()
        assert savings["n_nodes"] > 0
        assert savings["n_unique_computations"] > 0


class TestRepresentationCache:
    def test_stores_and_retrieves(self, tmp_path):
        cache = RepresentationCache(tmp_path)
        cache.put("key1", {"a": 1})
        assert cache.get("key1") == {"a": 1}
        assert cache.hits == 1

    def test_miss_returns_none(self, tmp_path):
        cache = RepresentationCache(tmp_path)
        assert cache.get("absent") is None
        assert cache.misses == 1

    def test_get_or_compute_calls_once(self, tmp_path):
        cache = RepresentationCache(tmp_path)
        calls = []

        def compute():
            calls.append(1)
            return [1, 2, 3]

        assert cache.get_or_compute("k", compute) == [1, 2, 3]
        assert cache.get_or_compute("k", compute) == [1, 2, 3]
        assert len(calls) == 1

    def test_can_be_disabled(self, tmp_path):
        cache = RepresentationCache(tmp_path, enabled=False)
        cache.put("k", 1)
        assert cache.get("k") is None

    def test_clear_removes_entries(self, tmp_path):
        cache = RepresentationCache(tmp_path)
        cache.put("a", 1)
        cache.put("b", 2)
        assert cache.clear() == 2
        assert cache.get("a") is None
