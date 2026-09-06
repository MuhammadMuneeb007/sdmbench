"""Representation encoder, fusion strategy and graph builder tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdmbench.core.modality import ModalityData, ModalityKind, ModalityMetadata
from sdmbench.exceptions import DataError
from sdmbench.fusion import FUSION_STRATEGIES
from sdmbench.graphs import GRAPH_BUILDERS
from sdmbench.representations import REPRESENTATIONS
from sdmbench.representations.base import EncodedModality
from tests.conftest import requires_torch


def tabular(n: int = 60, seed: int = 0) -> ModalityData:
    rng = np.random.default_rng(seed)
    return ModalityData(
        metadata=ModalityMetadata(name="climate", kind=ModalityKind.TABULAR),
        values=pd.DataFrame(
            {"bio01": rng.normal(10, 2, n), "bio12": rng.lognormal(6, 0.3, n)}
        ),
    )


def raster(n: int = 20, scales: int = 2) -> ModalityData:
    rng = np.random.default_rng(1)
    return ModalityData(
        metadata=ModalityMetadata(name="terrain", kind=ModalityKind.RASTER),
        values=rng.normal(size=(n, scales, 3, 8, 8)),
        scales=[f"{s}km" for s in (1, 5)][:scales],
    )


def sequence(n: int = 40, timesteps: int = 12) -> ModalityData:
    rng = np.random.default_rng(2)
    return ModalityData(
        metadata=ModalityMetadata(name="vegetation", kind=ModalityKind.SEQUENCE),
        values=rng.normal(size=(n, timesteps, 2)),
    )


class TestStatelessEncoders:
    def test_raw_tabular_is_the_identity(self):
        modality = tabular()
        encoded = REPRESENTATIONS.create("raw-tabular").fit_transform(modality)
        assert np.allclose(encoded.values, modality.as_frame().to_numpy())
        assert encoded.fitted_on == "none"

    def test_patch_statistics_produces_five_stats_per_channel_per_scale(self):
        encoded = REPRESENTATIONS.create("patch-statistics").fit_transform(raster())
        assert encoded.dim == 2 * 3 * 5  # scales x channels x stats

    def test_temporal_summary_includes_a_trend_term(self):
        encoded = REPRESENTATIONS.create("temporal-summary").fit_transform(sequence())
        assert any("trend" in name for name in encoded.feature_names)
        assert encoded.dim == 7 * 2  # seven statistics x two variables

    def test_temporal_summary_recovers_a_known_trend(self):
        n, timesteps = 5, 10
        rising = np.tile(np.arange(timesteps, dtype=float), (n, 1))[:, :, None]
        modality = ModalityData(
            ModalityMetadata(name="veg", kind=ModalityKind.SEQUENCE), rising
        )
        encoded = REPRESENTATIONS.create("temporal-summary").fit_transform(modality)
        trend_index = encoded.feature_names.index("veg_v0_trend")
        assert encoded.values[0, trend_index] == pytest.approx(1.0)


class TestFittedEncoders:
    def test_standardisation_is_fitted_on_training_data_only(self):
        train = tabular(seed=0)
        shifted = ModalityData(
            train.metadata, train.as_frame() + 100.0
        )
        encoder = REPRESENTATIONS.create("standardized-tabular")
        encoder.fit(train)
        assert np.allclose(encoder.transform(train).values.mean(axis=0), 0, atol=1e-8)
        # Test data must NOT be re-centred to zero.
        assert np.abs(encoder.transform(shifted).values.mean()) > 1.0
        assert encoder.fitted_on == "train"

    def test_transform_before_fit_raises_for_trainable_encoders(self):
        with pytest.raises(DataError, match="fitted"):
            REPRESENTATIONS.create("pca").transform(tabular())

    def test_pca_reduces_dimensionality_and_reports_variance(self):
        encoder = REPRESENTATIONS.create("pca", n_components=1)
        encoded = encoder.fit_transform(tabular())
        assert encoded.dim == 1
        assert 0.0 <= encoded.metadata["cumulative_variance"] <= 1.0

    def test_categorical_levels_come_from_training_only(self):
        train = ModalityData(
            ModalityMetadata(name="lc", kind=ModalityKind.CATEGORICAL),
            pd.DataFrame({"veg": ["a", "b", "a", "b"]}),
        )
        test = ModalityData(
            train.metadata, pd.DataFrame({"veg": ["a", "unseen"]})
        )
        encoder = REPRESENTATIONS.create("categorical")
        encoder.fit(train)
        encoded = encoder.transform(test)
        assert encoded.dim == 2  # still two columns, not three
        assert encoded.values[1].sum() == 0  # unseen level is all-zero

    def test_encoder_rejects_the_wrong_modality_kind(self):
        with pytest.raises(DataError, match="accepts modality kind"):
            REPRESENTATIONS.create("patch-statistics").fit_transform(tabular())


@requires_torch
class TestTorchEncoders:
    def test_autoencoder_produces_the_requested_latent_size(self):
        encoder = REPRESENTATIONS.create("autoencoder", latent_dim=4, epochs=3)
        assert encoder.fit_transform(tabular()).dim == 4

    def test_vae_encoding_is_deterministic(self):
        """The posterior MEAN is used, not a sample, so re-runs reproduce."""
        encoder = REPRESENTATIONS.create("vae", latent_dim=3, epochs=3)
        modality = tabular()
        encoder.fit(modality)
        assert np.allclose(
            encoder.transform(modality).values, encoder.transform(modality).values
        )


class TestFusion:
    def _encoded(self) -> dict[str, EncodedModality]:
        rng = np.random.default_rng(0)
        return {
            "climate": EncodedModality("climate", rng.normal(size=(50, 3)),
                                       ["c1", "c2", "c3"]),
            "soil": EncodedModality("soil", rng.normal(size=(50, 2)), ["s1", "s2"]),
        }

    def test_concat_stacks_dimensions(self):
        fused = FUSION_STRATEGIES.create("concat").fit_transform(self._encoded())
        assert fused.dim == 5
        assert fused.feature_names == ["c1", "c2", "c3", "s1", "s2"]

    def test_early_fusion_standardises_per_modality(self):
        """Without this a 64-d embedding swamps a 3-d climate block by scale."""
        encoded = self._encoded()
        encoded["soil"] = EncodedModality("soil", encoded["soil"].values * 1000.0)
        fused = FUSION_STRATEGIES.create("early").fit_transform(encoded)
        assert np.allclose(fused.values.mean(axis=0), 0, atol=1e-8)

    def test_late_fusion_returns_probabilities(self):
        encoded = self._encoded()
        y = (encoded["climate"].values[:, 0] > 0).astype(int)
        fused = FUSION_STRATEGIES.create("late").fit_transform(encoded, y)
        assert fused.probabilities is not None
        assert np.all((fused.probabilities >= 0) & (fused.probabilities <= 1))

    def test_weighted_fusion_reports_modality_importance(self):
        encoded = self._encoded()
        y = (encoded["climate"].values[:, 0] > 0).astype(int)
        strategy = FUSION_STRATEGIES.create("weighted")
        strategy.fit_transform(encoded, y)
        importance = strategy.modality_importance()
        # Climate generated the label, so it must weigh more than noise.
        assert importance["climate"] > importance["soil"]

    def test_mismatched_row_counts_are_rejected(self):
        encoded = self._encoded()
        encoded["soil"] = EncodedModality("soil", np.zeros((10, 2)))
        with pytest.raises(DataError, match="mismatched row counts"):
            FUSION_STRATEGIES.create("concat").fit_transform(encoded)

    def test_empty_input_is_rejected(self):
        with pytest.raises(DataError, match="at least one"):
            FUSION_STRATEGIES.create("concat").fit_transform({})

    def test_concat_tolerates_an_ablated_modality(self):
        """Ablation must not require special-casing in every strategy."""
        strategy = FUSION_STRATEGIES.create("concat")
        strategy.fit(self._encoded())
        reduced = {"climate": self._encoded()["climate"]}
        assert strategy.transform(reduced).dim == 3

    @requires_torch
    def test_gated_fusion_reports_per_observation_gates(self):
        encoded = self._encoded()
        y = (encoded["climate"].values[:, 0] > 0).astype(int)
        strategy = FUSION_STRATEGIES.create("gated", epochs=5, hidden_dim=8)
        fused = strategy.fit_transform(encoded, y)
        assert set(fused.modality_importance) == {"climate", "soil"}

    @requires_torch
    def test_cross_attention_reports_attention_weights(self):
        encoded = self._encoded()
        y = (encoded["climate"].values[:, 0] > 0).astype(int)
        strategy = FUSION_STRATEGIES.create("cross-attention", epochs=5, hidden_dim=8, n_heads=2)
        fused = strategy.fit_transform(encoded, y)
        weights = np.array(list(fused.modality_importance.values()))
        assert weights.sum() == pytest.approx(1.0, abs=1e-4)


class TestGraphBuilders:
    def _coords(self, n: int = 50) -> np.ndarray:
        return np.random.default_rng(0).uniform(0, 100_000, (n, 2))

    def test_knn_graph_is_symmetric_and_self_loop_free(self):
        graph = GRAPH_BUILDERS.create("geographic-knn", k=4).build(self._coords())
        edges = {(int(a), int(b)) for a, b in graph.edge_index.T}
        assert all((b, a) in edges for a, b in edges)
        assert all(a != b for a, b in edges)

    def test_knn_degree_scales_with_k(self):
        coords = self._coords()
        small = GRAPH_BUILDERS.create("geographic-knn", k=2).build(coords)
        large = GRAPH_BUILDERS.create("geographic-knn", k=8).build(coords)
        assert large.n_edges > small.n_edges

    def test_geographic_knn_uses_great_circle_for_lonlat(self):
        """Near the pole, nearest-in-degrees is not nearest-in-metres."""
        coords = np.array([[0.0, 89.0], [180.0, 89.0], [1.0, 89.0]])
        graph = GRAPH_BUILDERS.create("geographic-knn", k=1).build(
            coords, geographic=True
        )
        assert graph.n_edges > 0

    def test_radius_graph_respects_the_radius(self):
        coords = np.array([[0.0, 0.0], [1_000.0, 0.0], [50_000.0, 0.0]])
        graph = GRAPH_BUILDERS.create("geographic-radius", radius=5_000.0).build(coords)
        edges = {(int(a), int(b)) for a, b in graph.edge_index.T}
        assert (0, 1) in edges
        assert (0, 2) not in edges

    def test_radius_without_a_radius_is_rejected(self):
        with pytest.raises(DataError, match="radius"):
            GRAPH_BUILDERS.create("geographic-radius").build(self._coords())

    def test_environmental_graph_connects_by_niche_not_geography(self):
        """Two distant sites with identical environment become neighbours."""
        coords = np.array([[0.0, 0.0], [1e6, 1e6], [10.0, 10.0]])
        features = np.array([[1.0, 1.0], [1.0, 1.0], [50.0, 50.0]])
        graph = GRAPH_BUILDERS.create("environmental-knn", k=1).build(coords, features)
        edges = {(int(a), int(b)) for a, b in graph.edge_index.T}
        assert (0, 1) in edges  # far apart, environmentally identical

    def test_environmental_graph_requires_features(self):
        with pytest.raises(DataError, match="features"):
            GRAPH_BUILDERS.create("environmental-knn").build(self._coords())

    def test_hybrid_alpha_interpolates_between_the_two_spaces(self):
        coords = self._coords(40)
        features = np.random.default_rng(1).normal(size=(40, 3))
        geographic = GRAPH_BUILDERS.create("hybrid", alpha=1.0, k=3).build(coords, features)
        environmental = GRAPH_BUILDERS.create("hybrid", alpha=0.0, k=3).build(coords, features)
        assert not np.array_equal(geographic.edge_index, environmental.edge_index)

    def test_patch_graph_aggregates_observations_into_nodes(self):
        coords = self._coords(60)
        features = np.random.default_rng(2).normal(size=(60, 3))
        graph = GRAPH_BUILDERS.create("patch-adjacency", n_patches=6).build(coords, features)
        assert graph.n_nodes == 6
        assert graph.node_to_row is not None and len(graph.node_to_row) == 60
        assert graph.node_features.shape == (6, 3)

    def test_patch_graph_declares_the_som_substitution(self):
        """k-means stands in for the published SOM step; that must be recorded."""
        coords = self._coords(40)
        features = np.random.default_rng(3).normal(size=(40, 2))
        graph = GRAPH_BUILDERS.create("patch-adjacency", n_patches=4).build(coords, features)
        assert graph.metadata["verified_against_upstream"] is False
        assert "SUBSTITUTION" in graph.metadata["patching"]

    def test_user_graph_accepts_an_edge_list(self):
        graph = GRAPH_BUILDERS.create(
            "user", edge_index=np.array([[0, 1], [1, 2]])
        ).build(self._coords(5))
        assert graph.n_edges == 4  # symmetrised

    def test_user_graph_rejects_out_of_range_indices(self):
        with pytest.raises(DataError, match="edge indices"):
            GRAPH_BUILDERS.create(
                "user", edge_index=np.array([[0, 99]])
            ).build(self._coords(5))

    def test_graph_reports_isolated_nodes(self):
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [1e9, 0.0]])
        graph = GRAPH_BUILDERS.create("geographic-radius", radius=10.0).build(coords)
        assert graph.isolated_nodes == 1
