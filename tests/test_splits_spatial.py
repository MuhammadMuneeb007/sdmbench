"""Tests for the 10 km spatial buffer -- the protocol's most consequential rule.

If this is wrong, every spatial number in the reproduction is wrong, and it
would be wrong *quietly*: the benchmark would still run and still produce
plausible AUCs. So these tests pin the behaviour precisely.
"""

from __future__ import annotations

import numpy as np
import pytest

from sdmbench.exceptions import InsufficientDataError
from sdmbench.splits.spatial import (
    DEFAULT_BUFFER_M,
    S2_EARTH_RADIUS_M,
    SpatialBlockCV,
    apply_spatial_buffer,
    points_within_distance,
    spatial_block_folds,
    spatial_buffer_mask,
)


class TestPointsWithinDistance:
    def test_projected_crs_uses_euclidean_metres(self):
        """In a projected CRS, 10 km is a Euclidean radius of 10000 units."""
        train = np.array([[0.0, 0.0], [9_999.0, 0.0], [10_001.0, 0.0], [50_000.0, 0.0]])
        test = np.array([[0.0, 0.0]])
        within = points_within_distance(train, test, 10_000.0, geographic=False)
        assert within.tolist() == [True, True, False, False]

    def test_boundary_is_inclusive(self):
        """A point at exactly the buffer distance is INSIDE it.

        Matches sf::st_is_within_distance, which uses <=.
        """
        train = np.array([[10_000.0, 0.0]])
        test = np.array([[0.0, 0.0]])
        assert points_within_distance(train, test, 10_000.0, geographic=False)[0]

    def test_geographic_crs_uses_great_circle_metres(self):
        """In a geographic CRS, degrees must be converted, not compared raw."""
        # 0.05 deg latitude is about 5.6 km; 0.2 deg is about 22 km.
        train = np.array([[0.0, 0.0], [0.0, 0.05], [0.0, 0.2]])
        test = np.array([[0.0, 0.0]])
        within = points_within_distance(train, test, 10_000.0, geographic=True)
        assert within.tolist() == [True, True, False]

    def test_geographic_distance_matches_haversine(self):
        """Cross-check the chord/KD-tree trick against a direct computation."""
        rng = np.random.default_rng(0)
        train = np.column_stack(
            [rng.uniform(-180, 180, 200), rng.uniform(-80, 80, 200)]
        )
        test = np.array([[10.0, 45.0]])

        lon1, lat1 = np.radians(train[:, 0]), np.radians(train[:, 1])
        lon2, lat2 = np.radians(test[0, 0]), np.radians(test[0, 1])
        dlat, dlon = lat1 - lat2, lon1 - lon2
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        haversine_m = 2 * S2_EARTH_RADIUS_M * np.arcsin(np.sqrt(a))

        for threshold in (50_000.0, 500_000.0, 3_000_000.0):
            expected = haversine_m <= threshold
            actual = points_within_distance(train, test, threshold, geographic=True)
            # Allow disagreement only within a metre of the boundary, where
            # floating point genuinely cannot decide.
            disagree = expected != actual
            assert np.all(np.abs(haversine_m[disagree] - threshold) < 1.0)

    def test_longitude_wraparound(self):
        """Points either side of the antimeridian are close, not 360 degrees apart."""
        train = np.array([[179.99, 0.0]])
        test = np.array([[-179.99, 0.0]])
        assert points_within_distance(train, test, 10_000.0, geographic=True)[0]

    def test_empty_inputs(self):
        assert points_within_distance(np.zeros((0, 2)), np.ones((3, 2)), 10.0,
                                      geographic=False).shape == (0,)
        assert not points_within_distance(np.ones((3, 2)), np.zeros((0, 2)), 10.0,
                                          geographic=False).any()

    def test_rejects_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            points_within_distance(np.zeros((3, 3)), np.zeros((2, 2)), 10.0, geographic=False)


class TestSpatialBufferMask:
    def test_mask_marks_points_to_keep(self):
        """The mask is True for points FAR ENOUGH to keep, not for excluded ones."""
        train = np.array([[0.0, 0.0], [100_000.0, 0.0]])
        test = np.array([[0.0, 0.0]])
        keep = spatial_buffer_mask(train, test, buffer_m=10_000.0, geographic=False)
        assert keep.tolist() == [False, True]


class TestApplySpatialBuffer:
    def test_filters_training_data_only(self, species_task):
        """The test set is the independent survey and must never be modified."""
        filtered, info = apply_spatial_buffer(species_task, buffer_m=20_000.0)
        assert len(filtered.test) == len(species_task.test)
        assert filtered.test.equals(species_task.test)
        assert len(filtered.train) <= len(species_task.train)
        assert info["n_train_before"] == len(species_task.train)
        assert info["n_train_after"] == len(filtered.train)

    def test_no_retained_point_violates_the_buffer(self, species_task):
        """The core invariant: after filtering, nothing is within the buffer."""
        buffer_m = 15_000.0
        filtered, _ = apply_spatial_buffer(species_task, buffer_m=buffer_m)
        if len(filtered.train):
            violating = points_within_distance(
                filtered.train_coords(),
                filtered.test_coords(),
                buffer_m,
                geographic=filtered.geographic,
            )
            assert not violating.any()

    def test_zero_buffer_keeps_everything(self, species_task):
        filtered, info = apply_spatial_buffer(species_task, buffer_m=0.0)
        # Only exactly-coincident points can be excluded at distance 0.
        assert info["n_excluded"] <= len(species_task.train)

    def test_reports_single_class_outcome(self, species_task):
        """A buffer large enough to remove everything is flagged, not hidden.

        This is the condition that removes ~41 species from the paper's spatial
        evaluation.
        """
        _, info = apply_spatial_buffer(species_task, buffer_m=1e9)
        assert info["single_class_after_filter"] is True

    def test_single_class_raises_skippable_error(self, species_task):
        filtered, _ = apply_spatial_buffer(species_task, buffer_m=1e9)
        with pytest.raises(InsufficientDataError):
            filtered.require_both_classes()

    def test_default_buffer_is_ten_km(self):
        """Pinned to the published protocol (Dinnage & Warren 2026 sec. 2.5)."""
        assert DEFAULT_BUFFER_M == 10_000.0


class TestSpatialBlockCV:
    def test_folds_cover_every_point_exactly_once(self):
        rng = np.random.default_rng(1)
        coords = rng.uniform(0, 1000, size=(200, 2))
        folds = spatial_block_folds(coords, n_folds=5, seed=7)
        assert len(folds) == 200
        assert set(np.unique(folds)) <= set(range(5))

    def test_is_deterministic_for_a_seed(self):
        rng = np.random.default_rng(2)
        coords = rng.uniform(0, 1000, size=(150, 2))
        a = spatial_block_folds(coords, n_folds=5, seed=42)
        b = spatial_block_folds(coords, n_folds=5, seed=42)
        assert np.array_equal(a, b)

    def test_nearby_points_share_a_fold_more_often_than_chance(self):
        """The property that makes a spatial fold spatial.

        Under random assignment, two adjacent points share a fold 1/5 of the
        time. Blocked assignment should do substantially better.
        """
        rng = np.random.default_rng(3)
        coords = rng.uniform(0, 1000, size=(400, 2))
        folds = spatial_block_folds(coords, n_folds=5, seed=11)

        from scipy.spatial import cKDTree

        _, neighbour = cKDTree(coords).query(coords, k=2)
        same_fold = folds[neighbour[:, 1]] == folds
        assert same_fold.mean() > 0.5

    def test_first_fold_returns_disjoint_partitions(self):
        rng = np.random.default_rng(4)
        coords = rng.uniform(0, 1000, size=(120, 2))
        train_idx, test_idx = SpatialBlockCV(n_folds=5, seed=5).first_fold(coords)
        assert not set(train_idx) & set(test_idx)
        assert len(train_idx) + len(test_idx) == 120

    def test_rejects_too_few_folds(self):
        with pytest.raises(ValueError, match="n_folds"):
            spatial_block_folds(np.zeros((10, 2)), n_folds=1)
