"""Spatial separation between training and test data.

The published protocol
----------------------
Dinnage & Warren (2026) sec. 2.1.2 and 2.5:

    "we created spatially-filtered training sets by excluding training points
     within 10 km of any test location"

    "Spatial evaluation used a 10 km buffer for spatial separation. Training
     points within 10 km of any test location were excluded, testing model
     transferability to novel geographic areas."

Note precisely what this is and is not. It is a **one-sided buffer filter on
the training set**: the independent presence-absence survey data is fixed and
complete, and training records too close to any of it are dropped. It is *not*
a spatial block cross-validation, and it is *not* an sklearn ``train_test_split``
with a spatial flavour. Substituting either changes which species survive and
changes every reported number.

Distance is not a coordinate difference
---------------------------------------
``disdat`` regions do not share a CRS. AWT, NZ and SWI are projected in metres,
so a 10 km buffer is a Euclidean radius of 10000. CAN, NSW and SA are in
degrees, where 10000 is meaningless -- distance must be geodesic.

The authors built their spatial features with ``sf``. ``sf::st_is_within_distance``
dispatches on the CRS: planar distance for a projected CRS, great-circle
distance via s2 for a geographic one. :func:`points_within_distance` reproduces
that dispatch, using the same spherical earth radius s2 uses by default, so the
Python and R filters select the same rows. ``rbridge/scripts/spatial_split.R``
is the reference implementation the parity test compares against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sdmbench.data.base import SpeciesTask

__all__ = [
    "S2_EARTH_RADIUS_M",
    "DEFAULT_BUFFER_M",
    "points_within_distance",
    "spatial_buffer_mask",
    "apply_spatial_buffer",
    "SpatialBufferFilter",
    "SpatialBlockCV",
    "spatial_block_folds",
]

#: Earth radius used by s2, and therefore by ``sf`` for geographic CRSs
#: (``s2::s2_earth_radius_meters()``). Matching this constant is what keeps the
#: Python and R spatial filters in agreement to the metre.
S2_EARTH_RADIUS_M = 6_371_010.0

#: The buffer distance in Dinnage & Warren (2026). Metres, regardless of CRS.
DEFAULT_BUFFER_M = 10_000.0


def _lonlat_to_unit_sphere(coords: np.ndarray) -> np.ndarray:
    """Convert lon/lat degrees to Cartesian coordinates on the unit sphere."""
    lon = np.radians(np.asarray(coords[:, 0], dtype=float))
    lat = np.radians(np.asarray(coords[:, 1], dtype=float))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def _chord_for_arc(distance_m: float, radius_m: float = S2_EARTH_RADIUS_M) -> float:
    """Chord length on the unit sphere subtending a great-circle arc.

    Working in chord space lets a Euclidean KD-tree answer an exact
    great-circle radius query: the chord is a monotone function of the arc, so
    "within chord c" and "within arc d" select identical point sets.
    """
    arc = min(float(distance_m) / radius_m, np.pi)
    return 2.0 * np.sin(arc / 2.0)


def points_within_distance(
    query: np.ndarray,
    reference: np.ndarray,
    distance_m: float,
    *,
    geographic: bool,
    radius_m: float = S2_EARTH_RADIUS_M,
) -> np.ndarray:
    """Boolean mask: which ``query`` points lie within ``distance_m`` of ``reference``.

    Parameters
    ----------
    query, reference:
        ``(n, 2)`` arrays of ``(x, y)`` -- or ``(lon, lat)`` when ``geographic``.
    distance_m:
        Threshold in **metres** in both cases.
    geographic:
        True when coordinates are degrees; selects great-circle distance.

    Returns
    -------
    numpy.ndarray
        Boolean array of length ``len(query)``, True where the point is within
        the distance of at least one reference point.

    Notes
    -----
    The comparison is ``<=`` -- a point at exactly 10 km is inside the buffer,
    matching ``sf::st_is_within_distance``.
    """
    query = np.asarray(query, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError(f"query must have shape (n, 2), got {query.shape}")
    if len(query) == 0:
        return np.zeros(0, dtype=bool)
    if len(reference) == 0:
        return np.zeros(len(query), dtype=bool)
    if reference.ndim != 2 or reference.shape[1] != 2:
        raise ValueError(f"reference must have shape (n, 2), got {reference.shape}")

    if geographic:
        q = _lonlat_to_unit_sphere(query)
        r = _lonlat_to_unit_sphere(reference)
        threshold = _chord_for_arc(distance_m, radius_m)
    else:
        q, r = query, reference
        threshold = float(distance_m)

    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - scipy is a core dependency
        return _within_distance_bruteforce(q, r, threshold)

    tree = cKDTree(r)
    # `query` with distance_upper_bound returns inf when nothing is in range,
    # which is both faster and far lighter on memory than a full pair matrix.
    nearest, _ = tree.query(q, k=1, distance_upper_bound=threshold * (1 + 1e-12))
    return np.isfinite(nearest)


def _within_distance_bruteforce(
    query: np.ndarray, reference: np.ndarray, threshold: float
) -> np.ndarray:
    """Chunked fallback used when scipy is unavailable."""
    out = np.zeros(len(query), dtype=bool)
    chunk = max(1, int(2_000_000 // max(len(reference), 1)))
    for start in range(0, len(query), chunk):
        block = query[start : start + chunk]
        d2 = ((block[:, None, :] - reference[None, :, :]) ** 2).sum(axis=2)
        out[start : start + chunk] = np.sqrt(d2.min(axis=1)) <= threshold * (1 + 1e-12)
    return out


def spatial_buffer_mask(
    train_coords: np.ndarray,
    test_coords: np.ndarray,
    *,
    buffer_m: float = DEFAULT_BUFFER_M,
    geographic: bool = False,
) -> np.ndarray:
    """Return the mask of training rows to **keep** after buffering.

    ``True`` means "far enough from every test location to be used for
    training".
    """
    too_close = points_within_distance(
        train_coords, test_coords, buffer_m, geographic=geographic
    )
    return ~too_close


def apply_spatial_buffer(
    task: SpeciesTask,
    *,
    buffer_m: float = DEFAULT_BUFFER_M,
) -> tuple[SpeciesTask, dict[str, Any]]:
    """Apply the published 10 km filter to a task's training data.

    The test data is never modified -- it is the independent evaluation set and
    is frozen by definition.

    Returns
    -------
    tuple
        The filtered task and a diagnostics dict recorded in the result row.
    """
    keep = spatial_buffer_mask(
        task.train_coords(),
        task.test_coords(),
        buffer_m=buffer_m,
        geographic=task.geographic,
    )
    filtered = task.with_train(task.train.loc[keep].copy())
    diagnostics = {
        "buffer_m": float(buffer_m),
        "crs": task.crs,
        "geographic": task.geographic,
        "n_train_before": int(len(task.train)),
        "n_train_after": int(len(filtered.train)),
        "n_excluded": int((~keep).sum()),
        "train_presence_n_before": task.train_presence_n,
        "train_presence_n_after": filtered.train_presence_n,
        "train_background_n_after": filtered.train_background_n,
        "single_class_after_filter": (
            filtered.train_presence_n == 0 or filtered.train_background_n == 0
        ),
    }
    return filtered, diagnostics


@dataclass
class SpatialBufferFilter:
    """Reusable, configurable form of the published buffer filter."""

    buffer_m: float = DEFAULT_BUFFER_M

    def __call__(self, task: SpeciesTask) -> tuple[SpeciesTask, dict[str, Any]]:
        return apply_spatial_buffer(task, buffer_m=self.buffer_m)

    def describe(self) -> dict[str, Any]:
        return {
            "rule": "exclude training points within buffer of any test location",
            "buffer_m": self.buffer_m,
            "applies_to": "training data only",
            "source": "Dinnage & Warren 2026 sec. 2.1.2, 2.5",
        }


# ---------------------------------------------------------------------------
# Spatial block cross-validation
#
# Used for the "held-out versus independent test" comparison of sec. 2.7, where
# the authors used the R package `spatialsample` with v = 5 folds and kept the
# first fold as the holdout test set. This is a *different* procedure from the
# 10 km buffer above and is used for a different purpose -- do not conflate.
# ---------------------------------------------------------------------------


def spatial_block_folds(
    coords: np.ndarray,
    *,
    n_folds: int = 5,
    block_size: float | None = None,
    seed: int = 32639,
    geographic: bool = False,
) -> np.ndarray:
    """Assign points to spatial blocks, then blocks to folds.

    Points are binned onto a regular grid; whole grid cells are then dealt out
    to folds at random. Nearby points therefore land in the same fold, which is
    the property that makes the resulting estimate spatially honest.

    Parameters
    ----------
    block_size:
        Edge length of a grid cell, in coordinate units. Defaults to a grid
        with roughly ``4 * n_folds`` cells along the longer axis, which gives
        enough blocks to shuffle while keeping each block populated.

    Returns
    -------
    numpy.ndarray
        Integer fold index in ``[0, n_folds)`` per point.

    Notes
    -----
    This is sdmbench's own implementation of blocked assignment. It is *not*
    bit-compatible with ``spatialsample::spatial_block_cv``; sec. 2.7 results
    computed with it are flagged accordingly.
    """
    coords = np.asarray(coords, dtype=float)
    if len(coords) == 0:
        return np.zeros(0, dtype=int)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extent = np.maximum(maxs - mins, np.finfo(float).eps)
    if block_size is None:
        block_size = float(extent.max() / (4 * n_folds))
    block_size = max(float(block_size), float(np.finfo(float).eps))

    cell = np.floor((coords - mins) / block_size).astype(np.int64)
    # Encode 2-D cell coordinates into one integer id.
    n_cols = int(cell[:, 0].max()) + 1
    cell_id = cell[:, 1] * n_cols + cell[:, 0]

    unique_cells = np.unique(cell_id)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique_cells))
    cell_to_fold = {int(c): int(shuffled[i] % n_folds) for i, c in enumerate(unique_cells)}
    return np.array([cell_to_fold[int(c)] for c in cell_id], dtype=int)


@dataclass
class SpatialBlockCV:
    """Spatial block cross-validation over a set of coordinates."""

    n_folds: int = 5
    block_size: float | None = None
    seed: int = 32639

    def split(self, coords: np.ndarray, *, geographic: bool = False):
        """Yield ``(train_idx, test_idx)`` pairs, one per fold."""
        folds = spatial_block_folds(
            coords,
            n_folds=self.n_folds,
            block_size=self.block_size,
            seed=self.seed,
            geographic=geographic,
        )
        for fold in range(self.n_folds):
            test_idx = np.flatnonzero(folds == fold)
            train_idx = np.flatnonzero(folds != fold)
            yield train_idx, test_idx

    def first_fold(self, coords: np.ndarray, *, geographic: bool = False):
        """Return the first fold as ``(train_idx, test_idx)``.

        Matches sec. 2.7: "retaining the first fold as the holdout test set".
        """
        return next(iter(self.split(coords, geographic=geographic)))
