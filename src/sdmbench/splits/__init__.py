"""Split rules and the leakage auditor."""

from sdmbench.splits.leakage import LeakageAuditor, LeakageCheck, LeakageReport, Severity
from sdmbench.splits.random import identity_split, stratified_holdout, subsample_background
from sdmbench.splits.spatial import (
    DEFAULT_BUFFER_M,
    S2_EARTH_RADIUS_M,
    SpatialBlockCV,
    SpatialBufferFilter,
    apply_spatial_buffer,
    points_within_distance,
    spatial_block_folds,
    spatial_buffer_mask,
)

__all__ = [
    "LeakageAuditor",
    "LeakageCheck",
    "LeakageReport",
    "Severity",
    "identity_split",
    "stratified_holdout",
    "subsample_background",
    "DEFAULT_BUFFER_M",
    "S2_EARTH_RADIUS_M",
    "SpatialBlockCV",
    "SpatialBufferFilter",
    "apply_spatial_buffer",
    "points_within_distance",
    "spatial_block_folds",
    "spatial_buffer_mask",
]
