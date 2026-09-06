"""Modality fusion strategies."""

from sdmbench.fusion.base import (
    FUSION_STRATEGIES,
    FusedFeatures,
    FusionStrategy,
    fusion_spec,
)
from sdmbench.fusion.strategies import (  # noqa: F401 - registration side effects
    ConcatFusion,
    CrossAttentionFusion,
    EarlyFusion,
    GatedFusion,
    LateFusion,
    MixtureOfExpertsFusion,
    WeightedFusion,
)

__all__ = [
    "FUSION_STRATEGIES",
    "FusedFeatures",
    "FusionStrategy",
    "fusion_spec",
    "ConcatFusion",
    "EarlyFusion",
    "LateFusion",
    "WeightedFusion",
    "GatedFusion",
    "CrossAttentionFusion",
    "MixtureOfExpertsFusion",
]
