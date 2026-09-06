"""Fusion strategies -- how modalities are combined.

Once climate, terrain, soil and an EO embedding have each been encoded, they
have to be combined into something a predictor can consume. *How* is a genuine
research question, not plumbing:

``concat``
    Stack the vectors. The baseline. Cheap, and surprisingly hard to beat.
``early``
    Combine before a shared encoder, so the encoder sees raw cross-modal
    interactions.
``late``
    Fit one predictor per modality and combine their probabilities. Robust to a
    modality being missing or useless, but cannot model interactions.
``weighted``
    Learn one scalar weight per modality. Interpretable: the weights say which
    modality the model relies on.
``gated``
    Learn weights *per observation*. Lets the model use terrain in mountains
    and climate on plains -- an ecologically plausible behaviour.
``cross_attention``
    Treat each modality as a token and attend across them. Most expressive,
    most data-hungry.
``moe``
    Mixture of experts with a learned router.

Two properties every strategy must have, because the benchmark depends on them:

1. **Ablatable.** :meth:`FusionStrategy.fit` must cope with a modality being
   removed, so ``dataset.without("terrain")`` runs without special-casing.
2. **Interpretable where possible.** ``weighted``, ``gated`` and ``moe`` expose
   :meth:`FusionStrategy.modality_importance`, which is what turns a fusion
   experiment into an ecological statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from sdmbench.core.registry import Registry, Spec
from sdmbench.exceptions import DataError
from sdmbench.representations.base import EncodedModality

__all__ = ["FusionStrategy", "FusedFeatures", "FUSION_STRATEGIES", "fusion_spec"]

#: The fusion registry.
FUSION_STRATEGIES: Registry["FusionStrategy"] = Registry("fusion")


def fusion_spec(
    description: str,
    *,
    requires: tuple[str, ...] = (),
    extra: str | None = None,
    trainable: bool = False,
    maturity: str = "stable",
) -> Spec:
    return Spec(
        accepts=("encoded",),
        produces="vector",
        requires=requires,
        extra=extra,
        trainable=trainable,
        description=description,
        maturity=maturity,
    )


@dataclass
class FusedFeatures:
    """The output of a fusion strategy."""

    values: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    strategy: str = ""
    #: Per-modality contribution, when the strategy can report one.
    modality_importance: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Set when the strategy produces probabilities directly (late fusion).
    probabilities: np.ndarray | None = None

    @property
    def dim(self) -> int:
        array = np.asarray(self.values)
        return int(array.shape[1]) if array.ndim > 1 else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "dim": self.dim,
            "n_samples": int(len(self.values)),
            "modality_importance": self.modality_importance,
            **self.metadata,
        }


class FusionStrategy:
    """Base class for a fusion strategy.

    Subclasses implement :meth:`_fit` and :meth:`_transform`. Like encoders,
    trainable strategies may only be fitted on training data; ``fitted_on`` is
    recorded and audited.
    """

    name: str = "fusion"
    trainable: bool = False
    #: True when the strategy needs labels (late fusion, MoE routing).
    supervised: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.seed = int(params.get("seed", 32639))
        self._is_fitted = False
        self._fitted_on = "none"
        self._modality_order: list[str] = []

    # ------------------------------------------------------------- contract --
    def fit(
        self,
        encoded: Mapping[str, EncodedModality],
        y: np.ndarray | None = None,
    ) -> FusionStrategy:
        """Fit on training rows only."""
        self._validate(encoded)
        self._modality_order = sorted(encoded)
        if self.supervised and y is None:
            raise DataError(f"fusion strategy {self.name!r} requires labels")
        self._fit(encoded, y)
        self._is_fitted = True
        if self.trainable:
            self._fitted_on = "train"
        return self

    def transform(self, encoded: Mapping[str, EncodedModality]) -> FusedFeatures:
        self._validate(encoded)
        if self.trainable and not self._is_fitted:
            raise DataError(
                f"fusion strategy {self.name!r} is trainable and must be fitted first"
            )
        fused = self._transform(encoded)
        fused.strategy = self.name
        fused.metadata.setdefault("fitted_on", self._fitted_on)
        return fused

    def fit_transform(
        self, encoded: Mapping[str, EncodedModality], y: np.ndarray | None = None
    ) -> FusedFeatures:
        return self.fit(encoded, y).transform(encoded)

    # --------------------------------------------------------- for subclasses --
    def _fit(self, encoded: Mapping[str, EncodedModality], y: np.ndarray | None) -> None:
        """Stateless strategies need no fitting."""

    def _transform(self, encoded: Mapping[str, EncodedModality]) -> FusedFeatures:
        raise NotImplementedError(f"{type(self).__name__} must implement _transform()")

    # ----------------------------------------------------------------- utils --
    @staticmethod
    def _validate(encoded: Mapping[str, EncodedModality]) -> None:
        if not encoded:
            raise DataError("fusion needs at least one encoded modality")
        lengths = {name: len(m.values) for name, m in encoded.items()}
        if len(set(lengths.values())) > 1:
            raise DataError(f"encoded modalities have mismatched row counts: {lengths}")

    def _ordered(self, encoded: Mapping[str, EncodedModality]) -> list[str]:
        """Modality order fixed at fit time, so features stay aligned.

        Missing modalities (an ablation) are dropped; new ones are appended.
        """
        if not self._modality_order:
            return sorted(encoded)
        known = [n for n in self._modality_order if n in encoded]
        extra = [n for n in sorted(encoded) if n not in self._modality_order]
        return known + extra

    def modality_importance(self) -> dict[str, float]:
        """Per-modality importance, for strategies that learn one."""
        return {}

    def describe(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "trainable": self.trainable,
            "supervised": self.supervised,
            "fitted_on": self._fitted_on,
            "modalities": list(self._modality_order),
            "params": {k: str(v) for k, v in self.params.items()},
            "modality_importance": self.modality_importance(),
        }
