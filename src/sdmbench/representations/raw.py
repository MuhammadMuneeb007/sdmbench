"""Raw and statistical representations.

The baselines every learned representation has to beat. If ``raw-tabular``
matches an autoencoder embedding, the autoencoder has bought nothing but
compute -- and reporting that clearly is as much a result as the reverse.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sdmbench.core.modality import ModalityData, ModalityKind
from sdmbench.representations.base import (
    REPRESENTATIONS,
    BaseEncoder,
    EncodedModality,
    encoder_spec,
)

__all__ = [
    "RawTabularEncoder",
    "StandardizedTabularEncoder",
    "CategoricalEncoder",
    "PCAEncoder",
    "ICAEncoder",
    "IdentityEmbeddingEncoder",
]


@REPRESENTATIONS.register(
    "raw-tabular",
    "raw",
    spec=encoder_spec(
        accepts=("tabular",),
        description="Pass predictors through unchanged. The honest baseline.",
    ),
)
class RawTabularEncoder(BaseEncoder):
    """Identity on tabular data."""

    accepts = (ModalityKind.TABULAR,)
    trainable = False

    def _transform(self, modality: ModalityData) -> EncodedModality:
        frame = modality.as_frame()
        return EncodedModality(
            name=modality.name,
            values=frame.to_numpy(dtype=float),
            feature_names=list(frame.columns),
        )


@REPRESENTATIONS.register(
    "standardized-tabular",
    "standardized",
    "zscore",
    spec=encoder_spec(
        accepts=("tabular",),
        description="Centre and scale to zero mean and unit variance.",
        trainable=True,
    ),
)
class StandardizedTabularEncoder(BaseEncoder):
    """z-scoring, with the statistics estimated on training rows only."""

    accepts = (ModalityKind.TABULAR,)
    trainable = True

    def _fit(self, modality: ModalityData, *, y: np.ndarray | None = None) -> None:
        frame = modality.as_frame()
        array = frame.to_numpy(dtype=float)
        self._mean = np.nanmean(array, axis=0)
        std = np.nanstd(array, axis=0, ddof=1)
        # A constant column would divide by zero; leaving it centred-only keeps
        # it harmless rather than turning it into infinities.
        self._std = np.where(std > 0, std, 1.0)
        self._columns = list(frame.columns)

    def _transform(self, modality: ModalityData) -> EncodedModality:
        frame = modality.as_frame()
        missing = [c for c in self._columns if c not in frame.columns]
        if missing:
            raise ValueError(f"columns missing at transform time: {missing}")
        array = frame.loc[:, self._columns].to_numpy(dtype=float)
        return EncodedModality(
            name=modality.name,
            values=(array - self._mean) / self._std,
            feature_names=list(self._columns),
        )


@REPRESENTATIONS.register(
    "categorical",
    "one-hot",
    spec=encoder_spec(
        accepts=("categorical", "tabular"),
        description="One-hot encode categories against training levels.",
        trainable=True,
    ),
)
class CategoricalEncoder(BaseEncoder):
    """One-hot encoding with levels taken from training data only.

    An unseen test category becomes an all-zero row rather than a new column:
    adding a column at transform time would change the feature space between
    fit and predict.
    """

    accepts = (ModalityKind.CATEGORICAL, ModalityKind.TABULAR)
    trainable = True

    def _fit(self, modality: ModalityData, *, y: np.ndarray | None = None) -> None:
        frame = modality.as_frame()
        self._levels = {
            column: [str(v) for v in pd.unique(frame[column].astype("object").dropna())]
            for column in frame.columns
        }

    def _transform(self, modality: ModalityData) -> EncodedModality:
        frame = modality.as_frame()
        blocks: list[np.ndarray] = []
        names: list[str] = []
        for column, levels in self._levels.items():
            values = frame[column].astype("object").astype("string")
            block = np.zeros((len(frame), len(levels)), dtype=float)
            for i, level in enumerate(levels):
                block[:, i] = (values == level).to_numpy(dtype=float)
            blocks.append(block)
            names.extend(f"{column}={level}" for level in levels)
        stacked = np.hstack(blocks) if blocks else np.zeros((len(frame), 0))
        return EncodedModality(
            name=modality.name,
            values=stacked,
            feature_names=names,
            metadata={"n_levels": {k: len(v) for k, v in self._levels.items()}},
        )


@REPRESENTATIONS.register(
    "pca",
    spec=encoder_spec(
        accepts=("tabular", "embedding"),
        description="Principal components, fitted on training rows only.",
        trainable=True,
    ),
)
class PCAEncoder(BaseEncoder):
    """Linear dimensionality reduction (``sklearn.decomposition.PCA``).

    A classic leakage trap: fitting PCA on the pooled train+test matrix leaks
    the test distribution into the basis. Here the basis comes from training
    rows only, exactly like the rest of the preprocessing recipe.
    """

    accepts = (ModalityKind.TABULAR, ModalityKind.EMBEDDING)
    trainable = True

    def _fit(self, modality: ModalityData, *, y: np.ndarray | None = None) -> None:
        from sklearn.decomposition import PCA

        frame = modality.as_frame()
        array = np.nan_to_num(frame.to_numpy(dtype=float))
        n_components = self.params.get("n_components", min(10, array.shape[1]))
        if isinstance(n_components, int):
            n_components = max(1, min(n_components, min(array.shape)))
        self._model = PCA(
            n_components=n_components,
            random_state=int(self.params.get("seed", 32639)),
            whiten=bool(self.params.get("whiten", False)),
        )
        self._model.fit(array)
        self._columns = list(frame.columns)

    def _transform(self, modality: ModalityData) -> EncodedModality:
        frame = modality.as_frame().loc[:, self._columns]
        array = np.nan_to_num(frame.to_numpy(dtype=float))
        scores = self._model.transform(array)
        return EncodedModality(
            name=modality.name,
            values=scores,
            feature_names=[f"{modality.name}_pc{i + 1}" for i in range(scores.shape[1])],
            metadata={
                "explained_variance_ratio": [
                    float(v) for v in self._model.explained_variance_ratio_
                ],
                "cumulative_variance": float(sum(self._model.explained_variance_ratio_)),
            },
        )


@REPRESENTATIONS.register(
    "ica",
    spec=encoder_spec(
        accepts=("tabular", "embedding"),
        description="Independent components (FastICA), fitted on training rows only.",
        trainable=True,
    ),
)
class ICAEncoder(BaseEncoder):
    """Independent component analysis (``sklearn.decomposition.FastICA``)."""

    accepts = (ModalityKind.TABULAR, ModalityKind.EMBEDDING)
    trainable = True

    def _fit(self, modality: ModalityData, *, y: np.ndarray | None = None) -> None:
        from sklearn.decomposition import FastICA

        frame = modality.as_frame()
        array = np.nan_to_num(frame.to_numpy(dtype=float))
        n_components = int(self.params.get("n_components", min(10, array.shape[1])))
        self._model = FastICA(
            n_components=max(1, min(n_components, min(array.shape))),
            random_state=int(self.params.get("seed", 32639)),
            max_iter=int(self.params.get("max_iter", 500)),
        )
        self._model.fit(array)
        self._columns = list(frame.columns)

    def _transform(self, modality: ModalityData) -> EncodedModality:
        frame = modality.as_frame().loc[:, self._columns]
        scores = self._model.transform(np.nan_to_num(frame.to_numpy(dtype=float)))
        return EncodedModality(
            name=modality.name,
            values=scores,
            feature_names=[f"{modality.name}_ic{i + 1}" for i in range(scores.shape[1])],
        )


@REPRESENTATIONS.register(
    "pretrained-embedding",
    "identity-embedding",
    spec=encoder_spec(
        accepts=("embedding",),
        description="Use a precomputed embedding as-is (frozen foundation model output).",
        pretraining_source="external",
    ),
)
class IdentityEmbeddingEncoder(BaseEncoder):
    """Pass a precomputed embedding through unchanged.

    Used when a foundation-model provider has already produced the vectors.
    Marked ``fitted_on = "pretrained"``: the weights were estimated externally,
    so they cannot leak this benchmark's test data -- but the *provenance* of
    that pretraining still belongs in the manifest.
    """

    accepts = (ModalityKind.EMBEDDING,)
    trainable = False
    pretraining_source = "external"

    def _transform(self, modality: ModalityData) -> EncodedModality:
        array = np.asarray(modality.values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        names = modality.feature_names or [
            f"{modality.name}_e{i}" for i in range(array.shape[1])
        ]
        return EncodedModality(
            name=modality.name,
            values=array,
            feature_names=list(names),
            metadata={"source": modality.metadata.source, "provider": modality.metadata.provider},
        )
