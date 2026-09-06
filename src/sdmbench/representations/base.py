"""Representation encoders -- how a modality becomes machine-readable features.

An encoder turns one :class:`~sdmbench.core.modality.ModalityData` into a
feature matrix. Separating this from the model is the point of the whole
framework: it lets the benchmark ask *"is climate better as raw values, as
PCA scores, or as a contrastive embedding?"* while holding the classifier fixed.

The leakage contract
--------------------
Encoders come in two kinds, and the difference is a leakage question, not an
implementation detail:

**Stateless** (``trainable=False``) -- raw pass-through, categorical coding
against declared levels. Nothing is estimated, so nothing can leak.

**Fitted** (``trainable=True``) -- PCA, autoencoders, contrastive encoders,
standardisation. These estimate parameters, and those parameters **must** come
from training rows only. :meth:`BaseEncoder.fit` is the only method allowed to
see training data; :meth:`BaseEncoder.transform` must be a pure function of the
fitted state. :attr:`BaseEncoder.fitted_on` records which, and the leakage
auditor checks it.

A pretrained encoder (a frozen CNN, a foundation model) is a third case: it was
fitted, but on external data, so it cannot leak this benchmark's test set. It
declares ``pretraining_source`` and ``fitted_on = "pretrained"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sdmbench.core.modality import ModalityData, ModalityKind
from sdmbench.core.registry import Registry, Spec
from sdmbench.exceptions import DataError

__all__ = ["BaseEncoder", "EncodedModality", "REPRESENTATIONS"]

#: The representation registry.
REPRESENTATIONS: Registry["BaseEncoder"] = Registry("representation")


@dataclass
class EncodedModality:
    """The output of an encoder."""

    name: str
    values: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    encoder: str = ""
    #: ``"train"``, ``"pretrained"``, or ``"none"`` (stateless).
    fitted_on: str = "none"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        array = np.asarray(self.values)
        return int(array.shape[1]) if array.ndim > 1 else 1

    @property
    def n_samples(self) -> int:
        return int(len(self.values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "encoder": self.encoder,
            "dim": self.dim,
            "n_samples": self.n_samples,
            "fitted_on": self.fitted_on,
            "feature_names": list(self.feature_names),
            **self.metadata,
        }


class BaseEncoder:
    """Base class for every representation encoder.

    Subclasses implement :meth:`_fit` and :meth:`_transform`. The public
    :meth:`fit`/:meth:`transform` wrappers enforce the contract: transform
    before fit raises, and ``fitted_on`` is recorded automatically.
    """

    #: Registry name, set by the decorator.
    name: str = "encoder"
    #: Modality kinds this encoder accepts.
    accepts: tuple[ModalityKind, ...] = (ModalityKind.TABULAR,)
    #: Whether parameters are estimated from data.
    trainable: bool = False
    #: Non-empty when weights come from external pretraining.
    pretraining_source: str = ""

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self._is_fitted = False
        self._fitted_on = "pretrained" if self.pretraining_source else "none"
        self._output_dim: int | None = None

    # ------------------------------------------------------------- contract --
    def fit(self, modality: ModalityData, *, y: np.ndarray | None = None) -> BaseEncoder:
        """Estimate parameters. Must only ever be given training rows.

        ``y`` is available for supervised encoders but most ignore it; an
        encoder that uses it is doing supervised representation learning and
        must say so in its spec.
        """
        self._check_kind(modality)
        self._fit(modality, y=y)
        self._is_fitted = True
        if self.trainable:
            self._fitted_on = "train"
        return self

    def transform(self, modality: ModalityData) -> EncodedModality:
        """Apply the encoder. Safe on test data."""
        self._check_kind(modality)
        if self.trainable and not self._is_fitted:
            raise DataError(
                f"encoder {self.name!r} is trainable and must be fitted on training data "
                "before transform()"
            )
        encoded = self._transform(modality)
        self._output_dim = encoded.dim
        encoded.encoder = self.name
        encoded.fitted_on = self._fitted_on
        return encoded

    def fit_transform(
        self, modality: ModalityData, *, y: np.ndarray | None = None
    ) -> EncodedModality:
        return self.fit(modality, y=y).transform(modality)

    # --------------------------------------------------------- for subclasses --
    def _fit(self, modality: ModalityData, *, y: np.ndarray | None = None) -> None:
        """Stateless encoders need no fitting."""

    def _transform(self, modality: ModalityData) -> EncodedModality:
        raise NotImplementedError(f"{type(self).__name__} must implement _transform()")

    # ----------------------------------------------------------------- utils --
    def _check_kind(self, modality: ModalityData) -> None:
        if modality.kind not in self.accepts:
            accepted = ", ".join(k.value for k in self.accepts)
            raise DataError(
                f"encoder {self.name!r} accepts modality kind(s) [{accepted}] but "
                f"{modality.name!r} is {modality.kind.value!r}"
            )

    @property
    def output_dim(self) -> int | None:
        return self._output_dim

    @property
    def fitted_on(self) -> str:
        return self._fitted_on

    def describe(self) -> dict[str, Any]:
        return {
            "encoder": self.name,
            "accepts": [k.value for k in self.accepts],
            "trainable": self.trainable,
            "fitted_on": self._fitted_on,
            "pretraining_source": self.pretraining_source,
            "output_dim": self._output_dim,
            "params": {k: _jsonable(v) for k, v in self.params.items()},
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def encoder_spec(
    *,
    accepts: tuple[str, ...],
    description: str,
    requires: tuple[str, ...] = (),
    extra: str | None = None,
    trainable: bool = False,
    pretraining_source: str = "",
    maturity: str = "stable",
    output_dim: int | None = None,
) -> Spec:
    """Convenience constructor for an encoder's registry :class:`Spec`."""
    return Spec(
        accepts=accepts,
        produces="vector",
        requires=requires,
        extra=extra,
        trainable=trainable,
        pretraining_source=pretraining_source,
        description=description,
        maturity=maturity,
        output_dim=output_dim,
    )
