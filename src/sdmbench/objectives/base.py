"""Learning objectives -- what the model is actually asked to learn.

This is one of the framework's most consequential axes, and one the SDM
literature has largely collapsed. Almost every machine-learning SDM treats
presence-only data as **binary classification**: presences are class 1,
background points are class 0, minimise cross-entropy. But background points
are *not* absences. They are samples of available environment, and some of them
are certainly suitable habitat where the species simply was not recorded.

Point-process objectives take that seriously. MaxEnt's original formulation and
the inhomogeneous Poisson process likelihood both model the *intensity* of
occurrence over the landscape, with background points serving as quadrature
points for an integral rather than as negative examples. DeepMaxent extends
this to a neural, multi-species setting.

Making the objective pluggable is what lets the benchmark ask:

    Does the choice of objective matter more than the choice of architecture?

which is a question the field has not answered because the objective is
normally hard-coded.

Availability
------------
All objectives here define a differentiable loss for the neural models
(:mod:`sdmbench.models.neural`, :mod:`sdmbench.models.graph`). Classical
adapters -- scikit-learn, XGBoost, R MaxNet -- have their own fixed internal
objectives and simply report which one they use; a benchmark row records the
objective actually optimised, never an aspirational one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sdmbench.core.registry import Registry, Spec

__all__ = ["Objective", "OBJECTIVES", "objective_spec", "ObjectiveInfo"]

#: The objective registry.
OBJECTIVES: Registry["Objective"] = Registry("objective")


def objective_spec(
    description: str,
    *,
    requires: tuple[str, ...] = (),
    extra: str | None = None,
    maturity: str = "stable",
    tags: tuple[str, ...] = (),
) -> Spec:
    return Spec(
        accepts=("any",),
        produces="loss",
        requires=requires,
        extra=extra,
        description=description,
        maturity=maturity,
        tags=tags,
    )


@dataclass
class ObjectiveInfo:
    """What an objective assumes, for the results table and the report."""

    name: str
    treats_background_as: str
    output_meaning: str
    supports_multispecies: bool = False
    differentiable: bool = True
    reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.name,
            "treats_background_as": self.treats_background_as,
            "output_meaning": self.output_meaning,
            "supports_multispecies": self.supports_multispecies,
            "differentiable": self.differentiable,
            "reference": self.reference,
        }


class Objective:
    """Base class for a learning objective.

    Subclasses implement :meth:`loss` for torch tensors. :meth:`info` documents
    the modelling assumption, which is recorded with every result -- because
    "ROC-AUC 0.74" means something different under a classification objective
    than under a point-process one.
    """

    name: str = "objective"
    #: How this objective interprets background points.
    treats_background_as: str = "absence"
    #: What the model's output means under this objective.
    output_meaning: str = "probability of presence"
    supports_multispecies: bool = False
    reference: str = ""

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)

    def loss(self, logits: Any, target: Any, *, weights: Any = None, **kwargs: Any) -> Any:
        """Differentiable loss on torch tensors."""
        raise NotImplementedError(f"{type(self).__name__} must implement loss()")

    def class_weights(self, y: np.ndarray) -> np.ndarray:
        """Per-class weights implied by this objective. Default: balanced."""
        y = np.asarray(y, dtype=int)
        counts = np.bincount(y, minlength=2).astype(float)
        return (counts.sum() / np.maximum(counts, 1.0)) / 2.0

    def to_probability(self, logits: Any) -> Any:
        """Map raw model output to something comparable across objectives.

        Metrics such as ROC-AUC are rank-based and unaffected, but calibration
        metrics are not -- a point-process intensity is not a probability, and
        subclasses that produce one say so here.
        """
        torch = _torch()
        return torch.softmax(logits, dim=1)[:, 1] if logits.ndim == 2 else torch.sigmoid(logits)

    def info(self) -> ObjectiveInfo:
        return ObjectiveInfo(
            name=self.name,
            treats_background_as=self.treats_background_as,
            output_meaning=self.output_meaning,
            supports_multispecies=self.supports_multispecies,
            reference=self.reference,
        )

    def describe(self) -> dict[str, Any]:
        return {**self.info().to_dict(), "params": {k: str(v) for k, v in self.params.items()}}


def _torch():
    from sdmbench.optional import require

    return require("torch")
