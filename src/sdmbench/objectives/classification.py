"""Classification objectives.

The conventional treatment: background points are negatives, minimise
cross-entropy. These are the objectives almost every machine-learning SDM
implicitly uses, and they are the baseline the point-process objectives in
:mod:`sdmbench.objectives.pointprocess` must be compared against.

The assumption they make -- and it is an assumption, not a fact -- is that a
background point is evidence of absence. It is not. Including these objectives
explicitly, and labelling what they assume, is what makes that visible in the
results table rather than buried in a training loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sdmbench.objectives.base import OBJECTIVES, Objective, objective_spec
from sdmbench.optional import require

__all__ = ["BinaryCrossEntropy", "WeightedBCE", "FocalLoss", "PairwiseRanking"]


@OBJECTIVES.register(
    "bce",
    "binary-cross-entropy",
    "classification",
    spec=objective_spec(
        "Standard binary cross-entropy. Treats background points as absences."
    ),
)
class BinaryCrossEntropy(Objective):
    """Unweighted cross-entropy.

    Under 1:100 imbalance this collapses toward predicting background
    everywhere -- included precisely so that failure mode is measurable rather
    than assumed away.
    """

    treats_background_as = "absence"
    output_meaning = "probability of presence (conditional on the sampling design)"

    def loss(self, logits: Any, target: Any, *, weights: Any = None, **kwargs: Any) -> Any:
        torch = require("torch")
        return torch.nn.functional.cross_entropy(logits, target.long(), weight=weights)

    def class_weights(self, y: np.ndarray) -> np.ndarray:
        return np.ones(2, dtype=float)


@OBJECTIVES.register(
    "weighted-bce",
    "balanced-bce",
    spec=objective_spec(
        "Cross-entropy with inverse-frequency class weights. The usual imbalance fix."
    ),
)
class WeightedBCE(Objective):
    """Class-weighted cross-entropy.

    Equivalent in spirit to the presence-background weighting the published BRT
    and GAM baselines use (down-weighting background by the presence:background
    ratio), so the neural models are treated consistently with them.
    """

    treats_background_as = "absence (down-weighted)"
    output_meaning = "re-balanced probability of presence"

    def loss(self, logits: Any, target: Any, *, weights: Any = None, **kwargs: Any) -> Any:
        torch = require("torch")
        return torch.nn.functional.cross_entropy(logits, target.long(), weight=weights)


@OBJECTIVES.register(
    "focal",
    "focal-loss",
    spec=objective_spec(
        "Focal loss: down-weights easy examples to concentrate on hard ones.",
        tags=("imbalance",),
    ),
)
class FocalLoss(Objective):
    """Focal loss (Lin et al. 2017).

    Motivated for SDM by the observation that most background points are
    trivially unsuitable; the informative ones sit near the niche boundary.
    Focal loss shifts gradient toward those.
    """

    treats_background_as = "absence (difficulty-weighted)"
    output_meaning = "probability of presence"
    reference = "Lin et al. 2017, Focal Loss for Dense Object Detection"

    def loss(self, logits: Any, target: Any, *, weights: Any = None, **kwargs: Any) -> Any:
        torch = require("torch")
        gamma = float(self.params.get("gamma", 2.0))
        log_probability = torch.log_softmax(logits, dim=1)
        target = target.long()
        picked = log_probability.gather(1, target.unsqueeze(1)).squeeze(1)
        probability = picked.exp()
        loss = -((1.0 - probability) ** gamma) * picked
        if weights is not None:
            loss = loss * weights[target]
        return loss.mean()


@OBJECTIVES.register(
    "pairwise-ranking",
    "ranking",
    spec=objective_spec(
        "Rank presences above background points; optimises AUC directly.",
        tags=("ranking",),
        maturity="emerging",
    ),
)
class PairwiseRanking(Objective):
    """Pairwise ranking loss.

    Directly targets what ROC-AUC measures -- that a randomly chosen presence
    scores above a randomly chosen background point -- without asserting that
    background points are absences. A useful middle ground between
    classification and a full point-process model.

    Note the consequence: the output is a *score*, not a probability, so
    Miller's calibration slope is not meaningful for a model trained this way.
    :meth:`Objective.info` records that.
    """

    treats_background_as = "lower-ranked comparison points"
    output_meaning = "relative suitability score (NOT a calibrated probability)"
    reference = "Optimises the Wilcoxon-Mann-Whitney statistic underlying ROC-AUC"

    def loss(self, logits: Any, target: Any, *, weights: Any = None, **kwargs: Any) -> Any:
        torch = require("torch")
        scores = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits.squeeze(-1)
        target = target.long()
        positive = scores[target == 1]
        negative = scores[target == 0]
        if len(positive) == 0 or len(negative) == 0:
            # Single-class batch: no pairs exist. Return a zero that still
            # carries a gradient path so the training loop does not break.
            return scores.sum() * 0.0

        margin = float(self.params.get("margin", 1.0))
        max_pairs = int(self.params.get("max_pairs", 100_000))
        # Full outer product is O(P*N); subsample when that is too large.
        if len(positive) * len(negative) > max_pairs:
            generator = torch.Generator(device=scores.device)
            generator.manual_seed(int(self.params.get("seed", 32639)))
            n_draw = int(np.sqrt(max_pairs))
            positive = positive[
                torch.randint(len(positive), (min(n_draw, len(positive)),),
                              generator=generator, device=scores.device)
            ]
            negative = negative[
                torch.randint(len(negative), (min(n_draw, len(negative)),),
                              generator=generator, device=scores.device)
            ]
        differences = positive.unsqueeze(1) - negative.unsqueeze(0)
        return torch.nn.functional.relu(margin - differences).mean()
