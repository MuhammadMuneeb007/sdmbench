"""Point-process objectives: MaxEnt, Poisson, DeepMaxent.

Why these are different in kind
-------------------------------
Presence-only data is a realisation of a **point process**: species occur at a
rate that varies over the landscape, and we observe some of the points. Under
that view, background points are not negative examples -- they are quadrature
points that approximate the integral of the intensity function over the study
area.

This is not a reinterpretation invented here. It is the established
equivalence: MaxEnt with the default settings is exactly a fitted
inhomogeneous Poisson process (Renner & Warton 2013; Fithian & Hastie 2013),
and Warton & Shepherd (2010) showed the same for presence-background logistic
regression in the infinite-background limit.

The consequence for a benchmark is concrete: a model trained with a Poisson
objective outputs an *intensity*, not a probability. ROC-AUC still applies (it
is rank-based), but Miller's calibration slope does not mean the same thing.
Each objective declares this so the results table cannot mislead.

DeepMaxent
----------
:class:`DeepMaxentObjective` implements the normalised-Poisson objective of the
DeepMaxent line of work: a shared neural feature extractor across many species,
each with its own intensity head, trained under a per-species normalised
Poisson likelihood so that abundant and rare species contribute comparably.

**Provenance warning.** This implements the objective from its published
*description*, not from verified upstream source: at the time of writing the
reference implementation had not been located and cross-checked. It is
therefore registered with ``maturity="experimental"`` and
``verified_against_upstream=False``, and
:meth:`DeepMaxentObjective.info` says so. Do not report a DeepMaxent
reproduction from this implementation without first running the upstream
comparison described in ``docs/objectives.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sdmbench.objectives.base import OBJECTIVES, Objective, ObjectiveInfo, objective_spec
from sdmbench.optional import require

__all__ = ["MaxEntObjective", "PoissonProcessObjective", "DeepMaxentObjective"]


@OBJECTIVES.register(
    "maxent",
    "maximum-entropy",
    spec=objective_spec(
        "Maximum-entropy / Gibbs objective over background quadrature points.",
        requires=("torch",),
        extra="deep",
        tags=("presence-only", "point-process"),
    ),
)
class MaxEntObjective(Objective):
    """The maximum-entropy objective in its Gibbs-distribution form.

    Maximise the average score at presences minus the log-sum-exp of scores
    over the background, which is the log-likelihood of the Gibbs distribution
    normalised over available environment::

        L = -(1/n_p) * sum_presences f(x) + log( (1/n_b) * sum_background e^{f(x)} )

    Background points enter only through the normalising constant. Nothing in
    this objective asserts that they are absences -- which is exactly the
    difference from cross-entropy.
    """

    treats_background_as = "quadrature points for the normalising constant"
    output_meaning = "relative occurrence rate (Gibbs score, not a probability)"
    reference = "Phillips et al. 2006; Renner & Warton 2013 (Poisson equivalence)"

    def loss(self, logits: Any, target: Any, *, weights: Any = None, **kwargs: Any) -> Any:
        torch = require("torch")
        scores = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits.squeeze(-1)
        target = target.long()
        presence = scores[target == 1]
        background = scores[target == 0]
        if len(presence) == 0 or len(background) == 0:
            return scores.sum() * 0.0

        # logsumexp over background, minus log(n_b), is the log of the mean
        # exponentiated score -- computed stably.
        log_normaliser = torch.logsumexp(background, dim=0) - torch.log(
            torch.tensor(float(len(background)), device=scores.device)
        )
        loss = -presence.mean() + log_normaliser

        beta = float(self.params.get("l2_penalty", 0.0))
        if beta > 0:
            # The regularisation multiplier plays MaxEnt's `regmult` role.
            loss = loss + beta * (scores**2).mean()
        return loss

    def to_probability(self, logits: Any) -> Any:
        """Squash the Gibbs score to (0, 1) for metric computation.

        Rank-preserving, so ROC-AUC and PR-AUC are unaffected. Calibration
        metrics computed on this are *not* interpretable as probability
        calibration; :meth:`info` records that.
        """
        torch = require("torch")
        scores = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits.squeeze(-1)
        return torch.sigmoid(scores)

    def class_weights(self, y: np.ndarray) -> np.ndarray:
        return np.ones(2, dtype=float)


@OBJECTIVES.register(
    "poisson",
    "poisson-process",
    "ippm",
    spec=objective_spec(
        "Inhomogeneous Poisson point-process likelihood with background quadrature.",
        requires=("torch",),
        extra="deep",
        tags=("presence-only", "point-process"),
    ),
)
class PoissonProcessObjective(Objective):
    """Inhomogeneous Poisson process log-likelihood.

    The Berman-Turner / down-weighted-Poisson form::

        L = -(1/n) * sum_i w_i * ( y_i * log(lambda_i) - lambda_i )

    with presence points carrying weight ``1/A_p`` and background (quadrature)
    points weight ``A/n_b``. The integral of the intensity over the study area
    is approximated by the weighted sum over background points -- which is the
    formal statement of "background points are quadrature, not absences".
    """

    treats_background_as = "quadrature points approximating the intensity integral"
    output_meaning = "occurrence intensity (points per unit area), not a probability"
    reference = "Berman & Turner 1992; Renner et al. 2015"

    def loss(self, logits: Any, target: Any, *, weights: Any = None, **kwargs: Any) -> Any:
        torch = require("torch")
        log_intensity = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits.squeeze(-1)
        target = target.float()

        n_background = float((target == 0).sum().item())
        if n_background == 0:
            return log_intensity.sum() * 0.0

        area = float(self.params.get("area", 1.0))
        # Quadrature weights: presences get a small weight, background points
        # share the study area between them.
        quadrature = torch.where(
            target == 1,
            torch.full_like(target, 1e-6),
            torch.full_like(target, area / n_background),
        )
        # Clamp keeps exp() finite when the network briefly diverges early in
        # training; the bound is far outside any plausible log-intensity.
        intensity = torch.exp(torch.clamp(log_intensity, -30.0, 30.0))
        return -(target * log_intensity - quadrature * intensity).mean()

    def to_probability(self, logits: Any) -> Any:
        torch = require("torch")
        log_intensity = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits.squeeze(-1)
        # Complementary log-log: 1 - exp(-lambda), the probability of at least
        # one point. This is the cloglog link MaxEnt uses for its output.
        return 1.0 - torch.exp(-torch.exp(torch.clamp(log_intensity, -30.0, 30.0)))

    def class_weights(self, y: np.ndarray) -> np.ndarray:
        return np.ones(2, dtype=float)


@OBJECTIVES.register(
    "deepmaxent",
    "deep-maxent",
    spec=objective_spec(
        "Multi-species normalised Poisson objective with a shared neural encoder.",
        requires=("torch",),
        extra="deep",
        tags=("presence-only", "point-process", "multi-species"),
        maturity="experimental",
    ),
)
class DeepMaxentObjective(Objective):
    """The DeepMaxent normalised-Poisson objective (multi-species).

    Combines three ideas:

    1. **Shared representation.** One neural feature extractor serves every
       species, so rare species borrow strength from common ones.
    2. **Per-species intensity heads.** Each species has its own output.
    3. **Per-species normalisation.** The Poisson likelihood is normalised
       within each species, so a species with 900 presences does not dominate
       one with 12.

    Usage
    -----
    Multi-species mode needs ``species_index`` (one integer per row) passed to
    :meth:`loss`, and ``logits`` shaped ``(n, n_species)``. With a single
    species it degenerates to a normalised Poisson objective and works with the
    ordinary ``(n, 2)`` logits.

    NOT VERIFIED AGAINST UPSTREAM
    -----------------------------
    Implemented from the method's published description. The reference
    implementation has not been obtained and cross-checked, so numbers from
    this objective must not be reported as a DeepMaxent reproduction until the
    parity check in ``docs/objectives.md`` has been run.
    """

    treats_background_as = "shared quadrature points across species"
    output_meaning = "per-species occurrence intensity (not a probability)"
    supports_multispecies = True
    reference = "DeepMaxent (see docs/objectives.md -- implementation NOT verified upstream)"

    def loss(
        self,
        logits: Any,
        target: Any,
        *,
        weights: Any = None,
        species_index: Any = None,
        **kwargs: Any,
    ) -> Any:
        torch = require("torch")
        target = target.float()

        if species_index is None:
            # Single-species: reduce to a normalised Poisson objective.
            scores = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits.squeeze(-1)
            return self._normalised_poisson(scores, target)

        species_index = species_index.long()
        if logits.ndim != 2:
            raise ValueError("multi-species DeepMaxent expects logits of shape (n, n_species)")
        # Each row contributes through its own species' head.
        scores = logits.gather(1, species_index.unsqueeze(1)).squeeze(1)

        total = None
        n_species = 0
        for species in torch.unique(species_index):
            mask = species_index == species
            if mask.sum() < 2:
                continue
            species_loss = self._normalised_poisson(scores[mask], target[mask])
            total = species_loss if total is None else total + species_loss
            n_species += 1
        if total is None:
            return scores.sum() * 0.0
        return total / n_species

    def _normalised_poisson(self, scores: Any, target: Any) -> Any:
        """Poisson likelihood normalised by this species' presence count."""
        torch = require("torch")
        n_presence = target.sum()
        n_background = (1.0 - target).sum()
        if n_presence == 0 or n_background == 0:
            return scores.sum() * 0.0

        log_intensity = torch.clamp(scores, -30.0, 30.0)
        presence_term = (target * log_intensity).sum() / n_presence
        # Normalising by the mean exponentiated background score is what makes
        # the objective comparable between species of very different prevalence.
        background_term = torch.logsumexp(
            log_intensity[target == 0], dim=0
        ) - torch.log(n_background)
        loss = -presence_term + background_term

        penalty = float(self.params.get("l2_penalty", 0.0))
        if penalty > 0:
            loss = loss + penalty * (log_intensity**2).mean()
        return loss

    def to_probability(self, logits: Any) -> Any:
        torch = require("torch")
        scores = logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits.squeeze(-1)
        return torch.sigmoid(scores)

    def class_weights(self, y: np.ndarray) -> np.ndarray:
        return np.ones(2, dtype=float)

    def info(self) -> ObjectiveInfo:
        base = super().info()
        base.reference += " | verified_against_upstream=False"
        return base
