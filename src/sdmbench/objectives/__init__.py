"""Learning objectives.

Registered in two families:

**Classification** -- background points treated as absences (``bce``,
``weighted-bce``, ``focal``, ``pairwise-ranking``).

**Point process** -- background points treated as quadrature (``maxent``,
``poisson``, ``deepmaxent``).

Comparing across those two families is the point. See
:mod:`sdmbench.objectives.pointprocess` for why the distinction is substantive
rather than notational.
"""

from sdmbench.objectives.base import (
    OBJECTIVES,
    Objective,
    ObjectiveInfo,
    objective_spec,
)
from sdmbench.objectives.classification import (  # noqa: F401 - registration
    BinaryCrossEntropy,
    FocalLoss,
    PairwiseRanking,
    WeightedBCE,
)
from sdmbench.objectives.pointprocess import (  # noqa: F401 - registration
    DeepMaxentObjective,
    MaxEntObjective,
    PoissonProcessObjective,
)

__all__ = [
    "OBJECTIVES",
    "Objective",
    "ObjectiveInfo",
    "objective_spec",
    "BinaryCrossEntropy",
    "WeightedBCE",
    "FocalLoss",
    "PairwiseRanking",
    "MaxEntObjective",
    "PoissonProcessObjective",
    "DeepMaxentObjective",
]

#: Objectives that treat background points as quadrature rather than absences.
POINT_PROCESS_OBJECTIVES = ("maxent", "poisson", "deepmaxent")

#: Objectives that treat background points as negative examples.
CLASSIFICATION_OBJECTIVES = ("bce", "weighted-bce", "focal", "pairwise-ranking")
