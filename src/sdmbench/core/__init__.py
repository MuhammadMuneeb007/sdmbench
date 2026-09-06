"""Core abstractions.

The vocabulary the rest of the package is built from, and the distinction the
whole framework rests on::

    INFORMATION  !=  REPRESENTATION  !=  METHODOLOGY  !=  ALGORITHM  !=  VALIDATION

* :mod:`sdmbench.core.modality` -- what information is (and its provenance)
* :mod:`sdmbench.core.registry` -- how components are plugged in
* :mod:`sdmbench.core.methodology` -- a complete modelling approach
* :mod:`sdmbench.core.planner` -- the staged experimental design
* :mod:`sdmbench.core.experiment` -- the DAG and the representation cache
"""

from sdmbench.core.experiment import (
    ExperimentGraph,
    ExperimentNode,
    NodeKind,
    RepresentationCache,
    build_experiment_graph,
)
from sdmbench.core.methodology import (
    Methodology,
    MethodologyValidation,
    ModalitySpec,
    ValidationIssue,
)
from sdmbench.core.modality import (
    MODALITIES,
    STANDARD_MODALITIES,
    EcoDataset,
    ModalityData,
    ModalityDefinition,
    ModalityKind,
    ModalityMetadata,
)
from sdmbench.core.planner import BenchmarkPlan, Stage, StagedPlanner, StagePlan
from sdmbench.core.registry import REGISTRIES, ComponentEntry, Registry, Spec, all_registries

__all__ = [
    "ExperimentGraph",
    "ExperimentNode",
    "NodeKind",
    "RepresentationCache",
    "build_experiment_graph",
    "Methodology",
    "MethodologyValidation",
    "ModalitySpec",
    "ValidationIssue",
    "MODALITIES",
    "STANDARD_MODALITIES",
    "EcoDataset",
    "ModalityData",
    "ModalityDefinition",
    "ModalityKind",
    "ModalityMetadata",
    "BenchmarkPlan",
    "Stage",
    "StagedPlanner",
    "StagePlan",
    "REGISTRIES",
    "ComponentEntry",
    "Registry",
    "Spec",
    "all_registries",
]
