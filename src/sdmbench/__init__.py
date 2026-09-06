"""sdmbench -- a reproducible benchmarking framework for species distribution modelling.

sdmbench separates three things that are routinely conflated in the SDM
literature:

``Dataset``
    The raw observations and environmental covariates (e.g. ``disdat``).
``Benchmark``
    A *fixed* experimental design taken from a published paper: which species,
    which predictors, which preprocessing, which train/test protocol, which
    metrics.
``Model``
    The algorithm being evaluated.

Every model in an sdmbench run is forced through the *same* benchmark, so the
resulting leaderboard is a fair comparison rather than a collection of
independently-tuned numbers.

See :mod:`sdmbench.benchmarks` for the available benchmark recipes and
:mod:`sdmbench.models` for the model adapter registry.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Benchmark",
    "BenchmarkResults",
    "ModelAdapter",
    "ModelStatus",
    "RunResult",
    "SpeciesTask",
    "get_benchmark",
    "get_model",
    "list_benchmarks",
    "list_models",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy-import shim
    """Lazily expose the public API.

    Importing ``sdmbench`` must stay cheap and must not drag in optional
    dependencies, so the heavy modules are only imported on first attribute
    access.
    """
    if name in {"Benchmark", "BenchmarkResults"}:
        from sdmbench.evaluation.runner import Benchmark, BenchmarkResults

        return {"Benchmark": Benchmark, "BenchmarkResults": BenchmarkResults}[name]
    if name in {"ModelAdapter", "ModelStatus", "get_model", "list_models"}:
        from sdmbench.models.base import ModelAdapter, ModelStatus, get_model, list_models

        return {
            "ModelAdapter": ModelAdapter,
            "ModelStatus": ModelStatus,
            "get_model": get_model,
            "list_models": list_models,
        }[name]
    if name in {"get_benchmark", "list_benchmarks"}:
        from sdmbench.benchmarks import get_benchmark, list_benchmarks

        return {"get_benchmark": get_benchmark, "list_benchmarks": list_benchmarks}[name]
    if name == "RunResult":
        from sdmbench.results import RunResult

        return RunResult
    if name == "SpeciesTask":
        from sdmbench.data.base import SpeciesTask

        return SpeciesTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
