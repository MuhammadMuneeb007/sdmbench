"""Benchmark registry.

Each benchmark is one published paper's experimental design. Adding a paper
means adding a :class:`~sdmbench.benchmarks.base.PaperBenchmark` subclass here
-- the run engine, metrics, reporting and CLI need no changes.
"""

from __future__ import annotations

from typing import Any, Callable

from sdmbench.benchmarks.base import (
    PaperBenchmark,
    PaperMetadata,
    PublishedResult,
    ReferenceValues,
)
from sdmbench.config import RunConfig

__all__ = [
    "PaperBenchmark",
    "PaperMetadata",
    "PublishedResult",
    "ReferenceValues",
    "BENCHMARK_REGISTRY",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
    "benchmark_info",
]

BENCHMARK_REGISTRY: dict[str, type[PaperBenchmark]] = {}


def register_benchmark(
    name: str, *aliases: str
) -> Callable[[type[PaperBenchmark]], type[PaperBenchmark]]:
    """Class decorator registering a benchmark recipe."""

    def decorator(cls: type[PaperBenchmark]) -> type[PaperBenchmark]:
        for key in (name, *aliases):
            BENCHMARK_REGISTRY[key] = cls
        return cls

    return decorator


def _load_builtin() -> None:
    """Register the benchmarks that ship with sdmbench."""
    if BENCHMARK_REGISTRY:
        return
    from sdmbench.benchmarks.leopard_2025 import LeopardLeedham2025Benchmark
    from sdmbench.benchmarks.tabpfn_sdm_2026 import TabPFNSDM2026Benchmark

    for cls, keys in (
        (
            TabPFNSDM2026Benchmark,
            ("tabpfn-sdm-2026", "tabpfn_sdm_2026", "tabpfn-sdm", "dinnage-warren-2026"),
        ),
        (
            LeopardLeedham2025Benchmark,
            ("leopard-leedham-2025", "leopard_leedham_2025", "leopard", "leopard-2025"),
        ),
    ):
        for key in keys:
            BENCHMARK_REGISTRY[key] = cls


def get_benchmark(name: str, config: RunConfig | None = None, **kwargs: Any) -> PaperBenchmark:
    """Instantiate a benchmark recipe by name."""
    _load_builtin()
    key = name.strip().lower()
    cls = BENCHMARK_REGISTRY.get(key) or BENCHMARK_REGISTRY.get(key.replace("_", "-"))
    if cls is None:
        raise KeyError(
            f"unknown benchmark {name!r}. Available: {', '.join(list_benchmarks())}"
        )
    return cls(config=config, **kwargs)


def list_benchmarks() -> list[str]:
    """Canonical benchmark ids (aliases excluded)."""
    _load_builtin()
    return sorted({cls.benchmark_id for cls in BENCHMARK_REGISTRY.values()})


def benchmark_info() -> dict[str, dict[str, Any]]:
    """Summary of every registered benchmark, for ``sdmbench benchmarks``."""
    _load_builtin()
    out: dict[str, dict[str, Any]] = {}
    for cls in BENCHMARK_REGISTRY.values():
        if cls.benchmark_id in out:
            continue
        out[cls.benchmark_id] = {
            "version": cls.benchmark_version,
            "paper": cls.paper.title,
            "authors": list(cls.paper.authors),
            "year": cls.paper.year,
            "doi": cls.paper.doi,
            "scenarios": list(cls.scenarios),
            "metrics": list(cls.metrics),
            "default_models": list(cls.default_models),
        }
    return dict(sorted(out.items()))
