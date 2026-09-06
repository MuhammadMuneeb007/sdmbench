"""Configuration objects and YAML loading.

A benchmark run is fully described by a :class:`RunConfig`. The CLI builds one
from flags, ``sdmbench benchmark --config file.yaml`` builds one from YAML, and
both paths end up in the same object so the two entry points cannot drift.

The YAML schema mirrors the structure requested in the project brief::

    benchmark:
      name: tabpfn_sdm_2026
    evaluation:
      scenarios: [nonspatial, spatial]
    models:
      published: [maxnet, random_forest, brt, gam, tabpfn_sdm]
      classical: [knn, extra_trees, xgboost, lightgbm, catboost]
      graph:     [gcn, graph_transformer]
    metrics: [roc_auc, pr_auc, calibration_slope]
    compute:
      device: auto
      workers: 8
    seed: 32639

Model groups (``published``, ``classical``, ``graph``, ...) are purely
organisational -- they are flattened into one ordered, de-duplicated list. The
group name is *not* used to decide anything scientific.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sdmbench.exceptions import ConfigurationError

__all__ = [
    "ComputeConfig",
    "AutoMLConfig",
    "EvaluationConfig",
    "RunConfig",
    "load_config",
    "load_yaml",
]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict, with a clear error if PyYAML is missing."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - yaml is a core dependency
        raise ConfigurationError(
            "PyYAML is required to read configuration files (pip install pyyaml)"
        ) from exc
    p = Path(path)
    if not p.is_file():
        raise ConfigurationError(f"configuration file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigurationError(f"{p} must contain a YAML mapping at the top level")
    return data


@dataclass
class ComputeConfig:
    """Compute placement and parallelism.

    ``device`` accepts ``auto``/``cpu``/``cuda``/``cuda:N``/``mps``. Models that
    genuinely require a GPU become ``SKIPPED_NO_GPU`` rather than failing the
    run when none is present.
    """

    device: str = "auto"
    workers: int = 1
    gpu_workers: int = 1
    max_memory_gb: float | None = None

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ConfigurationError("compute.workers must be >= 1")


@dataclass
class AutoMLConfig:
    """Compute budget handed to AutoML adapters.

    These budgets are recorded in every result row. An AutoML result without a
    recorded budget is not comparable with anything.
    """

    time_limit_seconds: int = 3600
    cpu_limit: int = 8
    gpu_limit: int = 0
    memory_limit_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationConfig:
    """Which evaluation scenarios to run, and over what subset."""

    scenarios: list[str] = field(default_factory=lambda: ["nonspatial"])
    regions: list[str] | None = None
    species: list[str] | None = None
    max_species: int | None = None
    repeats: int = 1

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ConfigurationError("at least one evaluation scenario is required")
        bad = [s for s in self.scenarios if not isinstance(s, str) or not s.strip()]
        if bad:
            raise ConfigurationError(f"evaluation scenarios must be non-empty strings; got {bad}")
        # Which scenario NAMES are valid is a property of the benchmark recipe,
        # not of the configuration format -- `leopard-leedham-2025` defines
        # `spatial_cv`, for instance. `PaperBenchmark.validate_scenario` and
        # `BenchmarkRunner.run` reject names the chosen recipe does not define,
        # with a message naming the ones it does.
        self.scenarios = [s.strip() for s in self.scenarios]


@dataclass
class RunConfig:
    """A complete, self-contained description of a benchmark run."""

    benchmark: str
    models: list[str] = field(default_factory=list)
    metrics: list[str] | None = None
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    automl: AutoMLConfig = field(default_factory=AutoMLConfig)
    seed: int = 32639
    output_dir: str | None = None
    force: bool = False
    strict: bool = False
    audit_leakage: bool = True
    model_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Free-form extras a benchmark recipe may consume.
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ API --
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunConfig:
        """Build a :class:`RunConfig` from a parsed YAML mapping."""
        data = dict(data)

        bench = data.get("benchmark")
        if isinstance(bench, Mapping):
            benchmark_name = bench.get("name")
        else:
            benchmark_name = bench
        if not benchmark_name:
            raise ConfigurationError("configuration must specify benchmark.name")

        models = _flatten_models(data.get("models"))

        evaluation_raw = data.get("evaluation") or {}
        if not isinstance(evaluation_raw, Mapping):
            raise ConfigurationError("evaluation must be a mapping")
        evaluation = EvaluationConfig(
            scenarios=list(evaluation_raw.get("scenarios", ["nonspatial"])),
            regions=_as_list_or_none(evaluation_raw.get("regions")),
            species=_as_list_or_none(evaluation_raw.get("species")),
            max_species=evaluation_raw.get("max_species"),
            repeats=int(evaluation_raw.get("repeats", 1)),
        )

        compute_raw = data.get("compute") or {}
        compute = ComputeConfig(
            device=str(compute_raw.get("device", "auto")),
            workers=int(compute_raw.get("workers", 1)),
            gpu_workers=int(compute_raw.get("gpu_workers", 1)),
            max_memory_gb=compute_raw.get("max_memory_gb"),
        )

        automl_raw = data.get("automl") or {}
        automl = AutoMLConfig(
            time_limit_seconds=int(automl_raw.get("time_limit_seconds", 3600)),
            cpu_limit=int(automl_raw.get("cpu_limit", 8)),
            gpu_limit=int(automl_raw.get("gpu_limit", 0)),
            memory_limit_gb=automl_raw.get("memory_limit_gb"),
        )

        known = {
            "benchmark",
            "models",
            "metrics",
            "evaluation",
            "compute",
            "automl",
            "seed",
            "output_dir",
            "force",
            "strict",
            "audit_leakage",
            "model_options",
        }
        return cls(
            benchmark=str(benchmark_name),
            models=models,
            metrics=_as_list_or_none(data.get("metrics")),
            evaluation=evaluation,
            compute=compute,
            automl=automl,
            seed=int(data.get("seed", 32639)),
            output_dir=data.get("output_dir"),
            force=bool(data.get("force", False)),
            strict=bool(data.get("strict", False)),
            audit_leakage=bool(data.get("audit_leakage", True)),
            model_options=dict(data.get("model_options") or {}),
            extra={k: v for k, v in data.items() if k not in known},
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        return cls.from_dict(load_yaml(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": {"name": self.benchmark},
            "models": list(self.models),
            "metrics": list(self.metrics) if self.metrics else None,
            "evaluation": asdict(self.evaluation),
            "compute": asdict(self.compute),
            "automl": asdict(self.automl),
            "seed": self.seed,
            "output_dir": self.output_dir,
            "force": self.force,
            "strict": self.strict,
            "audit_leakage": self.audit_leakage,
            "model_options": self.model_options,
            "extra": self.extra,
        }

    def canonical_json(self) -> str:
        """Deterministic JSON used for configuration hashing."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    def options_for(self, model_name: str) -> dict[str, Any]:
        """Per-model keyword overrides declared under ``model_options``."""
        return dict(self.model_options.get(model_name, {}))


def _as_list_or_none(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    raise ConfigurationError(f"expected a list, got {type(value).__name__}")


def _flatten_models(value: Any) -> list[str]:
    """Flatten the grouped ``models:`` mapping into one ordered unique list."""
    if value is None:
        return []
    if isinstance(value, str):
        return _split_csv(value)
    if isinstance(value, Mapping):
        out: list[str] = []
        for group in value.values():
            out.extend(_flatten_models(group))
        return _dedupe(out)
    if isinstance(value, Iterable):
        out = []
        for item in value:
            if isinstance(item, str):
                out.extend(_split_csv(item))
            else:
                out.extend(_flatten_models(item))
        return _dedupe(out)
    raise ConfigurationError(f"cannot interpret models: {value!r}")


def _split_csv(value: str) -> list[str]:
    """Split a comma-separated model list, tolerating whitespace and newlines."""
    return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> RunConfig:
    """Load a :class:`RunConfig` from YAML with optional top-level overrides.

    ``overrides`` values that are ``None`` are ignored, so CLI flags that were
    not supplied never clobber the file.
    """
    data: dict[str, Any] = load_yaml(path) if path else {}
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        data[key] = value
    return RunConfig.from_dict(data)
