"""The paper-benchmark abstraction.

A *benchmark* is not a dataset and not a model. It is the fixed experimental
design of a published study::

    paper + dataset + preprocessing + split protocol + models + evaluation

Encoding that as an object is what lets sdmbench run a new algorithm under
exactly the rules of an existing paper, and what lets a second paper be added
without touching the run engine.

To add a benchmark, subclass :class:`PaperBenchmark`, implement
:meth:`~PaperBenchmark.iter_tasks` and :meth:`~PaperBenchmark.prepare`, declare
the published reference values, and register it. See
``docs/adding_benchmarks.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from sdmbench.config import RunConfig
from sdmbench.data.base import PreparedSplit, SpeciesTask
from sdmbench.provenance import ProvenanceReport

__all__ = ["PaperBenchmark", "PaperMetadata", "PublishedResult", "ReferenceValues"]


@dataclass(frozen=True)
class PaperMetadata:
    """Bibliographic identity of the study being reproduced."""

    paper_id: str
    title: str
    authors: tuple[str, ...]
    year: int
    venue: str = ""
    doi: str = ""
    code_url: str = ""
    data_source: str = ""

    def citation(self) -> str:
        names = ", ".join(self.authors)
        parts = [f"{names} ({self.year}). {self.title}."]
        if self.venue:
            parts.append(f"{self.venue}.")
        if self.doi:
            parts.append(f"https://doi.org/{self.doi}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "venue": self.venue,
            "doi": self.doi,
            "code_url": self.code_url,
            "data_source": self.data_source,
            "citation": self.citation(),
        }


@dataclass(frozen=True)
class PublishedResult:
    """One value reported in the original paper.

    These are **reference targets only**. They are stored, displayed and
    compared against, but they are never written into a computed result: the
    reproduction report keeps ``published_reference`` and ``reproduced_result``
    in separate namespaces precisely so the two cannot be confused.
    """

    model: str
    scenario: str
    metric: str
    value: float
    n_species: int | None = None
    source: str = ""
    statistic: str = "mean"

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model, self.scenario, self.metric)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scenario": self.scenario,
            "metric": self.metric,
            "statistic": self.statistic,
            "published_value": self.value,
            "n_species": self.n_species,
            "source": self.source,
        }


@dataclass
class ReferenceValues:
    """The published results for one benchmark, keyed for lookup."""

    entries: list[PublishedResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, result: PublishedResult) -> None:
        self.entries.append(result)

    def get(self, model: str, scenario: str, metric: str) -> PublishedResult | None:
        for entry in self.entries:
            if entry.key == (model, scenario, metric):
                return entry
        return None

    def for_scenario(self, scenario: str) -> list[PublishedResult]:
        return [e for e in self.entries if e.scenario == scenario]

    def to_dict(self) -> dict[str, Any]:
        return {
            "published_reference": [e.to_dict() for e in self.entries],
            "notes": self.notes,
        }


class PaperBenchmark:
    """Base class for a reproducible published benchmark.

    Subclass responsibilities
    -------------------------
    :meth:`iter_tasks`
        Yield one :class:`~sdmbench.data.base.SpeciesTask` per species, using
        the paper's species set and predictor selection.
    :meth:`prepare`
        Turn a task into a :class:`~sdmbench.data.base.PreparedSplit` for a
        given scenario -- this is where the split protocol and the preprocessing
        recipe are applied, in the paper's order.
    """

    #: Registry key, e.g. ``"tabpfn-sdm-2026"``.
    benchmark_id: str = "benchmark"
    #: Bumped whenever a change would alter computed numbers.
    benchmark_version: str = "1"
    #: Bibliographic metadata.
    paper: PaperMetadata
    #: Evaluation scenarios this benchmark defines.
    scenarios: tuple[str, ...] = ("nonspatial",)
    #: Metrics of the published protocol.
    metrics: tuple[str, ...] = ("roc_auc", "pr_auc", "calibration_slope")
    #: Models to run when the user does not name any.
    default_models: tuple[str, ...] = ()
    #: The seed used throughout the original study.
    seed: int = 32639

    def __init__(self, config: RunConfig | None = None) -> None:
        self.config = config or RunConfig(benchmark=self.benchmark_id)

    # ------------------------------------------------------------------ data --
    def iter_tasks(self) -> Iterator[SpeciesTask]:
        raise NotImplementedError

    def n_tasks(self) -> int:
        """Number of species tasks; override when it can be known cheaply."""
        return sum(1 for _ in self.iter_tasks())

    def dataset_hash(self) -> str:
        raise NotImplementedError

    # ------------------------------------------------------------- protocol --
    def prepare(self, task: SpeciesTask, scenario: str) -> PreparedSplit:
        raise NotImplementedError

    def validate_scenario(self, scenario: str) -> None:
        if scenario not in self.scenarios:
            raise ValueError(
                f"{self.benchmark_id} does not define scenario {scenario!r}; "
                f"available: {list(self.scenarios)}"
            )

    def model_options(self, model_name: str, scenario: str) -> dict[str, Any]:
        """Per-model, per-scenario overrides the protocol requires.

        For example, the TabPFN-SDM 2026 recipe must load the *spatial*
        finetuned checkpoint in the spatial scenario.
        """
        return {}

    # ------------------------------------------------------------ references --
    def published_results(self) -> ReferenceValues:
        return ReferenceValues()

    def provenance(self) -> ProvenanceReport:
        """Where every protocol constant in this recipe came from."""
        return ProvenanceReport(benchmark_id=self.benchmark_id)

    # -------------------------------------------------------------- metadata --
    def describe(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "paper": self.paper.to_dict(),
            "scenarios": list(self.scenarios),
            "metrics": list(self.metrics),
            "default_models": list(self.default_models),
            "seed": self.seed,
        }

    @classmethod
    def config_path(cls) -> Path | None:
        """Location of the shipped YAML configuration, if any."""
        root = Path(__file__).resolve().parents[3] / "configs"
        candidate = root / f"{cls.benchmark_id.replace('-', '_')}.yaml"
        return candidate if candidate.is_file() else None
