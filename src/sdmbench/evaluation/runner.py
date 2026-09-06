"""The run engine.

Owns the loop that every model goes through, identically::

    for task in benchmark.iter_tasks():
        for scenario in scenarios:
            split = benchmark.prepare(task, scenario)   # protocol, not model
            audit  = auditor.audit_split(split)         # leakage gate
            for model in models:
                result = model.run(split)               # fit + predict only
                metrics = compute_metrics(y_test, ...)  # benchmark's metrics

The ordering is what makes the comparison fair: the split and the preprocessing
are computed **once per task**, before any model is chosen, so no model can
influence the data it is given. Models see only the finished split, and never
``y_test``.

Failure policy
--------------
A single species or a single missing dependency must not end a run of thousands
of jobs. Skippable conditions become ``SKIPPED_*`` rows; unexpected exceptions
become ``FAILED`` rows carrying the message. Both are persisted, so the
leaderboard can report coverage honestly. ``strict=True`` re-raises instead,
which is what the test suite and CI use.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pandas as pd

from sdmbench.benchmarks import get_benchmark
from sdmbench.benchmarks.base import PaperBenchmark, PaperMetadata
from sdmbench.config import RunConfig
from sdmbench.data.base import PreparedSplit, SpeciesTask
from sdmbench.evaluation.store import ResultStore, job_id_for
from sdmbench.exceptions import (
    InsufficientDataError,
    LeakageError,
    LicenseError,
    MissingDependencyError,
    NoGpuError,
    SkippableError,
)
from sdmbench.metrics import compute_metrics
from sdmbench.models.base import ModelAdapter, ModelStatus, get_model, resolve_model_names
from sdmbench.paths import ensure_dir, runs_dir
from sdmbench.provenance import ProvenanceReport
from sdmbench.reproducibility.environment import git_commit, python_packages
from sdmbench.reproducibility.hashes import hash_json, hash_split
from sdmbench.reproducibility.manifest import RunManifest, new_run_id
from sdmbench.results import RunResult
from sdmbench.splits.leakage import LeakageAuditor
from sdmbench.splits.spatial import DEFAULT_BUFFER_M

__all__ = ["BenchmarkRunner", "Benchmark", "BenchmarkResults"]


@dataclass
class BenchmarkResults:
    """Everything a finished (or partial) run produced."""

    results: list[RunResult] = field(default_factory=list)
    manifest: RunManifest | None = None
    run_dir: Path | None = None
    provenance: ProvenanceReport | None = None

    def to_dataframe(self) -> pd.DataFrame:
        from sdmbench.results import results_to_dataframe

        return results_to_dataframe(self.results)

    def leaderboard(self, **kwargs: Any) -> pd.DataFrame:
        """Aggregate across species into a model leaderboard."""
        from sdmbench.evaluation.aggregate import build_leaderboard

        return build_leaderboard(self.to_dataframe(), **kwargs)

    def compare(self, model_a: str, model_b: str, **kwargs: Any):
        """Paired comparison of two models across species."""
        from sdmbench.evaluation.compare import compare_models

        return compare_models(self.to_dataframe(), model_a, model_b, **kwargs)

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return dict(sorted(counts.items()))

    def __len__(self) -> int:
        return len(self.results)


class BenchmarkRunner:
    """Executes a benchmark over a set of models.

    Parameters
    ----------
    benchmark:
        A recipe instance, or a registered name.
    config:
        The run configuration. When ``benchmark`` is a name, this also
        parameterises the recipe.
    run_dir:
        Output directory. Defaults to ``<cache>/runs/<run_id>``.
    """

    def __init__(
        self,
        benchmark: PaperBenchmark | str,
        config: RunConfig | None = None,
        *,
        run_dir: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        if isinstance(benchmark, str):
            config = config or RunConfig(benchmark=benchmark)
            self.benchmark = get_benchmark(benchmark, config=config)
        else:
            self.benchmark = benchmark
            config = config or benchmark.config
        self.config = config
        self.run_id = run_id or new_run_id(self.benchmark.benchmark_id)
        self.run_dir = Path(run_dir) if run_dir else runs_dir(self.run_id)
        self.store = ResultStore(self.run_dir)
        self.manifest: RunManifest | None = None
        self._adapters: dict[str, ModelAdapter] = {}
        self._unavailable: dict[str, tuple[str, str]] = {}

    # ---------------------------------------------------------------- models --
    def model_names(self) -> list[str]:
        """Requested models, expanded from any set aliases."""
        names = self.config.models or list(self.benchmark.default_models)
        return resolve_model_names(names)

    def _adapter(self, name: str, scenario: str) -> ModelAdapter:
        """Instantiate an adapter with the protocol's and user's options.

        Benchmark options come first so that a user override wins -- but the
        override is recorded in the hyperparameters, so a deviation from the
        published protocol is always visible in the results.
        """
        key = f"{name}@{scenario}"
        if key not in self._adapters:
            options = {
                **self.benchmark.model_options(name, scenario),
                **self.config.options_for(name),
            }
            if name in {"autogluon", "h2o-automl", "flaml", "h2o"}:
                options.setdefault("time_limit_seconds", self.config.automl.time_limit_seconds)
                options.setdefault("cpu_limit", self.config.automl.cpu_limit)
                options.setdefault("gpu_limit", self.config.automl.gpu_limit)
            options.setdefault("seed", self.config.seed)
            options.setdefault("device", self.config.compute.device)
            self._adapters[key] = get_model(name, **options)
        return self._adapters[key]

    def _availability(self, name: str, scenario: str) -> tuple[str, str]:
        """Check once per model whether it can run at all here."""
        if name in self._unavailable:
            return self._unavailable[name]
        try:
            self._adapter(name, scenario).check_available()
        except MissingDependencyError as exc:
            self._unavailable[name] = (ModelStatus.SKIPPED_DEPENDENCY.value, str(exc))
        except NoGpuError as exc:
            self._unavailable[name] = (ModelStatus.SKIPPED_NO_GPU.value, str(exc))
        except LicenseError as exc:
            self._unavailable[name] = (ModelStatus.SKIPPED_LICENSE.value, str(exc))
        except KeyError as exc:
            self._unavailable[name] = (ModelStatus.FAILED.value, str(exc))
        else:
            self._unavailable[name] = (ModelStatus.OK.value, "")
        return self._unavailable[name]

    # ------------------------------------------------------------------- run --
    def run(self, *, progress: bool = True) -> BenchmarkResults:
        """Execute the benchmark, writing results as they complete."""
        self.store.ensure()
        models = self.model_names()
        scenarios = [s for s in self.config.evaluation.scenarios if s in self.benchmark.scenarios]
        if not scenarios:
            raise ValueError(
                f"none of {self.config.evaluation.scenarios} are defined by "
                f"{self.benchmark.benchmark_id} (which offers {list(self.benchmark.scenarios)})"
            )

        provenance = self.benchmark.provenance()
        manifest = RunManifest.start(
            benchmark_id=self.benchmark.benchmark_id,
            benchmark_version=self.benchmark.benchmark_version,
            config=self.config.to_dict(),
            run_id=self.run_id,
        )
        manifest.models_requested = models
        manifest.scenarios = scenarios
        manifest.seeds = {"global": self.config.seed}
        manifest.provenance = provenance.to_dict()
        if not provenance.is_fully_verified:
            manifest.note(
                f"{len(provenance.unverified)} protocol constant(s) are UNVERIFIED: "
                f"{sorted(provenance.unverified)}. This run is an approximation of the "
                "published protocol, not a byte-exact reproduction."
            )
        try:
            manifest.dataset_hashes = {"primary": self.benchmark.dataset_hash()}
        except Exception as exc:  # noqa: BLE001 - hashing must not block a run
            manifest.note(f"dataset hash unavailable: {exc}")
        manifest.write(self.run_dir)
        self.manifest = manifest

        already_done = (
            set() if self.config.force else self.store.completed_job_ids(include_failed=True)
        )
        package_versions = python_packages()
        commit = git_commit()
        results: list[RunResult] = []

        n_tasks = self.benchmark.n_tasks()
        for index, task in enumerate(self.benchmark.iter_tasks(), start=1):
            if progress:
                print(
                    f"[{index}/{n_tasks}] {task.region}/{task.species_id} "
                    f"(presences={task.train_presence_n}, test={len(task.test)})",
                    flush=True,
                )
            for scenario in scenarios:
                for result in self._run_task_scenario(
                    task,
                    scenario,
                    models,
                    already_done=already_done,
                    package_versions=package_versions,
                    git_sha=commit,
                ):
                    results.append(result)
                    self.store.write(result)
                    if result.status == ModelStatus.OK.value:
                        manifest.n_jobs_completed += 1
                    elif result.status.startswith("SKIPPED"):
                        manifest.n_jobs_skipped += 1
                    else:
                        manifest.n_jobs_failed += 1
                    manifest.n_jobs_total += 1

        manifest.finish("COMPLETED")
        manifest.write(self.run_dir)
        self.store.write_parquet()

        # Include anything an earlier interrupted run left behind, so the
        # returned object matches what is on disk.
        return BenchmarkResults(
            results=self.store.load_all(),
            manifest=manifest,
            run_dir=self.run_dir,
            provenance=provenance,
        )

    def _run_task_scenario(
        self,
        task: SpeciesTask,
        scenario: str,
        models: Sequence[str],
        *,
        already_done: set[str],
        package_versions: dict[str, str],
        git_sha: str,
    ) -> Iterator[RunResult]:
        """Prepare one split, then evaluate every model on it."""
        pending = [
            name
            for name in models
            if job_id_for(
                benchmark_id=self.benchmark.benchmark_id,
                scenario=scenario,
                region=task.region,
                species_id=task.species_id,
                model_name=name,
            )
            not in already_done
        ]
        if not pending:
            return

        base = dict(
            benchmark_id=self.benchmark.benchmark_id,
            benchmark_version=self.benchmark.benchmark_version,
            run_id=self.run_id,
            species_id=task.species_id,
            region=task.region,
            group=task.group,
            evaluation_scenario=scenario,
            split_id=scenario,
            seed=self.config.seed,
            package_versions=package_versions,
            git_commit=git_sha,
        )

        # --- prepare the split once, before any model is involved ------------
        try:
            split = self.benchmark.prepare(task, scenario)
        except SkippableError as exc:
            for name in pending:
                yield self._skipped(base, name, scenario, exc)
            return
        except Exception as exc:  # noqa: BLE001
            if self.config.strict:
                raise
            for name in pending:
                yield self._failed(base, name, scenario, exc)
            return

        # --- audit the split -------------------------------------------------
        audit_status = "NOT_RUN"
        if self.config.audit_leakage:
            auditor = LeakageAuditor(
                spatial_buffer_m=DEFAULT_BUFFER_M if scenario == "spatial" else None,
                allow_coordinate_features=bool(self.config.extra.get("include_coordinates", False)),
            )
            report = auditor.audit_task(task if scenario == "nonspatial" else _task_from(split, task))
            for check in auditor.audit_split(split).checks:
                report.add(check)
            audit_status = report.status
            self.store.write_leakage(
                job_id_for(
                    benchmark_id=self.benchmark.benchmark_id,
                    scenario=scenario,
                    region=task.region,
                    species_id=task.species_id,
                    model_name="__split__",
                ),
                report.to_dict(),
            )
            if not report.passed:
                if self.config.strict:
                    report.raise_if_failed()
                for name in pending:
                    yield self._failed(
                        base,
                        name,
                        scenario,
                        LeakageError("; ".join(c.detail for c in report.failures)),
                        audit=audit_status,
                    )
                return

        split_hash = hash_split(
            [f"{r}" for r in task.train.get("siteid", pd.Series(range(len(task.train))))],
            [f"{r}" for r in task.test.get("siteid", pd.Series(range(len(task.test))))],
            extra={"scenario": scenario, "buffer_m": DEFAULT_BUFFER_M if scenario == "spatial" else None},
        )
        dataset_hash = self.manifest.dataset_hashes.get("primary", "") if self.manifest else ""

        for name in pending:
            yield self._run_one_model(
                name,
                split,
                base=base,
                scenario=scenario,
                split_hash=split_hash,
                dataset_hash=dataset_hash,
                audit_status=audit_status,
            )

    def _run_one_model(
        self,
        name: str,
        split: PreparedSplit,
        *,
        base: dict[str, Any],
        scenario: str,
        split_hash: str,
        dataset_hash: str,
        audit_status: str,
    ) -> RunResult:
        job_id = job_id_for(
            benchmark_id=base["benchmark_id"],
            scenario=scenario,
            region=base["region"],
            species_id=base["species_id"],
            model_name=name,
        )
        status, detail = self._availability(name, scenario)
        if status != ModelStatus.OK.value:
            return RunResult(
                **base,
                job_id=job_id,
                model_name=name,
                model_family=_family_of(name),
                status=status,
                status_detail=detail,
                split_hash=split_hash,
                dataset_hash=dataset_hash,
                leakage_audit=audit_status,
                train_n=split.n_train,
                train_presence_n=split.train_presence_n,
                train_background_n=split.train_background_n,
                test_n=split.n_test,
                test_presence_n=split.test_presence_n,
                test_absence_n=split.test_absence_n,
            )

        adapter = self._adapter(name, scenario)
        counts = dict(
            train_n=split.n_train,
            train_presence_n=split.train_presence_n,
            train_background_n=split.train_background_n,
            test_n=split.n_test,
            test_presence_n=split.test_presence_n,
            test_absence_n=split.test_absence_n,
        )

        started = time.perf_counter()
        try:
            fit_result = adapter.run(split)
        except SkippableError as exc:
            return RunResult(
                **base, **counts,
                job_id=job_id,
                model_name=name,
                model_family=adapter.family,
                status=getattr(exc, "status", ModelStatus.SKIPPED_DATA.value),
                status_detail=str(exc),
                split_hash=split_hash,
                dataset_hash=dataset_hash,
                leakage_audit=audit_status,
            )
        except Exception as exc:  # noqa: BLE001
            if self.config.strict:
                raise
            return RunResult(
                **base, **counts,
                job_id=job_id,
                model_name=name,
                model_family=adapter.family,
                status=ModelStatus.FAILED.value,
                status_detail=f"{type(exc).__name__}: {exc}",
                split_hash=split_hash,
                dataset_hash=dataset_hash,
                leakage_audit=audit_status,
                metrics={},
                total_time_seconds=time.perf_counter() - started,
            )

        metrics = compute_metrics(
            split.y_test,
            fit_result.probabilities,
            self.config.metrics or list(self.benchmark.metrics),
        )
        metadata = adapter.get_metadata()
        return RunResult(
            **base, **counts,
            job_id=job_id,
            model_name=name,
            model_family=adapter.family,
            model_version=metadata.get("model_version", ""),
            hyperparameters=metadata.get("hyperparameters", {}),
            status=ModelStatus.OK.value,
            metrics=metrics,
            fit_time_seconds=fit_result.fit_seconds,
            predict_time_seconds=fit_result.predict_seconds,
            total_time_seconds=time.perf_counter() - started,
            peak_memory_mb=fit_result.peak_memory_mb,
            device=fit_result.device,
            compute_budget=(
                adapter.compute_budget() if hasattr(adapter, "compute_budget") else {}
            ),
            split_hash=split_hash,
            dataset_hash=dataset_hash,
            model_config_hash=hash_json(metadata.get("hyperparameters", {})),
            leakage_audit=audit_status,
        )

    # -------------------------------------------------------------- helpers --
    @staticmethod
    def _skipped(base: dict[str, Any], name: str, scenario: str, exc: Exception) -> RunResult:
        return RunResult(
            **base,
            job_id=job_id_for(
                benchmark_id=base["benchmark_id"],
                scenario=scenario,
                region=base["region"],
                species_id=base["species_id"],
                model_name=name,
            ),
            model_name=name,
            model_family=_family_of(name),
            status=getattr(exc, "status", ModelStatus.SKIPPED_DATA.value),
            status_detail=str(exc),
        )

    @staticmethod
    def _failed(
        base: dict[str, Any],
        name: str,
        scenario: str,
        exc: Exception,
        *,
        audit: str = "NOT_RUN",
    ) -> RunResult:
        return RunResult(
            **base,
            job_id=job_id_for(
                benchmark_id=base["benchmark_id"],
                scenario=scenario,
                region=base["region"],
                species_id=base["species_id"],
                model_name=name,
            ),
            model_name=name,
            model_family=_family_of(name),
            status=ModelStatus.FAILED.value,
            status_detail=f"{type(exc).__name__}: {exc}",
            leakage_audit=audit,
        )


def _task_from(split: PreparedSplit, original: SpeciesTask) -> SpeciesTask:
    """Rebuild a task view of a prepared split, for the spatial buffer check.

    The auditor verifies the buffer against the *filtered* training set, which
    is what the split holds; the original task still has the unfiltered rows.
    """
    train = original.train.iloc[: 0].copy()
    train = pd.DataFrame(
        {
            "region": original.region,
            "group": original.group,
            "siteid": [f"t{i}" for i in range(split.n_train)],
            "spid": original.species_id,
            "x": split.coords_train[:, 0],
            "y": split.coords_train[:, 1],
            "occ": np.asarray(split.y_train, dtype=int),
        }
    )
    for column in original.predictors:
        train[column] = np.nan
    test = pd.DataFrame(
        {
            "region": original.region,
            "group": original.group,
            "siteid": [f"e{i}" for i in range(split.n_test)],
            "spid": original.species_id,
            "x": split.coords_test[:, 0],
            "y": split.coords_test[:, 1],
            "occ": np.asarray(split.y_test, dtype=int),
        }
    )
    for column in original.predictors:
        test[column] = np.nan
    return SpeciesTask(
        region=original.region,
        species_id=original.species_id,
        train=train,
        test=test,
        predictors=list(original.predictors),
        categorical_predictors=[],
        group=original.group,
        crs=original.crs,
        geographic=original.geographic,
    )


def _family_of(name: str) -> str:
    """Best-effort family lookup that tolerates an unimportable backend."""
    from sdmbench.models.base import MODEL_REGISTRY, _load_all_adapters

    if not MODEL_REGISTRY:
        _load_all_adapters()
    cls = MODEL_REGISTRY.get(name)
    return getattr(cls, "family", "unknown") if cls else "unknown"


class Benchmark:
    """High-level entry point for the Python API.

    Examples
    --------
    >>> from sdmbench import Benchmark                     # doctest: +SKIP
    >>> bench = Benchmark.from_csv(                        # doctest: +SKIP
    ...     "species.csv", target="presence",
    ...     coordinates=("longitude", "latitude"),
    ... )
    >>> results = bench.run(models=["knn", "random-forest-sklearn", "xgboost"])
    >>> print(results.leaderboard())                       # doctest: +SKIP
    """

    def __init__(self, benchmark: PaperBenchmark, config: RunConfig | None = None) -> None:
        self.benchmark = benchmark
        self.config = config or benchmark.config

    @classmethod
    def from_name(cls, name: str, **config_kwargs: Any) -> Benchmark:
        config = RunConfig(benchmark=name, **config_kwargs)
        return cls(get_benchmark(name, config=config), config)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        target: str = "presence",
        coordinates: tuple[str, str] | None = ("longitude", "latitude"),
        predictors: Sequence[str] | None = None,
        include_coordinates: bool = False,
        test_csv: str | Path | None = None,
        **config_kwargs: Any,
    ) -> Benchmark:
        """Build an ad-hoc benchmark around a user CSV.

        ``include_coordinates`` must be set explicitly to allow longitude and
        latitude to be used as predictors.
        """
        from sdmbench.data.csv import CsvDataset

        dataset = CsvDataset(
            path,
            target=target,
            coordinates=coordinates,
            predictors=predictors,
            include_coordinates=include_coordinates,
        )
        test_frame = pd.read_csv(test_csv) if test_csv else None
        config = RunConfig(benchmark="csv", **config_kwargs)
        return cls(_AdHocBenchmark(dataset, config, test_frame=test_frame), config)

    def run(
        self,
        models: Sequence[str] | None = None,
        *,
        scenarios: Sequence[str] | None = None,
        run_dir: str | Path | None = None,
        progress: bool = False,
        **kwargs: Any,
    ) -> BenchmarkResults:
        config = self.config
        if models:
            config.models = resolve_model_names(models)
        if scenarios:
            config.evaluation.scenarios = list(scenarios)
        runner = BenchmarkRunner(self.benchmark, config, run_dir=run_dir, **kwargs)
        return runner.run(progress=progress)


class _AdHocBenchmark(PaperBenchmark):
    """Wraps a user dataset in the benchmark interface.

    Deliberately *not* registered: it has no paper, no published reference
    values, and no reproduction claim. It exists so a user CSV can travel
    through the same engine -- and so its results carry the same leakage audit
    and manifest as a published benchmark.
    """

    benchmark_id = "csv"
    benchmark_version = "1"
    scenarios = ("nonspatial", "spatial")
    metrics = ("roc_auc", "pr_auc", "calibration_slope", "boyce")
    default_models = ("logistic-regression", "knn", "random-forest-sklearn")

    paper = PaperMetadata(
        paper_id="user_csv",
        title="User-supplied dataset",
        authors=("user",),
        year=0,
    )

    def __init__(self, dataset: Any, config: RunConfig, *, test_frame: pd.DataFrame | None = None):
        super().__init__(config)
        self.dataset = dataset
        self.test_frame = test_frame

    def iter_tasks(self) -> Iterator[SpeciesTask]:
        yield self.dataset.to_task(test_frame=self.test_frame, seed=self.config.seed)

    def n_tasks(self) -> int:
        return 1

    def dataset_hash(self) -> str:
        return self.dataset.content_hash()

    def prepare(self, task: SpeciesTask, scenario: str) -> PreparedSplit:
        from sdmbench.preprocessing import SdmRecipe
        from sdmbench.splits.random import identity_split
        from sdmbench.splits.spatial import apply_spatial_buffer

        self.validate_scenario(scenario)
        if scenario == "spatial":
            task, info = apply_spatial_buffer(
                task, buffer_m=float(self.config.extra.get("spatial_buffer_m", DEFAULT_BUFFER_M))
            )
        else:
            task, info = identity_split(task)
        task.require_both_classes()

        recipe = SdmRecipe(
            numeric_predictors=task.numeric_predictors,
            categorical_predictors=task.categorical_predictors,
        )
        X_train = recipe.fit_transform(task.train)
        X_test = recipe.transform(task.test)
        meta = recipe.describe()
        meta["fitted_on"] = "train"

        return PreparedSplit(
            X_train=X_train,
            y_train=np.asarray(task.train["occ"], dtype=int),
            X_test=X_test,
            y_test=np.asarray(task.test["occ"], dtype=int),
            coords_train=task.train_coords(),
            coords_test=task.test_coords(),
            feature_names=recipe.output_columns,
            categorical_features=list(task.categorical_predictors),
            scenario=scenario,
            split_id=scenario,
            crs=task.crs,
            geographic=task.geographic,
            metadata={
                "region": task.region,
                "species_id": task.species_id,
                "split": info,
                "preprocessing": meta,
                **task.metadata,
            },
        )
