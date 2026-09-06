"""``run_manifest.json`` -- the record that makes a run re-runnable.

One manifest per run, written when the run starts and updated when it
finishes. It records the environment, configuration, seeds, input hashes, model
checkpoint hashes and the upstream commit, so that a later reader can tell
whether a re-run is comparable.
"""

from __future__ import annotations

import datetime as _dt
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sdmbench.reproducibility.environment import capture_environment
from sdmbench.reproducibility.hashes import hash_json

__all__ = ["RunManifest", "new_run_id"]

MANIFEST_FILENAME = "run_manifest.json"


def new_run_id(prefix: str = "run") -> str:
    """Time-ordered, collision-resistant run identifier."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass
class RunManifest:
    """Everything needed to interpret and repeat a benchmark run."""

    run_id: str
    benchmark_id: str
    benchmark_version: str = "0"
    sdmbench_version: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    dataset_hashes: dict[str, str] = field(default_factory=dict)
    split_hashes: dict[str, str] = field(default_factory=dict)
    model_checkpoints: dict[str, dict[str, Any]] = field(default_factory=dict)
    upstream: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    models_requested: list[str] = field(default_factory=list)
    scenarios: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    status: str = "RUNNING"
    n_jobs_total: int = 0
    n_jobs_completed: int = 0
    n_jobs_skipped: int = 0
    n_jobs_failed: int = 0
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------- lifecycle --
    @classmethod
    def start(
        cls,
        *,
        benchmark_id: str,
        benchmark_version: str = "0",
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
        include_r: bool = True,
    ) -> RunManifest:
        from sdmbench import __version__

        return cls(
            run_id=run_id or new_run_id(),
            benchmark_id=benchmark_id,
            benchmark_version=benchmark_version,
            sdmbench_version=__version__,
            config=config or {},
            environment=capture_environment(include_r=include_r),
            started_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            status="RUNNING",
        )

    def finish(self, status: str = "COMPLETED") -> None:
        self.finished_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        self.status = status

    def note(self, message: str) -> None:
        """Attach a human-readable caveat to the run.

        Used, for example, to record that the upstream repository could not be
        fetched, so the run used sdmbench's native protocol implementation.
        """
        self.notes.append(message)

    def record_checkpoint(self, name: str, info: dict[str, Any]) -> None:
        self.model_checkpoints[name] = info

    # ------------------------------------------------------------------- io --
    @property
    def config_hash(self) -> str:
        return hash_json(self.config)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_version": self.benchmark_version,
            "sdmbench_version": self.sdmbench_version,
            "config": self.config,
            "config_hash": self.config_hash,
            "seeds": self.seeds,
            "dataset_hashes": self.dataset_hashes,
            "split_hashes": self.split_hashes,
            "model_checkpoints": self.model_checkpoints,
            "upstream": self.upstream,
            "provenance": self.provenance,
            "environment": self.environment,
            "models_requested": self.models_requested,
            "scenarios": self.scenarios,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "jobs": {
                "total": self.n_jobs_total,
                "completed": self.n_jobs_completed,
                "skipped": self.n_jobs_skipped,
                "failed": self.n_jobs_failed,
            },
            "notes": self.notes,
        }
        return data

    def write(self, directory: str | Path) -> Path:
        """Write (atomically) to ``<directory>/run_manifest.json``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_FILENAME
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return path

    @classmethod
    def read(cls, directory: str | Path) -> RunManifest:
        path = Path(directory)
        if path.is_dir():
            path = path / MANIFEST_FILENAME
        data = json.loads(path.read_text(encoding="utf-8"))
        jobs = data.get("jobs", {})
        return cls(
            run_id=data["run_id"],
            benchmark_id=data["benchmark_id"],
            benchmark_version=data.get("benchmark_version", "0"),
            sdmbench_version=data.get("sdmbench_version", ""),
            config=data.get("config", {}),
            seeds=data.get("seeds", {}),
            dataset_hashes=data.get("dataset_hashes", {}),
            split_hashes=data.get("split_hashes", {}),
            model_checkpoints=data.get("model_checkpoints", {}),
            upstream=data.get("upstream", {}),
            provenance=data.get("provenance", {}),
            environment=data.get("environment", {}),
            models_requested=data.get("models_requested", []),
            scenarios=data.get("scenarios", []),
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
            status=data.get("status", "UNKNOWN"),
            n_jobs_total=jobs.get("total", 0),
            n_jobs_completed=jobs.get("completed", 0),
            n_jobs_skipped=jobs.get("skipped", 0),
            n_jobs_failed=jobs.get("failed", 0),
            notes=data.get("notes", []),
        )
