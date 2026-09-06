"""Result storage and checkpointing.

A full run is ``226 species x N models x 2 scenarios`` -- thousands of jobs,
many of them slow. It has to be interruptible and resumable, so every job
writes its own small JSON file the moment it finishes::

    <run_dir>/
        run_manifest.json
        results/<scenario>/<region>/<species>__<model>.json
        results.parquet          consolidated table
        leakage/<job_id>.json    audit reports

On restart, :meth:`ResultStore.completed_job_ids` tells the runner what already
exists and only the missing jobs are run. A completed successful job is never
recomputed unless ``--force`` is given.

Failed and skipped jobs are also persisted, so a resumed run does not retry a
model whose dependency is still missing -- but ``--force`` and
``--retry-failed`` both bring them back.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from sdmbench.paths import ensure_dir
from sdmbench.results import RESULT_COLUMNS, RunResult, results_to_dataframe

__all__ = ["ResultStore", "job_id_for"]

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(value: str) -> str:
    """Filesystem-safe fragment (Windows included)."""
    return _SAFE.sub("-", str(value)).strip("-") or "na"


def job_id_for(
    *,
    benchmark_id: str,
    scenario: str,
    region: str,
    species_id: str,
    model_name: str,
    repeat_id: int = 0,
) -> str:
    """Stable identifier for one unit of work.

    Deterministic, so a resumed run recognises previously completed jobs.
    """
    parts = [benchmark_id, scenario, region, species_id, model_name]
    if repeat_id:
        parts.append(f"r{repeat_id}")
    return "__".join(_slug(p) for p in parts)


class ResultStore:
    """Reads and writes per-job results under a run directory."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.results_dir = self.run_dir / "results"
        self.leakage_dir = self.run_dir / "leakage"

    def ensure(self) -> None:
        ensure_dir(self.results_dir)
        ensure_dir(self.leakage_dir)

    # ------------------------------------------------------------------ paths --
    def path_for(self, result: RunResult) -> Path:
        return (
            self.results_dir
            / _slug(result.evaluation_scenario)
            / _slug(result.region or "na")
            / f"{_slug(result.species_id)}__{_slug(result.model_name)}"
            f"{'' if not result.repeat_id else f'__r{result.repeat_id}'}.json"
        )

    # ------------------------------------------------------------------- io --
    def write(self, result: RunResult) -> Path:
        """Persist one result atomically."""
        path = self.path_for(result)
        result.write_json(path)
        return path

    def write_leakage(self, job_id: str, report: dict) -> Path:
        ensure_dir(self.leakage_dir)
        path = self.leakage_dir / f"{_slug(job_id)}.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        tmp.replace(path)
        return path

    def iter_results(self) -> Iterator[RunResult]:
        """Yield every persisted result, skipping unreadable files."""
        if not self.results_dir.is_dir():
            return
        for path in sorted(self.results_dir.rglob("*.json")):
            try:
                yield RunResult.read_json(path)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    def load_all(self) -> list[RunResult]:
        return list(self.iter_results())

    def completed_job_ids(self, *, include_failed: bool = True) -> set[str]:
        """Job ids already on disk.

        ``include_failed=False`` makes a resumed run retry jobs that previously
        failed while still skipping successful ones.
        """
        out: set[str] = set()
        for result in self.iter_results():
            if not result.job_id:
                continue
            if include_failed or result.status == "OK":
                out.add(result.job_id)
        return out

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for result in self.iter_results():
            counts[result.status] = counts.get(result.status, 0) + 1
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------ consolidate --
    def to_dataframe(self) -> pd.DataFrame:
        """All results as one DataFrame in the canonical column order."""
        return results_to_dataframe(self.iter_results())

    def write_parquet(self, path: str | Path | None = None) -> Path:
        """Consolidate every per-job result into a single Parquet table."""
        target = Path(path) if path else self.run_dir / "results.parquet"
        ensure_dir(target.parent)
        frame = self.to_dataframe()
        frame.to_parquet(target, index=False)
        return target

    def write_csv(self, path: str | Path | None = None) -> Path:
        """CSV alongside the Parquet, for eyeballing without pyarrow."""
        target = Path(path) if path else self.run_dir / "results.csv"
        ensure_dir(target.parent)
        self.to_dataframe().to_csv(target, index=False)
        return target

    @staticmethod
    def read_parquet(path: str | Path) -> pd.DataFrame:
        """Load a consolidated results table, checking its schema."""
        frame = pd.read_parquet(path)
        missing = [c for c in RESULT_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{path} is not an sdmbench results table; missing columns: {missing[:8]}"
            )
        return frame

    @classmethod
    def find_latest(cls, runs_root: str | Path, benchmark_id: str | None = None) -> ResultStore | None:
        """Most recent run directory, optionally filtered by benchmark.

        Run ids begin with a UTC timestamp, so lexical order is chronological.
        """
        root = Path(runs_root)
        if not root.is_dir():
            return None
        candidates = [p for p in root.iterdir() if p.is_dir() and (p / "run_manifest.json").is_file()]
        if benchmark_id:
            filtered = []
            for path in candidates:
                try:
                    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if manifest.get("benchmark_id") == benchmark_id:
                    filtered.append(path)
            candidates = filtered
        if not candidates:
            return None
        return cls(sorted(candidates, key=lambda p: p.name)[-1])
