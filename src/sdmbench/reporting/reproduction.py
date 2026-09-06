"""The reproduction report.

Compares what sdmbench computed against what the paper reported, while keeping
the two in strictly separate namespaces:

``published_reference``
    Values quoted from the paper. Never computed, never overwritten.
``reproduced_result``
    Values sdmbench measured in this run.

A model with no reproduced result is reported ``NOT RUN`` -- the published
number is never echoed back as though it had been measured.

On tolerances
-------------
The brief asks not to invent arbitrary MATCH thresholds. The ones used here are
stated, justified and configurable, and the report prints them every time:

``MATCH``     |difference| <= 0.010
``CLOSE``     |difference| <= 0.025
``MISMATCH``  otherwise

The 0.010 band is the order of the Monte-Carlo variation you would expect from
the protocol's own stochastic components -- background subsampling, random
forest bagging, BRT's internal CV fold assignment, the K = 16 ensemble draw --
when the seed cannot be matched exactly. The 0.025 band is roughly the spread
between the four traditional baselines in the paper (0.717-0.732): a
discrepancy that large is the size of a real between-method difference and
therefore cannot be dismissed as noise.

These are heuristics for triage, not statistical tests. A ``MATCH`` is evidence
that the pipeline is behaving, not proof of exact reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sdmbench.benchmarks.base import PaperBenchmark
from sdmbench.evaluation.aggregate import build_leaderboard, summarise_metric
from sdmbench.provenance import ProvenanceReport

__all__ = [
    "ReproductionReport",
    "ReproductionRow",
    "build_reproduction_report",
    "MATCH_TOLERANCE",
    "CLOSE_TOLERANCE",
]

#: See the module docstring for the justification of these bands.
MATCH_TOLERANCE = 0.010
CLOSE_TOLERANCE = 0.025


@dataclass
class ReproductionRow:
    """One published value and its reproduction status."""

    model: str
    scenario: str
    metric: str
    published: float
    reproduced: float | None = None
    n_species_published: int | None = None
    n_species_reproduced: int | None = None
    reproduced_sd: float | None = None
    reproduced_ci: tuple[float, float] | None = None
    status: str = "NOT RUN"
    source: str = ""
    note: str = ""

    @property
    def difference(self) -> float | None:
        """``reproduced - published``, or ``None`` when not run."""
        if self.reproduced is None or not np.isfinite(self.reproduced):
            return None
        return float(self.reproduced - self.published)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "scenario": self.scenario,
            "metric": self.metric,
            "published_reference": self.published,
            "reproduced_result": self.reproduced,
            "difference": self.difference,
            "n_species_published": self.n_species_published,
            "n_species_reproduced": self.n_species_reproduced,
            "reproduced_sd": self.reproduced_sd,
            "reproduced_ci": list(self.reproduced_ci) if self.reproduced_ci else None,
            "status": self.status,
            "source": self.source,
            "note": self.note,
        }


@dataclass
class ReproductionReport:
    """The full comparison for one benchmark."""

    benchmark_id: str
    paper: dict[str, Any]
    rows: list[ReproductionRow] = field(default_factory=list)
    provenance: ProvenanceReport | None = None
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    upstream: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    notes: list[str] = field(default_factory=list)
    match_tolerance: float = MATCH_TOLERANCE
    close_tolerance: float = CLOSE_TOLERANCE

    @property
    def overall_status(self) -> str:
        statuses = {r.status for r in self.rows}
        if not statuses or statuses == {"NOT RUN"}:
            return "NOT RUN"
        if "MISMATCH" in statuses:
            return "MISMATCH"
        if "CLOSE" in statuses:
            return "CLOSE"
        if "MATCH" in statuses:
            return "MATCH"
        return "PARTIAL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "run_id": self.run_id,
            "paper": self.paper,
            "overall_status": self.overall_status,
            "tolerances": {
                "match": self.match_tolerance,
                "close": self.close_tolerance,
                "justification": (
                    "MATCH band approximates Monte-Carlo variation from the protocol's own "
                    "stochastic components; CLOSE band approximates the observed spread "
                    "between the paper's traditional baselines. Heuristics for triage, not "
                    "statistical tests."
                ),
            },
            "comparisons": [r.to_dict() for r in self.rows],
            "dataset": self.dataset_summary,
            "upstream": self.upstream,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "notes": self.notes,
        }

    def render(self) -> str:
        lines: list[str] = []
        title = f"REPRODUCTION REPORT -- {self.benchmark_id}"
        lines.append("=" * len(title))
        lines.append(title)
        lines.append("=" * len(title))
        lines.append("")

        lines.append("PAPER")
        lines.append(f"  {self.paper.get('citation', self.paper.get('title', ''))}")
        if self.paper.get("doi"):
            lines.append(f"  DOI: {self.paper['doi']}")
        if self.paper.get("code_url"):
            lines.append(f"  Code: {self.paper['code_url']}")
        lines.append("")

        lines.append("DATA")
        if self.dataset_summary:
            for key in ("dataset", "version", "total_species", "disdat_version"):
                if key in self.dataset_summary:
                    lines.append(f"  {key}: {self.dataset_summary[key]}")
            regions = self.dataset_summary.get("regions")
            if isinstance(regions, dict):
                lines.append(f"  regions: {len(regions)} ({', '.join(sorted(regions))})")
        else:
            lines.append("  (dataset not available; run `sdmbench data fetch disdat`)")
        lines.append("")

        lines.append("UPSTREAM VERSION")
        if self.upstream:
            commit = self.upstream.get("commit") or "(not fetched)"
            lines.append(f"  repository: {self.upstream.get('repo_url', '')}")
            lines.append(f"  commit:     {commit}")
            if self.upstream.get("note"):
                lines.append(f"  note:       {self.upstream['note']}")
        else:
            lines.append("  (no upstream repository registered)")
        lines.append("")

        lines.append("RESULTS")
        lines.append(
            f"  Tolerances: MATCH <= {self.match_tolerance:.3f}, "
            f"CLOSE <= {self.close_tolerance:.3f} (absolute difference)"
        )
        lines.append("")
        header = (
            f"  {'Model':<20} {'Scenario':<12} {'Metric':<18} "
            f"{'Published':>10} {'Reproduced':>11} {'Diff':>8}  Status"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for row in self.rows:
            reproduced = "-" if row.reproduced is None else f"{row.reproduced:.3f}"
            difference = "-" if row.difference is None else f"{row.difference:+.3f}"
            lines.append(
                f"  {row.model:<20} {row.scenario:<12} {row.metric:<18} "
                f"{row.published:>10.3f} {reproduced:>11} {difference:>8}  {row.status}"
            )
            if row.n_species_reproduced is not None and row.n_species_published is not None:
                if row.n_species_reproduced != row.n_species_published:
                    lines.append(
                        f"  {'':<20} species: published n={row.n_species_published}, "
                        f"reproduced n={row.n_species_reproduced} -- means are over "
                        "different species sets"
                    )
            if row.note:
                lines.append(f"  {'':<20} {row.note}")
        lines.append("")
        lines.append(f"OVERALL STATUS: {self.overall_status}")

        if self.provenance is not None and not self.provenance.is_fully_verified:
            lines.append("")
            lines.append("PROTOCOL VERIFICATION")
            lines.append(
                f"  {len(self.provenance.unverified)} of {len(self.provenance.entries)} "
                "protocol constants could not be verified against a public source:"
            )
            for key, sourced in sorted(self.provenance.unverified.items()):
                lines.append(f"    ! {key}: {sourced.citation}")
            lines.append(
                "  This run therefore approximates the published protocol. It is not a "
                "byte-exact reproduction, and differences above may reflect these gaps "
                "rather than an implementation error."
            )

        if self.notes:
            lines.append("")
            lines.append("NOTES")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


def _status_for(difference: float | None, *, match: float, close: float) -> str:
    if difference is None or not np.isfinite(difference):
        return "NOT RUN"
    magnitude = abs(difference)
    if magnitude <= match:
        return "MATCH"
    if magnitude <= close:
        return "CLOSE"
    return "MISMATCH"


def build_reproduction_report(
    benchmark: PaperBenchmark,
    results: pd.DataFrame | None,
    *,
    run_id: str = "",
    dataset_summary: dict[str, Any] | None = None,
    upstream: dict[str, Any] | None = None,
    match_tolerance: float = MATCH_TOLERANCE,
    close_tolerance: float = CLOSE_TOLERANCE,
) -> ReproductionReport:
    """Compare a run's results against the benchmark's published values."""
    references = benchmark.published_results()
    report = ReproductionReport(
        benchmark_id=benchmark.benchmark_id,
        paper=benchmark.paper.to_dict(),
        provenance=benchmark.provenance(),
        dataset_summary=dataset_summary or {},
        upstream=upstream or {},
        run_id=run_id,
        notes=list(references.notes),
        match_tolerance=match_tolerance,
        close_tolerance=close_tolerance,
    )

    have_results = results is not None and not results.empty
    for entry in references.entries:
        row = ReproductionRow(
            model=entry.model,
            scenario=entry.scenario,
            metric=entry.metric,
            published=entry.value,
            n_species_published=entry.n_species,
            source=entry.source,
        )
        if have_results:
            subset = results[
                (results["model_name"] == entry.model)
                & (results["evaluation_scenario"] == entry.scenario)
                & (results["status"] == "OK")
            ]
            if not subset.empty and entry.metric in subset.columns:
                stats_ = summarise_metric(subset[entry.metric])
                if stats_["n"]:
                    row.reproduced = stats_["mean"]
                    row.reproduced_sd = stats_["sd"]
                    row.reproduced_ci = (stats_["ci_low"], stats_["ci_high"])
                    row.n_species_reproduced = int(subset["species_id"].nunique())
            elif not subset.empty:
                row.note = f"metric {entry.metric!r} was not computed in this run"
            else:
                statuses = (
                    results[
                        (results["model_name"] == entry.model)
                        & (results["evaluation_scenario"] == entry.scenario)
                    ]["status"]
                    .value_counts()
                    .to_dict()
                )
                if statuses:
                    row.note = "no successful jobs: " + ", ".join(
                        f"{k}={v}" for k, v in statuses.items()
                    )
        row.status = _status_for(
            row.difference, match=match_tolerance, close=close_tolerance
        )
        report.rows.append(row)

    if not have_results:
        report.notes.append(
            "No computed results were supplied, so every row is NOT RUN. The published "
            "values above are reference targets only."
        )
    return report


def extra_models_table(
    benchmark: PaperBenchmark,
    results: pd.DataFrame,
    *,
    scenario: str = "nonspatial",
    metric: str = "roc_auc",
) -> pd.DataFrame:
    """Models evaluated in this run that the paper did not report.

    The Mode B output: everything measured under the paper's exact rules but
    not part of the original comparison.
    """
    if results.empty:
        return pd.DataFrame()
    published = {e.model for e in benchmark.published_results().entries}
    board = build_leaderboard(results, metric=metric, scenario=scenario)
    if board.empty:
        return board
    return board[~board["model_name"].isin(published)].reset_index(drop=True)
