"""Upstream source metadata.

Where each benchmark's authoritative implementation lives, and what sdmbench
should look at inside it when it becomes available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["UpstreamSource", "UPSTREAM_SOURCES", "get_upstream"]


@dataclass(frozen=True)
class UpstreamSource:
    """A benchmark's upstream repository."""

    benchmark_id: str
    repo_url: str
    #: Files worth inspecting once the repository is available, and what each
    #: is expected to confirm.
    files_of_interest: dict[str, str] = field(default_factory=dict)
    #: Constants that sdmbench currently marks UNVERIFIED and that this
    #: repository should resolve.
    resolves: tuple[str, ...] = ()
    #: Status as of the last check.
    availability: str = "unknown"
    note: str = ""
    license: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "repo_url": self.repo_url,
            "files_of_interest": self.files_of_interest,
            "resolves": list(self.resolves),
            "availability": self.availability,
            "note": self.note,
            "license": self.license,
        }


UPSTREAM_SOURCES: dict[str, UpstreamSource] = {
    "tabpfn-sdm-2026": UpstreamSource(
        benchmark_id="tabpfn-sdm-2026",
        repo_url="https://github.com/rdinnager/TabPFN-SDM",
        files_of_interest={
            "R/run_tabpfn_finetuned.R": (
                "run_tabpfn_finetuned_ensemble() -- resolves whether inference partitions "
                "the background or draws balanced samples with overlap"
            ),
            "_targets.R": (
                "the targets pipeline (~7,500 targets) -- authoritative order of "
                "preprocessing, splitting and evaluation"
            ),
            "R/": "model wrappers for maxnet, BRT, RF and GAM, including the GAM formulas",
            "R/metrics.R": "exact yardstick metric calls and the Miller calibration slope",
            "R/spatial.R": "construction of the 10 km buffer with sf",
            "python/": "the finetuning loop, sub-batching and ensemble implementation",
        },
        resolves=(
            "gam_region_formulas",
            "random_forest_replace",
            "ensemble_scheme",
            "pr_auc_estimator",
        ),
        availability="not_public",
        note=(
            "Cited in Dinnage & Warren (2026) sec. 2.8 as the code repository, but it "
            "returned HTTP 404 and does not appear in the owner's public repository list. "
            "It is presumably still private. sdmbench's native implementation is built from "
            "the paper's Methods section and the Hugging Face model card; constants that "
            "depend on this repository are tagged UNVERIFIED."
        ),
        license="unknown -- do not vendor source until a licence is published",
    ),
}


def get_upstream(benchmark_id: str) -> UpstreamSource | None:
    """Upstream source for a benchmark, if one is registered."""
    return UPSTREAM_SOURCES.get(benchmark_id) or UPSTREAM_SOURCES.get(
        benchmark_id.replace("_", "-")
    )
