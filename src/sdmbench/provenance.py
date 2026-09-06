"""Provenance tracking for reproduction constants.

A reproduction framework is only trustworthy if it is explicit about *where
each protocol constant came from*. Some values in the TabPFN-SDM 2026 recipe
are stated verbatim in the paper, some only in the Hugging Face model card,
some are inferable from ``disdat`` itself, and some could not be verified from
any public source at the time of writing.

Rather than silently guessing, every non-obvious constant in a benchmark recipe
is wrapped in a :class:`Sourced` value carrying a :class:`Provenance` tag and a
citation. ``sdmbench report`` prints the provenance table, and any recipe that
still contains ``UNVERIFIED`` values is reported as such instead of being
presented as an exact reproduction.

Why this exists
---------------
The upstream repository named in the preprint (``github.com/rdinnager/TabPFN-SDM``)
returned HTTP 404 when this package was written -- it is not public. The
:mod:`sdmbench.upstream` manager will fetch and diff it automatically once it
becomes available; until then the ``UNVERIFIED`` tags mark exactly which
decisions need to be re-checked against the authors' code.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

__all__ = ["Provenance", "Sourced", "ProvenanceReport", "unwrap"]

T = TypeVar("T")


class Provenance(enum.Enum):
    """Where a protocol constant was verified from."""

    #: Stated explicitly in the text of Dinnage & Warren (2026).
    PAPER = "paper"
    #: Stated in the Hugging Face model card or its ``config.json``.
    MODEL_CARD = "model_card"
    #: Read directly out of the ``disdat`` R package source or documentation.
    DISDAT = "disdat"
    #: Read from the authors' source repository (available once it is public).
    UPSTREAM_CODE = "upstream_code"
    #: Taken from a cited third-party reference (e.g. Valavi et al. 2022).
    CITED_REFERENCE = "cited_reference"
    #: A default chosen by sdmbench because no public source specifies it.
    #: Any recipe containing these cannot claim to be an exact reproduction.
    UNVERIFIED = "unverified"

    @property
    def is_verified(self) -> bool:
        return self is not Provenance.UNVERIFIED


@dataclass(frozen=True)
class Sourced(Generic[T]):
    """A protocol constant together with its citation.

    Examples
    --------
    >>> buf = Sourced(10_000.0, Provenance.PAPER, "Dinnage & Warren 2026 sec. 2.5")
    >>> float(buf.value)
    10000.0
    """

    value: T
    provenance: Provenance
    citation: str
    note: str = ""

    def __post_init__(self) -> None:
        if not self.citation:
            raise ValueError("a Sourced value requires a citation")

    @property
    def is_verified(self) -> bool:
        return self.provenance.is_verified

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "provenance": self.provenance.value,
            "citation": self.citation,
            "note": self.note,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Sourced({self.value!r}, {self.provenance.value}, {self.citation!r})"


def unwrap(value: Any) -> Any:
    """Return the underlying value of a :class:`Sourced`, or the value itself.

    Lets call sites treat sourced and plain constants uniformly.
    """
    return value.value if isinstance(value, Sourced) else value


@dataclass
class ProvenanceReport:
    """Collected provenance for one benchmark recipe."""

    benchmark_id: str
    entries: dict[str, Sourced[Any]] = field(default_factory=dict)

    def add(self, key: str, sourced: Sourced[Any]) -> None:
        self.entries[key] = sourced

    @property
    def unverified(self) -> dict[str, Sourced[Any]]:
        """Constants that no public source could confirm."""
        return {k: v for k, v in self.entries.items() if not v.is_verified}

    @property
    def is_fully_verified(self) -> bool:
        return not self.unverified

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "fully_verified": self.is_fully_verified,
            "n_entries": len(self.entries),
            "n_unverified": len(self.unverified),
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }

    def render(self) -> str:
        """Render a human-readable provenance table."""
        if not self.entries:
            return f"No provenance recorded for {self.benchmark_id}."
        width = max(len(k) for k in self.entries)
        lines = [f"Provenance for benchmark {self.benchmark_id!r}", ""]
        for key in sorted(self.entries):
            src = self.entries[key]
            flag = " " if src.is_verified else "!"
            lines.append(f"{flag} {key:<{width}}  {src.provenance.value:<16}  {src.citation}")
            if src.note:
                lines.append(f"  {'':<{width}}  {'':<16}  ({src.note})")
        n_bad = len(self.unverified)
        lines.append("")
        if n_bad:
            lines.append(
                f"{n_bad} of {len(self.entries)} constants are UNVERIFIED (marked '!'). "
                "This recipe is an approximation, not an exact reproduction."
            )
        else:
            lines.append(f"All {len(self.entries)} constants verified against a public source.")
        return "\n".join(lines)
