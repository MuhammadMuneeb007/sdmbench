"""Exception hierarchy for sdmbench.

The distinction that matters scientifically is between *this cannot run here*
(a missing optional dependency, no GPU, an unaccepted licence) and *this ran
and was wrong*. The former must degrade to a ``SKIPPED_*`` row in the results
table rather than aborting a benchmark that may span thousands of jobs; the
latter must be loud.
"""

from __future__ import annotations

__all__ = [
    "SdmbenchError",
    "ConfigurationError",
    "DataError",
    "DataNotFetchedError",
    "LeakageError",
    "SkippableError",
    "MissingDependencyError",
    "NoGpuError",
    "LicenseError",
    "InsufficientDataError",
    "RBridgeError",
    "UpstreamUnavailableError",
]


class SdmbenchError(Exception):
    """Base class for every error raised by sdmbench."""


class ConfigurationError(SdmbenchError):
    """A configuration file or CLI combination is invalid."""


class DataError(SdmbenchError):
    """Benchmark data is missing, malformed, or fails its schema contract."""


class DataNotFetchedError(DataError):
    """The requested dataset has not been downloaded into the cache yet."""


class LeakageError(SdmbenchError):
    """A leakage audit found a violation that invalidates a run."""


class SkippableError(SdmbenchError):
    """Base class for conditions that skip a single job instead of failing a run.

    Subclasses map onto the ``SKIPPED_*`` values of
    :class:`sdmbench.models.base.ModelStatus`.
    """

    status = "SKIPPED"


class MissingDependencyError(SkippableError):
    """An optional third-party package required by an adapter is not installed."""

    status = "SKIPPED_DEPENDENCY"

    def __init__(self, package: str, extra: str | None = None, hint: str | None = None):
        self.package = package
        self.extra = extra
        msg = f"sdmbench needs the optional package {package!r}, which is not installed."
        if extra:
            msg += f'\n  Install it with:  pip install "sdmbench[{extra}]"'
        else:
            msg += f"\n  Install it with:  pip install {package}"
        if hint:
            msg += f"\n  {hint}"
        super().__init__(msg)


class NoGpuError(SkippableError):
    """A model that requires a GPU was asked to run without one."""

    status = "SKIPPED_NO_GPU"


class LicenseError(SkippableError):
    """A model artefact requires licence acceptance or authentication.

    sdmbench never attempts to bypass authentication or licence gates. It
    reports what the user must do and skips the job.
    """

    status = "SKIPPED_LICENSE"


class InsufficientDataError(SkippableError):
    """A species/scenario combination has too little (or single-class) data.

    In the TabPFN-SDM 2026 protocol this is the condition that removes ~41
    species from the spatial evaluation.
    """

    status = "SKIPPED_DATA"


class RBridgeError(SdmbenchError):
    """An R subprocess failed or R is not available."""


class UpstreamUnavailableError(SdmbenchError):
    """The upstream source repository for a benchmark could not be obtained."""
