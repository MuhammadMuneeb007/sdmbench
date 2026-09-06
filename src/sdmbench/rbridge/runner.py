"""The R bridge.

Several models in the TabPFN-SDM 2026 protocol have no faithful Python
equivalent. ``maxnet``, ``gbm.step()`` from ``dismo`` and ``mgcv``'s penalised
thin-plate GAMs are not merely "a MaxEnt", "a GBM" and "a GAM" -- their
defaults, regularisation and smoothing-parameter selection are part of the
published result. Strict reproduction therefore calls the authors' actual R
implementations rather than substituting scikit-learn look-alikes.

Protocol
--------
Each R script under ``rbridge/scripts/`` is invoked as::

    Rscript <script>.R <input.json> <output.json>

The Python side writes ``input.json`` (data paths + parameters), the R side
writes ``output.json`` (predictions + diagnostics). Data crosses the boundary
as Feather/CSV files rather than on stdin, so large tables stay off the pipe
and the exchange is inspectable after a failure.

This module never installs R packages. If one is missing it reports exactly
which, and the job becomes ``SKIPPED_DEPENDENCY``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from sdmbench.exceptions import MissingDependencyError, RBridgeError

__all__ = ["RRunner", "RScriptResult", "SCRIPTS_DIR", "find_rscript"]

SCRIPTS_DIR = Path(__file__).parent / "scripts"

#: Places Rscript commonly lives on Windows, where it is usually not on PATH.
_WINDOWS_R_GLOBS = (
    r"C:\Program Files\R\R-*\bin\Rscript.exe",
    r"C:\Program Files\R\R-*\bin\x64\Rscript.exe",
    r"C:\Program Files (x86)\R\R-*\bin\Rscript.exe",
)


def find_rscript(explicit: str | None = None) -> str | None:
    """Locate the ``Rscript`` executable.

    Resolution order: explicit argument, ``$SDMBENCH_RSCRIPT``, ``PATH``, then
    the standard Windows install locations (newest version first).
    """
    for candidate in (explicit, os.environ.get("SDMBENCH_RSCRIPT")):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    on_path = shutil.which("Rscript")
    if on_path:
        return on_path
    if os.name == "nt":
        import glob

        found: list[str] = []
        for pattern in _WINDOWS_R_GLOBS:
            found.extend(glob.glob(pattern))
        if found:
            return sorted(found)[-1]
    return None


@dataclass
class RScriptResult:
    """Outcome of one R script invocation."""

    ok: bool
    data: dict[str, Any]
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    workdir: str | None = None

    def require(self) -> dict[str, Any]:
        """Return the payload, raising :class:`RBridgeError` on failure."""
        if not self.ok:
            detail = self.data.get("error") if isinstance(self.data, dict) else None
            raise RBridgeError(
                f"R script failed (exit {self.returncode}): {detail or self.stderr[-2000:]}"
            )
        return self.data


class RRunner:
    """Invoke R scripts and exchange tabular data with them.

    Parameters
    ----------
    rscript:
        Path to ``Rscript``. Auto-detected when omitted.
    timeout:
        Per-invocation timeout in seconds.
    keep_workdir:
        Keep the temporary exchange directory for debugging.
    """

    def __init__(
        self,
        rscript: str | None = None,
        *,
        timeout: int = 3600,
        keep_workdir: bool = False,
        scripts_dir: Path | None = None,
    ) -> None:
        self.rscript = find_rscript(rscript)
        self.timeout = timeout
        self.keep_workdir = keep_workdir
        self.scripts_dir = Path(scripts_dir) if scripts_dir else SCRIPTS_DIR

    # ----------------------------------------------------------- availability --
    @property
    def available(self) -> bool:
        return self.rscript is not None

    def require_available(self) -> str:
        if not self.rscript:
            raise MissingDependencyError(
                "R (Rscript)",
                extra="r",
                hint=(
                    "Install R from https://cran.r-project.org/ and make sure Rscript is on "
                    "PATH, or set $SDMBENCH_RSCRIPT to its full path."
                ),
            )
        return self.rscript

    def r_version(self) -> str | None:
        if not self.rscript:
            return None
        out = subprocess.run(
            [self.rscript, "-e", "cat(paste(R.version$major, R.version$minor, sep='.'))"],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return out.stdout.strip() if out.returncode == 0 else None

    def check_packages(self, packages: Sequence[str]) -> dict[str, str | None]:
        """Report installed versions of R packages. Never installs anything.

        Returns a mapping ``package -> version`` where ``None`` means missing.
        """
        self.require_available()
        expr = (
            "pkgs <- commandArgs(trailingOnly=TRUE); "
            "res <- sapply(pkgs, function(p) "
            "tryCatch(as.character(utils::packageVersion(p)), error=function(e) NA_character_)); "
            "cat(paste(pkgs, res, sep='=', collapse='\\n'))"
        )
        # NOTE: no "--args" separator here. With `Rscript -e <expr>`, everything
        # after the expression is already passed through to the script, and a
        # literal "--args" is picked up by commandArgs(trailingOnly=TRUE) as if
        # it were a package name -- producing the baffling error
        # "missing R package(s): --args". The separator is only needed for
        # `R --no-save -e`, not for Rscript.
        out = subprocess.run(
            [self.rscript, "-e", expr, *packages],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        result: dict[str, str | None] = {p: None for p in packages}
        if out.returncode != 0:
            return result
        for line in out.stdout.strip().splitlines():
            if "=" not in line:
                continue
            name, _, version = line.partition("=")
            result[name.strip()] = None if version.strip() in {"NA", ""} else version.strip()
        return result

    def require_packages(self, packages: Sequence[str]) -> None:
        """Raise :class:`MissingDependencyError` listing every missing R package."""
        versions = self.check_packages(packages)
        missing = [p for p, v in versions.items() if v is None]
        if missing:
            raise MissingDependencyError(
                "R packages: " + ", ".join(missing),
                extra="r",
                hint=(
                    "Install them with:  Rscript scripts/setup_r_packages.R\n"
                    "  sdmbench never installs R packages automatically."
                ),
            )

    # ------------------------------------------------------------------- run --
    def run_script(
        self,
        script: str,
        payload: dict[str, Any],
        *,
        frames: dict[str, pd.DataFrame] | None = None,
        timeout: int | None = None,
    ) -> RScriptResult:
        """Run ``scripts/<script>`` with ``payload`` and optional data frames.

        ``frames`` are written as CSV into the exchange directory; their paths
        are injected into the payload under ``inputs.<name>`` so the R side can
        read them with ``read.csv``.
        """
        self.require_available()
        script_path = self.scripts_dir / script
        if not script_path.is_file():
            raise RBridgeError(f"R script not found: {script_path}")

        workdir = Path(tempfile.mkdtemp(prefix="sdmbench-r-"))
        try:
            payload = dict(payload)
            inputs: dict[str, str] = {}
            for name, frame in (frames or {}).items():
                csv_path = workdir / f"{name}.csv"
                frame.to_csv(csv_path, index=False, na_rep="NA")
                inputs[name] = str(csv_path)
            payload["inputs"] = {**payload.get("inputs", {}), **inputs}
            payload.setdefault("workdir", str(workdir))

            in_path = workdir / "input.json"
            out_path = workdir / "output.json"
            in_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

            proc = subprocess.run(
                [self.rscript, "--vanilla", str(script_path), str(in_path), str(out_path)],
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                check=False,
            )

            data: dict[str, Any] = {}
            if out_path.is_file():
                try:
                    data = json.loads(out_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    data = {"error": f"R produced unparseable JSON: {exc}"}

            ok = proc.returncode == 0 and bool(data) and not data.get("error")
            return RScriptResult(
                ok=ok,
                data=data,
                stdout=proc.stdout,
                stderr=proc.stderr,
                returncode=proc.returncode,
                workdir=str(workdir) if self.keep_workdir else None,
            )
        except subprocess.TimeoutExpired as exc:
            raise RBridgeError(f"R script {script} timed out after {timeout or self.timeout}s") from exc
        finally:
            if not self.keep_workdir:
                shutil.rmtree(workdir, ignore_errors=True)
