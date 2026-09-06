"""The upstream repository manager.

sdmbench does **not** vendor upstream source code. Copying an unknown-licence
repository into this package would be a licensing problem and a maintenance
one, and it would hide which lines are the authors' and which are ours.

Instead the upstream repository is cloned into the cache, pinned by commit SHA,
and recorded in every run manifest. That gives two independent things:

1. **Attribution and provenance** -- results state exactly which upstream
   commit they correspond to.
2. **Two implementations to cross-check** -- the authors' code and sdmbench's
   native one. Agreement between them is far stronger evidence of a correct
   reproduction than either alone.

Current status for ``tabpfn-sdm-2026``: the repository cited in the paper is not
public (HTTP 404). :meth:`UpstreamManager.check_availability` reports that
plainly, and :meth:`UpstreamManager.fetch` gives actionable instructions rather
than failing obscurely.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sdmbench.exceptions import UpstreamUnavailableError
from sdmbench.paths import ensure_dir, upstream_dir
from sdmbench.upstream.metadata import UpstreamSource, get_upstream

__all__ = ["UpstreamManager", "UpstreamCheckout"]


@dataclass
class UpstreamCheckout:
    """A cloned upstream repository, pinned to a commit."""

    benchmark_id: str
    repo_url: str
    path: Path
    commit: str = ""
    branch: str = ""
    fetched_at: str = ""
    files_present: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "repo_url": self.repo_url,
            "path": str(self.path),
            "commit": self.commit,
            "branch": self.branch,
            "fetched_at": self.fetched_at,
            "files_present": self.files_present,
        }


class UpstreamManager:
    """Clones, pins and inspects a benchmark's upstream repository."""

    def __init__(self, benchmark_id: str, *, root: str | Path | None = None) -> None:
        self.benchmark_id = benchmark_id
        source = get_upstream(benchmark_id)
        if source is None:
            raise UpstreamUnavailableError(
                f"no upstream repository is registered for benchmark {benchmark_id!r}"
            )
        self.source: UpstreamSource = source
        slug = benchmark_id.replace("-", "_")
        self.root = Path(root) if root else upstream_dir(slug)

    # ------------------------------------------------------------ availability --
    def check_availability(self, *, timeout: int = 20) -> dict[str, Any]:
        """Probe whether the upstream repository is reachable.

        A plain HTTP HEAD, so it works without git configured and without
        cloning anything.
        """
        info: dict[str, Any] = {
            "benchmark_id": self.benchmark_id,
            "repo_url": self.source.repo_url,
            "recorded_status": self.source.availability,
            "note": self.source.note,
        }
        request = urllib.request.Request(self.source.repo_url, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                info["http_status"] = int(response.status)
                info["available"] = 200 <= response.status < 400
        except urllib.error.HTTPError as exc:
            info["http_status"] = int(exc.code)
            info["available"] = False
        except Exception as exc:  # noqa: BLE001 - offline is a normal outcome
            info["http_status"] = None
            info["available"] = False
            info["error"] = str(exc)
        return info

    # ------------------------------------------------------------------ fetch --
    def fetch(
        self,
        *,
        revision: str | None = None,
        force: bool = False,
        timeout: int = 600,
    ) -> UpstreamCheckout:
        """Clone or update the upstream repository into the cache."""
        git = shutil.which("git")
        if not git:
            raise UpstreamUnavailableError(
                "git is required to fetch upstream repositories but was not found on PATH"
            )

        checkout_dir = self.root / "repo"
        if checkout_dir.exists() and force:
            shutil.rmtree(checkout_dir, ignore_errors=True)

        if not checkout_dir.exists():
            availability = self.check_availability()
            if not availability.get("available"):
                raise UpstreamUnavailableError(
                    f"cannot fetch {self.source.repo_url}\n"
                    f"  HTTP status: {availability.get('http_status')}\n"
                    f"  {self.source.note}\n"
                    "  sdmbench's native implementation of this protocol is used instead; "
                    "constants it could not verify are listed by "
                    f"`sdmbench report {self.benchmark_id}`.\n"
                    "  Re-run this command once the repository is published."
                )
            ensure_dir(self.root)
            self._git(git, ["clone", self.source.repo_url, str(checkout_dir)], timeout=timeout)
        else:
            self._git(git, ["fetch", "--all", "--tags"], cwd=checkout_dir, timeout=timeout)

        if revision:
            self._git(git, ["checkout", revision], cwd=checkout_dir, timeout=timeout)

        commit = self._git(
            git, ["rev-parse", "HEAD"], cwd=checkout_dir, timeout=60
        ).stdout.strip()
        branch = self._git(
            git, ["rev-parse", "--abbrev-ref", "HEAD"], cwd=checkout_dir, timeout=60
        ).stdout.strip()

        import datetime as _dt

        checkout = UpstreamCheckout(
            benchmark_id=self.benchmark_id,
            repo_url=self.source.repo_url,
            path=checkout_dir,
            commit=commit,
            branch=branch,
            fetched_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            files_present=self._present_files(checkout_dir),
        )
        self._write_record(checkout)
        return checkout

    @staticmethod
    def _git(git: str, args: list[str], *, cwd: Path | None = None, timeout: int = 300):
        result = subprocess.run(
            [git, *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            raise UpstreamUnavailableError(
                f"git {' '.join(args)} failed: {(result.stderr or result.stdout).strip()[:1000]}"
            )
        return result

    def _present_files(self, checkout_dir: Path) -> list[str]:
        """Which of the files we care about actually exist in the checkout."""
        return [
            name
            for name in self.source.files_of_interest
            if (checkout_dir / name).exists()
        ]

    def _write_record(self, checkout: UpstreamCheckout) -> Path:
        ensure_dir(self.root)
        path = self.root / "upstream.json"
        payload = {
            **checkout.to_dict(),
            "source": self.source.to_dict(),
            "attribution": (
                "Upstream code remains the property of its authors and is used here "
                "under its own licence. sdmbench does not redistribute it."
            ),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    # ------------------------------------------------------------------ read --
    def record(self) -> dict[str, Any] | None:
        """The stored checkout record, if the repository has been fetched."""
        path = self.root / "upstream.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def commit(self) -> str:
        """Pinned upstream commit, or an empty string when not fetched."""
        record = self.record()
        return str(record.get("commit", "")) if record else ""

    def inspect(self) -> dict[str, Any]:
        """Summarise what is present and what remains unverified."""
        record = self.record()
        if record is None:
            return {
                "fetched": False,
                "benchmark_id": self.benchmark_id,
                "repo_url": self.source.repo_url,
                "availability": self.source.availability,
                "note": self.source.note,
                "would_resolve": list(self.source.resolves),
                "files_of_interest": self.source.files_of_interest,
            }
        checkout_dir = Path(record["path"])
        present = self._present_files(checkout_dir)
        return {
            "fetched": True,
            "benchmark_id": self.benchmark_id,
            "repo_url": record["repo_url"],
            "commit": record.get("commit", ""),
            "branch": record.get("branch", ""),
            "fetched_at": record.get("fetched_at", ""),
            "files_present": present,
            "files_missing": [
                name for name in self.source.files_of_interest if name not in present
            ],
            "would_resolve": list(self.source.resolves),
        }

    def manifest_entry(self) -> dict[str, Any]:
        """Compact record for ``run_manifest.json``."""
        record = self.record()
        if record is None:
            return {
                "repo_url": self.source.repo_url,
                "commit": "",
                "status": self.source.availability,
                "note": self.source.note,
            }
        return {
            "repo_url": record["repo_url"],
            "commit": record.get("commit", ""),
            "fetched_at": record.get("fetched_at", ""),
            "status": "fetched",
        }
