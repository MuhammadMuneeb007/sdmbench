#!/usr/bin/env python3
"""Prepare an sdmbench environment.

    python scripts/setup_environment.py --plan          # print commands only
    python scripts/setup_environment.py --gpu --plan
    python scripts/setup_environment.py --run           # actually execute

Nothing is installed or modified unless ``--run`` is given explicitly. The
default is ``--plan``: it detects your tooling, works out the right commands,
and prints them for you to inspect and run yourself.

That default is deliberate. Creating or mutating a conda environment is not a
reversible action, and a research tool should not do it as a side effect of
being asked what it would do.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Preference order. micromamba and mamba resolve far faster than conda on the
#: geospatial stack, which is the slowest part of this environment.
CONDA_TOOLS = ("micromamba", "mamba", "conda")


def find_conda_tool() -> tuple[str, str] | None:
    """Return ``(tool_name, path)`` for the fastest available conda tool."""
    for tool in CONDA_TOOLS:
        path = shutil.which(tool)
        if path:
            return tool, path
    return None


def detect_cuda() -> str | None:
    """Report the CUDA version from ``nvidia-smi``, if there is a GPU."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run(
            [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - probing is best-effort
        return None
    return None


def find_r() -> str | None:
    rscript = os.environ.get("SDMBENCH_RSCRIPT") or shutil.which("Rscript")
    if rscript:
        return rscript
    if os.name == "nt":
        import glob

        found = glob.glob(r"C:\Program Files\R\R-*\bin\Rscript.exe")
        if found:
            return sorted(found)[-1]
    return None


def build_plan(*, gpu: bool, extras: list[str], name: str | None) -> list[tuple[str, str]]:
    """Build the ``(description, command)`` list for this machine."""
    tool = find_conda_tool()
    env_file = "environment-gpu.yml" if gpu else "environment.yml"
    env_name = name or ("sdmbench-gpu" if gpu else "sdmbench")
    steps: list[tuple[str, str]] = []

    if tool:
        tool_name, _ = tool
        steps.append(
            (
                f"Create the conda environment with {tool_name}",
                f"{tool_name} env create -f {env_file}",
            )
        )
        steps.append(
            ("Activate it", f"conda activate {env_name}"
             if tool_name != "micromamba" else f"micromamba activate {env_name}")
        )
    else:
        steps.append(
            (
                "No conda tool found -- create a virtual environment instead",
                f"{sys.executable} -m venv .venv",
            )
        )
        activate = ".venv\\Scripts\\activate" if os.name == "nt" else "source .venv/bin/activate"
        steps.append(("Activate it", activate))
        steps.append(
            (
                "NOTE: the geospatial extras (rasterio/geopandas) build from source "
                "under pip and often fail; conda-forge is strongly preferred for those",
                "",
            )
        )

    extra_spec = f"[{','.join(extras)}]" if extras else ""
    steps.append(
        ("Install sdmbench in editable mode", f'pip install -e ".{extra_spec}"' if extra_spec
         else "pip install -e .")
    )

    if gpu:
        steps.append(
            (
                "Install torch-geometric's compiled companions, matched to your torch build",
                "pip install pyg-lib torch-scatter torch-sparse "
                "-f https://data.pyg.org/whl/torch-$(python -c "
                "'import torch;print(torch.__version__.split(\"+\")[0])')+cu121.html",
            )
        )

    rscript = find_r()
    if rscript:
        steps.append(
            ("Install the R packages for the published baselines",
             f'"{rscript}" scripts/setup_r_packages.R')
        )
    else:
        steps.append(
            (
                "R was not found. The published MaxNet/BRT/GAM/RF baselines will be "
                "SKIPPED_DEPENDENCY without it. Install R from https://cran.r-project.org/, "
                "then run",
                "Rscript scripts/setup_r_packages.R",
            )
        )

    steps.append(("Verify the installation", "sdmbench env check"))
    steps.append(("Prepare the benchmark data", "sdmbench data fetch disdat"))
    return steps


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan (or perform) an sdmbench environment setup.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gpu", action="store_true", help="use the CUDA environment file")
    parser.add_argument("--name", help="environment name override")
    parser.add_argument(
        "--extras",
        default="boosting,raster,plots",
        help="comma-separated pip extras (default: boosting,raster,plots)",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="EXECUTE the plan. Without this flag nothing is installed or modified.",
    )
    parser.add_argument("--plan", action="store_true", help="print the plan (default)")
    args = parser.parse_args()

    extras = [e.strip() for e in args.extras.split(",") if e.strip()]

    print("sdmbench environment setup")
    print("=" * 26)
    print(f"  platform      {platform.platform()}")
    print(f"  python        {sys.version.split()[0]} ({sys.executable})")
    tool = find_conda_tool()
    print(f"  conda tool    {tool[0] + ' -> ' + tool[1] if tool else 'not found'}")
    cuda = detect_cuda()
    print(f"  nvidia driver {cuda or 'no GPU detected'}")
    rscript = find_r()
    print(f"  Rscript       {rscript or 'not found'}")
    print(f"  repo root     {REPO_ROOT}")
    print()

    if args.gpu and not cuda:
        print("WARNING: --gpu was requested but no NVIDIA GPU was detected.")
        print("         The CUDA environment will install a torch that falls back to CPU.")
        print()

    steps = build_plan(gpu=args.gpu, extras=extras, name=args.name)

    print("PLAN")
    print("-" * 4)
    for i, (description, command) in enumerate(steps, start=1):
        print(f"{i}. {description}")
        if command:
            print(f"     $ {command}")
    print()

    if not args.run:
        print("Nothing was installed or modified (this is the default).")
        print("Review the commands above and run them yourself, or re-run with --run.")
        return 0

    print("--run given: executing the plan.")
    print("NOTE: 'conda activate' cannot be executed from inside this process; steps that")
    print("      depend on an activated environment are printed for you to run manually.")
    print()
    for description, command in steps:
        if not command or command.startswith(("conda activate", "micromamba activate", "source ")):
            print(f"SKIP (manual): {description}")
            continue
        print(f"$ {command}")
        result = subprocess.run(command, shell=True, check=False)  # noqa: S602
        if result.returncode != 0:
            print(f"\nStep failed with exit code {result.returncode}: {description}")
            print("Stopping. Fix the problem and re-run.")
            return result.returncode
    print("\nDone. Verify with:  sdmbench env check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
