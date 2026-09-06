#!/usr/bin/env python3
"""Install every sdmbench dependency, group by group, and report what landed.

    python scripts/install_all.py                  # everything except AutoML
    python scripts/install_all.py --with-automl    # + autogluon and h2o
    python scripts/install_all.py --dry-run        # print the plan only
    python scripts/install_all.py --check          # report state, install nothing

Why this exists rather than just `pip install -e ".[all]"`
----------------------------------------------------------
A single fat pip command either succeeds or fails as a unit. When it fails --
which on a cluster it eventually does, usually because one package pins a
version range that conflicts with the rest -- you get a resolver error and no
information about which of the twenty things you wanted actually installed.

This installs group by group, keeps going when a group fails, and finishes with
a table of what is present and what is not. A failure in the AutoML group does
not stop the boosting group from installing.

It never touches R. Run `Rscript scripts/setup_r_packages.R` for that.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: (group, [pip specs], [importable module names], why it matters)
GROUPS: list[tuple[str, list[str], list[str], str]] = [
    (
        "core",
        ["numpy>=1.24", "pandas>=2.0", "scipy>=1.10", "scikit-learn>=1.3",
         "pyarrow>=12.0", "pyyaml>=6.0"],
        ["numpy", "pandas", "scipy", "sklearn", "pyarrow", "yaml"],
        "schema, splits, metrics, aggregation, reporting -- required",
    ),
    (
        "boosting",
        ["xgboost>=2.0", "lightgbm>=4.0", "catboost>=1.2"],
        ["xgboost", "lightgbm", "catboost"],
        "XGBoost / LightGBM / CatBoost adapters",
    ),
    (
        "deep",
        ["torch>=2.0", "torchvision>=0.15"],
        ["torch", "torchvision"],
        "MLP, FT-Transformer, autoencoders, neural fusion, CNN patch encoders",
    ),
    (
        "graph",
        ["torch-geometric>=2.4"],
        ["torch_geometric"],
        "GCN, GraphSAGE, GATv2, Graph Transformer (needs torch first)",
    ),
    (
        "tabpfn",
        ["tabpfn>=2.0", "huggingface-hub>=0.20"],
        ["tabpfn", "huggingface_hub"],
        "TabPFN + the finetuned TabPFN-SDM checkpoints (Prior Labs License v1.1)",
    ),
    (
        "raster",
        ["rasterio>=1.3", "rioxarray>=0.15", "xarray>=2023.1", "geopandas>=0.14",
         "shapely>=2.0", "pyproj>=3.5"],
        ["rasterio", "rioxarray", "xarray", "geopandas", "shapely", "pyproj"],
        "raster extraction, background sampling, suitability maps",
    ),
    (
        "explain",
        ["shap>=0.44", "matplotlib>=3.7"],
        ["shap", "matplotlib"],
        "SHAP attribution and plots",
    ),
    (
        "flaml",
        ["flaml>=2.1"],
        ["flaml"],
        "lightweight AutoML (fast, no JVM)",
    ),
]

#: Installed only with --with-automl. Both are disruptive in their own way.
AUTOML_GROUPS: list[tuple[str, list[str], list[str], str]] = [
    (
        "autogluon",
        ["autogluon.tabular>=1.0"],
        ["autogluon.tabular"],
        "AutoGluon -- ~200 packages; may conflict with a modern sklearn/numpy",
    ),
    (
        "h2o",
        ["h2o>=3.44"],
        ["h2o"],
        "H2O AutoML -- installs via pip but needs a Java runtime to RUN",
    ),
]


def installed(module: str) -> str | None:
    """Return a version string if the module imports, else None."""
    import importlib
    import importlib.metadata as md

    try:
        importlib.import_module(module)
    except Exception:  # noqa: BLE001 - a broken install counts as absent
        return None
    for candidate in (module, module.replace("_", "-"), module.split(".")[0]):
        try:
            return md.version(candidate)
        except Exception:  # noqa: BLE001
            continue
    return "installed"


def pip_install(specs: list[str], *, upgrade: bool = False) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(specs)
    print(f"    $ {' '.join(cmd[3:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return True, ""
    tail = (result.stderr or result.stdout).strip().splitlines()
    return False, "\n".join(tail[-6:])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install sdmbench dependencies group by group.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--with-automl", action="store_true",
                        help="also install autogluon and h2o (slow, may conflict)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument("--check", action="store_true",
                        help="report what is installed; install nothing")
    parser.add_argument("--upgrade", action="store_true", help="pass --upgrade to pip")
    parser.add_argument("--editable", action="store_true",
                        help="also install sdmbench itself in editable mode")
    args = parser.parse_args()

    groups = list(GROUPS) + (AUTOML_GROUPS if args.with_automl else [])

    print("sdmbench dependency installer")
    print("=" * 29)
    print(f"  python  {sys.version.split()[0]}")
    print(f"  pip     {sys.executable} -m pip")
    print(f"  repo    {REPO_ROOT}")
    print()

    if args.check or args.dry_run:
        print(f"{'group':<12} {'status':<14} packages")
        print("-" * 78)
        for name, specs, modules, why in groups:
            missing = [m for m in modules if installed(m) is None]
            status = "OK" if not missing else f"MISSING {len(missing)}"
            print(f"{name:<12} {status:<14} {', '.join(modules)}")
            if missing:
                print(f"{'':<12} {'':<14} -> {why}")
        if not args.with_automl:
            print()
            print("AutoML (autogluon, h2o) not included. Add --with-automl.")
        print()
        print("check only -- nothing was installed." if args.check
              else "dry run -- nothing was installed.")
        return 0

    failures: list[tuple[str, str]] = []
    for name, specs, modules, why in groups:
        missing = [m for m in modules if installed(m) is None]
        if not missing and not args.upgrade:
            print(f"[skip] {name}: already installed")
            continue
        print(f"[install] {name}  ({why})")
        ok, detail = pip_install(specs, upgrade=args.upgrade)
        if ok:
            print(f"    ok")
        else:
            print(f"    FAILED")
            failures.append((name, detail))
            # Keep going: one failed group must not block the others.

    if args.editable:
        print("[install] sdmbench (editable)")
        ok, detail = pip_install(["-e", str(REPO_ROOT)])
        if not ok:
            failures.append(("sdmbench", detail))

    print()
    print("FINAL STATE")
    print(f"{'group':<12} {'status':<14} detail")
    print("-" * 78)
    for name, specs, modules, why in groups:
        versions = {m: installed(m) for m in modules}
        missing = [m for m, v in versions.items() if v is None]
        status = "OK" if not missing else "INCOMPLETE"
        detail = ", ".join(f"{m}={v}" for m, v in versions.items() if v) or "none"
        print(f"{name:<12} {status:<14} {detail}")
        if missing:
            print(f"{'':<12} {'':<14} missing: {', '.join(missing)}")

    if failures:
        print()
        print("FAILURES")
        for name, detail in failures:
            print(f"  {name}:")
            for line in detail.splitlines():
                print(f"    {line}")
        print()
        print("A failed group means those models report SKIPPED_DEPENDENCY.")
        print("The rest of the benchmark still runs.")

    print()
    print("R packages are installed separately (sdmbench never installs them):")
    print("    Rscript scripts/setup_r_packages.R")
    print()
    print("Verify with:  sdmbench env check")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
