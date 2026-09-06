#!/usr/bin/env python3
"""Report what sdmbench can and cannot run on this machine.

    python scripts/check_environment.py
    python scripts/check_environment.py --json

Read-only: imports things and asks R for version numbers. It installs nothing
and changes nothing.

The output is organised by *what you lose* rather than by package, because
"lightgbm is missing" is less useful than "LightGBM will be SKIPPED_DEPENDENCY
in every run".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def gather() -> dict:
    from sdmbench import __version__
    from sdmbench.optional import package_version
    from sdmbench.rbridge.runner import RRunner
    from sdmbench.reproducibility.environment import capture_environment

    report: dict = {"sdmbench_version": __version__, "environment": capture_environment(
        include_r=False
    )}

    capabilities = {
        "core (sklearn models, metrics, splits, reporting)": [
            ("numpy", None), ("pandas", None), ("scipy", None),
            ("scikit-learn", None), ("pyarrow", None), ("pyyaml", None),
        ],
        "gradient boosting (xgboost, lightgbm, catboost)": [
            ("xgboost", "boosting"), ("lightgbm", "boosting"), ("catboost", "boosting"),
        ],
        "deep tabular + learned representations": [
            ("torch", "deep"), ("torchvision", "deep"),
        ],
        "graph neural networks": [
            ("torch", "graph"), ("torch-geometric", "graph"),
        ],
        "TabPFN and the finetuned SDM checkpoints": [
            ("tabpfn", "tabpfn"), ("huggingface-hub", "tabpfn"), ("torch", "tabpfn"),
        ],
        "raster / spatial workflows": [
            ("rasterio", "raster"), ("rioxarray", "raster"), ("geopandas", "raster"),
            ("shapely", "raster"), ("pyproj", "raster"),
        ],
        "AutoML": [
            ("autogluon.tabular", "automl"), ("flaml", "automl"), ("h2o", "automl"),
        ],
        "geospatial foundation embeddings": [
            ("ee", "foundation"), ("transformers", "foundation"),
        ],
        "explainability": [
            ("shap", "explain"), ("captum", "explain"),
        ],
        "plots": [("matplotlib", "plots")],
    }

    report["capabilities"] = {}
    for capability, packages in capabilities.items():
        entries = []
        for name, extra in packages:
            version = package_version(name)
            entries.append(
                {"package": name, "version": version, "extra": extra,
                 "installed": version != "not-installed"}
            )
        report["capabilities"][capability] = {
            "available": all(e["installed"] for e in entries),
            "packages": entries,
            "install": (
                f'pip install "sdmbench[{entries[0]["extra"]}]"'
                if entries and entries[0]["extra"]
                else None
            ),
        }

    runner = RRunner()
    r_packages = [
        "jsonlite", "disdat", "maxnet", "dismo", "gbm", "randomForest",
        "mgcv", "sf", "terra", "yardstick", "tidysdm",
    ]
    report["r"] = {
        "rscript": runner.rscript,
        "available": runner.available,
        "version": runner.r_version() if runner.available else None,
        "packages": runner.check_packages(r_packages) if runner.available else {},
    }

    from sdmbench.data.disdat import DisdatDataset

    dataset = DisdatDataset()
    report["data"] = {"disdat": dataset.summary()}
    return report


def render(report: dict) -> str:
    lines = [
        f"sdmbench {report['sdmbench_version']} environment check",
        "=" * 44,
        "",
    ]
    env = report["environment"]
    lines.append(f"  python  {env['python_version']}  ({env['platform']})")
    gpu = env["gpu"]
    lines.append(
        f"  gpu     {'yes: ' + ', '.join(gpu.get('devices', [])) if gpu['cuda_available'] else 'none detected'}"
    )
    if gpu.get("cuda_version"):
        lines.append(f"  cuda    {gpu['cuda_version']}")
    lines.append("")

    lines.append("CAPABILITIES")
    for capability, info in report["capabilities"].items():
        mark = "OK  " if info["available"] else "MISS"
        lines.append(f"  [{mark}] {capability}")
        if not info["available"]:
            missing = [p["package"] for p in info["packages"] if not p["installed"]]
            lines.append(f"         missing: {', '.join(missing)}")
            if info["install"]:
                lines.append(f"         {info['install']}")
    lines.append("")

    r = report["r"]
    lines.append("R (published MaxNet / BRT / GAM / Random Forest baselines)")
    if not r["available"]:
        lines.append("  [MISS] Rscript not found.")
        lines.append("         Those four baselines will be SKIPPED_DEPENDENCY in every run,")
        lines.append("         so a TabPFN-SDM reproduction would have nothing to compare against.")
        lines.append("         Install R from https://cran.r-project.org/, then:")
        lines.append("             Rscript scripts/setup_r_packages.R")
    else:
        lines.append(f"  [OK  ] {r['rscript']}  (R {r['version']})")
        missing = [name for name, version in r["packages"].items() if not version]
        for name, version in r["packages"].items():
            lines.append(f"         {name:<14} {version or 'MISSING'}")
        if missing:
            lines.append("         Install the missing ones with:")
            lines.append("             Rscript scripts/setup_r_packages.R")
    lines.append("")

    disdat = report["data"]["disdat"]
    lines.append("BENCHMARK DATA")
    if disdat.get("available"):
        lines.append(
            f"  [OK  ] disdat {disdat.get('disdat_version', '')} -- "
            f"{disdat.get('total_species', '?')} species in {disdat['root']}"
        )
    else:
        lines.append(f"  [MISS] disdat is not prepared in {disdat['root']}")
        lines.append("         sdmbench data fetch disdat")
    lines.append("")

    ready = all(
        info["available"]
        for name, info in report["capabilities"].items()
        if name.startswith("core")
    )
    lines.append(
        "Core benchmark is runnable." if ready
        else "Core dependencies are incomplete; install sdmbench first."
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report sdmbench's available capabilities.")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    report = gather()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
