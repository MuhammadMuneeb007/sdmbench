"""The ``sdmbench`` command-line interface.

Built on ``argparse`` so the core package needs no CLI dependency.

Command groups::

    sdmbench data fetch disdat          prepare benchmark data
    sdmbench data info disdat           what is in the cache
    sdmbench inspect data.csv           suggest a schema for a user table
    sdmbench benchmark --benchmark ...  run a benchmark
    sdmbench benchmark-csv species.csv  run on a user CSV
    sdmbench reproduce tabpfn-sdm-2026  strict reproduction
    sdmbench report tabpfn-sdm-2026     published vs reproduced
    sdmbench leaderboard                ranked results
    sdmbench compare A B                paired comparison
    sdmbench upstream fetch <id>        clone and pin upstream code
    sdmbench models / benchmarks        registries
    sdmbench env check                  environment diagnosis

Every command that could run for hours prints its plan first and is resumable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from sdmbench import __version__

__all__ = ["main", "build_parser"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _load_results(args: argparse.Namespace):
    """Locate a results table from --run-dir, --results, or the latest run."""
    import pandas as pd

    from sdmbench.evaluation.store import ResultStore
    from sdmbench.paths import runs_dir

    if getattr(args, "results", None):
        return ResultStore.read_parquet(args.results), Path(args.results).parent
    if getattr(args, "run_dir", None):
        store = ResultStore(args.run_dir)
        return store.to_dataframe(), Path(args.run_dir)
    store = ResultStore.find_latest(runs_dir(), getattr(args, "benchmark", None))
    if store is None:
        return pd.DataFrame(), None
    return store.to_dataframe(), store.run_dir


def _build_run_config(args: argparse.Namespace):
    """Build a RunConfig from --config plus CLI overrides."""
    from sdmbench.config import RunConfig, load_yaml

    data: dict[str, Any] = {}
    if getattr(args, "config", None):
        data = load_yaml(args.config)
    if getattr(args, "benchmark", None):
        data["benchmark"] = {"name": args.benchmark}
    if getattr(args, "models", None):
        data["models"] = args.models
    if getattr(args, "scenarios", None):
        data.setdefault("evaluation", {})["scenarios"] = args.scenarios
    if getattr(args, "regions", None):
        data.setdefault("evaluation", {})["regions"] = args.regions
    if getattr(args, "max_species", None) is not None:
        data.setdefault("evaluation", {})["max_species"] = args.max_species
    if getattr(args, "species", None):
        data.setdefault("evaluation", {})["species"] = args.species
    if getattr(args, "device", None):
        data.setdefault("compute", {})["device"] = args.device
    if getattr(args, "workers", None) is not None:
        data.setdefault("compute", {})["workers"] = args.workers
    if getattr(args, "seed", None) is not None:
        data["seed"] = args.seed
    if getattr(args, "force", False):
        data["force"] = True
    if getattr(args, "strict", False):
        data["strict"] = True
    if getattr(args, "no_audit", False):
        data["audit_leakage"] = False
    if getattr(args, "metrics", None):
        data["metrics"] = args.metrics
    return RunConfig.from_dict(data)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def cmd_data_fetch(args: argparse.Namespace) -> int:
    from sdmbench.data.disdat import DisdatDataset, fetch_from_osf
    from sdmbench.paths import datasets_dir

    name = args.dataset.lower()
    if name != "disdat":
        print(f"unknown dataset {args.dataset!r}; currently only 'disdat' is available")
        return 2

    if args.source == "osf":
        target = datasets_dir("disdat") / "raw"
        try:
            path = fetch_from_osf(target, url=args.url)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            print(f"error: {exc}")
            return 1
        print(f"downloaded {path}")
        print(
            "NOTE: the OSF archive is not byte-identical to the R package data, so results "
            "prepared this way carry a different dataset_hash."
        )
        return 0

    print("Preparing disdat via the R package (this reads the installed package; it does")
    print("not install anything). This can take a couple of minutes.")
    try:
        dataset = DisdatDataset.fetch(
            regions=args.regions, rscript=args.rscript, force=args.force
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        return 1
    summary = dataset.summary()
    print(f"\nPrepared {summary.get('total_species', '?')} species in {dataset.root}")
    _print_json(summary)
    return 0


def cmd_data_info(args: argparse.Namespace) -> int:
    from sdmbench.data.disdat import DisdatDataset

    dataset = DisdatDataset()
    summary = dataset.summary()
    if not summary["available"]:
        print(f"disdat is not prepared in {dataset.root}")
        print("Run:  sdmbench data fetch disdat")
        return 1
    _print_json(summary)
    return 0


def cmd_data_list(args: argparse.Namespace) -> int:
    from sdmbench.data.disdat import DisdatDataset

    dataset = DisdatDataset()
    print(f"{'dataset':<12} {'status':<14} description")
    print("-" * 74)
    status = "ready" if dataset.is_available() else "not fetched"
    print(f"{'disdat':<12} {status:<14} 226 species, 6 regions (Elith et al. 2020)")
    print(f"{'csv':<12} {'n/a':<14} any user-supplied table (see sdmbench inspect)")
    print(f"{'raster':<12} {'n/a':<14} occurrences + raster stack (sdmbench[raster])")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Suggest a schema for a user table without committing to it."""
    import pandas as pd

    from sdmbench.data.inspector import DatasetInspector

    frame = pd.read_csv(args.path)
    report = DatasetInspector(frame).inspect()
    if args.json:
        _print_json(report.to_dict())
    else:
        print(report.render())
    return 0


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------


def cmd_benchmark(args: argparse.Namespace) -> int:
    from sdmbench.evaluation.runner import BenchmarkRunner

    config = _build_run_config(args)
    runner = BenchmarkRunner(config.benchmark, config, run_dir=args.output)

    print(f"benchmark : {config.benchmark}")
    print(f"models    : {', '.join(runner.model_names())}")
    print(f"scenarios : {', '.join(config.evaluation.scenarios)}")
    print(f"seed      : {config.seed}")
    print(f"run dir   : {runner.run_dir}")
    print()
    if args.dry_run:
        print("(dry run: nothing executed)")
        return 0

    results = runner.run(progress=not args.quiet)
    print()
    print(f"finished: {len(results)} result row(s)")
    _print_json(results.status_counts)
    print(f"\nresults: {runner.run_dir / 'results.parquet'}")
    print(f"manifest: {runner.run_dir / 'run_manifest.json'}")
    print(f"\nNext:  sdmbench leaderboard --run-dir {runner.run_dir}")
    return 0


def cmd_benchmark_csv(args: argparse.Namespace) -> int:
    from sdmbench.evaluation.runner import Benchmark

    coordinates = (
        (args.longitude, args.latitude) if args.longitude and args.latitude else None
    )
    benchmark = Benchmark.from_csv(
        args.path,
        target=args.target,
        coordinates=coordinates,
        include_coordinates=args.include_coordinates,
        test_csv=args.test_csv,
        seed=args.seed or 32639,
    )
    models = args.models or ["standard"]
    scenarios = args.scenarios or (["nonspatial"] if not coordinates else ["nonspatial", "spatial"])

    print(f"table   : {args.path}")
    print(f"target  : {args.target}")
    print(f"models  : {', '.join(models)}")
    if not args.include_coordinates and coordinates:
        print(
            "note    : coordinates are used for splitting only, not as predictors "
            "(pass --include-coordinates to change this)"
        )
    print()
    results = benchmark.run(models=models, scenarios=scenarios, run_dir=args.output)

    from sdmbench.reporting.leaderboard import render_leaderboard

    print(render_leaderboard(results.to_dataframe(), metric=args.metric))
    return 0


def cmd_reproduce(args: argparse.Namespace) -> int:
    """Strict reproduction: the paper's own models, under the paper's protocol."""
    from sdmbench.benchmarks import get_benchmark
    from sdmbench.evaluation.runner import BenchmarkRunner

    args.benchmark = args.benchmark_id
    config = _build_run_config(args)
    if not config.models:
        config.models = list(get_benchmark(config.benchmark, config=config).default_models)
    if not args.scenarios:
        config.evaluation.scenarios = list(
            get_benchmark(config.benchmark, config=config).scenarios
        )

    runner = BenchmarkRunner(config.benchmark, config, run_dir=args.output)
    benchmark = runner.benchmark
    provenance = benchmark.provenance()

    print(f"Reproducing: {benchmark.paper.citation()}")
    print(f"models    : {', '.join(runner.model_names())}")
    print(f"scenarios : {', '.join(config.evaluation.scenarios)}")
    print(f"run dir   : {runner.run_dir}")
    print()
    if not provenance.is_fully_verified:
        print(provenance.render())
        print()
    if args.dry_run:
        print("(dry run: nothing executed)")
        return 0

    results = runner.run(progress=not args.quiet)
    print()
    _print_json(results.status_counts)
    print(f"\nNext:  sdmbench report {args.benchmark_id} --run-dir {runner.run_dir}")
    return 0


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    from sdmbench.benchmarks import get_benchmark
    from sdmbench.reporting.reproduction import build_reproduction_report
    from sdmbench.upstream.manager import UpstreamManager

    args.benchmark = args.benchmark_id
    benchmark = get_benchmark(args.benchmark_id)
    results, run_dir = _load_results(args)

    dataset_summary: dict[str, Any] = {}
    try:
        dataset_summary = benchmark.dataset.summary()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - a missing dataset is reported, not fatal
        pass

    upstream: dict[str, Any] = {}
    try:
        upstream = UpstreamManager(args.benchmark_id).manifest_entry()
    except Exception:  # noqa: BLE001
        pass

    report = build_reproduction_report(
        benchmark,
        results if not results.empty else None,
        run_id=str(run_dir) if run_dir else "",
        dataset_summary=dataset_summary,
        upstream=upstream,
        match_tolerance=args.match_tolerance,
        close_tolerance=args.close_tolerance,
    )
    if args.json:
        _print_json(report.to_dict())
    else:
        print(report.render())

    if not results.empty and args.show_extra:
        from sdmbench.reporting.reproduction import extra_models_table
        from sdmbench.reporting.leaderboard import render_dataframe

        extra = extra_models_table(benchmark, results, scenario=args.scenario or "nonspatial")
        if not extra.empty:
            print()
            print("EXTENDED BENCHMARK -- models not reported in the paper,")
            print("evaluated under identical rules:")
            print()
            print(
                render_dataframe(
                    extra,
                    [
                        ("rank", "#"),
                        ("model_name", "Model"),
                        ("roc_auc_mean", "ROC-AUC"),
                        ("pr_auc_mean", "PR-AUC"),
                        ("calibration_slope_mean", "Calib"),
                        ("n_species", "Species"),
                    ],
                )
            )

    if args.output:
        Path(args.output).write_text(
            json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        print(f"\nwrote {args.output}")
    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    from sdmbench.evaluation.aggregate import common_species_leaderboard
    from sdmbench.reporting.leaderboard import render_leaderboard, render_dataframe

    results, run_dir = _load_results(args)
    if results.empty:
        print("no results found. Run a benchmark first, or pass --run-dir / --results.")
        return 1
    if run_dir:
        print(f"run: {run_dir}\n")

    if args.common_species:
        board = common_species_leaderboard(
            results, metric=args.metric, scenario=args.scenario
        )
        print("Leaderboard -- species completed by EVERY model (like-for-like)")
        print()
        print(
            render_dataframe(
                board,
                [
                    ("rank", "#"),
                    ("model_name", "Model"),
                    (f"{args.metric}_mean", args.metric.upper()),
                    (f"{args.metric}_sd", "SD"),
                    (f"{args.metric}_ci_low", "CI low"),
                    (f"{args.metric}_ci_high", "CI high"),
                    ("n_species", "Species"),
                ],
            )
        )
    else:
        print(render_leaderboard(results, metric=args.metric, scenario=args.scenario))

    if args.output:
        from sdmbench.evaluation.aggregate import build_leaderboard

        build_leaderboard(results, metric=args.metric, scenario=args.scenario).to_csv(
            args.output, index=False
        )
        print(f"\nwrote {args.output}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from sdmbench.evaluation.compare import compare_all_pairs, compare_models

    results, _ = _load_results(args)
    if results.empty:
        print("no results found.")
        return 1

    if args.all_pairs or args.reference:
        frame = compare_all_pairs(
            results,
            metric=args.metric,
            scenario=args.scenario,
            reference=args.reference,
            correction=args.correction,
        )
        if frame.empty:
            print("no comparable model pairs.")
            return 1
        from sdmbench.reporting.leaderboard import render_dataframe

        print(
            render_dataframe(
                frame,
                [
                    ("model_a", "Model A"),
                    ("model_b", "Model B"),
                    ("mean_difference", "Diff"),
                    ("ci_low", "CI low"),
                    ("ci_high", "CI high"),
                    ("wins", "W"),
                    ("ties", "T"),
                    ("losses", "L"),
                    ("wilcoxon_p_adjusted", "p(adj)"),
                    ("verdict", "Verdict"),
                ],
            )
        )
        print(f"\nmultiple-comparison correction: {args.correction}")
        return 0

    if not args.model_b:
        print("provide two models, or --all-pairs / --reference MODEL")
        return 2
    comparison = compare_models(
        results, args.model_a, args.model_b, metric=args.metric, scenario=args.scenario
    )
    print(comparison.render())
    return 0


# ---------------------------------------------------------------------------
# registries and environment
# ---------------------------------------------------------------------------


def cmd_models(args: argparse.Namespace) -> int:
    from sdmbench.models.base import MODEL_SETS, model_info

    info = model_info()
    seen: set[str] = set()
    print(f"{'model':<26} {'family':<12} {'gpu':<5} class")
    print("-" * 88)
    for name, meta in info.items():
        if meta["class"] in seen and not args.all:
            continue
        seen.add(meta["class"])
        gpu = "yes" if meta["requires_gpu"] else "-"
        print(f"{name:<26} {meta['family']:<12} {gpu:<5} {meta['class']}")
    print()
    print("Model set aliases (usable anywhere a model list is accepted):")
    for key, members in sorted(MODEL_SETS.items()):
        print(f"  {key:<12} {', '.join(members)}")
    return 0


def cmd_benchmarks(args: argparse.Namespace) -> int:
    from sdmbench.benchmarks import benchmark_info

    for benchmark_id, meta in benchmark_info().items():
        print(f"{benchmark_id}  (v{meta['version']})")
        print(f"  {meta['paper']}")
        print(f"  {', '.join(meta['authors'])} ({meta['year']})"
              + (f"  doi:{meta['doi']}" if meta["doi"] else ""))
        print(f"  scenarios: {', '.join(meta['scenarios'])}")
        print(f"  metrics:   {', '.join(meta['metrics'])}")
        print(f"  models:    {', '.join(meta['default_models'])}")
        print()
    return 0


def cmd_provenance(args: argparse.Namespace) -> int:
    from sdmbench.benchmarks import get_benchmark

    benchmark = get_benchmark(args.benchmark_id)
    report = benchmark.provenance()
    if args.json:
        _print_json(report.to_dict())
    else:
        print(report.render())
    return 0 if report.is_fully_verified else 3


def cmd_upstream(args: argparse.Namespace) -> int:
    from sdmbench.upstream.manager import UpstreamManager

    manager = UpstreamManager(args.benchmark_id)
    if args.upstream_action == "check":
        _print_json(manager.check_availability())
        return 0
    if args.upstream_action == "info":
        _print_json(manager.inspect())
        return 0
    try:
        checkout = manager.fetch(revision=args.revision, force=args.force)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}")
        return 1
    print(f"fetched {checkout.repo_url}")
    print(f"  commit: {checkout.commit}")
    print(f"  path:   {checkout.path}")
    print(f"  files of interest present: {checkout.files_present}")
    return 0


def cmd_env_check(args: argparse.Namespace) -> int:
    from sdmbench.optional import have, package_version
    from sdmbench.rbridge.runner import RRunner
    from sdmbench.reproducibility.environment import capture_environment

    env = capture_environment()
    print(f"sdmbench {__version__}")
    print(f"python   {env['python_version']}  ({env['platform']})")
    print()
    print("Core dependencies:")
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "pyyaml"):
        print(f"  {name:<16} {package_version(name)}")
    print()
    print("Optional dependencies:")
    for name, extra in (
        ("xgboost", "boosting"), ("lightgbm", "boosting"), ("catboost", "boosting"),
        ("torch", "deep"), ("torch-geometric", "graph"), ("tabpfn", "tabpfn"),
        ("huggingface-hub", "tabpfn"), ("rasterio", "raster"), ("geopandas", "raster"),
        ("autogluon.tabular", "automl"), ("h2o", "automl"), ("flaml", "automl"),
        ("matplotlib", "plots"),
    ):
        version = package_version(name)
        mark = " " if version != "not-installed" else "!"
        print(f"  {mark} {name:<20} {version:<14} (sdmbench[{extra}])")
    print()
    gpu = env["gpu"]
    print(f"GPU: cuda_available={gpu['cuda_available']} devices={gpu.get('devices', [])}")
    print()
    runner = RRunner()
    if runner.available:
        print(f"R: {runner.rscript}  (version {runner.r_version()})")
        packages = runner.check_packages(
            ["disdat", "jsonlite", "maxnet", "dismo", "gbm", "randomForest", "mgcv",
             "sf", "yardstick", "tidysdm"]
        )
        for name, version in packages.items():
            mark = " " if version else "!"
            print(f"  {mark} {name:<16} {version or 'not installed'}")
        missing = [n for n, v in packages.items() if not v]
        if missing:
            print()
            print("  Install missing R packages with:")
            print("    Rscript scripts/setup_r_packages.R")
    else:
        print("R: not found. Published baselines (maxnet/BRT/GAM/RF) will be SKIPPED.")
        print("   Install R and put Rscript on PATH, or set $SDMBENCH_RSCRIPT.")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdmbench",
        description="Reproducible benchmarking for species distribution modelling.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  sdmbench data fetch disdat\n"
            "  sdmbench reproduce tabpfn-sdm-2026 --max-species 5\n"
            "  sdmbench benchmark --benchmark tabpfn-sdm-2026 --models knn,xgboost\n"
            "  sdmbench leaderboard --scenario nonspatial\n"
            "  sdmbench compare tabpfn-sdm knn --metric roc_auc\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"sdmbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_results_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-dir", help="run directory to read results from")
        p.add_argument("--results", help="path to a results.parquet file")

    def add_run_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", help="YAML configuration file")
        p.add_argument("--models", type=_csv_list, help="comma-separated model names or set alias")
        p.add_argument("--scenarios", type=_csv_list, help="nonspatial,spatial")
        p.add_argument("--regions", type=_csv_list, help="restrict to these regions")
        p.add_argument("--species", type=_csv_list, help="restrict to these species ids")
        p.add_argument("--max-species", type=int, help="cap the number of species (smoke tests)")
        p.add_argument("--metrics", type=_csv_list, help="override the metric set")
        p.add_argument("--device", help="auto|cpu|cuda|cuda:N|mps")
        p.add_argument("--workers", type=int)
        p.add_argument("--seed", type=int)
        p.add_argument("--output", "-o", help="run directory")
        p.add_argument("--force", action="store_true", help="recompute completed jobs")
        p.add_argument("--strict", action="store_true", help="fail instead of skipping")
        p.add_argument("--no-audit", action="store_true", help="disable the leakage auditor")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--quiet", "-q", action="store_true")

    # -- data ---------------------------------------------------------------
    p_data = sub.add_parser("data", help="fetch and inspect benchmark data")
    data_sub = p_data.add_subparsers(dest="data_action", required=True)

    p_fetch = data_sub.add_parser("fetch", help="download/prepare a dataset")
    p_fetch.add_argument("dataset")
    p_fetch.add_argument("--source", choices=["r", "osf"], default="r")
    p_fetch.add_argument("--url", help="explicit archive URL for --source osf")
    p_fetch.add_argument("--regions", type=_csv_list)
    p_fetch.add_argument("--rscript", help="path to Rscript")
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=cmd_data_fetch)

    p_info = data_sub.add_parser("info", help="describe the prepared dataset")
    p_info.add_argument("dataset", nargs="?", default="disdat")
    p_info.set_defaults(func=cmd_data_info)

    p_list = data_sub.add_parser("list", help="list known datasets")
    p_list.set_defaults(func=cmd_data_list)

    # -- inspect ------------------------------------------------------------
    p_inspect = sub.add_parser("inspect", help="suggest a schema for a user table")
    p_inspect.add_argument("path")
    p_inspect.add_argument("--json", action="store_true")
    p_inspect.set_defaults(func=cmd_inspect)

    # -- benchmark ----------------------------------------------------------
    p_bench = sub.add_parser("benchmark", help="run a benchmark")
    p_bench.add_argument("--benchmark", "-b", help="benchmark id")
    add_run_args(p_bench)
    p_bench.set_defaults(func=cmd_benchmark)

    p_csv = sub.add_parser("benchmark-csv", help="benchmark a user CSV")
    p_csv.add_argument("path")
    p_csv.add_argument("--target", default="presence")
    p_csv.add_argument("--longitude", default="longitude")
    p_csv.add_argument("--latitude", default="latitude")
    p_csv.add_argument("--test-csv", help="independent evaluation table")
    p_csv.add_argument("--models", type=_csv_list)
    p_csv.add_argument("--scenarios", type=_csv_list)
    p_csv.add_argument("--metric", default="roc_auc")
    p_csv.add_argument("--seed", type=int)
    p_csv.add_argument("--output", "-o")
    p_csv.add_argument(
        "--include-coordinates",
        action="store_true",
        help="allow longitude/latitude to be used as predictors (off by default)",
    )
    p_csv.set_defaults(func=cmd_benchmark_csv)

    # -- reproduce ----------------------------------------------------------
    p_repro = sub.add_parser("reproduce", help="strict reproduction of a paper")
    p_repro.add_argument("benchmark_id")
    add_run_args(p_repro)
    p_repro.set_defaults(func=cmd_reproduce)

    # -- report -------------------------------------------------------------
    p_report = sub.add_parser("report", help="published vs reproduced")
    p_report.add_argument("benchmark_id")
    add_results_args(p_report)
    p_report.add_argument("--scenario")
    p_report.add_argument("--json", action="store_true")
    p_report.add_argument("--output", "-o", help="write the report as JSON")
    p_report.add_argument("--show-extra", action="store_true",
                          help="also list models the paper did not report")
    p_report.add_argument("--match-tolerance", type=float, default=0.010)
    p_report.add_argument("--close-tolerance", type=float, default=0.025)
    p_report.set_defaults(func=cmd_report)

    # -- leaderboard --------------------------------------------------------
    p_board = sub.add_parser("leaderboard", help="ranked results")
    add_results_args(p_board)
    p_board.add_argument("--benchmark", "-b")
    p_board.add_argument("--metric", default="roc_auc")
    p_board.add_argument("--scenario")
    p_board.add_argument("--common-species", action="store_true",
                         help="restrict to species every model completed")
    p_board.add_argument("--output", "-o", help="write the leaderboard as CSV")
    p_board.set_defaults(func=cmd_leaderboard)

    # -- compare ------------------------------------------------------------
    p_cmp = sub.add_parser("compare", help="paired model comparison")
    p_cmp.add_argument("model_a", nargs="?")
    p_cmp.add_argument("model_b", nargs="?")
    add_results_args(p_cmp)
    p_cmp.add_argument("--benchmark", "-b")
    p_cmp.add_argument("--metric", default="roc_auc")
    p_cmp.add_argument("--scenario")
    p_cmp.add_argument("--all-pairs", action="store_true")
    p_cmp.add_argument("--reference", help="compare every model against this one")
    p_cmp.add_argument("--correction", default="holm", choices=["holm", "bonferroni", "none"])
    p_cmp.set_defaults(func=cmd_compare)

    # -- registries ---------------------------------------------------------
    p_models = sub.add_parser("models", help="list model adapters")
    p_models.add_argument("--all", action="store_true", help="include aliases")
    p_models.set_defaults(func=cmd_models)

    p_benchmarks = sub.add_parser("benchmarks", help="list benchmark recipes")
    p_benchmarks.set_defaults(func=cmd_benchmarks)

    p_prov = sub.add_parser("provenance", help="show where a recipe's constants came from")
    p_prov.add_argument("benchmark_id")
    p_prov.add_argument("--json", action="store_true")
    p_prov.set_defaults(func=cmd_provenance)

    # -- upstream -----------------------------------------------------------
    p_up = sub.add_parser("upstream", help="manage upstream source repositories")
    up_sub = p_up.add_subparsers(dest="upstream_action", required=True)
    for action, help_text in (
        ("fetch", "clone/update and pin the upstream repository"),
        ("check", "probe whether the repository is reachable"),
        ("info", "show what has been fetched"),
    ):
        sp = up_sub.add_parser(action, help=help_text)
        sp.add_argument("benchmark_id")
        sp.add_argument("--revision")
        sp.add_argument("--force", action="store_true")
        sp.set_defaults(func=cmd_upstream)

    # -- environment --------------------------------------------------------
    p_env = sub.add_parser("env", help="environment diagnostics")
    env_sub = p_env.add_subparsers(dest="env_action", required=True)
    p_check = env_sub.add_parser("check", help="report what is and is not available")
    p_check.set_defaults(func=cmd_env_check)

    return parser


def _csv_list(value: str) -> list[str]:
    return [part.strip() for part in value.replace("\n", ",").split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted. Re-run the same command to resume from where it stopped.")
        return 130
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not traceback
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        if "--debug" in (argv or sys.argv):
            raise
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
