"""Shared fixtures and tiny synthetic datasets.

Every fixture here is small and deterministic. The default suite must never
download 226 species, need a GPU, or touch R -- those are marked with
``@pytest.mark.integration`` / ``gpu`` / ``r`` / ``network`` and skipped unless
the dependency is genuinely present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sdmbench.data.base import SpeciesTask

SEED = 32639


def _skip_without(module: str):
    """Skip marker for an optional backend."""
    from sdmbench.optional import have

    return pytest.mark.skipif(not have(module), reason=f"{module} is not installed")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_frame(rng: np.random.Generator) -> pd.DataFrame:
    """A small presence-background table with a real environmental signal.

    The signal is deliberately learnable but not trivial, so that a working
    model scores well above 0.5 and a broken pipeline is visible.
    """
    n = 300
    lon = rng.uniform(-5.0, 5.0, n)
    lat = rng.uniform(-5.0, 5.0, n)
    bio01 = 10.0 + 0.8 * lat + rng.normal(0, 1.0, n)
    bio12 = 800.0 + 30.0 * lon + rng.normal(0, 50.0, n)
    vegetation = rng.choice(["forest", "grass", "shrub"], n)

    logit = -1.5 + 0.5 * (bio01 - 10.0) + 0.004 * (bio12 - 800.0)
    presence = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "longitude": lon,
            "latitude": lat,
            "bio01": bio01,
            "bio12": bio12,
            "vegetation": vegetation,
            "presence": presence,
        }
    )


@pytest.fixture
def species_task(rng: np.random.Generator) -> SpeciesTask:
    """A minimal :class:`SpeciesTask` in the internal schema.

    Training data is presence-only + background (imbalanced, as real data is);
    test data is a separate presence-absence survey.
    """

    def build(n: int, prefix: str, presence_rate: float) -> pd.DataFrame:
        x = rng.uniform(0.0, 100_000.0, n)
        y = rng.uniform(0.0, 100_000.0, n)
        bio01 = 10.0 + y / 20_000.0 + rng.normal(0, 0.5, n)
        bio12 = 500.0 + x / 100.0 + rng.normal(0, 20.0, n)
        return pd.DataFrame(
            {
                "region": "TEST",
                "group": None,
                "siteid": [f"{prefix}{i}" for i in range(n)],
                "spid": "sp01",
                "x": x,
                "y": y,
                "occ": (rng.uniform(size=n) < presence_rate).astype(int),
                "bio01": bio01,
                "bio12": bio12,
                "vegsys": rng.choice(["1", "2", "3"], n),
            }
        )

    train = build(240, "tr", 0.15)
    test = build(80, "te", 0.35)
    return SpeciesTask(
        region="TEST",
        species_id="sp01",
        train=train,
        test=test,
        predictors=["bio01", "bio12", "vegsys"],
        categorical_predictors=["vegsys"],
        crs="EPSG:28355",
        geographic=False,
    )


@pytest.fixture
def results_frame() -> pd.DataFrame:
    """A synthetic results table for aggregation and comparison tests.

    Model A is systematically better than model B, by a consistent margin, so
    a correct paired test must detect it and an incorrect one will not.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for region in ("AWT", "NSW"):
        for species in range(15):
            base = rng.uniform(0.60, 0.85)
            for model, offset in (("model-a", 0.03), ("model-b", 0.0), ("model-c", -0.05)):
                for scenario in ("nonspatial", "spatial"):
                    penalty = 0.06 if scenario == "spatial" else 0.0
                    rows.append(
                        {
                            "benchmark_id": "test",
                            "benchmark_version": "1",
                            "run_id": "r1",
                            "job_id": f"{region}-{species}-{model}-{scenario}",
                            "species_id": f"{region}_sp{species:02d}",
                            "region": region,
                            "group": None,
                            "evaluation_scenario": scenario,
                            "split_id": scenario,
                            "repeat_id": 0,
                            "seed": SEED,
                            "model_family": "classical",
                            "model_name": model,
                            "model_version": "1.0",
                            "hyperparameters_json": "{}",
                            "train_n": 200,
                            "train_presence_n": 30,
                            "train_background_n": 170,
                            "test_n": 80,
                            "test_presence_n": 28,
                            "test_absence_n": 52,
                            "status": "OK",
                            "status_detail": "",
                            "roc_auc": np.clip(base + offset - penalty + rng.normal(0, 0.01), 0, 1),
                            "pr_auc": np.clip(base + offset - penalty - 0.2, 0, 1),
                            "calibration_slope": 1.0 + rng.normal(0, 0.15),
                            "calibration_intercept": rng.normal(0, 0.2),
                            "boyce": np.clip(base + offset, 0, 1),
                            "brier": 0.2,
                            "log_loss": 0.5,
                            "sensitivity": None,
                            "specificity": None,
                            "balanced_accuracy": None,
                            "metrics_json": "{}",
                            "fit_time_seconds": 1.0,
                            "predict_time_seconds": 0.1,
                            "total_time_seconds": 1.1,
                            "peak_memory_mb": None,
                            "device": "cpu",
                            "compute_budget_json": "{}",
                            "dataset_hash": "abc",
                            "split_hash": "def",
                            "model_config_hash": "ghi",
                            "leakage_audit": "PASS",
                            "package_versions_json": "{}",
                            "git_commit": "",
                            "upstream_commit": "",
                            "timestamp": "2026-01-01T00:00:00+00:00",
                        }
                    )
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_run_dir(tmp_path):
    return tmp_path / "run"


# Convenience skip markers used across the suite.
requires_torch = _skip_without("torch")
requires_pyg = _skip_without("torch_geometric")
requires_xgboost = _skip_without("xgboost")
requires_tabpfn = _skip_without("tabpfn")
requires_rasterio = _skip_without("rasterio")


def pytest_configure(config):
    """Register markers so ``--strict-markers`` accepts them."""
    for marker, description in (
        ("integration", "end-to-end tests spanning several subsystems"),
        ("slow", "tests taking more than a few seconds"),
        ("gpu", "requires a CUDA device"),
        ("r", "requires R and specific R packages"),
        ("network", "downloads data or model weights"),
        ("tabpfn", "requires tabpfn and its licensed weights"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")
