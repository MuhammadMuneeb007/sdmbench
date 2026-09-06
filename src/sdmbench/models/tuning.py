"""Hyperparameter selection that cannot see the evaluation data.

The rule from the project brief, and the only defensible one for a benchmark:

    Hyperparameters must be selected WITHOUT accessing independent evaluation
    data.

:class:`InnerCvTuner` therefore searches a grid using cross-validation **inside
the training partition only**. It never receives ``split.X_test`` or
``split.y_test``; the constructor signature makes that impossible rather than
merely discouraged.

For the spatial scenario the inner folds are spatially blocked as well, so that
the selected configuration is the one that generalises across space rather than
the one that best exploits spatial autocorrelation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from sdmbench.splits.spatial import spatial_block_folds

__all__ = ["InnerCvTuner", "TuningResult", "expand_grid"]


def expand_grid(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a hyperparameter grid, in deterministic order."""
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


@dataclass
class TuningResult:
    """What the search chose, and the evidence for it."""

    best_params: dict[str, Any]
    best_score: float
    all_scores: list[tuple[dict[str, Any], float]] = field(default_factory=list)
    n_candidates: int = 0
    n_folds: int = 0
    metric: str = "roc_auc"
    used_test_data: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "n_candidates": self.n_candidates,
            "n_folds": self.n_folds,
            "metric": self.metric,
            "selection_used_test_data": self.used_test_data,
        }


class InnerCvTuner:
    """Grid search by cross-validation within the training partition.

    Parameters
    ----------
    build:
        ``params -> fitted-able estimator`` factory. Must return a fresh
        estimator each call.
    grid:
        Hyperparameter grid.
    n_folds:
        Number of inner folds.
    spatial:
        Use spatially blocked inner folds (for the spatial scenario).
    metric:
        Currently ``"roc_auc"`` or ``"average_precision"``.
    """

    def __init__(
        self,
        build: Callable[[dict[str, Any]], Any],
        grid: dict[str, Sequence[Any]],
        *,
        n_folds: int = 5,
        spatial: bool = False,
        metric: str = "roc_auc",
        seed: int = 32639,
    ) -> None:
        self.build = build
        self.grid = dict(grid)
        self.n_folds = n_folds
        self.spatial = spatial
        self.metric = metric
        self.seed = seed

    def search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        coords_train: np.ndarray | None = None,
    ) -> TuningResult:
        """Run the search. Only training data is ever passed in."""
        from sdmbench.metrics import METRIC_FUNCTIONS

        score_fn = METRIC_FUNCTIONS[self.metric]
        X = np.asarray(X_train, dtype=float)
        y = np.asarray(y_train, dtype=int)
        folds = self._make_folds(y, coords_train)

        candidates = expand_grid(self.grid)
        scored: list[tuple[dict[str, Any], float]] = []
        for params in candidates:
            fold_scores: list[float] = []
            for train_idx, valid_idx in folds:
                if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[valid_idx])) < 2:
                    continue
                estimator = self.build(params)
                try:
                    estimator.fit(X[train_idx], y[train_idx])
                    proba = estimator.predict_proba(X[valid_idx])
                    prob = proba[:, 1] if getattr(proba, "ndim", 1) == 2 else proba
                    fold_scores.append(float(score_fn(y[valid_idx], prob)))
                except Exception:  # noqa: BLE001 - a bad corner of the grid is not fatal
                    continue
            mean_score = float(np.nanmean(fold_scores)) if fold_scores else float("nan")
            scored.append((params, mean_score))

        usable = [(p, s) for p, s in scored if np.isfinite(s)]
        if not usable:
            return TuningResult(
                best_params={},
                best_score=float("nan"),
                all_scores=scored,
                n_candidates=len(candidates),
                n_folds=len(folds),
                metric=self.metric,
            )
        best_params, best_score = max(usable, key=lambda item: item[1])
        return TuningResult(
            best_params=best_params,
            best_score=best_score,
            all_scores=scored,
            n_candidates=len(candidates),
            n_folds=len(folds),
            metric=self.metric,
            used_test_data=False,
        )

    def _make_folds(
        self, y: np.ndarray, coords: np.ndarray | None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.spatial and coords is not None and len(coords) == len(y):
            assignments = spatial_block_folds(coords, n_folds=self.n_folds, seed=self.seed)
            return [
                (np.flatnonzero(assignments != f), np.flatnonzero(assignments == f))
                for f in range(self.n_folds)
                if np.any(assignments == f)
            ]
        from sklearn.model_selection import StratifiedKFold

        n_splits = min(self.n_folds, int(np.min(np.bincount(y))) if len(y) else self.n_folds)
        if n_splits < 2:
            # Too few members of a class to cross-validate; fall back to a
            # single split so the search still returns something honest.
            cut = max(1, len(y) // 5)
            return [(np.arange(cut, len(y)), np.arange(cut))]
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        return list(cv.split(np.zeros(len(y)), y))
