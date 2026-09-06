"""Random (non-spatial) partitioning.

Two distinct things live here, and conflating them is a common source of
invalid SDM benchmarks:

**The non-spatial benchmark scenario** does *no* random splitting at all. In
Dinnage & Warren (2026) the training data is the presence-only + background
records and the test data is the independent presence-absence survey -- two
different data collections. Sec. 2.5: "Standard evaluation used all available
training data without spatial considerations". :func:`identity_split` documents
that explicitly so nobody later "fixes" it by inserting a train_test_split.

**Holdout splits within the training data** are a separate diagnostic from sec.
2.7, used to measure how much a same-distribution estimate inflates apparent
performance relative to the independent survey. Those *are* random, and
:func:`stratified_holdout` implements the 80/20 stratified split the paper used
via ``rsample``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sdmbench.data.base import SpeciesTask
from sdmbench.exceptions import InsufficientDataError

__all__ = [
    "identity_split",
    "stratified_holdout",
    "subsample_background",
]


def identity_split(task: SpeciesTask) -> tuple[SpeciesTask, dict[str, Any]]:
    """The non-spatial scenario: use all training data, test on the survey data.

    A no-op by design. It exists so that both scenarios go through the same
    interface and so the absence of a random split is a documented decision
    rather than an omission.
    """
    return task, {
        "rule": "no filtering; all training data used",
        "source": "Dinnage & Warren 2026 sec. 2.5",
        "n_train": int(len(task.train)),
        "n_test": int(len(task.test)),
    }


def stratified_holdout(
    y: np.ndarray,
    *,
    test_size: float = 0.2,
    seed: int = 32639,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified random holdout preserving class proportions.

    Matches sec. 2.7: "Random holdout splits used stratified sampling
    (80% train, 20% test) with the rsample package ... Both split types
    required both classes present in both partitions."

    Returns
    -------
    tuple
        ``(train_idx, test_idx)`` arrays of positional indices.
    """
    y = np.asarray(y).astype(int)
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    rng = np.random.default_rng(seed)

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        # Never let a stratum contribute zero rows to either side.
        n_test = min(max(n_test, 1), len(idx) - 1) if len(idx) > 1 else 0
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])

    train_idx = np.sort(np.concatenate(train_parts)) if train_parts else np.array([], dtype=int)
    test_idx = np.sort(np.concatenate(test_parts)) if test_parts else np.array([], dtype=int)

    if require_both_classes:
        for name, idx in (("train", train_idx), ("test", test_idx)):
            classes = np.unique(y[idx]) if len(idx) else np.array([])
            if len(classes) < 2:
                raise InsufficientDataError(
                    f"stratified holdout produced a single-class {name} partition "
                    f"(n={len(idx)}, classes={classes.tolist()})"
                )
    return train_idx, test_idx


def subsample_background(
    y: np.ndarray,
    *,
    n_background: int | None = None,
    ratio: float = 1.0,
    seed: int = 32639,
    replace: bool = False,
) -> np.ndarray:
    """Indices of all presences plus a random subsample of background records.

    The asymmetry is deliberate and is the basis of the paper's ensemble class
    balancing (sec. 2.3): presences are scarce and individually informative,
    while any given background point "could be replaced by another randomly
    drawn background point without loss of information".

    Parameters
    ----------
    n_background:
        Absolute number of background records to draw. Overrides ``ratio``.
    ratio:
        Background records per presence. ``1.0`` gives a balanced sample.
    """
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    pres_idx = np.flatnonzero(y == 1)
    bg_idx = np.flatnonzero(y == 0)

    if n_background is None:
        n_background = int(round(len(pres_idx) * ratio))
    n_background = max(0, min(n_background, len(bg_idx) if not replace else n_background))

    chosen = (
        rng.choice(bg_idx, size=n_background, replace=replace)
        if n_background and len(bg_idx)
        else np.array([], dtype=int)
    )
    return np.sort(np.concatenate([pres_idx, chosen]).astype(int))
