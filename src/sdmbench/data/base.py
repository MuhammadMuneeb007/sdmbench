"""The standardised internal data model.

Every dataset sdmbench understands -- ``disdat``, a user CSV, a raster stack --
is normalised into the same two objects:

:class:`SpeciesTask`
    All data for one species: a training table (presences + background) and a
    test table (independent presence/absence survey data), plus the predictor
    names, which of them are categorical, and the coordinate reference system.
:class:`PreparedSplit`
    What a model actually sees: fitted-and-applied design matrices, labels,
    coordinates, and the categorical column indices. Produced by a split rule
    plus a preprocessing recipe.

Column conventions
------------------
Long-format occurrence tables use the ``disdat`` names so that the R bridge and
the Python side agree without translation:

==========  ==========================================================
``region``  region code (``AWT``, ``CAN``, ``NSW``, ``NZ``, ``SA``, ``SWI``)
``group``   taxonomic/survey group, or ``None`` where a region has one
``siteid``  site identifier as supplied by the source data
``spid``    anonymised species identifier
``x``, ``y``  coordinates *in the region's own CRS* (see :func:`region_crs`)
``occ``     1 = presence, 0 = background (train) or absence (test)
==========  ==========================================================

Coordinates are deliberately kept out of the predictor list. They are used to
build splits and graph topology; making them features is a separate, explicit
opt-in (see ``include_coordinates``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from sdmbench.exceptions import DataError, InsufficientDataError

__all__ = [
    "META_COLUMNS",
    "COORD_COLUMNS",
    "TARGET_COLUMN",
    "SpeciesTask",
    "PreparedSplit",
    "Dataset",
    "validate_occurrence_frame",
]

#: Non-predictor columns recognised in a standardised occurrence table.
META_COLUMNS = ("region", "group", "siteid", "spid", "x", "y", "occ")
COORD_COLUMNS = ("x", "y")
TARGET_COLUMN = "occ"


def validate_occurrence_frame(
    df: pd.DataFrame,
    *,
    predictors: Sequence[str],
    name: str = "table",
    require_target: bool = True,
) -> None:
    """Raise :class:`DataError` if ``df`` violates the standardised schema."""
    if not isinstance(df, pd.DataFrame):
        raise DataError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")
    missing_coords = [c for c in COORD_COLUMNS if c not in df.columns]
    if missing_coords:
        raise DataError(f"{name} is missing coordinate column(s): {missing_coords}")
    if require_target and TARGET_COLUMN not in df.columns:
        raise DataError(f"{name} is missing the target column {TARGET_COLUMN!r}")
    missing_pred = [p for p in predictors if p not in df.columns]
    if missing_pred:
        raise DataError(f"{name} is missing predictor column(s): {missing_pred}")
    if require_target:
        values = pd.unique(df[TARGET_COLUMN].dropna())
        bad = set(np.asarray(values).tolist()) - {0, 1, 0.0, 1.0, True, False}
        if bad:
            raise DataError(
                f"{name}[{TARGET_COLUMN!r}] must be binary 0/1; found unexpected values: "
                f"{sorted(bad, key=str)[:10]}"
            )


@dataclass
class SpeciesTask:
    """All data for a single species, before splitting or preprocessing.

    Parameters
    ----------
    train:
        Presence-only records for this species plus the region's background
        sample. ``occ`` is 1 for presences and 0 for background. Background
        records are *not* absences -- they are samples of available environment.
    test:
        Independent presence/absence survey data. ``occ`` is 1/0 and here 0
        genuinely means absence. These rows are frozen: nothing in the fitting
        pipeline may touch them.
    """

    region: str
    species_id: str
    train: pd.DataFrame
    test: pd.DataFrame
    predictors: list[str]
    categorical_predictors: list[str] = field(default_factory=list)
    group: str | None = None
    crs: str | None = None
    #: True when x/y are longitude/latitude in degrees (affects buffer maths).
    geographic: bool = False
    benchmark_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_occurrence_frame(self.train, predictors=self.predictors, name="train")
        validate_occurrence_frame(self.test, predictors=self.predictors, name="test")
        unknown_cat = set(self.categorical_predictors) - set(self.predictors)
        if unknown_cat:
            raise DataError(
                f"categorical_predictors not present in predictors: {sorted(unknown_cat)}"
            )
        self.train = self.train.reset_index(drop=True)
        self.test = self.test.reset_index(drop=True)

    # ---------------------------------------------------------------- counts --
    @property
    def numeric_predictors(self) -> list[str]:
        cats = set(self.categorical_predictors)
        return [p for p in self.predictors if p not in cats]

    @property
    def train_presence_n(self) -> int:
        return int((self.train[TARGET_COLUMN] == 1).sum())

    @property
    def train_background_n(self) -> int:
        return int((self.train[TARGET_COLUMN] == 0).sum())

    @property
    def test_presence_n(self) -> int:
        return int((self.test[TARGET_COLUMN] == 1).sum())

    @property
    def test_absence_n(self) -> int:
        return int((self.test[TARGET_COLUMN] == 0).sum())

    @property
    def task_id(self) -> str:
        """Stable identifier used for checkpoint filenames."""
        parts = [self.region, self.species_id]
        if self.group:
            parts.insert(1, self.group)
        return "_".join(parts)

    # ------------------------------------------------------------- accessors --
    def train_coords(self) -> np.ndarray:
        return self.train.loc[:, list(COORD_COLUMNS)].to_numpy(dtype=float)

    def test_coords(self) -> np.ndarray:
        return self.test.loc[:, list(COORD_COLUMNS)].to_numpy(dtype=float)

    def with_train(self, train: pd.DataFrame) -> SpeciesTask:
        """Return a copy with a different (e.g. spatially filtered) training set."""
        return SpeciesTask(
            region=self.region,
            species_id=self.species_id,
            train=train.reset_index(drop=True),
            test=self.test,
            predictors=list(self.predictors),
            categorical_predictors=list(self.categorical_predictors),
            group=self.group,
            crs=self.crs,
            geographic=self.geographic,
            benchmark_id=self.benchmark_id,
            metadata=dict(self.metadata),
        )

    def require_both_classes(self, *, where: str = "training data") -> None:
        """Raise :class:`InsufficientDataError` if a partition is single-class.

        This is the condition that removes ~41 species from the spatial
        evaluation in the TabPFN-SDM 2026 protocol: after excluding training
        points within the buffer, some species retain only one class and
        classification is undefined.
        """
        if self.train_presence_n == 0 or self.train_background_n == 0:
            raise InsufficientDataError(
                f"{self.task_id}: {where} is single-class after filtering "
                f"(presences={self.train_presence_n}, background={self.train_background_n})"
            )
        if self.test_presence_n == 0 or self.test_absence_n == 0:
            raise InsufficientDataError(
                f"{self.task_id}: test data is single-class "
                f"(presences={self.test_presence_n}, absences={self.test_absence_n})"
            )

    def describe(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "group": self.group,
            "species_id": self.species_id,
            "n_predictors": len(self.predictors),
            "n_categorical": len(self.categorical_predictors),
            "train_presence_n": self.train_presence_n,
            "train_background_n": self.train_background_n,
            "test_presence_n": self.test_presence_n,
            "test_absence_n": self.test_absence_n,
            "crs": self.crs,
        }


@dataclass
class PreparedSplit:
    """Design matrices handed to a model adapter.

    ``X_train``/``X_test`` are DataFrames rather than arrays so that adapters
    which support native categoricals (LightGBM, CatBoost, the R models) can
    use them, while array-only adapters call :meth:`as_arrays`.
    """

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    coords_train: np.ndarray
    coords_test: np.ndarray
    feature_names: list[str]
    categorical_features: list[str] = field(default_factory=list)
    scenario: str = "nonspatial"
    split_id: str = "default"
    repeat_id: int = 0
    crs: str | None = None
    geographic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.X_train) != len(self.y_train):
            raise DataError(
                f"X_train ({len(self.X_train)}) and y_train ({len(self.y_train)}) length mismatch"
            )
        if len(self.X_test) != len(self.y_test):
            raise DataError(
                f"X_test ({len(self.X_test)}) and y_test ({len(self.y_test)}) length mismatch"
            )
        if len(self.coords_train) != len(self.X_train):
            raise DataError("coords_train does not match X_train length")
        if len(self.coords_test) != len(self.X_test):
            raise DataError("coords_test does not match X_test length")

    @property
    def categorical_indices(self) -> list[int]:
        """Positional indices of categorical features.

        TabPFN needs these explicitly (paper sec. 2.1.2: "Only the local TabPFN
        implementation required explicit specification of categorical feature
        indices").
        """
        lookup = {name: i for i, name in enumerate(self.feature_names)}
        return sorted(lookup[c] for c in self.categorical_features if c in lookup)

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(X_train, y_train, X_test, y_test)`` as float arrays.

        Categorical columns are integer-coded on the *training* levels; unseen
        test levels become -1. Encoding must never be fitted on test data.
        """
        Xtr = self.X_train.copy()
        Xte = self.X_test.copy()
        for col in self.categorical_features:
            if col not in Xtr.columns:
                continue
            levels = pd.Index(pd.unique(Xtr[col].astype("object").dropna()))
            Xtr[col] = pd.Categorical(Xtr[col].astype("object"), categories=levels).codes
            Xte[col] = pd.Categorical(Xte[col].astype("object"), categories=levels).codes
        return (
            Xtr.to_numpy(dtype=float),
            np.asarray(self.y_train, dtype=int),
            Xte.to_numpy(dtype=float),
            np.asarray(self.y_test, dtype=int),
        )

    @property
    def n_train(self) -> int:
        return len(self.X_train)

    @property
    def n_test(self) -> int:
        return len(self.X_test)

    @property
    def train_presence_n(self) -> int:
        return int(np.sum(np.asarray(self.y_train) == 1))

    @property
    def train_background_n(self) -> int:
        return int(np.sum(np.asarray(self.y_train) == 0))

    @property
    def test_presence_n(self) -> int:
        return int(np.sum(np.asarray(self.y_test) == 1))

    @property
    def test_absence_n(self) -> int:
        return int(np.sum(np.asarray(self.y_test) == 0))


class Dataset:
    """Interface implemented by every data source.

    A dataset knows how to enumerate species tasks; it knows nothing about
    splits, preprocessing, models or metrics -- those belong to the benchmark
    recipe. This is what allows one dataset to serve several benchmarks.
    """

    #: Short identifier, e.g. ``"disdat"``.
    name: str = "dataset"

    def regions(self) -> list[str]:
        raise NotImplementedError

    def species(self, region: str) -> list[str]:
        raise NotImplementedError

    def iter_tasks(self, regions: Iterable[str] | None = None) -> Iterable[SpeciesTask]:
        raise NotImplementedError

    def get_task(self, region: str, species_id: str) -> SpeciesTask:
        raise NotImplementedError

    def content_hash(self) -> str:
        """Hash identifying the exact data content, recorded in every result."""
        raise NotImplementedError
