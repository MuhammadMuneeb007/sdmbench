"""Benchmarking a user-supplied CSV.

For a table the user already has -- extracted covariates, a presence column,
and optionally coordinates::

    longitude, latitude, bio01, bio04, bio12, vegetation, elevation, presence

The validation here is deliberately strict, because the failure modes it
catches are the ones that quietly produce a wrong benchmark rather than an
error:

* a target column that is not binary, or is 95% one class
* coordinates outside plausible bounds, or swapped lon/lat
* duplicated rows, which inflate the apparent sample size
* constant or all-missing predictors
* **longitude/latitude silently becoming predictors** -- the single most common
  way to get an impressive-looking SDM that has learned nothing but where the
  surveys happened. Coordinates are excluded unless
  ``include_coordinates=True`` is passed explicitly.

Where there is no independent survey data, the test set has to come from the
same table. That is a weaker design than the ``disdat`` benchmark and
:meth:`CsvDataset.to_task` records it in the task metadata so a report never
implies otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from sdmbench.data.base import SpeciesTask
from sdmbench.exceptions import DataError
from sdmbench.reproducibility.hashes import hash_dataframe, hash_file

__all__ = ["CsvDataset", "ValidationReport", "validate_table"]


@dataclass
class ValidationReport:
    """Findings from validating a user table."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            raise DataError(
                "the supplied table failed validation:\n"
                + "\n".join(f"  - {e}" for e in self.errors)
            )

    def render(self) -> str:
        lines = [f"Validation: {'PASS' if self.ok else 'FAIL'}"]
        for e in self.errors:
            lines.append(f"  [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"  [WARN ] {w}")
        for key, value in self.summary.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


def validate_table(
    df: pd.DataFrame,
    *,
    target: str,
    longitude: str | None = None,
    latitude: str | None = None,
    predictors: Sequence[str] | None = None,
    min_positives: int = 5,
) -> ValidationReport:
    """Check a user table against the requirements of a fair benchmark."""
    report = ValidationReport()

    if target not in df.columns:
        report.errors.append(f"target column {target!r} not found; columns: {list(df.columns)}")
        return report

    values = df[target].dropna()
    unique = set(pd.unique(values).tolist())
    if not unique <= {0, 1, 0.0, 1.0, True, False}:
        as_text = {str(v).strip().lower() for v in unique}
        if as_text <= {"yes", "no", "true", "false", "presence", "absence", "present", "absent"}:
            report.warnings.append(
                f"target {target!r} holds labels {sorted(unique, key=str)[:6]}; "
                "they will be mapped to 1/0"
            )
        else:
            report.errors.append(
                f"target {target!r} must be binary; found {sorted(unique, key=str)[:10]}"
            )
    if df[target].isna().any():
        report.errors.append(f"target {target!r} contains {int(df[target].isna().sum())} missing values")

    n_pos = int((pd.to_numeric(df[target], errors="coerce") == 1).sum())
    n_neg = int((pd.to_numeric(df[target], errors="coerce") == 0).sum())
    if n_pos < min_positives:
        report.errors.append(f"only {n_pos} presence record(s); at least {min_positives} needed")
    if n_neg == 0:
        report.errors.append("no background/absence records")
    if n_pos and n_neg:
        ratio = n_neg / n_pos
        if ratio > 200:
            report.warnings.append(
                f"severe class imbalance: {n_pos} presences to {n_neg} background "
                f"({ratio:.0f}:1)"
            )

    # --- coordinates ------------------------------------------------------
    for name, column in (("longitude", longitude), ("latitude", latitude)):
        if column is None:
            continue
        if column not in df.columns:
            report.errors.append(f"{name} column {column!r} not found")
            continue
        col = pd.to_numeric(df[column], errors="coerce")
        if col.isna().any():
            report.warnings.append(
                f"{name} column {column!r} has {int(col.isna().sum())} non-numeric/missing values"
            )
        limit = 180 if name == "longitude" else 90
        if col.abs().max() is not np.nan and float(col.abs().max() or 0) > limit:
            report.warnings.append(
                f"{name} values exceed +/-{limit}; the data may be projected rather than "
                "geographic, or the columns may be swapped"
            )

    # --- predictors -------------------------------------------------------
    meta = {c for c in (target, longitude, latitude) if c}
    preds = list(predictors) if predictors else [c for c in df.columns if c not in meta]
    missing = [p for p in preds if p not in df.columns]
    if missing:
        report.errors.append(f"predictor column(s) not found: {missing}")
    if not preds:
        report.errors.append("no predictor columns remain after excluding target and coordinates")

    for col in preds:
        if col not in df.columns:
            continue
        series = df[col]
        if series.isna().all():
            report.errors.append(f"predictor {col!r} is entirely missing")
        elif series.nunique(dropna=True) <= 1:
            report.warnings.append(f"predictor {col!r} is constant and carries no information")
        elif series.isna().mean() > 0.5:
            report.warnings.append(
                f"predictor {col!r} is {series.isna().mean():.0%} missing"
            )

    coordinate_like = [
        c
        for c in preds
        if c.lower() in {"x", "y", "lon", "long", "longitude", "lat", "latitude", "easting", "northing"}
    ]
    if coordinate_like:
        report.warnings.append(
            f"predictor(s) {coordinate_like} look like coordinates. sdmbench excludes "
            "coordinates from the feature set unless include_coordinates=True is passed "
            "explicitly, because coordinate features let a model memorise survey locations."
        )

    n_duplicates = int(df.duplicated().sum())
    if n_duplicates:
        report.warnings.append(f"{n_duplicates} duplicate row(s)")

    report.summary = {
        "n_rows": int(len(df)),
        "n_presences": n_pos,
        "n_background": n_neg,
        "n_predictors": len(preds),
        "n_duplicates": n_duplicates,
    }
    return report


class CsvDataset:
    """A single-species dataset backed by a user CSV.

    Parameters
    ----------
    path:
        CSV file, or an in-memory DataFrame via :meth:`from_frame`.
    target:
        Binary presence column.
    coordinates:
        ``(longitude, latitude)`` column names, or ``None`` when the table has
        no coordinates -- in which case the spatial scenario is unavailable.
    predictors:
        Explicit predictor list. Defaults to every column that is not the
        target or a coordinate.
    include_coordinates:
        Opt in to using coordinates as predictors. Off by default.
    """

    name = "csv"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        frame: pd.DataFrame | None = None,
        target: str = "presence",
        coordinates: tuple[str, str] | None = ("longitude", "latitude"),
        predictors: Sequence[str] | None = None,
        categorical: Sequence[str] | None = None,
        include_coordinates: bool = False,
        species_id: str = "species",
        region: str = "user",
        crs: str = "EPSG:4326",
        geographic: bool = True,
    ) -> None:
        if (path is None) == (frame is None):
            raise DataError("provide exactly one of `path` or `frame`")
        self.path = Path(path) if path else None
        self._frame = frame.copy() if frame is not None else pd.read_csv(self.path)
        self.target = target
        self.coordinates = tuple(coordinates) if coordinates else None
        self.include_coordinates = include_coordinates
        self.species_id = species_id
        self.region = region
        self.crs = crs
        self.geographic = geographic

        meta = {target, *(self.coordinates or ())}
        self.predictors = (
            list(predictors)
            if predictors
            else [c for c in self._frame.columns if c not in meta]
        )
        if include_coordinates and self.coordinates:
            for col in self.coordinates:
                if col not in self.predictors:
                    self.predictors.append(col)
        self.categorical = list(categorical) if categorical else self._infer_categorical()

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, **kwargs: Any) -> CsvDataset:
        return cls(frame=frame, **kwargs)

    def _infer_categorical(self) -> list[str]:
        """Treat object/string/low-cardinality-integer columns as categorical."""
        out: list[str] = []
        for col in self.predictors:
            series = self._frame[col]
            if series.dtype == object or pd.api.types.is_string_dtype(series):
                out.append(col)
            elif pd.api.types.is_integer_dtype(series) and series.nunique(dropna=True) <= 12:
                # An integer code with few levels is far more often a class
                # label (vegetation type) than a magnitude.
                out.append(col)
        return out

    # ------------------------------------------------------------ validation --
    def validate(self) -> ValidationReport:
        lon, lat = self.coordinates if self.coordinates else (None, None)
        return validate_table(
            self._frame,
            target=self.target,
            longitude=lon,
            latitude=lat,
            predictors=self.predictors,
        )

    # ------------------------------------------------------------------ data --
    def to_task(
        self,
        *,
        test_frame: pd.DataFrame | None = None,
        test_size: float = 0.2,
        seed: int = 32639,
        benchmark_id: str = "csv",
    ) -> SpeciesTask:
        """Build a :class:`SpeciesTask`.

        ``test_frame`` supplies genuinely independent evaluation data. Without
        it, a stratified holdout is carved from the same table -- a weaker
        design, recorded in the task metadata as ``test_data="holdout"`` so
        reports do not overstate what was measured.
        """
        self.validate().raise_if_failed()
        frame = self._standardise(self._frame)

        if test_frame is not None:
            train = frame
            test = self._standardise(test_frame)
            provenance = "independent"
        else:
            from sdmbench.splits.random import stratified_holdout

            train_idx, test_idx = stratified_holdout(
                frame["occ"].to_numpy(), test_size=test_size, seed=seed
            )
            train = frame.iloc[train_idx].reset_index(drop=True)
            test = frame.iloc[test_idx].reset_index(drop=True)
            provenance = "holdout"

        return SpeciesTask(
            region=self.region,
            species_id=self.species_id,
            train=train,
            test=test,
            predictors=list(self.predictors),
            categorical_predictors=list(self.categorical),
            crs=self.crs,
            geographic=self.geographic,
            benchmark_id=benchmark_id,
            metadata={
                "dataset": self.name,
                "source": str(self.path) if self.path else "in-memory",
                "test_data": provenance,
                "include_coordinates": self.include_coordinates,
                "note": (
                    "Test data was held out from the same table, so it is not an "
                    "independent survey; performance here is a same-distribution estimate."
                    if provenance == "holdout"
                    else "Test data was supplied separately."
                ),
            },
        )

    def _standardise(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rename user columns onto the internal schema."""
        out = pd.DataFrame(index=range(len(frame)))
        out["region"] = self.region
        out["group"] = None
        out["siteid"] = [f"row{i}" for i in range(len(frame))]
        out["spid"] = self.species_id
        if self.coordinates:
            lon, lat = self.coordinates
            out["x"] = pd.to_numeric(frame[lon], errors="coerce").to_numpy()
            out["y"] = pd.to_numeric(frame[lat], errors="coerce").to_numpy()
        else:
            # Splits and graphs need coordinates; without them only the
            # non-spatial scenario is meaningful.
            out["x"] = np.nan
            out["y"] = np.nan
        out["occ"] = _coerce_binary(frame[self.target]).to_numpy()
        for col in self.predictors:
            out[col] = frame[col].to_numpy()
        return out

    def iter_tasks(self, **kwargs: Any) -> Iterator[SpeciesTask]:
        yield self.to_task(**kwargs)

    def content_hash(self) -> str:
        if self.path and self.path.is_file():
            return hash_file(self.path)
        return hash_dataframe(self._frame)


def _coerce_binary(series: pd.Series) -> pd.Series:
    """Map common presence encodings onto 1/0."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int)
    mapping = {
        "yes": 1, "no": 0, "true": 1, "false": 0,
        "presence": 1, "absence": 0, "present": 1, "absent": 0,
        "1": 1, "0": 0,
    }
    coerced = series.astype(str).str.strip().str.lower().map(mapping)
    if coerced.isna().any():
        bad = sorted(set(series[coerced.isna()].astype(str)))[:10]
        raise DataError(f"cannot interpret target values as binary: {bad}")
    return coerced.astype(int)
