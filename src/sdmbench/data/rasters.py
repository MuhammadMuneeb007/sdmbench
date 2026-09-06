"""Raster / spatial workflows (optional, ``sdmbench[raster]``).

The workflow this supports::

    occurrences.csv + environment_*.tif
        -> extract covariates at occurrence coordinates
        -> sample background / pseudo-absences
        -> build spatial validation splits
        -> fit models
        -> predict across raster cells
        -> write a suitability map

Everything geospatial is delegated to mature libraries -- ``rasterio`` for I/O
and sampling, ``pyproj`` for transforms, ``geopandas``/``shapely`` for vector
masks. sdmbench implements no GIS primitives of its own.

Implementation status
---------------------
Extraction, background sampling and prediction-to-raster are implemented.
Nothing in this module has been executed, because these dependencies were not
installed in the session that wrote it, so treat it as reviewed-but-untested
code rather than verified behaviour. The API is stable enough for the rest of
the package to build against.

Coordinate handling
-------------------
Occurrence coordinates are reprojected to the raster CRS before sampling, not
the reverse: resampling a raster to match points would change the pixel values
being extracted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from sdmbench.data.base import SpeciesTask
from sdmbench.exceptions import DataError
from sdmbench.optional import require

__all__ = [
    "RasterStack",
    "RasterDataset",
    "extract_at_points",
    "sample_background",
    "predict_to_raster",
]


@dataclass
class RasterStack:
    """A set of co-registered raster predictors.

    Parameters
    ----------
    paths:
        Raster files, one per predictor. The band name defaults to the file
        stem, so ``bio01.tif`` becomes the predictor ``bio01``.
    names:
        Explicit predictor names, overriding the file stems.
    """

    paths: list[Path]
    names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.paths = [Path(p) for p in self.paths]
        missing = [str(p) for p in self.paths if not p.is_file()]
        if missing:
            raise DataError(f"raster file(s) not found: {missing}")
        if not self.names:
            self.names = [p.stem for p in self.paths]
        if len(self.names) != len(self.paths):
            raise DataError("names and paths must be the same length")

    @classmethod
    def from_directory(cls, directory: str | Path, pattern: str = "*.tif") -> RasterStack:
        """Build a stack from every matching raster in a directory."""
        files = sorted(Path(directory).glob(pattern))
        if not files:
            raise DataError(f"no rasters matching {pattern!r} in {directory}")
        return cls(paths=files)

    def profile(self) -> dict[str, Any]:
        """CRS, transform, shape and nodata of the first raster."""
        rasterio = require("rasterio")
        with rasterio.open(self.paths[0]) as src:
            return {
                "crs": str(src.crs),
                "transform": tuple(src.transform),
                "width": src.width,
                "height": src.height,
                "nodata": src.nodata,
                "dtype": src.dtypes[0],
            }

    def check_alignment(self) -> list[str]:
        """Report any raster that does not share the first one's grid.

        Misaligned rasters silently produce covariates sampled from different
        locations, which is a wrong benchmark rather than an error, so this is
        worth checking up front.
        """
        rasterio = require("rasterio")
        problems: list[str] = []
        reference = None
        for path in self.paths:
            with rasterio.open(path) as src:
                signature = (str(src.crs), tuple(src.transform), src.width, src.height)
            if reference is None:
                reference = signature
            elif signature != reference:
                problems.append(f"{path.name} does not share the grid of {self.paths[0].name}")
        return problems


def extract_at_points(
    stack: RasterStack,
    coords: np.ndarray,
    *,
    source_crs: str | None = None,
) -> pd.DataFrame:
    """Sample every raster in ``stack`` at ``coords``.

    Parameters
    ----------
    coords:
        ``(n, 2)`` array of ``(x, y)``.
    source_crs:
        CRS of ``coords``. When it differs from the raster CRS the points are
        reprojected to the raster's CRS before sampling.

    Returns
    -------
    pandas.DataFrame
        One column per predictor; nodata becomes ``NaN``.
    """
    rasterio = require("rasterio")
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise DataError(f"coords must have shape (n, 2), got {coords.shape}")

    out: dict[str, np.ndarray] = {}
    for name, path in zip(stack.names, stack.paths):
        with rasterio.open(path) as src:
            points = coords
            if source_crs and str(src.crs) and str(src.crs) != str(source_crs):
                pyproj = require("pyproj")
                transformer = pyproj.Transformer.from_crs(
                    source_crs, str(src.crs), always_xy=True
                )
                xs, ys = transformer.transform(coords[:, 0], coords[:, 1])
                points = np.column_stack([xs, ys])
            sampled = np.array(
                [v[0] for v in src.sample(points.tolist())], dtype=float
            )
            if src.nodata is not None:
                sampled = np.where(sampled == src.nodata, np.nan, sampled)
            out[name] = sampled
    return pd.DataFrame(out)


def sample_background(
    stack: RasterStack,
    n: int = 10000,
    *,
    seed: int = 32639,
    exclude_coords: np.ndarray | None = None,
    exclusion_radius: float = 0.0,
    mask_path: str | Path | None = None,
) -> pd.DataFrame:
    """Draw ``n`` background points from the raster extent.

    Only cells with data in the first raster are eligible, so background points
    never land in the sea or outside the study area.

    Parameters
    ----------
    exclude_coords, exclusion_radius:
        Optionally reject background points within ``exclusion_radius`` of a
        given set of coordinates (typically the presences).
    mask_path:
        A vector file restricting the sampling region.

    Returns
    -------
    pandas.DataFrame
        Columns ``x``, ``y`` and one per predictor.
    """
    rasterio = require("rasterio")
    rng = np.random.default_rng(seed)

    with rasterio.open(stack.paths[0]) as src:
        band = src.read(1, masked=True)
        rows, cols = np.where(~np.ma.getmaskarray(band))
        if len(rows) == 0:
            raise DataError(f"{stack.paths[0].name} has no valid cells to sample")
        transform = src.transform
        raster_crs = str(src.crs)

    geometry_mask = None
    if mask_path is not None:
        geopandas = require("geopandas")
        shapes = geopandas.read_file(mask_path)
        geometry_mask = shapes.to_crs(raster_crs).union_all()

    chosen_x: list[float] = []
    chosen_y: list[float] = []
    attempts = 0
    # Oversample and filter: rejection sampling converges quickly unless the
    # exclusions remove most of the extent, and the attempt cap stops an
    # impossible request from looping forever.
    while len(chosen_x) < n and attempts < 50:
        attempts += 1
        take = min(len(rows), (n - len(chosen_x)) * 4)
        picks = rng.choice(len(rows), size=take, replace=len(rows) < take)
        xs, ys = rasterio.transform.xy(transform, rows[picks], cols[picks])
        candidates = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])

        if geometry_mask is not None:
            shapely = require("shapely")
            keep = np.array(
                [geometry_mask.contains(shapely.geometry.Point(x, y)) for x, y in candidates]
            )
            candidates = candidates[keep]

        if exclude_coords is not None and exclusion_radius > 0 and len(candidates):
            from sdmbench.splits.spatial import points_within_distance

            too_close = points_within_distance(
                candidates,
                np.asarray(exclude_coords, dtype=float),
                exclusion_radius,
                geographic=False,
            )
            candidates = candidates[~too_close]

        for x, y in candidates[: n - len(chosen_x)]:
            chosen_x.append(float(x))
            chosen_y.append(float(y))

    coords = np.column_stack([chosen_x, chosen_y])
    values = extract_at_points(stack, coords)
    values.insert(0, "y", coords[:, 1])
    values.insert(0, "x", coords[:, 0])
    return values


def predict_to_raster(
    model: Any,
    stack: RasterStack,
    output_path: str | Path,
    *,
    recipe: Any = None,
    block_size: int = 512,
    dtype: str = "float32",
) -> Path:
    """Predict suitability for every raster cell and write a GeoTIFF.

    Processed in windows so a continental raster does not have to fit in RAM.

    Parameters
    ----------
    model:
        Anything with ``predict_proba``.
    recipe:
        A fitted :class:`~sdmbench.preprocessing.SdmRecipe`, applied per window
        with the parameters estimated at training time.
    """
    rasterio = require("rasterio")
    from rasterio.windows import Window

    output_path = Path(output_path)
    with rasterio.open(stack.paths[0]) as reference:
        profile = reference.profile.copy()
        width, height = reference.width, reference.height
        nodata_value = reference.nodata
    profile.update(count=1, dtype=dtype, nodata=np.nan, compress="lzw")

    sources = [rasterio.open(p) for p in stack.paths]
    try:
        with rasterio.open(output_path, "w", **profile) as dst:
            for row_off in range(0, height, block_size):
                for col_off in range(0, width, block_size):
                    window = Window(
                        col_off,
                        row_off,
                        min(block_size, width - col_off),
                        min(block_size, height - row_off),
                    )
                    columns: dict[str, np.ndarray] = {}
                    for name, src in zip(stack.names, sources):
                        block = src.read(1, window=window).astype(float)
                        if src.nodata is not None:
                            block = np.where(block == src.nodata, np.nan, block)
                        columns[name] = block.ravel()

                    frame = pd.DataFrame(columns)
                    valid = frame.notna().all(axis=1).to_numpy()
                    predictions = np.full(len(frame), np.nan, dtype=float)
                    if valid.any():
                        features = frame.loc[valid]
                        if recipe is not None:
                            features = recipe.transform(features)
                        proba = model.predict_proba(features)
                        proba = np.asarray(proba, dtype=float)
                        predictions[valid] = proba[:, 1] if proba.ndim == 2 else proba.ravel()

                    dst.write(
                        predictions.reshape(window.height, window.width).astype(dtype),
                        1,
                        window=window,
                    )
    finally:
        for src in sources:
            src.close()
    return output_path


class RasterDataset:
    """Occurrence records plus a raster predictor stack.

    Parameters
    ----------
    occurrences:
        CSV or DataFrame with coordinate columns and, optionally, a presence
        column. Rows without a presence column are treated as presences.
    stack:
        The predictor rasters.
    n_background:
        Background points to draw when the occurrences contain no absences.
    """

    name = "raster"

    def __init__(
        self,
        occurrences: str | Path | pd.DataFrame,
        stack: RasterStack,
        *,
        longitude: str = "longitude",
        latitude: str = "latitude",
        target: str | None = None,
        n_background: int = 10000,
        species_id: str = "species",
        region: str = "user",
        seed: int = 32639,
    ) -> None:
        self.occurrences = (
            occurrences.copy()
            if isinstance(occurrences, pd.DataFrame)
            else pd.read_csv(occurrences)
        )
        self.stack = stack
        self.longitude = longitude
        self.latitude = latitude
        self.target = target
        self.n_background = n_background
        self.species_id = species_id
        self.region = region
        self.seed = seed

        for col in (longitude, latitude):
            if col not in self.occurrences.columns:
                raise DataError(
                    f"coordinate column {col!r} not found; columns: "
                    f"{list(self.occurrences.columns)}"
                )

    def build_training_frame(self) -> pd.DataFrame:
        """Extract covariates at occurrences and add background points."""
        profile = self.stack.profile()
        coords = self.occurrences.loc[:, [self.longitude, self.latitude]].to_numpy(dtype=float)
        values = extract_at_points(self.stack, coords)

        frame = pd.DataFrame(
            {
                "region": self.region,
                "group": None,
                "siteid": [f"occ{i}" for i in range(len(coords))],
                "spid": self.species_id,
                "x": coords[:, 0],
                "y": coords[:, 1],
                "occ": (
                    pd.to_numeric(self.occurrences[self.target], errors="coerce")
                    .fillna(1)
                    .astype(int)
                    .to_numpy()
                    if self.target and self.target in self.occurrences.columns
                    else 1
                ),
            }
        )
        frame = pd.concat([frame, values], axis=1)

        if int((frame["occ"] == 0).sum()) == 0:
            background = sample_background(self.stack, self.n_background, seed=self.seed)
            bg_frame = pd.DataFrame(
                {
                    "region": self.region,
                    "group": None,
                    "siteid": [f"bg{i}" for i in range(len(background))],
                    "spid": self.species_id,
                    "x": background["x"].to_numpy(),
                    "y": background["y"].to_numpy(),
                    "occ": 0,
                }
            )
            for name in self.stack.names:
                bg_frame[name] = background[name].to_numpy()
            frame = pd.concat([frame, bg_frame], ignore_index=True)

        frame.attrs["crs"] = profile["crs"]
        return frame

    def to_task(self, *, test_size: float = 0.2, benchmark_id: str = "raster") -> SpeciesTask:
        """Build a task, holding out a spatially blocked test partition."""
        from sdmbench.splits.spatial import SpatialBlockCV

        frame = self.build_training_frame()
        crs = frame.attrs.get("crs", "")
        geographic = "4326" in str(crs) or "longlat" in str(crs).lower()

        coords = frame.loc[:, ["x", "y"]].to_numpy(dtype=float)
        n_folds = max(2, int(round(1 / test_size)))
        train_idx, test_idx = SpatialBlockCV(n_folds=n_folds, seed=self.seed).first_fold(
            coords, geographic=geographic
        )

        return SpeciesTask(
            region=self.region,
            species_id=self.species_id,
            train=frame.iloc[train_idx].reset_index(drop=True),
            test=frame.iloc[test_idx].reset_index(drop=True),
            predictors=list(self.stack.names),
            crs=str(crs),
            geographic=geographic,
            benchmark_id=benchmark_id,
            metadata={
                "dataset": self.name,
                "rasters": [str(p) for p in self.stack.paths],
                "test_data": "spatial_holdout",
                "note": (
                    "Test data is a spatially blocked holdout from the same records, "
                    "not an independent survey."
                ),
            },
        )

    def iter_tasks(self, **kwargs: Any) -> Iterable[SpeciesTask]:
        yield self.to_task(**kwargs)

    def content_hash(self) -> str:
        from sdmbench.reproducibility.hashes import hash_dataframe, hash_file, hash_json

        return hash_json(
            {
                "occurrences": hash_dataframe(self.occurrences),
                "rasters": {p.name: hash_file(p) for p in self.stack.paths},
            }
        )
