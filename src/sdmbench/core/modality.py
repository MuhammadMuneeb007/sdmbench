"""Ecological modalities -- typed information sources.

A **modality** is one kind of ecological information: climate, terrain, soil,
vegetation, land cover, remote sensing, hydrology, human pressure, species
traits, community context, or a learned geospatial embedding.

Why this type exists
--------------------
The tempting shortcut is to concatenate everything into one wide DataFrame and
hand it to a classifier. That throws away exactly the information the central
research question needs:

* **Semantics.** Climate at a point and a 25 km terrain patch are different
  kinds of evidence and belong in different encoders.
* **Scale.** A modality has a spatial resolution and a support (point vs patch).
  Losing it makes the scale-ablation question unanswerable.
* **Provenance.** Source, licence, CRS, download time and checksum have to
  travel with the data or the run is not reproducible.
* **Ablation.** "Does terrain add anything beyond climate?" is only a
  well-posed question if terrain is a nameable, removable unit.

So :class:`ModalityData` keeps each source separate and typed all the way to
the fusion step, and :class:`ModalityMetadata` carries its provenance.
"""

from __future__ import annotations

import datetime as _dt
import enum
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from sdmbench.core.registry import Registry, Spec
from sdmbench.exceptions import DataError

__all__ = [
    "ModalityKind",
    "ModalityMetadata",
    "ModalityData",
    "EcoDataset",
    "MODALITIES",
    "ModalityDefinition",
    "STANDARD_MODALITIES",
]


class ModalityKind(str, enum.Enum):
    """The structural shape of a modality's data.

    This is what an encoder dispatches on: a CNN needs ``RASTER``, an LSTM
    needs ``SEQUENCE``, a gradient-boosted tree needs ``TABULAR``.
    """

    #: ``(n_samples, n_features)`` -- one row per observation.
    TABULAR = "tabular"
    #: ``(n_samples, channels, height, width)`` -- a patch per observation.
    RASTER = "raster"
    #: ``(n_samples, timesteps, n_features)``.
    SEQUENCE = "sequence"
    #: ``(n_samples, dim)`` from a pretrained model; usually frozen.
    EMBEDDING = "embedding"
    #: Node features plus an edge index.
    GRAPH = "graph"
    #: Categorical/text attributes, e.g. taxonomy.
    CATEGORICAL = "categorical"


@dataclass
class ModalityMetadata:
    """Provenance and physical description of one modality.

    Recorded in the run manifest and hashed into the result rows. A modality
    without provenance cannot be part of a reproducible benchmark.
    """

    name: str
    kind: ModalityKind = ModalityKind.TABULAR
    source: str = ""
    provider: str = ""
    variables: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    crs: str | None = None
    spatial_resolution_m: float | None = None
    #: Patch edge length in metres, for raster modalities. ``None`` = point.
    spatial_support_m: float | None = None
    temporal_resolution: str | None = None
    temporal_coverage: str | None = None
    coverage: str = ""
    license: str = "unknown"
    citation: str = ""
    url: str = ""
    downloaded_at: str = ""
    checksum: str = ""
    missingness: float | None = None
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(cls, **kwargs: Any) -> ModalityMetadata:
        kwargs.setdefault(
            "downloaded_at",
            _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        )
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "source": self.source,
            "provider": self.provider,
            "variables": list(self.variables),
            "units": dict(self.units),
            "crs": self.crs,
            "spatial_resolution_m": self.spatial_resolution_m,
            "spatial_support_m": self.spatial_support_m,
            "temporal_resolution": self.temporal_resolution,
            "temporal_coverage": self.temporal_coverage,
            "coverage": self.coverage,
            "license": self.license,
            "citation": self.citation,
            "url": self.url,
            "downloaded_at": self.downloaded_at,
            "checksum": self.checksum,
            "missingness": self.missingness,
            "notes": self.notes,
            **self.extras,
        }


@dataclass
class ModalityData:
    """One modality's values for a set of observations.

    ``values`` shape depends on :attr:`ModalityMetadata.kind`:

    ==============  ==========================================
    ``TABULAR``     ``(n, f)`` DataFrame or array
    ``RASTER``      ``(n, c, h, w)`` array, or ``(n, s, c, h, w)`` multi-scale
    ``SEQUENCE``    ``(n, t, f)`` array
    ``EMBEDDING``   ``(n, d)`` array
    ``CATEGORICAL`` ``(n, f)`` DataFrame of categories
    ==============  ==========================================

    Multi-scale raster modalities carry one entry per scale in :attr:`scales`,
    which is what makes the "at what scale?" ablation expressible.
    """

    metadata: ModalityMetadata
    values: Any
    #: Scale labels for a multi-scale raster modality, e.g. ``["1km", "5km"]``.
    scales: list[str] = field(default_factory=list)
    #: Column names for tabular/categorical modalities.
    feature_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.metadata.kind in {ModalityKind.TABULAR, ModalityKind.CATEGORICAL}:
            if isinstance(self.values, pd.DataFrame) and not self.feature_names:
                self.feature_names = list(self.values.columns)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def kind(self) -> ModalityKind:
        return self.metadata.kind

    @property
    def n_samples(self) -> int:
        return int(len(self.values))

    @property
    def is_multiscale(self) -> bool:
        return len(self.scales) > 1

    def as_frame(self) -> pd.DataFrame:
        """Tabular view. Raises for modalities that are not tabular."""
        if isinstance(self.values, pd.DataFrame):
            return self.values
        if self.kind in {ModalityKind.TABULAR, ModalityKind.EMBEDDING}:
            array = np.asarray(self.values)
            names = self.feature_names or [
                f"{self.name}_{i}" for i in range(array.shape[1] if array.ndim > 1 else 1)
            ]
            return pd.DataFrame(array.reshape(len(array), -1), columns=names)
        raise DataError(
            f"modality {self.name!r} has kind {self.kind.value!r} and has no tabular view; "
            "encode it first with a representation encoder"
        )

    def as_array(self) -> np.ndarray:
        if isinstance(self.values, pd.DataFrame):
            return self.values.to_numpy()
        return np.asarray(self.values)

    def subset(self, indices: Sequence[int] | np.ndarray) -> ModalityData:
        """Row subset, preserving metadata. Used by every split."""
        idx = np.asarray(indices)
        values = (
            self.values.iloc[idx] if hasattr(self.values, "iloc") else np.asarray(self.values)[idx]
        )
        return ModalityData(
            metadata=self.metadata,
            values=values,
            scales=list(self.scales),
            feature_names=list(self.feature_names),
        )

    def describe(self) -> dict[str, Any]:
        array_shape = getattr(self.values, "shape", (self.n_samples,))
        return {
            **self.metadata.to_dict(),
            "n_samples": self.n_samples,
            "shape": tuple(array_shape),
            "scales": list(self.scales),
            "n_features": len(self.feature_names) or None,
        }


@dataclass
class EcoDataset:
    """Observations described by several modalities.

    The multimodal generalisation of
    :class:`~sdmbench.data.base.SpeciesTask`. Each modality keeps its own type,
    scale and provenance; ``coordinates`` and ``target`` are shared.

    Coordinates are stored here, deliberately outside the modality dict, so
    that geography can build splits and graph topology without ever being
    mistaken for a predictor.
    """

    observations: pd.DataFrame
    target: np.ndarray
    coordinates: np.ndarray
    modalities: dict[str, ModalityData] = field(default_factory=dict)
    species_id: str = ""
    region: str = ""
    group: str | None = None
    crs: str | None = None
    geographic: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.observations)
        if len(self.target) != n:
            raise DataError(f"target has {len(self.target)} rows, observations have {n}")
        if len(self.coordinates) != n:
            raise DataError(f"coordinates have {len(self.coordinates)} rows, observations have {n}")
        for name, modality in self.modalities.items():
            if modality.n_samples != n:
                raise DataError(
                    f"modality {name!r} has {modality.n_samples} rows, observations have {n}"
                )

    @property
    def n_samples(self) -> int:
        return len(self.observations)

    @property
    def modality_names(self) -> list[str]:
        return sorted(self.modalities)

    def add_modality(self, modality: ModalityData) -> EcoDataset:
        if modality.n_samples != self.n_samples:
            raise DataError(
                f"modality {modality.name!r} has {modality.n_samples} rows, "
                f"dataset has {self.n_samples}"
            )
        self.modalities[modality.name] = modality
        return self

    def without(self, *names: str) -> EcoDataset:
        """A copy with the named modalities removed -- the ablation primitive."""
        dropped = set(names)
        unknown = dropped - set(self.modalities)
        if unknown:
            raise DataError(f"cannot drop unknown modality/modalities: {sorted(unknown)}")
        return EcoDataset(
            observations=self.observations,
            target=self.target,
            coordinates=self.coordinates,
            modalities={k: v for k, v in self.modalities.items() if k not in dropped},
            species_id=self.species_id,
            region=self.region,
            group=self.group,
            crs=self.crs,
            geographic=self.geographic,
            metadata={**self.metadata, "ablated_modalities": sorted(dropped)},
        )

    def only(self, *names: str) -> EcoDataset:
        """A copy keeping only the named modalities."""
        keep = set(names)
        unknown = keep - set(self.modalities)
        if unknown:
            raise DataError(f"unknown modality/modalities: {sorted(unknown)}")
        return EcoDataset(
            observations=self.observations,
            target=self.target,
            coordinates=self.coordinates,
            modalities={k: v for k, v in self.modalities.items() if k in keep},
            species_id=self.species_id,
            region=self.region,
            group=self.group,
            crs=self.crs,
            geographic=self.geographic,
            metadata={**self.metadata, "modality_subset": sorted(keep)},
        )

    def subset(self, indices: Sequence[int] | np.ndarray) -> EcoDataset:
        """Row subset across every modality at once."""
        idx = np.asarray(indices)
        return EcoDataset(
            observations=self.observations.iloc[idx].reset_index(drop=True),
            target=np.asarray(self.target)[idx],
            coordinates=np.asarray(self.coordinates)[idx],
            modalities={k: v.subset(idx) for k, v in self.modalities.items()},
            species_id=self.species_id,
            region=self.region,
            group=self.group,
            crs=self.crs,
            geographic=self.geographic,
            metadata=dict(self.metadata),
        )

    def concat_tabular(self) -> pd.DataFrame:
        """Flatten every tabular/embedding modality into one frame.

        The escape hatch for models that only accept a matrix. Column names are
        prefixed with the modality so provenance survives the flattening.
        """
        frames: list[pd.DataFrame] = []
        for name in self.modality_names:
            modality = self.modalities[name]
            if modality.kind not in {
                ModalityKind.TABULAR,
                ModalityKind.EMBEDDING,
                ModalityKind.CATEGORICAL,
            }:
                continue
            frame = modality.as_frame().copy()
            frame.columns = [f"{name}__{c}" for c in frame.columns]
            frames.append(frame.reset_index(drop=True))
        if not frames:
            raise DataError("no tabular-compatible modalities to concatenate")
        return pd.concat(frames, axis=1)

    def describe(self) -> dict[str, Any]:
        return {
            "species_id": self.species_id,
            "region": self.region,
            "group": self.group,
            "n_samples": self.n_samples,
            "n_presences": int(np.sum(np.asarray(self.target) == 1)),
            "n_background": int(np.sum(np.asarray(self.target) == 0)),
            "crs": self.crs,
            "geographic": self.geographic,
            "modalities": {k: v.describe() for k, v in self.modalities.items()},
        }

    def __iter__(self) -> Iterator[ModalityData]:
        return iter(self.modalities.values())


# ---------------------------------------------------------------------------
# The modality registry
# ---------------------------------------------------------------------------

MODALITIES: Registry[Any] = Registry("modality")


@dataclass
class ModalityDefinition:
    """A known ecological modality and where its data usually comes from.

    Registering a definition does not download anything -- it declares that the
    modality exists, what shape it takes, and which providers can supply it.
    """

    name: str
    kind: ModalityKind
    description: str
    typical_variables: tuple[str, ...] = ()
    providers: tuple[str, ...] = ()
    default_representation: str = "raw-tabular"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "description": self.description,
            "typical_variables": list(self.typical_variables),
            "providers": list(self.providers),
            "default_representation": self.default_representation,
            "notes": self.notes,
        }


#: The modalities sdmbench understands out of the box. Providers listed here
#: are *candidates* -- see :mod:`sdmbench.data.providers` for which are
#: actually implemented, which are scaffolds, and which need credentials.
STANDARD_MODALITIES: dict[str, ModalityDefinition] = {
    d.name: d
    for d in (
        ModalityDefinition(
            "climate",
            ModalityKind.TABULAR,
            "Temperature, precipitation, seasonality and derived bioclimatic variables.",
            typical_variables=("bio01", "bio04", "bio05", "bio12", "bio15"),
            providers=("worldclim", "chelsa", "pastclim", "disdat"),
        ),
        ModalityDefinition(
            "terrain",
            ModalityKind.RASTER,
            "Elevation and its derivatives: slope, aspect, ruggedness, curvature.",
            typical_variables=("elevation", "slope", "aspect", "rugosity", "tri", "tpi"),
            providers=("copernicus_dem", "srtm", "local_raster"),
            default_representation="multiscale-cnn",
            notes="The clearest case for multi-scale encoding: species respond to "
                  "topography at several scales at once.",
        ),
        ModalityDefinition(
            "soil",
            ModalityKind.TABULAR,
            "Edaphic properties: pH, organic carbon, texture, bulk density.",
            typical_variables=("phh2o", "soc", "clay", "sand", "silt", "bdod"),
            providers=("soilgrids", "local_raster"),
        ),
        ModalityDefinition(
            "vegetation",
            ModalityKind.SEQUENCE,
            "Vegetation structure and productivity, often as a seasonal series.",
            typical_variables=("ndvi", "evi", "lai", "npp"),
            providers=("modis", "local_raster"),
            default_representation="temporal-summary",
        ),
        ModalityDefinition(
            "landcover",
            ModalityKind.CATEGORICAL,
            "Discrete land cover classes, or class proportions within a buffer.",
            typical_variables=("landcover_class",),
            providers=("esa_worldcover", "local_raster"),
            default_representation="categorical",
        ),
        ModalityDefinition(
            "remote_sensing",
            ModalityKind.RASTER,
            "Multispectral or radar imagery and derived indices.",
            typical_variables=("B2", "B3", "B4", "B8", "VV", "VH"),
            providers=("sentinel2", "landsat", "local_raster"),
            default_representation="cnn",
        ),
        ModalityDefinition(
            "hydrology",
            ModalityKind.TABULAR,
            "Water availability and proximity: rivers, wetlands, flow, distance to water.",
            typical_variables=("distance_to_water", "flow_accumulation", "water_occurrence"),
            providers=("hydrosheds", "jrc_surface_water", "local_raster"),
        ),
        ModalityDefinition(
            "human_pressure",
            ModalityKind.TABULAR,
            "Anthropogenic influence: roads, population, night lights, built-up area.",
            typical_variables=("population_density", "road_density", "nightlights", "footprint"),
            providers=("ghsl", "human_footprint", "local_raster"),
        ),
        ModalityDefinition(
            "species_traits",
            ModalityKind.CATEGORICAL,
            "Attributes of the species itself: body size, dispersal, trophic level, taxonomy.",
            typical_variables=("body_mass", "dispersal_distance", "trophic_level", "family"),
            providers=("user_table",),
            default_representation="categorical",
            notes="Only informative in a multi-species setting, where it lets one model "
                  "share strength across taxa.",
        ),
        ModalityDefinition(
            "community",
            ModalityKind.TABULAR,
            "Co-occurring species: co-occurrence vectors or a learned community embedding.",
            providers=("derived",),
        ),
        ModalityDefinition(
            "temporal_climate",
            ModalityKind.SEQUENCE,
            "Climate through time: annual series, palaeoclimate, or future projections.",
            providers=("pastclim", "chelsa_trace", "cmip6"),
            default_representation="temporal-transformer",
        ),
        ModalityDefinition(
            "earth_observation",
            ModalityKind.EMBEDDING,
            "Pretrained geospatial foundation-model embeddings for a location.",
            providers=("alphaearth", "terramind", "prithvi"),
            default_representation="pretrained-embedding",
            notes="Usually frozen. Most require credentials or a cloud workflow; "
                  "sdmbench never bypasses authentication.",
        ),
    )
}
