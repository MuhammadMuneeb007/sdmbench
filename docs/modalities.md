# Modalities

A **modality** is one kind of ecological information. sdmbench keeps them typed
and separate all the way to the fusion step rather than flattening everything
into one wide table.

---

## Why not just concatenate?

The shortcut discards four things the central research question needs:

- **Semantics** — climate at a point and a 25 km terrain patch are different
  kinds of evidence and belong in different encoders.
- **Scale** — a modality has a resolution and a support. Lose it and "at what
  scale does terrain matter?" is unanswerable.
- **Provenance** — source, licence, CRS, download time, checksum. Without them
  the run is not reproducible.
- **Ablation** — "does terrain add anything beyond climate?" is only a
  well-posed question if terrain is a nameable, removable unit.

---

## The registry

| Modality | Kind | Typical variables | Candidate providers |
|---|---|---|---|
| `climate` | tabular | bio01, bio04, bio05, bio12, bio15 | WorldClim, CHELSA, pastclim, **disdat** |
| `terrain` | raster | elevation, slope, aspect, rugosity, TRI, TPI | Copernicus DEM, SRTM, local rasters |
| `soil` | tabular | pH, organic carbon, clay, sand, silt, bulk density | SoilGrids |
| `vegetation` | sequence | NDVI, EVI, LAI, NPP | MODIS |
| `landcover` | categorical | class, or proportions in a buffer | ESA WorldCover |
| `remote_sensing` | raster | Sentinel-2 bands, Sentinel-1 VV/VH | Sentinel, Landsat |
| `hydrology` | tabular | distance to water, flow accumulation | HydroSHEDS, JRC surface water |
| `human_pressure` | tabular | population, roads, night lights, footprint | GHSL, Human Footprint |
| `species_traits` | categorical | body mass, dispersal, trophic level, taxonomy | user table |
| `community` | tabular | co-occurrence vector, community embedding | derived |
| `temporal_climate` | sequence | annual series, palaeoclimate, projections | pastclim, CHELSA-TraCE, CMIP6 |
| `earth_observation` | embedding | pretrained location vectors | AlphaEarth, TerraMind, Prithvi |

**Status:** only `climate` and `soil` (via `disdat` columns) and user-supplied
rasters are wired to working providers today. The rest are **declared but not
downloadable** — see [`public_datasets.md`](public_datasets.md). Declaring a
modality does not download anything; it states that the modality exists, what
shape it takes, and which providers could supply it.

---

## Structural kinds

What an encoder dispatches on:

| Kind | Shape | Encoder family |
|---|---|---|
| `TABULAR` | (n, f) | raw, standardised, PCA, autoencoder |
| `RASTER` | (n, c, h, w) or (n, s, c, h, w) | patch statistics, CNN, multi-scale CNN, ViT |
| `SEQUENCE` | (n, t, f) | temporal summary, LSTM, temporal CNN, transformer |
| `EMBEDDING` | (n, d) | identity (frozen foundation output) |
| `CATEGORICAL` | (n, f) | one-hot, categorical embedding |
| `GRAPH` | nodes + edges | graph builders |

`Methodology.validate()` uses these to reject `cnn` on a tabular modality
*before* compute is spent, with a message naming the compatible encoders.

---

## Provenance

Every modality carries `ModalityMetadata`, recorded in the run manifest and
hashed into the results:

```python
ModalityMetadata.now(
    name="climate",
    kind=ModalityKind.TABULAR,
    source="WorldClim v2.1",
    provider="worldclim",
    variables=["bio01", "bio04", "bio12"],
    units={"bio01": "degC", "bio12": "mm"},
    crs="EPSG:4326",
    spatial_resolution_m=1000.0,
    spatial_support_m=None,          # None = point sample
    temporal_coverage="1970-2000",
    license="CC BY-SA 4.0",
    citation="Fick & Hijmans 2017",
    checksum="sha256:...",
)
```

`spatial_support_m` deserves attention: `None` means a point sample, a number
means a patch of that edge length. It is what distinguishes "elevation here"
from "the terrain within 5 km", which are different pieces of evidence.

---

## `EcoDataset`

The multimodal generalisation of `SpeciesTask`:

```python
dataset = EcoDataset(
    observations=frame,
    target=y,
    coordinates=xy,           # for SPLITS and TOPOLOGY, never features
    modalities={"climate": ..., "terrain": ...},
)

dataset.without("terrain")        # the ablation primitive
dataset.only("climate", "soil")
dataset.subset(train_idx)         # row subset across every modality at once
dataset.concat_tabular()          # escape hatch: prefixed flat frame
```

Coordinates live outside the modality dict deliberately, so geography can build
splits and graph topology without ever being mistaken for a predictor.

`concat_tabular` prefixes columns with the modality (`climate__bio01`), so
provenance survives the flattening and downstream attribution can be grouped
back by modality.

---

## Adding a modality

Two separate things:

**1. Declare the modality type** (what it is):

```python
from sdmbench.core.modality import ModalityDefinition, ModalityKind, STANDARD_MODALITIES

STANDARD_MODALITIES["fire"] = ModalityDefinition(
    "fire",
    ModalityKind.SEQUENCE,
    "Fire history: frequency, time since last burn, severity.",
    typical_variables=("fire_frequency", "years_since_burn"),
    providers=("modis_burned_area",),
    default_representation="temporal-summary",
)
```

**2. Supply the data** — either as columns in a user CSV mapped through a schema
file, or by writing a provider. See
[`adding_a_dataset.md`](adding_a_dataset.md).

For a user table, the schema maps columns to modalities:

```yaml
target: presence
coordinates: {longitude: longitude, latitude: latitude}
modalities:
  climate:    {columns: [bio04, bio05, bio08]}
  vegetation: {columns: [lai]}
  terrain:    {columns: [rugosity]}
```

`sdmbench inspect data.csv` proposes such a schema — with a confidence and a
reason per column — but never acts on it. Guessing a column's role wrong does
not produce an error; it produces a plausible number.
