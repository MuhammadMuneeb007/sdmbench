# Adding a dataset

Three routes, in increasing order of effort.

---

## 1. A user table (no code)

```bash
sdmbench inspect my_species.csv
```

Proposes a schema with a confidence and a reason per column — and acts on
nothing. Guessing a column's role wrong does not produce an error; it produces a
plausible number.

```
column      inferred role             conf   reason
----------  ------------------------  -----  ---------------------------------
presence    target                    0.95   binary values and a target-like name
longitude   longitude                 0.90   longitude-like name, values within ±180
bio04       continuous_predictor      0.80   continuous numeric; name suggests climate
vegetation  categorical_predictor     0.70   non-numeric with 3 levels
record_id   identifier                0.80   every value is distinct
```

Then either:

```bash
sdmbench benchmark-csv my_species.csv --target presence --models standard
```

or write the schema for modality-aware runs:

```yaml
target: presence
coordinates: {longitude: longitude, latitude: latitude}
modalities:
  climate:    {columns: [bio04, bio05, bio08]}
  vegetation: {columns: [lai]}
  terrain:    {columns: [rugosity]}
```

**Coordinates are not predictors** unless you pass `--include-coordinates`.

---

## 2. Occurrences plus a raster stack

```
occurrences.csv
climate/*.tif
terrain/*.tif
```

```python
from sdmbench.data.rasters import RasterDataset, RasterStack

stack = RasterStack.from_directory("terrain/")
print(stack.check_alignment())     # do this FIRST

dataset = RasterDataset(
    "occurrences.csv", stack,
    longitude="decimalLongitude", latitude="decimalLatitude",
    n_background=10_000,
)
task = dataset.to_task()
```

`check_alignment()` matters: misaligned rasters silently produce covariates
sampled from *different locations*. That is a wrong benchmark, not an error.

Occurrence coordinates are reprojected to the raster CRS before sampling, not
the reverse — resampling a raster to match points would change the pixel values
being extracted.

> Status: written against the documented rasterio/pyproj APIs, **never
> executed**. Treat as reviewed-but-untested.

---

## 3. A new data provider (code)

For a public dataset others should be able to fetch.

```python
from sdmbench.core.registry import Registry, Spec

@DATA_PROVIDERS.register(
    "worldclim",
    spec=Spec(produces="raster", requires=("rasterio",), extra="raster"),
)
class WorldClimProvider:
    name = "worldclim"
    modalities = ("climate",)
    requires_auth = False
    license = "CC BY-SA 4.0"

    def availability(self) -> dict: ...
    def download(self, dest, **kwargs): ...
    def prepare(self, coords, crs) -> ModalityData: ...
    def describe(self) -> dict: ...
```

### Rules

**Never embed credentials.** If a source needs authentication, detect it and
raise `LicenseError` with setup instructions — which becomes a `SKIPPED_LICENSE`
row. sdmbench never bypasses a gate. See `AlphaEarthProvider` for the pattern.

**Never overwrite source data.** Raw payloads land in `raw/` and are immutable;
every derived artefact is written beside them.

**Always checksum.** Record a sha256 per file in `checksums.json`.

**Fill in `ModalityMetadata` completely.** Source, licence, CRS, resolution,
temporal coverage, download time, checksum. Without them the run is not
reproducible.

**Do not write a downloader for data you cannot legally redistribute or
publicly retrieve.** Where availability is unclear (GeoPlant, for example),
`references/software.yaml` records `access: UNVERIFIED` and no downloader is
written until it is established.

---

## The `disdat` pattern

Worth reading as a worked example, because it solves a common problem: the data
ships as `.rds` inside an R package and Python cannot read it.

```
sdmbench data fetch disdat
        │
        ├─ R bridge: rbridge/scripts/fetch_disdat.R
        │    reads the INSTALLED package, exports long-format CSV + metadata.json
        │    (never downloads, installs or modifies anything)
        │
        ├─ Python: convert CSV -> Parquet with stable dtypes
        │
        └─ checksums.json: sha256 per file
```

Cache layout:

```
<cache>/datasets/disdat/1.1-0/
    raw/                    untouched R export
    AWT_po.parquet  AWT_bg.parquet  AWT_pa.parquet  AWT_env.parquet
    ...
    metadata.json           regions, groups, species, predictors, CRS
    checksums.json
```

The wide presence-absence table (one column per species) is reshaped to long
format during export, so a species task becomes a simple filter on `spid`.

A second acquisition route (`--source osf`) exists for users without R. The
`disdat` vignette notes the two data releases are "similar", not identical — so
a benchmark prepared that way records a **different `dataset_hash`** and is not
automatically comparable with an R-prepared one.

---

## The internal schema

Whatever the source, everything normalises to:

| Column | Meaning |
|---|---|
| `region` | region code |
| `group` | taxonomic/survey group, or `None` |
| `siteid` | site identifier from the source |
| `spid` | species identifier |
| `x`, `y` | coordinates **in the region's own CRS** |
| `occ` | 1 = presence, 0 = background (train) or absence (test) |
| *predictors…* | environmental covariates |

Coordinates stay in the source CRS rather than being reprojected to a common
one, because the split rule needs to know the units — see
[`validation.md`](validation.md).

---

## Checklist

- [ ] Raw data cached immutably, with checksums
- [ ] `ModalityMetadata` complete (source, licence, CRS, resolution, checksum)
- [ ] Authentication detected and reported, never bypassed
- [ ] Licence recorded in `references/software.yaml`
- [ ] `content_hash()` implemented, so results are keyed to the exact data
- [ ] Tests use a tiny synthetic fixture, not a download
