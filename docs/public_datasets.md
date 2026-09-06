# Public datasets

What sdmbench can fetch today, what is scaffolded, and what is not wired at all.

**Status is stated honestly.** A "planned" provider is a stub with no
downloader; do not read the modality registry as a list of things that work.

---

## Working today

### `disdat` — the primary benchmark

```bash
sdmbench data fetch disdat
```

226 anonymised species across six regions (Elith et al. 2020). Presence-only
training records, 10,000 background points per region, and **independent
presence-absence survey data** for evaluation — which is what makes it a serious
benchmark rather than a cross-validation exercise.

| | |
|---|---|
| Source | R package (CRAN), or an OSF CSV archive |
| Licence | GPL (≥ 3) |
| Size | small — ships inside the R package |
| Needs | R with `disdat` installed (or `--source osf`) |
| Regions | AWT, CAN, NSW, NZ, SA, SWI |

This **is** the NCEAS SDM benchmark data (project 4980), already integrated.

The two release routes are "similar", not byte-identical, so an OSF-prepared
benchmark records a different `dataset_hash` and is not automatically comparable
with an R-prepared one.

### User CSV and user rasters

See [`adding_a_dataset.md`](adding_a_dataset.md). Raster support is written
against the documented rasterio/pyproj APIs but has **never been executed**.

---

## Scaffolded — code written, never run, needs setup

| Provider | Modality | Blocker |
|---|---|---|
| `alphaearth` | earth_observation | **your** Google Earth Engine account |
| `terramind` | earth_observation | `terratorch` install; API varies by release |
| `prithvi` | earth_observation | `transformers`; band layout is model-specific |
| `cached` | earth_observation | **none — works today** |

### AlphaEarth

Published as an Earth Engine ImageCollection of annual 64-band embeddings.
Retrieval is a cloud query, not a file download.

```bash
earthengine authenticate
export SDMBENCH_EE_PROJECT=your-project-id
```

sdmbench never embeds, requests or stores credentials. Without an authenticated
client it raises `LicenseError` → `SKIPPED_LICENSE`.

**The practical path for 226 species:** compute embeddings once, cache them, and
load via the `cached` provider. That is the only route with an executable path
today.

```yaml
earth_observation:
  provider: cached
  representation: pretrained-embedding
  options: {path: embeddings/alphaearth_disdat.npz}
```

---

## Planned — declared, no downloader

These appear in the modality registry as *candidates*. Nothing fetches them yet.

| Provider | Modality | Access |
|---|---|---|
| WorldClim, CHELSA | climate | public HTTP |
| pastclim, CHELSA-TraCE | temporal_climate | R package / public HTTP |
| Copernicus DEM, SRTM | terrain | public (AWS open data) |
| SoilGrids | soil | public WCS / STAC |
| MODIS | vegetation | NASA Earthdata (free account) |
| ESA WorldCover | landcover | public |
| Sentinel-1/2, Landsat | remote_sensing | public STAC |
| HydroSHEDS, JRC surface water | hydrology | public |
| GHSL, Human Footprint | human_pressure | public |

Recommended order, by value per unit of work:

1. **Copernicus DEM** — unlocks the terrain multi-scale question, which is the
   most interesting untested axis, and is a clean public download.
2. **SoilGrids** — straightforward, adds a genuinely different modality.
3. **WorldClim/CHELSA** — lets climate be re-derived rather than taken from
   `disdat`, enabling a resolution ablation.
4. **MODIS** — the first real temporal modality; unlocks the temporal encoders.

---

## Not wired, and why

### GeoPlant

`access: UNVERIFIED` in `references/software.yaml`. Licensing and public
availability have not been established.

**No downloader will be written until that is confirmed.** Fabricating an
automated fetch for data that cannot legally or publicly be retrieved would be
worse than having nothing.

### The leopard dataset (Leedham et al. 2025)

The recipe structure is implemented; the **data loader is not**.
`sdmbench reproduce leopard-leedham-2025` raises a clear error saying what to
implement.

It is kept for two reasons: it proves the `PaperBenchmark` abstraction
generalises beyond `disdat` (different data, different split scheme, different
primary metric), and our existing leopard scripts should be **refactored into**
it rather than copied in — the reusable parts (pseudo-absence sampling, spatial
folds, the Boyce index) already live in `sdmbench.splits` and
`sdmbench.metrics`.

---

## Cache layout

```
~/.cache/sdmbench/           # or %LOCALAPPDATA%\sdmbench\Cache on Windows
    datasets/disdat/1.1-0/
        raw/                 untouched source export (never overwritten)
        *.parquet            standardised tables
        metadata.json
        checksums.json       sha256 per file
    embeddings/<provider>/   cached foundation embeddings
    models/tabpfn_sdm/       downloaded checkpoints (hash-verified)
    upstream/<benchmark>/    cloned repositories, pinned by commit
    runs/<run_id>/           results, manifests, per-job checkpoints
```

Override with `$SDMBENCH_CACHE`. Source data is never modified.

---

## Licensing

sdmbench downloads artefacts on your behalf and **never redistributes them**.
Each carries its own licence:

| Artefact | Licence |
|---|---|
| sdmbench source | MIT |
| `disdat` data | GPL (≥ 3) |
| TabPFN weights (base and finetuned) | Prior Labs License v1.1 |
| R packages | each its own |
| AlphaEarth | Google Earth Engine dataset terms |

`maxent.jar` cannot be installed by sdmbench at all — its licence forbids
redistribution, so the R script checks for it and prints instructions.

sdmbench never bypasses an authentication or licence gate.
