"""The ``disdat`` benchmark dataset.

``disdat`` (Elith et al. 2020) provides 226 anonymised species across six
regions, each with presence-only training records, a background sample, and
*independent* presence-absence survey data for evaluation. It is the dataset
underlying both Valavi et al. (2022) and Dinnage & Warren (2026).

Two acquisition strategies
--------------------------
1. **R package** (default, authoritative). ``disdat`` ships its data as ``.rds``
   files that Python cannot read. :meth:`DisdatDataset.fetch` calls
   ``rbridge/scripts/fetch_disdat.R`` to export every region to long-format CSV
   plus a ``metadata.json``, then converts to Parquet.
2. **Raw OSF archive** (fallback). The same data is published on OSF as CSVs.
   :func:`fetch_from_osf` downloads that archive for users without R.

Both land in the same standardised cache, so everything downstream is identical::

    <cache>/datasets/disdat/<version>/
        raw/                    untouched source payloads (never overwritten)
        AWT_po.parquet  AWT_bg.parquet  AWT_pa.parquet  AWT_env.parquet
        ...
        metadata.json           regions, groups, species, predictors, CRS
        checksums.json          sha256 of every file

Region geometry
---------------
Coordinate reference systems differ by region and, critically, some are
projected (metres) while others are geographic (degrees). The 10 km spatial
buffer therefore cannot be a constant in coordinate units -- see
:data:`REGION_CRS` and :mod:`sdmbench.splits.spatial`.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

from sdmbench.data.base import SpeciesTask
from sdmbench.exceptions import DataError, DataNotFetchedError
from sdmbench.paths import datasets_dir, ensure_dir
from sdmbench.rbridge.runner import RRunner
from sdmbench.reproducibility.hashes import hash_file, hash_json

__all__ = [
    "DisdatDataset",
    "REGIONS",
    "REGION_GROUPS",
    "REGION_CRS",
    "CATEGORICAL_VARIABLES",
    "RegionMeta",
    "fetch_from_osf",
]

#: The six benchmark regions. Source: ``disdat:::.checkRegion``.
REGIONS = ("AWT", "CAN", "NSW", "NZ", "SA", "SWI")

#: Regions whose test data is split across multiple survey groups. For these,
#: a species must be evaluated against the survey for *its own* group.
#: Source: ``disdat:::.getDisData`` argument validation.
REGION_GROUPS: dict[str, tuple[str, ...]] = {
    "AWT": ("bird", "plant"),
    "NSW": ("ba", "db", "nb", "ot", "ou", "rt", "ru", "sr"),
}

#: Human-readable expansion of the NSW group codes (from ``man/NSW.Rd``).
NSW_GROUP_NAMES = {
    "ba": "bats",
    "db": "diurnal birds",
    "nb": "nocturnal birds",
    "ot": "open-forest trees",
    "ou": "open-forest understorey plants",
    "rt": "rainforest trees",
    "ru": "rainforest understorey plants",
    "sr": "small reptiles",
}


@dataclass(frozen=True)
class RegionCRS:
    """Coordinate reference system of one region.

    ``geographic`` is the field that matters for distance: when True the
    coordinates are degrees and a 10 km buffer requires geodesic distance, not
    a Euclidean radius of 10000.
    """

    epsg: str
    proj4: str
    geographic: bool
    units: str


#: Per-region CRS. Source: ``disdat::disCRS`` (``R/disOther.R``) and the region
#: help pages. NSW and SA fall through disCRS's final branch to WGS84.
REGION_CRS: dict[str, RegionCRS] = {
    "AWT": RegionCRS(
        "EPSG:28355", "+proj=utm +zone=55 +south +ellps=GRS80", geographic=False, units="m"
    ),
    "CAN": RegionCRS(
        "EPSG:4008", "+proj=longlat +ellps=clrk66", geographic=True, units="degrees"
    ),
    "NSW": RegionCRS(
        "EPSG:4326", "+proj=longlat +datum=WGS84", geographic=True, units="degrees"
    ),
    "NZ": RegionCRS(
        "EPSG:27200",
        "+proj=nzmg +lat_0=-41 +lon_0=173 +x_0=2510000 +y_0=6023150 +ellps=intl "
        "+towgs84=59.47,-5.04,187.44,0.47,-0.1,1.024,-4.5993 +units=m",
        geographic=False,
        units="m",
    ),
    "SA": RegionCRS(
        "EPSG:4326", "+proj=longlat +datum=WGS84", geographic=True, units="degrees"
    ),
    "SWI": RegionCRS(
        "EPSG:21781",
        "+proj=tmerc +lat_0=46.95228333333333 +lon_0=7.439583333333333 +k=1 +x_0=600000 "
        "+y_0=200000 +ellps=bessel +units=m +no_defs",
        geographic=False,
        units="m",
    ),
}

#: The five categorical predictors in the benchmark. Source: the ``disdat``
#: modelling vignette and Dinnage & Warren (2026) sec. 2.1.1 -- which agree.
CATEGORICAL_VARIABLES = ("ontveg", "vegsys", "toxicats", "age", "calc")

#: R packages ``fetch_disdat.R`` needs.
R_FETCH_PACKAGES = ("disdat", "jsonlite")

_TABLES = ("po", "bg", "pa", "env")
_DEFAULT_VERSION = "1.1-0"


@dataclass
class RegionMeta:
    """Metadata for one region, read from ``metadata.json``."""

    region: str
    predictors: list[str]
    categorical_predictors: list[str]
    groups: list[str]
    species: list[str]
    species_group: dict[str, str | None]
    crs_epsg: str
    crs_proj4: str
    n_species: int = 0

    @property
    def geographic(self) -> bool:
        crs = REGION_CRS.get(self.region)
        return crs.geographic if crs else False


class DisdatDataset:
    """Loader for the standardised ``disdat`` cache.

    Parameters
    ----------
    root:
        Directory holding the standardised Parquet files. Defaults to
        ``<cache>/datasets/disdat/<version>``.
    """

    name = "disdat"

    def __init__(self, root: str | Path | None = None, version: str = _DEFAULT_VERSION) -> None:
        self.version = version
        self.root = Path(root) if root else datasets_dir("disdat") / version
        self._meta: dict[str, Any] | None = None
        self._frames: dict[tuple[str, str], pd.DataFrame] = {}

    # ------------------------------------------------------------ acquisition --
    @classmethod
    def fetch(
        cls,
        *,
        root: str | Path | None = None,
        regions: Sequence[str] | None = None,
        rscript: str | None = None,
        force: bool = False,
        version: str = _DEFAULT_VERSION,
    ) -> DisdatDataset:
        """Export ``disdat`` from R into the standardised cache.

        Raises a clear, actionable error if R or the ``disdat`` package is
        missing -- it never attempts to install either.
        """
        dataset = cls(root=root, version=version)
        if dataset.is_available() and not force:
            return dataset

        runner = RRunner(rscript)
        runner.require_available()
        runner.require_packages(R_FETCH_PACKAGES)

        raw_dir = ensure_dir(dataset.root / "raw")
        result = runner.run_script(
            "fetch_disdat.R",
            {
                "output_dir": str(raw_dir),
                "regions": list(regions) if regions else list(REGIONS),
            },
            timeout=3600,
        ).require()

        dataset._convert_raw_to_parquet(raw_dir)
        dataset._write_checksums(extra={"r_export": result})
        dataset._meta = None
        return dataset

    def _convert_raw_to_parquet(self, raw_dir: Path) -> None:
        """Convert exported CSVs to Parquet. Raw CSVs are left untouched."""
        ensure_dir(self.root)
        meta_src = raw_dir / "metadata.json"
        if not meta_src.is_file():
            raise DataError(f"R export did not produce {meta_src}")
        shutil.copyfile(meta_src, self.root / "metadata.json")

        n_converted = 0
        for csv_path in sorted(raw_dir.glob("*_*.csv")):
            stem = csv_path.stem
            region, _, table = stem.rpartition("_")
            if region not in REGIONS or table not in _TABLES:
                continue
            frame = pd.read_csv(csv_path)
            frame = _coerce_types(frame, table=table)
            frame.to_parquet(self.root / f"{region}_{table}.parquet", index=False)
            n_converted += 1
        if n_converted == 0:
            raise DataError(f"no disdat tables found to convert in {raw_dir}")

    def _write_checksums(self, extra: dict[str, Any] | None = None) -> None:
        """Record a sha256 for every standardised file."""
        checksums = {
            p.name: hash_file(p)
            for p in sorted(self.root.glob("*"))
            if p.is_file() and p.name != "checksums.json"
        }
        payload = {
            "dataset": self.name,
            "version": self.version,
            "algorithm": "sha256",
            "files": checksums,
        }
        if extra:
            payload["export"] = extra
        (self.root / "checksums.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    def is_available(self) -> bool:
        """Whether the standardised cache looks complete."""
        if not (self.root / "metadata.json").is_file():
            return False
        return all((self.root / f"{r}_po.parquet").is_file() for r in REGIONS)

    def require_available(self) -> None:
        if not self.is_available():
            raise DataNotFetchedError(
                f"disdat has not been prepared in {self.root}.\n"
                "  Run:  sdmbench data fetch disdat\n"
                "  (this needs R with the 'disdat' package installed, or use "
                "'sdmbench data fetch disdat --source osf')"
            )

    # -------------------------------------------------------------- metadata --
    @property
    def metadata(self) -> dict[str, Any]:
        if self._meta is None:
            self.require_available()
            self._meta = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        return self._meta

    def regions(self) -> list[str]:
        """Regions actually present in the cache, in canonical order."""
        present = set(self.metadata.get("regions", {}))
        return [r for r in REGIONS if r in present]

    def region_meta(self, region: str) -> RegionMeta:
        region = region.upper()
        raw = self.metadata.get("regions", {}).get(region)
        if raw is None:
            raise DataError(f"region {region!r} is not present in {self.root}")
        species = [str(s) for s in _as_list(raw.get("species", []))]
        species_group = _parse_species_group(raw.get("species_group"), species)
        predictors = [str(p) for p in _as_list(raw.get("predictors", []))]
        return RegionMeta(
            region=region,
            predictors=predictors,
            categorical_predictors=[p for p in predictors if p in CATEGORICAL_VARIABLES],
            groups=[str(g) for g in _as_list(raw.get("groups", []))],
            species=species,
            species_group=species_group,
            crs_epsg=str(raw.get("crs_epsg", REGION_CRS[region].epsg)),
            crs_proj4=str(raw.get("crs_proj4", REGION_CRS[region].proj4)),
            n_species=int(raw.get("n_species", len(species))),
        )

    def species(self, region: str) -> list[str]:
        return list(self.region_meta(region).species)

    def all_species(self) -> list[tuple[str, str]]:
        """Every ``(region, species_id)`` pair in the dataset."""
        return [(r, s) for r in self.regions() for s in self.species(r)]

    def predictors(self, region: str) -> list[str]:
        return list(self.region_meta(region).predictors)

    # ----------------------------------------------------------------- data --
    def table(self, region: str, table: str) -> pd.DataFrame:
        """Load one standardised table, memoised per process."""
        region = region.upper()
        if table not in _TABLES:
            raise DataError(f"unknown table {table!r}; expected one of {_TABLES}")
        key = (region, table)
        if key not in self._frames:
            self.require_available()
            path = self.root / f"{region}_{table}.parquet"
            if not path.is_file():
                raise DataError(f"missing table file: {path}")
            self._frames[key] = pd.read_parquet(path)
        return self._frames[key]

    def get_task(
        self,
        region: str,
        species_id: str,
        *,
        predictors: Sequence[str] | None = None,
        benchmark_id: str = "",
    ) -> SpeciesTask:
        """Assemble the :class:`SpeciesTask` for one species.

        The training table is this species' presence records plus the region's
        full background sample. The test table is the *independent* survey data
        for this species' own group, joined to the survey environmental data.

        ``predictors`` lets a benchmark recipe request a documented subset --
        Dinnage & Warren (2026) sec. 2.1.1 uses fewer predictors than
        ``disPredictors()`` returns for several regions.
        """
        region = region.upper()
        meta = self.region_meta(region)
        preds = list(predictors) if predictors else list(meta.predictors)
        missing = [p for p in preds if p not in meta.predictors]
        if missing:
            raise DataError(
                f"requested predictors not available for {region}: {missing}\n"
                f"  available: {meta.predictors}"
            )

        po = self.table(region, "po")
        presences = po[po["spid"].astype(str) == str(species_id)]
        if presences.empty:
            raise DataError(f"species {species_id!r} has no presence records in {region}")
        group = meta.species_group.get(species_id)
        if group is None:
            observed = presences["group"].dropna().unique()
            group = str(observed[0]) if len(observed) else None

        bg = self.table(region, "bg")
        train = pd.concat([presences, bg], ignore_index=True, sort=False)
        train["occ"] = train["occ"].astype(int)

        test = self._build_test_frame(region, species_id, group, preds)

        keep = ["region", "group", "siteid", "spid", "x", "y", "occ", *preds]
        train = train.loc[:, [c for c in keep if c in train.columns]].copy()
        test = test.loc[:, [c for c in keep if c in test.columns]].copy()

        crs = REGION_CRS[region]
        return SpeciesTask(
            region=region,
            species_id=str(species_id),
            train=train,
            test=test,
            predictors=preds,
            categorical_predictors=[p for p in preds if p in CATEGORICAL_VARIABLES],
            group=group,
            crs=crs.epsg,
            geographic=crs.geographic,
            benchmark_id=benchmark_id,
            metadata={
                "dataset": self.name,
                "dataset_version": self.version,
                "crs_proj4": crs.proj4,
                "crs_units": crs.units,
            },
        )

    def _build_test_frame(
        self, region: str, species_id: str, group: str | None, preds: Sequence[str]
    ) -> pd.DataFrame:
        """Join this species' presence/absence records to the survey covariates."""
        pa = self.table(region, "pa")
        env = self.table(region, "env")

        rows = pa[pa["spid"].astype(str) == str(species_id)]
        if group is not None and "group" in pa.columns:
            in_group = rows["group"].astype(str) == str(group)
            if in_group.any():
                rows = rows[in_group]
        if rows.empty:
            raise DataError(
                f"species {species_id!r} ({region}, group={group}) has no "
                "presence-absence test records"
            )

        env_cols = ["siteid", *[p for p in preds if p in env.columns]]
        env_side = env
        if group is not None and "group" in env.columns:
            in_group = env_side["group"].astype(str) == str(group)
            if in_group.any():
                env_side = env_side[in_group]
        env_side = env_side.loc[:, env_cols].drop_duplicates(subset="siteid")

        merged = rows.merge(env_side, on="siteid", how="inner", validate="many_to_one")
        if merged.empty:
            raise DataError(
                f"no test sites matched between pa and env tables for {region}/{species_id}"
            )
        merged["occ"] = merged["occ"].astype(int)
        return merged

    def iter_tasks(
        self,
        regions: Iterable[str] | None = None,
        *,
        predictors_by_region: dict[str, Sequence[str]] | None = None,
        benchmark_id: str = "",
    ) -> Iterator[SpeciesTask]:
        """Yield every species task, region by region."""
        for region in (list(regions) if regions else self.regions()):
            region = region.upper()
            preds = (predictors_by_region or {}).get(region)
            for species_id in self.species(region):
                yield self.get_task(
                    region, species_id, predictors=preds, benchmark_id=benchmark_id
                )

    # --------------------------------------------------------------- hashing --
    def content_hash(self) -> str:
        """Hash of the standardised files, from ``checksums.json`` when present."""
        checks = self.root / "checksums.json"
        if checks.is_file():
            payload = json.loads(checks.read_text(encoding="utf-8"))
            return hash_json(payload.get("files", {}))
        self.require_available()
        return hash_json(
            {p.name: hash_file(p) for p in sorted(self.root.glob("*.parquet")) if p.is_file()}
        )

    def summary(self) -> dict[str, Any]:
        """Counts per region, for ``sdmbench data info``."""
        out: dict[str, Any] = {
            "dataset": self.name,
            "version": self.version,
            "root": str(self.root),
            "available": self.is_available(),
        }
        if not self.is_available():
            return out
        regions: dict[str, Any] = {}
        total = 0
        for region in self.regions():
            meta = self.region_meta(region)
            regions[region] = {
                "n_species": meta.n_species,
                "n_predictors": len(meta.predictors),
                "categorical": meta.categorical_predictors,
                "groups": meta.groups,
                "crs": meta.crs_epsg,
                "geographic": meta.geographic,
            }
            total += meta.n_species
        out["regions"] = regions
        out["total_species"] = total
        out["disdat_version"] = self.metadata.get("disdat_version")
        return out


def _as_list(value: Any) -> list[Any]:
    """Normalise jsonlite output, which may unbox length-1 lists to scalars."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    return [value]


def _clean_group(value: Any) -> str | None:
    """Normalise an R group label, mapping R's ``NA`` onto ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    return None if text in {"", "NA", "nan", "None"} else text


def _parse_species_group(raw: Any, species: Sequence[str]) -> dict[str, str | None]:
    """Map species id -> survey group.

    ``jsonlite`` serialises R's named character vector as a JSON *object*, but
    an unnamed one as an array. Accept both so the loader does not depend on
    that detail.
    """
    if isinstance(raw, dict):
        return {str(k): _clean_group(v) for k, v in raw.items()}
    values = _as_list(raw)
    if len(values) == len(species):
        return {sp: _clean_group(g) for sp, g in zip(species, values)}
    return {sp: None for sp in species}


def _coerce_types(frame: pd.DataFrame, *, table: str) -> pd.DataFrame:
    """Apply stable dtypes so Parquet round-trips deterministically."""
    out = frame.copy()
    for col in ("region", "group", "siteid", "spid"):
        if col in out.columns:
            out[col] = out[col].astype("string")
    for col in ("x", "y"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    if "occ" in out.columns:
        out["occ"] = pd.to_numeric(out["occ"], errors="coerce").fillna(0).astype("int64")
    # Categorical predictors arrive as integer codes in several regions; keep
    # them as strings so downstream code never treats a class label as an
    # ordered magnitude.
    for col in CATEGORICAL_VARIABLES:
        if col in out.columns:
            out[col] = out[col].astype("string")
    return out


def fetch_from_osf(
    dest: str | Path,
    *,
    url: str | None = None,
    timeout: int = 600,
) -> Path:
    """Download the raw Elith et al. (2020) CSV archive from OSF.

    Fallback for users without R. The ``disdat`` vignette states the data is
    "released both on Open Science Framework primarily as .csv files ... and
    here as an R package. Both sets of data are similar" -- *similar*, not
    byte-identical, so a benchmark prepared this way records a different
    ``dataset_hash`` and is not automatically comparable with an R-prepared one.

    Parameters
    ----------
    url:
        Override the archive URL. Required when the default OSF endpoint has
        moved; sdmbench does not guess at mirror locations.
    """
    import urllib.request

    dest = ensure_dir(Path(dest))
    if url is None:
        raise DataError(
            "No OSF archive URL is configured for the raw disdat download.\n"
            "  The R-package route is authoritative and is what the paper used:\n"
            "      sdmbench data fetch disdat\n"
            "  To use a raw archive instead, pass its URL explicitly:\n"
            "      sdmbench data fetch disdat --source osf --url <archive-url>\n"
            "  (The OSF project for Elith et al. 2020 is linked from "
            "https://doi.org/10.17161/bi.v15i2.13384)"
        )
    target = dest / Path(url).name
    if target.exists():
        return target
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as resp, tmp.open("wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)
    tmp.replace(target)
    return target
