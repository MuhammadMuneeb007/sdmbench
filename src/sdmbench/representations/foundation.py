"""Geospatial foundation-model embeddings.

Adapters for pretrained models that turn a *location* into a vector:
AlphaEarth Foundations, TerraMind, Prithvi. The scientific question they exist
to answer is sharp and testable:

    Does a general-purpose Earth-observation embedding carry more ecological
    signal than the bioclim variables ecologists have used for twenty years?

Status, stated plainly
----------------------
None of these adapters can be executed by sdmbench without setup the user must
do themselves, and **none of them has been run**:

``alphaearth``
    Google DeepMind's AlphaEarth Foundations Satellite Embedding is published
    as an Earth Engine image collection. Access needs a Google Earth Engine
    account and an authenticated ``earthengine`` client. sdmbench never embeds
    or requests credentials; it detects an authenticated client and otherwise
    fails with setup instructions.
``terramind``
    IBM/ESA's multimodal EO foundation model, distributed via ``terratorch`` /
    Hugging Face. The adapter loads the official implementation -- it does not
    reimplement it.
``prithvi``
    NASA/IBM geospatial foundation model on Hugging Face, loaded through
    ``transformers``/``terratorch``.

All three are **frozen feature extractors**: weights are pretrained externally,
nothing is fitted on benchmark data, so no leakage is possible from them. What
they do require is provenance -- model id, revision, embedding dimension, and
the date the embedding was computed -- because an embedding is only reproducible
if the exact model version is recorded.

Precomputed embeddings
----------------------
The practical path for a large benchmark is to compute embeddings once, cache
them, and then treat them as an ``EMBEDDING`` modality that
:class:`~sdmbench.representations.raw.IdentityEmbeddingEncoder` passes straight
through. :class:`CachedEmbeddingProvider` reads such a cache, and is the only
class here that runs today without external credentials.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from sdmbench.core.modality import ModalityData, ModalityKind, ModalityMetadata
from sdmbench.core.registry import Registry, Spec
from sdmbench.exceptions import DataError, LicenseError, MissingDependencyError
from sdmbench.optional import package_version, require, try_import
from sdmbench.paths import cache_root, ensure_dir
from sdmbench.reproducibility.hashes import hash_array, hash_json

__all__ = [
    "FoundationEmbeddingProvider",
    "EmbeddingResult",
    "FOUNDATION_PROVIDERS",
    "AlphaEarthProvider",
    "TerraMindProvider",
    "PrithviProvider",
    "CachedEmbeddingProvider",
]

#: Registry of foundation embedding providers.
FOUNDATION_PROVIDERS: Registry["FoundationEmbeddingProvider"] = Registry(
    "foundation_provider", entry_point_group="sdmbench.foundation_providers"
)


@dataclass
class EmbeddingResult:
    """Embeddings plus the provenance needed to reproduce them."""

    values: np.ndarray
    provider: str
    model_id: str = ""
    revision: str = ""
    dim: int = 0
    computed_at: str = ""
    crs: str = ""
    year: int | None = None
    checksum: str = ""
    license: str = "unknown"
    notes: str = ""

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        self.dim = int(self.values.shape[1])
        if not self.computed_at:
            self.computed_at = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        if not self.checksum:
            self.checksum = hash_array(self.values)

    def to_modality(self, name: str = "earth_observation") -> ModalityData:
        """Wrap as an ``EMBEDDING`` modality ready for the benchmark."""
        return ModalityData(
            metadata=ModalityMetadata(
                name=name,
                kind=ModalityKind.EMBEDDING,
                source=self.model_id or self.provider,
                provider=self.provider,
                crs=self.crs or None,
                license=self.license,
                downloaded_at=self.computed_at,
                checksum=self.checksum,
                notes=self.notes,
                extras={"revision": self.revision, "embedding_dim": self.dim},
            ),
            values=self.values,
            feature_names=[f"{name}_e{i}" for i in range(self.dim)],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "dim": self.dim,
            "computed_at": self.computed_at,
            "crs": self.crs,
            "year": self.year,
            "checksum": self.checksum,
            "license": self.license,
            "notes": self.notes,
        }


class FoundationEmbeddingProvider:
    """Base class: turn coordinates (or patches) into pretrained embeddings.

    Subclasses implement :meth:`encode_locations` and optionally
    :meth:`encode_patches`. Both are pure inference against frozen weights.
    """

    name: str = "foundation"
    model_id: str = ""
    embedding_dim: int | None = None
    license: str = "unknown"
    requires_auth: bool = True
    #: One-line statement of what the user must set up first.
    setup_instructions: str = ""

    def __init__(self, **params: Any) -> None:
        self.params = dict(params)
        self.cache_dir = ensure_dir(
            Path(params.get("cache_dir") or cache_root() / "embeddings" / self.name)
        )

    # ---------------------------------------------------------- availability --
    def check_available(self) -> None:
        """Raise a skippable error if this provider cannot run here."""
        raise NotImplementedError

    def _auth_error(self, detail: str = "") -> LicenseError:
        return LicenseError(
            f"{self.name} requires authentication or access approval that sdmbench will "
            "not bypass.\n"
            f"  {self.setup_instructions}\n"
            + (f"  underlying error: {detail}\n" if detail else "")
            + "  Alternatively, precompute embeddings yourself and load them with "
            "the 'cached' provider."
        )

    # ------------------------------------------------------------- inference --
    def encode_locations(
        self, coords: np.ndarray, *, crs: str = "EPSG:4326", year: int | None = None
    ) -> EmbeddingResult:
        raise NotImplementedError

    def encode_patches(self, patches: np.ndarray) -> EmbeddingResult:
        raise NotImplementedError(
            f"{self.name} does not implement patch encoding; use encode_locations()"
        )

    # ---------------------------------------------------------------- cache --
    def cache_key(self, coords: np.ndarray, *, crs: str, year: int | None) -> str:
        return hash_json(
            {
                "provider": self.name,
                "model_id": self.model_id,
                "crs": crs,
                "year": year,
                "coords": hash_array(np.asarray(coords, dtype=float)),
                "params": {k: str(v) for k, v in sorted(self.params.items())},
            }
        )

    def load_cached(self, key: str) -> EmbeddingResult | None:
        path = self.cache_dir / f"{key}.npz"
        meta_path = self.cache_dir / f"{key}.json"
        if not (path.is_file() and meta_path.is_file()):
            return None
        payload = np.load(path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return EmbeddingResult(values=payload["values"], **{
            k: v for k, v in meta.items() if k in {
                "provider", "model_id", "revision", "computed_at", "crs", "year",
                "checksum", "license", "notes",
            }
        })

    def save_cached(self, key: str, result: EmbeddingResult) -> Path:
        path = self.cache_dir / f"{key}.npz"
        np.savez_compressed(path, values=result.values)
        (self.cache_dir / f"{key}.json").write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        return path

    def describe(self) -> dict[str, Any]:
        try:
            self.check_available()
            available, detail = True, ""
        except Exception as exc:  # noqa: BLE001
            available, detail = False, str(exc).splitlines()[0]
        return {
            "provider": self.name,
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
            "license": self.license,
            "requires_auth": self.requires_auth,
            "available": available,
            "detail": detail,
            "setup": self.setup_instructions,
        }


@FOUNDATION_PROVIDERS.register(
    "alphaearth",
    "satellite-embedding",
    spec=Spec(
        accepts=("coordinates",),
        produces="embedding",
        requires=("ee",),
        extra="foundation",
        output_dim=64,
        pretraining_source="Google DeepMind AlphaEarth Foundations",
        description="64-dimensional annual satellite embedding per location.",
        maturity="emerging",
    ),
)
class AlphaEarthProvider(FoundationEmbeddingProvider):
    """AlphaEarth Foundations satellite embeddings, via Google Earth Engine.

    The embedding is published as an Earth Engine ImageCollection of annual,
    64-band embedding images. Retrieval is therefore a cloud query, not a file
    download, and it needs the user's own authenticated Earth Engine project.

    NOT EXECUTED. This adapter is written against the documented Earth Engine
    API but has not been run, because doing so requires credentials that
    sdmbench must never hold.
    """

    name = "alphaearth"
    model_id = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    embedding_dim = 64
    license = "see Google Earth Engine dataset terms"
    requires_auth = True
    setup_instructions = (
        "Register for Google Earth Engine (https://earthengine.google.com/), then run "
        "`earthengine authenticate` and set $SDMBENCH_EE_PROJECT to your cloud project id."
    )

    def check_available(self) -> None:
        ee = try_import("ee")
        if ee is None:
            raise MissingDependencyError(
                "earthengine-api",
                extra="foundation",
                hint=self.setup_instructions,
            )
        try:
            ee.Number(1).getInfo()
        except Exception as exc:  # noqa: BLE001 - any failure here is an auth failure
            raise self._auth_error(str(exc)) from exc

    def encode_locations(
        self, coords: np.ndarray, *, crs: str = "EPSG:4326", year: int | None = None
    ) -> EmbeddingResult:
        coords = np.asarray(coords, dtype=float)
        key = self.cache_key(coords, crs=crs, year=year)
        cached = self.load_cached(key)
        if cached is not None:
            return cached

        self.check_available()
        ee = require("ee")

        year = year or int(self.params.get("year", 2023))
        collection = ee.ImageCollection(self.model_id).filterDate(
            f"{year}-01-01", f"{year + 1}-01-01"
        )
        image = collection.mosaic()

        points = [
            ee.Feature(ee.Geometry.Point([float(x), float(y)], proj=crs))
            for x, y in coords
        ]
        # Earth Engine caps a single getInfo() payload, so sample in chunks.
        chunk_size = int(self.params.get("chunk_size", 1000))
        rows: list[list[float]] = []
        for start in range(0, len(points), chunk_size):
            block = ee.FeatureCollection(points[start : start + chunk_size])
            sampled = image.sampleRegions(
                collection=block,
                scale=int(self.params.get("scale", 10)),
                geometries=False,
            ).getInfo()
            for feature in sampled.get("features", []):
                properties = feature.get("properties", {})
                rows.append([float(properties.get(f"A{i:02d}", np.nan)) for i in range(64)])

        if len(rows) != len(coords):
            raise DataError(
                f"AlphaEarth returned {len(rows)} embeddings for {len(coords)} coordinates; "
                "some points may fall outside the collection's coverage"
            )

        result = EmbeddingResult(
            values=np.asarray(rows, dtype=float),
            provider=self.name,
            model_id=self.model_id,
            revision=str(year),
            crs=crs,
            year=year,
            license=self.license,
            notes="Retrieved via Google Earth Engine sampleRegions.",
        )
        self.save_cached(key, result)
        return result


@FOUNDATION_PROVIDERS.register(
    "terramind",
    spec=Spec(
        accepts=("raster",),
        produces="embedding",
        requires=("terratorch",),
        extra="foundation",
        pretraining_source="IBM/ESA TerraMind",
        description="Multimodal EO foundation model embeddings via terratorch.",
        maturity="emerging",
    ),
)
class TerraMindProvider(FoundationEmbeddingProvider):
    """TerraMind embeddings, loaded through the official ``terratorch`` package.

    sdmbench does not reimplement TerraMind. It loads the published model and
    runs frozen inference over raster patches.

    NOT EXECUTED -- ``terratorch`` was not installed in the session that wrote
    this adapter.
    """

    name = "terramind"
    model_id = "ibm-esa-geospatial/TerraMind-1.0-base"
    license = "see the TerraMind model card"
    requires_auth = False
    setup_instructions = (
        'pip install "sdmbench[foundation]" (installs terratorch), then accept any model '
        "terms on Hugging Face."
    )

    def check_available(self) -> None:
        if try_import("terratorch") is None:
            raise MissingDependencyError(
                "terratorch", extra="foundation", hint=self.setup_instructions
            )

    def encode_patches(self, patches: np.ndarray) -> EmbeddingResult:
        self.check_available()
        torch = require("torch")
        terratorch = require("terratorch")

        model_id = str(self.params.get("model_id", self.model_id))
        try:
            model = terratorch.BACKBONE_REGISTRY.build(
                model_id, pretrained=True
            )
        except Exception as exc:  # noqa: BLE001
            raise DataError(
                f"could not build TerraMind backbone {model_id!r} via terratorch "
                f"({package_version('terratorch')}): {exc}\n"
                "  The terratorch API has changed between releases; pass model_id= or "
                "supply precomputed embeddings via the 'cached' provider."
            ) from exc

        model.eval()
        array = np.nan_to_num(np.asarray(patches, dtype=float))
        if array.ndim == 5:  # collapse a multi-scale axis
            array = array.reshape(-1, *array.shape[2:])
        with torch.no_grad():
            output = model(torch.tensor(array, dtype=torch.float32))
            features = output[-1] if isinstance(output, (list, tuple)) else output
            features = features.flatten(1).cpu().numpy()

        return EmbeddingResult(
            values=features,
            provider=self.name,
            model_id=model_id,
            revision=package_version("terratorch"),
            license=self.license,
            notes="Frozen TerraMind backbone features.",
        )


@FOUNDATION_PROVIDERS.register(
    "prithvi",
    spec=Spec(
        accepts=("raster",),
        produces="embedding",
        requires=("transformers",),
        extra="foundation",
        pretraining_source="NASA/IBM Prithvi",
        description="Prithvi geospatial foundation model embeddings.",
        maturity="emerging",
    ),
)
class PrithviProvider(FoundationEmbeddingProvider):
    """Prithvi embeddings via Hugging Face.

    NOT EXECUTED. Prithvi's input expectations (band order, normalisation,
    temporal stacking) are model-version specific; the adapter reports clearly
    rather than silently feeding it wrongly-shaped data.
    """

    name = "prithvi"
    model_id = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M"
    license = "Apache-2.0 (verify on the model card)"
    requires_auth = False
    setup_instructions = 'pip install "sdmbench[foundation]" (installs transformers).'

    def check_available(self) -> None:
        if try_import("transformers") is None:
            raise MissingDependencyError(
                "transformers", extra="foundation", hint=self.setup_instructions
            )

    def encode_patches(self, patches: np.ndarray) -> EmbeddingResult:
        self.check_available()
        torch = require("torch")
        transformers = require("transformers")

        model_id = str(self.params.get("model_id", self.model_id))
        try:
            model = transformers.AutoModel.from_pretrained(model_id, trust_remote_code=True)
        except Exception as exc:  # noqa: BLE001
            raise DataError(
                f"could not load Prithvi model {model_id!r}: {exc}\n"
                "  Prithvi requires trust_remote_code and a specific band layout; check the "
                "model card, or supply precomputed embeddings via the 'cached' provider."
            ) from exc

        model.eval()
        array = np.nan_to_num(np.asarray(patches, dtype=float))
        if array.ndim == 5:
            array = array.reshape(-1, *array.shape[2:])
        with torch.no_grad():
            output = model(torch.tensor(array, dtype=torch.float32))
            hidden = getattr(output, "last_hidden_state", output)
            features = (
                hidden.mean(dim=1) if hidden.ndim == 3 else hidden.flatten(1)
            ).cpu().numpy()

        return EmbeddingResult(
            values=features,
            provider=self.name,
            model_id=model_id,
            revision=package_version("transformers"),
            license=self.license,
            notes="Mean-pooled Prithvi encoder states.",
        )


@FOUNDATION_PROVIDERS.register(
    "cached",
    "precomputed",
    spec=Spec(
        accepts=("coordinates",),
        produces="embedding",
        description="Load embeddings the user computed elsewhere (NPZ/Parquet/CSV).",
    ),
)
class CachedEmbeddingProvider(FoundationEmbeddingProvider):
    """Load precomputed embeddings from disk.

    The practical route for a 226-species benchmark: compute the embeddings
    once (in a notebook, on a cluster, in Earth Engine), save them, and point
    sdmbench at the file. This is the only provider here with no external
    dependency, and the only one that has an executable path today.

    The file must have one row per observation, in the same order as the
    dataset's coordinates. That ordering assumption is checked by row count and
    recorded in the manifest; sdmbench cannot verify the correspondence itself.
    """

    name = "cached"
    requires_auth = False
    setup_instructions = "Pass path=<file.npz|.parquet|.csv> when constructing the provider."

    def check_available(self) -> None:
        path = self.params.get("path")
        if not path:
            raise DataError("the 'cached' embedding provider needs path=<file>")
        if not Path(path).is_file():
            raise DataError(f"embedding file not found: {path}")

    def encode_locations(
        self, coords: np.ndarray, *, crs: str = "EPSG:4326", year: int | None = None
    ) -> EmbeddingResult:
        self.check_available()
        path = Path(self.params["path"])

        if path.suffix == ".npz":
            payload = np.load(path)
            key = self.params.get("key") or list(payload.keys())[0]
            values = np.asarray(payload[key], dtype=float)
        elif path.suffix in {".parquet", ".pq"}:
            import pandas as pd

            values = pd.read_parquet(path).to_numpy(dtype=float)
        elif path.suffix in {".csv", ".tsv"}:
            import pandas as pd

            values = pd.read_csv(
                path, sep="\t" if path.suffix == ".tsv" else ","
            ).to_numpy(dtype=float)
        else:
            raise DataError(f"unsupported embedding file type: {path.suffix}")

        if len(values) != len(coords):
            raise DataError(
                f"{path.name} has {len(values)} rows but the dataset has {len(coords)} "
                "observations. Precomputed embeddings must be row-aligned with the data."
            )
        return EmbeddingResult(
            values=values,
            provider=self.name,
            model_id=str(self.params.get("model_id", path.name)),
            revision=str(self.params.get("revision", "")),
            crs=crs,
            year=year,
            license=str(self.params.get("license", "user-supplied")),
            notes=f"Loaded from {path}; row order assumed to match the dataset.",
        )
