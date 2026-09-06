"""Schema inference for user tables -- suggestions, never assumptions.

``sdmbench inspect data.csv`` looks at a table and proposes what each column
probably is. It is a convenience, not an authority:

* Every suggestion carries a **confidence** and the reason for it.
* Nothing is acted on. The user confirms, or supplies a schema file.
* A supplied schema always overrides inference.

The reason for that strictness: guessing a column's role wrong does not produce
an error, it produces a plausible number. If ``elevation`` were silently taken
as the target, or ``site_id`` as a predictor, the benchmark would still run and
still print a leaderboard.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["DatasetInspector", "ColumnGuess", "InspectionReport", "ColumnRole"]


class ColumnRole(str, enum.Enum):
    """The role a column might play."""

    TARGET = "target"
    LONGITUDE = "longitude"
    LATITUDE = "latitude"
    SPECIES_ID = "species_id"
    GROUP = "group"
    SITE_ID = "site_id"
    TIMESTAMP = "timestamp"
    CONTINUOUS_PREDICTOR = "continuous_predictor"
    CATEGORICAL_PREDICTOR = "categorical_predictor"
    IDENTIFIER = "identifier"
    UNKNOWN = "unknown"


@dataclass
class ColumnGuess:
    """One column's inferred role."""

    column: str
    role: ColumnRole
    confidence: float
    reason: str
    dtype: str = ""
    n_unique: int = 0
    n_missing: int = 0
    sample: list[Any] = field(default_factory=list)
    modality_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "role": self.role.value,
            "confidence": round(self.confidence, 2),
            "reason": self.reason,
            "dtype": self.dtype,
            "n_unique": self.n_unique,
            "n_missing": self.n_missing,
            "modality_hint": self.modality_hint,
        }


@dataclass
class InspectionReport:
    """All guesses for one table, plus a draft schema."""

    n_rows: int
    guesses: list[ColumnGuess] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def by_role(self, role: ColumnRole) -> list[ColumnGuess]:
        return [g for g in self.guesses if g.role is role]

    def best(self, role: ColumnRole) -> ColumnGuess | None:
        candidates = sorted(self.by_role(role), key=lambda g: g.confidence, reverse=True)
        return candidates[0] if candidates else None

    def draft_schema(self) -> dict[str, Any]:
        """A schema.yaml skeleton the user can edit and pass back in."""
        target = self.best(ColumnRole.TARGET)
        lon = self.best(ColumnRole.LONGITUDE)
        lat = self.best(ColumnRole.LATITUDE)

        modalities: dict[str, dict[str, list[str]]] = {}
        for guess in self.guesses:
            if guess.role not in {
                ColumnRole.CONTINUOUS_PREDICTOR,
                ColumnRole.CATEGORICAL_PREDICTOR,
            }:
                continue
            modality = guess.modality_hint or "environment"
            modalities.setdefault(modality, {"columns": []})["columns"].append(guess.column)

        schema: dict[str, Any] = {"target": target.column if target else "REPLACE_ME"}
        if lon and lat:
            schema["coordinates"] = {"longitude": lon.column, "latitude": lat.column}
        species = self.best(ColumnRole.SPECIES_ID)
        if species:
            schema["species_id"] = species.column
        schema["modalities"] = modalities
        return schema

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "columns": [g.to_dict() for g in self.guesses],
            "draft_schema": self.draft_schema(),
            "warnings": self.warnings,
        }

    def render(self) -> str:
        lines = [f"Inspected {self.n_rows} rows, {len(self.guesses)} columns.", ""]
        width = max((len(g.column) for g in self.guesses), default=8)
        lines.append(f"{'column'.ljust(width)}  {'inferred role':<24} {'conf':<6} reason")
        lines.append("-" * (width + 60))
        for guess in self.guesses:
            lines.append(
                f"{guess.column.ljust(width)}  {guess.role.value:<24} "
                f"{guess.confidence:<6.2f} {guess.reason}"
            )
        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        lines.append("")
        lines.append("These are SUGGESTIONS, not decisions. Nothing has been assumed.")
        lines.append("Confirm by writing a schema file:")
        lines.append("")
        try:
            import yaml

            lines.append(yaml.safe_dump(self.draft_schema(), sort_keys=False).rstrip())
        except ImportError:  # pragma: no cover
            lines.append(str(self.draft_schema()))
        lines.append("")
        lines.append("then run:  sdmbench benchmark-csv <file> --target <target>")
        return "\n".join(lines)


#: Name fragments that hint at a role. Names are weak evidence, so a name match
#: alone never exceeds moderate confidence -- the column's *values* must agree.
_NAME_HINTS: dict[ColumnRole, tuple[str, ...]] = {
    ColumnRole.TARGET: ("presence", "occurrence", "occ", "present", "pa", "target", "label", "y"),
    ColumnRole.LONGITUDE: ("longitude", "lon", "long", "x", "easting", "decimallongitude"),
    ColumnRole.LATITUDE: ("latitude", "lat", "y", "northing", "decimallatitude"),
    ColumnRole.SPECIES_ID: ("species", "spid", "sp", "taxon", "scientificname", "binomial"),
    ColumnRole.GROUP: ("group", "grp", "taxa", "guild", "clade"),
    ColumnRole.SITE_ID: ("siteid", "site", "plot", "station", "id", "record"),
    ColumnRole.TIMESTAMP: ("date", "time", "year", "timestamp", "eventdate", "observed"),
}

#: Name fragments that hint at an ecological modality.
_MODALITY_HINTS: dict[str, tuple[str, ...]] = {
    "climate": ("bio", "bc", "temp", "prec", "rain", "tmin", "tmax", "chelsa", "worldclim",
                "seas", "clim", "ddeg", "pday"),
    "terrain": ("elev", "alt", "slope", "aspect", "rugged", "topo", "tri", "dem", "curvature",
                "hillshade"),
    "soil": ("soil", "ph", "clay", "sand", "silt", "carbon", "nutri", "calc", "bulkdens"),
    "vegetation": ("ndvi", "evi", "lai", "npp", "canopy", "veg", "forest", "tree"),
    "landcover": ("landcover", "lc", "cover", "worldcover", "ontveg", "vegsys"),
    "hydrology": ("water", "river", "wetland", "flow", "drainage", "watdist", "swb", "deficit"),
    "human": ("pop", "road", "urban", "nightlight", "footprint", "hfp", "crop", "disturb"),
    "remote_sensing": ("band", "sentinel", "landsat", "modis", "reflect", "sr_b"),
}


class DatasetInspector:
    """Infers likely column roles from names, dtypes and value distributions."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def inspect(self) -> InspectionReport:
        report = InspectionReport(n_rows=len(self.frame))
        for column in self.frame.columns:
            report.guesses.append(self._guess(column))

        # Cross-column sanity checks: a lone coordinate is a schema problem.
        has_lon = any(g.role is ColumnRole.LONGITUDE for g in report.guesses)
        has_lat = any(g.role is ColumnRole.LATITUDE for g in report.guesses)
        if has_lon != has_lat:
            report.warnings.append(
                "only one coordinate column was identified; spatial evaluation needs both "
                "longitude and latitude"
            )
        targets = report.by_role(ColumnRole.TARGET)
        if not targets:
            report.warnings.append(
                "no binary target column identified -- specify one with --target"
            )
        elif len(targets) > 1:
            report.warnings.append(
                f"several columns look like targets: {[g.column for g in targets]}; "
                "specify which with --target"
            )
        if not any(
            g.role in {ColumnRole.CONTINUOUS_PREDICTOR, ColumnRole.CATEGORICAL_PREDICTOR}
            for g in report.guesses
        ):
            report.warnings.append("no predictor columns identified")
        return report

    def _guess(self, column: str) -> ColumnGuess:
        series = self.frame[column]
        lowered = column.strip().lower().replace("_", "").replace(" ", "")
        n_unique = int(series.nunique(dropna=True))
        n_missing = int(series.isna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        is_numeric = numeric.notna().mean() > 0.95

        base = dict(
            column=column,
            dtype=str(series.dtype),
            n_unique=n_unique,
            n_missing=n_missing,
            sample=series.dropna().head(3).tolist(),
        )

        def name_matches(role: ColumnRole) -> bool:
            return any(h == lowered or lowered.startswith(h) for h in _NAME_HINTS[role])

        # --- target: binary values are the strong signal ---------------------
        values = set(pd.unique(numeric.dropna()).tolist()) if is_numeric else set()
        looks_binary = values <= {0.0, 1.0} and len(values) == 2
        if not looks_binary and not is_numeric:
            text = {str(v).strip().lower() for v in pd.unique(series.dropna())}
            looks_binary = text <= {"yes", "no", "true", "false", "present", "absent",
                                    "presence", "absence"} and len(text) == 2
        if looks_binary:
            confident = name_matches(ColumnRole.TARGET)
            return ColumnGuess(
                **base,
                role=ColumnRole.TARGET,
                confidence=0.95 if confident else 0.6,
                reason=(
                    "binary values and a target-like name"
                    if confident
                    else "binary values, but the name does not obviously say 'presence'"
                ),
            )

        # --- coordinates: name AND a plausible numeric range ------------------
        if is_numeric:
            values_array = numeric.dropna().to_numpy()
            in_lon_range = bool(len(values_array)) and np.all(np.abs(values_array) <= 180)
            in_lat_range = bool(len(values_array)) and np.all(np.abs(values_array) <= 90)
            if name_matches(ColumnRole.LONGITUDE) and lowered not in {"y"}:
                return ColumnGuess(
                    **base,
                    role=ColumnRole.LONGITUDE,
                    confidence=0.9 if in_lon_range else 0.5,
                    reason=(
                        "longitude-like name, values within +/-180"
                        if in_lon_range
                        else "longitude-like name, but values exceed +/-180 (projected?)"
                    ),
                )
            if name_matches(ColumnRole.LATITUDE) and lowered not in {"x"}:
                return ColumnGuess(
                    **base,
                    role=ColumnRole.LATITUDE,
                    confidence=0.9 if in_lat_range else 0.5,
                    reason=(
                        "latitude-like name, values within +/-90"
                        if in_lat_range
                        else "latitude-like name, but values exceed +/-90 (projected?)"
                    ),
                )

        # --- identifiers -----------------------------------------------------
        if name_matches(ColumnRole.SPECIES_ID):
            return ColumnGuess(
                **base,
                role=ColumnRole.SPECIES_ID,
                confidence=0.8 if n_unique < max(2, len(self.frame) // 2) else 0.4,
                reason=f"species-like name with {n_unique} distinct values",
            )
        if name_matches(ColumnRole.GROUP) and n_unique <= 50:
            return ColumnGuess(
                **base, role=ColumnRole.GROUP, confidence=0.7,
                reason=f"group-like name with {n_unique} levels",
            )
        if name_matches(ColumnRole.TIMESTAMP):
            return ColumnGuess(
                **base, role=ColumnRole.TIMESTAMP, confidence=0.7,
                reason="time-like name",
            )
        if n_unique == len(self.frame) and len(self.frame) > 1:
            return ColumnGuess(
                **base, role=ColumnRole.IDENTIFIER, confidence=0.8,
                reason="every value is distinct -- looks like a row identifier, not a predictor",
            )
        if name_matches(ColumnRole.SITE_ID):
            return ColumnGuess(
                **base, role=ColumnRole.SITE_ID, confidence=0.6,
                reason=f"site-like name with {n_unique} distinct values",
            )

        # --- predictors ------------------------------------------------------
        modality = self._modality_hint(lowered)
        if not is_numeric or (n_unique <= 12 and pd.api.types.is_integer_dtype(numeric)):
            return ColumnGuess(
                **base,
                role=ColumnRole.CATEGORICAL_PREDICTOR,
                confidence=0.7 if not is_numeric else 0.5,
                reason=(
                    f"non-numeric with {n_unique} levels"
                    if not is_numeric
                    else f"integer with only {n_unique} distinct values -- likely a class code"
                ),
                modality_hint=modality,
            )
        return ColumnGuess(
            **base,
            role=ColumnRole.CONTINUOUS_PREDICTOR,
            confidence=0.8 if modality else 0.6,
            reason=(
                f"continuous numeric; name suggests the {modality} modality"
                if modality
                else "continuous numeric"
            ),
            modality_hint=modality,
        )

    @staticmethod
    def _modality_hint(lowered: str) -> str | None:
        for modality, fragments in _MODALITY_HINTS.items():
            if any(fragment in lowered for fragment in fragments):
                return modality
        return None
