"""Data sources and the standardised internal data model."""

from sdmbench.data.base import (
    COORD_COLUMNS,
    META_COLUMNS,
    TARGET_COLUMN,
    Dataset,
    PreparedSplit,
    SpeciesTask,
    validate_occurrence_frame,
)

__all__ = [
    "COORD_COLUMNS",
    "META_COLUMNS",
    "TARGET_COLUMN",
    "Dataset",
    "PreparedSplit",
    "SpeciesTask",
    "validate_occurrence_frame",
    "DisdatDataset",
    "CsvDataset",
    "RasterDataset",
]


def __getattr__(name: str):  # pragma: no cover - lazy import shim
    if name == "DisdatDataset":
        from sdmbench.data.disdat import DisdatDataset

        return DisdatDataset
    if name == "CsvDataset":
        from sdmbench.data.csv import CsvDataset

        return CsvDataset
    if name == "RasterDataset":
        from sdmbench.data.rasters import RasterDataset

        return RasterDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
