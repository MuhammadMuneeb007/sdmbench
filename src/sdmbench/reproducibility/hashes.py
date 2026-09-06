"""Content hashing for data, splits and configurations.

Every result row carries ``dataset_hash``, ``split_hash`` and
``model_config_hash``. Together they answer the question a reviewer will ask:
*was this number computed on the same data, with the same split, under the same
settings as that one?*

Hashes are content-based and order-stable. Two runs on the same inputs produce
identical hashes on any machine and any Python version, which is why the
dataframe hasher goes through a canonical CSV-like serialisation rather than
``pandas.util.hash_pandas_object`` (whose output has changed across releases).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "hash_bytes",
    "hash_file",
    "hash_json",
    "hash_dataframe",
    "hash_array",
    "hash_split",
    "short",
]

_ALGO = "sha256"
_CHUNK = 1 << 20


def hash_bytes(payload: bytes) -> str:
    return hashlib.new(_ALGO, payload).hexdigest()


def hash_file(path: str | Path) -> str:
    """Streaming hash of a file's contents."""
    h = hashlib.new(_ALGO)
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_json(obj: Any) -> str:
    """Hash of any JSON-serialisable object, key-order independent."""
    return hash_bytes(json.dumps(obj, sort_keys=True, default=str).encode("utf-8"))


def hash_array(arr: np.ndarray) -> str:
    """Hash a numpy array by shape, dtype and contiguous bytes."""
    a = np.ascontiguousarray(arr)
    h = hashlib.new(_ALGO)
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def hash_dataframe(df: pd.DataFrame, *, columns: Sequence[str] | None = None) -> str:
    """Stable content hash of a DataFrame.

    Columns are hashed in sorted name order so that column ordering does not
    change the hash, while values, dtypes and row order do.
    """
    cols = sorted(columns if columns is not None else df.columns)
    h = hashlib.new(_ALGO)
    h.update(f"nrows={len(df)}".encode())
    for col in cols:
        h.update(f"|col={col}:{df[col].dtype}|".encode())
        series = df[col]
        if pd.api.types.is_float_dtype(series):
            # Round to 12 significant digits: protects the hash from
            # last-bit float noise introduced by different BLAS builds.
            values = np.round(series.to_numpy(dtype=float), 12)
            h.update(np.ascontiguousarray(values).tobytes())
        elif pd.api.types.is_integer_dtype(series) or pd.api.types.is_bool_dtype(series):
            h.update(np.ascontiguousarray(series.to_numpy()).tobytes())
        else:
            h.update("\x1f".join(map(str, series.tolist())).encode("utf-8"))
    return h.hexdigest()


def hash_split(
    train_ids: Iterable[Any],
    test_ids: Iterable[Any],
    *,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """Hash identifying a concrete train/test partition.

    Identifiers are sorted, so the hash describes *membership* rather than the
    incidental order in which rows were emitted.
    """
    payload = {
        "train": sorted(map(str, train_ids)),
        "test": sorted(map(str, test_ids)),
        "extra": dict(extra or {}),
    }
    return hash_json(payload)


def short(digest: str, n: int = 12) -> str:
    """Abbreviate a hex digest for display."""
    return digest[:n]
