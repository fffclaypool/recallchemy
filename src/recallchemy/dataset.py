from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from .metrics import canonical_metric
from .types import DatasetBundle


def _decode_attr(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            scalar = value.item()  # type: ignore[call-arg]
        except Exception:
            scalar = value
        if isinstance(scalar, bytes):
            return scalar.decode("utf-8")
        return str(scalar)
    return str(value)


def _split_train_queries(
    array: NDArray[np.float32], query_fraction: float, seed: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    if not (0.0 < query_fraction < 1.0):
        raise ValueError("query_fraction must be in (0, 1)")
    n = array.shape[0]
    if n < 2:
        raise ValueError("dataset must contain at least 2 vectors")
    query_count = max(1, int(n * query_fraction))
    rng = np.random.default_rng(seed)
    query_idx = rng.choice(n, size=query_count, replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[query_idx] = False
    return array[train_mask], array[query_idx]


def _maybe_sample(
    data: NDArray[np.float32],
    max_rows: int | None,
    seed: int,
) -> NDArray[np.float32]:
    if max_rows is None or data.shape[0] <= max_rows:
        return data
    rng = np.random.default_rng(seed)
    idx = rng.choice(data.shape[0], size=max_rows, replace=False)
    return data[idx]


def _sample_indices(size: int, max_rows: int, seed: int) -> NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(size, size=max_rows, replace=False)
    return np.asarray(idx, dtype=np.int64)


def load_dataset(
    path: str | Path,
    metric: str | None = None,
    query_fraction: float = 0.1,
    seed: int = 42,
    max_train: int | None = None,
    max_queries: int | None = None,
) -> DatasetBundle:
    source = Path(path)
    suffix = source.suffix.lower()

    train: NDArray[np.float32]
    queries: NDArray[np.float32]
    ground_truth: NDArray[np.int64] | None = None
    inferred_metric: str | None = None

    if suffix in {".hdf5", ".h5"}:
        with h5py.File(source, "r") as f:
            if "train" not in f:
                raise ValueError("HDF5 dataset must contain 'train'")
            train = np.asarray(f["train"], dtype=np.float32)
            if "test" in f:
                queries = np.asarray(f["test"], dtype=np.float32)
            elif "queries" in f:
                queries = np.asarray(f["queries"], dtype=np.float32)
            else:
                train, queries = _split_train_queries(train, query_fraction, seed)

            if "neighbors" in f:
                ground_truth = np.asarray(f["neighbors"], dtype=np.int64)
            elif "ground_truth" in f:
                ground_truth = np.asarray(f["ground_truth"], dtype=np.int64)

            inferred_metric = _decode_attr(f.attrs.get("distance"))
    elif suffix == ".npz":
        data = np.load(source)
        if "train" not in data:
            raise ValueError("NPZ dataset must contain 'train'")
        train = np.asarray(data["train"], dtype=np.float32)
        if "test" in data:
            queries = np.asarray(data["test"], dtype=np.float32)
        elif "queries" in data:
            queries = np.asarray(data["queries"], dtype=np.float32)
        else:
            train, queries = _split_train_queries(train, query_fraction, seed)

        if "neighbors" in data:
            ground_truth = np.asarray(data["neighbors"], dtype=np.int64)
        elif "ground_truth" in data:
            ground_truth = np.asarray(data["ground_truth"], dtype=np.int64)

        if "distance" in data:
            inferred_metric = _decode_attr(data["distance"])
    elif suffix == ".npy":
        all_vectors = np.asarray(np.load(source), dtype=np.float32)
        if all_vectors.ndim != 2:
            raise ValueError("NPY dataset must be a 2-D array of vectors")
        train, queries = _split_train_queries(all_vectors, query_fraction, seed)
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    if train.ndim != 2 or queries.ndim != 2:
        raise ValueError("train and queries must be 2-D arrays")
    if train.shape[1] != queries.shape[1]:
        raise ValueError("train and queries dimensionality mismatch")

    metric_value = canonical_metric(metric or inferred_metric or "euclidean")
    train = _maybe_sample(train, max_train, seed)

    if ground_truth is not None:
        if max_train is not None:
            # Ground-truth indices point to the original train set, so they are invalid after sampling.
            ground_truth = None
        elif max_queries is not None and queries.shape[0] > max_queries:
            idx = _sample_indices(queries.shape[0], max_queries, seed + 1)
            queries = queries[idx]
            ground_truth = ground_truth[idx]
    else:
        queries = _maybe_sample(queries, max_queries, seed + 1)

    return DatasetBundle(
        train=np.ascontiguousarray(train, dtype=np.float32),
        queries=np.ascontiguousarray(queries, dtype=np.float32),
        metric=metric_value,
        ground_truth=ground_truth,
    )
