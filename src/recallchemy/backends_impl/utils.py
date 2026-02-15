from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from numpy.typing import NDArray

from ..priors import FAISS_PQ_M


def _ensure_k(ids: NDArray[np.int64], k: int) -> NDArray[np.int64]:
    if ids.shape[0] >= k:
        return ids[:k]
    padded = np.full((k,), -1, dtype=np.int64)
    padded[: ids.shape[0]] = ids
    return padded


def _bounds(low: int, high: int) -> tuple[int, int]:
    if high < low:
        return low, low
    return low, high


def _local_int_window(
    center: int,
    low: int,
    high: int,
    *,
    scale: float = 2.0,
    min_span: int = 2,
) -> tuple[int, int]:
    if center <= 0:
        center = low
    raw_low = int(center / scale)
    raw_high = int(center * scale)
    window_low = max(low, raw_low)
    window_high = min(high, raw_high)
    if window_high - window_low < min_span:
        half = max(1, min_span // 2)
        window_low = max(low, center - half)
        window_high = min(high, center + half)
    if window_high < window_low:
        return window_low, window_low
    return window_low, window_high


def _suggest_int_log(trial: optuna.Trial, name: str, low: int, high: int) -> int:
    low, high = _bounds(low, high)
    if low == high:
        return low
    if low <= 1:
        return trial.suggest_int(name, low, high)
    return trial.suggest_int(name, low, high, log=True)


def _pq_m_candidates(dim: int, preferred_values: list[int] | None = None) -> list[int]:
    base_values = preferred_values if preferred_values is not None else FAISS_PQ_M
    preferred = [m for m in base_values if m <= dim and dim % m == 0]
    if preferred:
        return preferred
    allowed = set(base_values)
    return [m for m in range(2, min(dim - 1, 128) + 1) if dim % m == 0 and m in allowed]


def _neighbor_choices(values: list[int], anchor: int, radius: int = 1) -> list[int]:
    uniq = sorted(set(values))
    if not uniq:
        return []
    if anchor not in uniq:
        return uniq
    idx = uniq.index(anchor)
    lo = max(0, idx - radius)
    hi = min(len(uniq), idx + radius + 1)
    return uniq[lo:hi]


def _rerank_ids(
    query: NDArray[np.float32],
    candidate_ids: NDArray[np.int64],
    train: NDArray[np.float32],
    metric: str,
    top_k: int,
) -> NDArray[np.int64]:
    valid_ids = candidate_ids[candidate_ids >= 0]
    if valid_ids.size <= top_k:
        return valid_ids[:top_k]

    vectors = train[valid_ids]
    q = query.reshape(1, -1)
    if metric == "euclidean":
        distances = np.sum((vectors - q) * (vectors - q), axis=1)
        order = np.argsort(distances)
    else:
        # For angular metric, train/query are already normalized before this call.
        scores = (vectors @ q.T).reshape(-1)
        order = np.argsort(-scores)
    return valid_ids[order[:top_k]]


__all__ = [
    "_bounds",
    "_ensure_k",
    "_local_int_window",
    "_neighbor_choices",
    "_pq_m_candidates",
    "_rerank_ids",
    "_suggest_int_log",
]
