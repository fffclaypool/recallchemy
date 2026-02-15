from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def canonical_metric(metric: str) -> str:
    normalized = metric.lower().strip()
    aliases = {
        "l2": "euclidean",
        "cosine": "angular",
        "ip": "dot",
        "inner_product": "dot",
    }
    return aliases.get(normalized, normalized)


def _normalize_rows(x: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Keep zero vectors unchanged.
    np.divide(x, norms, out=x, where=norms > 0)
    return x


def compute_ground_truth(
    train: NDArray[np.float32],
    queries: NDArray[np.float32],
    k: int,
    metric: str,
    batch_size: int = 64,
) -> NDArray[np.int64]:
    if k <= 0:
        raise ValueError("k must be positive")
    if train.ndim != 2 or queries.ndim != 2:
        raise ValueError("train and queries must be 2-D arrays")
    if train.shape[1] != queries.shape[1]:
        raise ValueError("train and queries must share dimensionality")

    metric = canonical_metric(metric)
    if metric not in {"euclidean", "angular", "dot"}:
        raise ValueError(f"Unsupported metric for exact search: {metric}")

    k = min(k, train.shape[0])
    train_f = np.asarray(train, dtype=np.float32)
    queries_f = np.asarray(queries, dtype=np.float32)

    if metric == "angular":
        train_work = _normalize_rows(train_f.copy())
        train_sq = None
    else:
        train_work = train_f
        train_sq = np.sum(train_work * train_work, axis=1)

    output = np.empty((queries_f.shape[0], k), dtype=np.int64)
    for start in range(0, queries_f.shape[0], batch_size):
        end = min(start + batch_size, queries_f.shape[0])
        q = queries_f[start:end]

        if metric == "angular":
            q = _normalize_rows(q.copy())
            scores = q @ train_work.T
            partial = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            rows = np.arange(partial.shape[0])[:, None]
            ranking = np.argsort(-scores[rows, partial], axis=1)
            output[start:end] = partial[rows, ranking]
            continue

        if metric == "dot":
            scores = q @ train_work.T
            partial = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            rows = np.arange(partial.shape[0])[:, None]
            ranking = np.argsort(-scores[rows, partial], axis=1)
            output[start:end] = partial[rows, ranking]
            continue

        # Squared Euclidean distance.
        q_sq = np.sum(q * q, axis=1, keepdims=True)
        distances = q_sq + train_sq[None, :] - (2.0 * (q @ train_work.T))
        partial = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(partial.shape[0])[:, None]
        ranking = np.argsort(distances[rows, partial], axis=1)
        output[start:end] = partial[rows, ranking]

    return output


def recall_at_k(predictions: NDArray[np.int64], ground_truth: NDArray[np.int64], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if predictions.shape[0] != ground_truth.shape[0]:
        raise ValueError("predictions and ground_truth must have equal number of rows")

    pred_k = predictions[:, :k]
    truth_k = ground_truth[:, :k]
    hits = 0
    for i in range(pred_k.shape[0]):
        pred_row = pred_k[i][pred_k[i] >= 0]
        hits += np.intersect1d(pred_row, truth_k[i], assume_unique=False).size
    return float(hits / (pred_k.shape[0] * k))


def ndcg_at_k(predictions: NDArray[np.int64], ground_truth: NDArray[np.int64], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if predictions.shape[0] != ground_truth.shape[0]:
        raise ValueError("predictions and ground_truth must have equal number of rows")

    pred_k = predictions[:, :k]
    truth_k = ground_truth[:, :k]
    ideal_dcg = float(np.sum(1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))))
    if ideal_dcg <= 0.0:
        return 0.0

    total = 0.0
    for i in range(pred_k.shape[0]):
        truth_set = set(int(x) for x in truth_k[i] if int(x) >= 0)
        dcg = 0.0
        for rank, pred_id in enumerate(pred_k[i]):
            pid = int(pred_id)
            if pid < 0:
                continue
            if pid in truth_set:
                dcg += 1.0 / np.log2(float(rank + 2))
        total += dcg / ideal_dcg
    return float(total / pred_k.shape[0])


def mrr_at_k(predictions: NDArray[np.int64], ground_truth: NDArray[np.int64], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if predictions.shape[0] != ground_truth.shape[0]:
        raise ValueError("predictions and ground_truth must have equal number of rows")

    pred_k = predictions[:, :k]
    truth_k = ground_truth[:, :k]
    total = 0.0
    for i in range(pred_k.shape[0]):
        truth_set = set(int(x) for x in truth_k[i] if int(x) >= 0)
        rr = 0.0
        for rank, pred_id in enumerate(pred_k[i]):
            pid = int(pred_id)
            if pid < 0:
                continue
            if pid in truth_set:
                rr = 1.0 / float(rank + 1)
                break
        total += rr
    return float(total / pred_k.shape[0])
