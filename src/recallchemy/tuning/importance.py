from __future__ import annotations

from typing import Any

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    i = 0
    while i < values.shape[0]:
        j = i + 1
        while j < values.shape[0] and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _numeric_importance(values: list[float], target: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if np.all(x == x[0]) or np.all(y == y[0]):
        return 0.0
    xr = _rankdata(x)
    yr = _rankdata(y)
    corr = np.corrcoef(xr, yr)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def _categorical_importance(values: list[str], target: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    y = np.asarray(target, dtype=np.float64)
    y_mean = float(np.mean(y))
    ss_total = float(np.sum((y - y_mean) ** 2))
    if ss_total <= 0.0:
        return 0.0
    ss_between = 0.0
    groups: dict[str, list[float]] = {}
    for value, score in zip(values, target):
        groups.setdefault(str(value), []).append(float(score))
    for group_scores in groups.values():
        g = np.asarray(group_scores, dtype=np.float64)
        ss_between += float(len(g) * (np.mean(g) - y_mean) ** 2)
    return float(max(0.0, min(1.0, ss_between / ss_total)))


def _compute_param_importance(
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
) -> dict[str, float]:
    if len(rows) < 2:
        return {}

    targets = [float(row[metric_key]) for row in rows]
    by_param: dict[str, dict[str, list[Any]]] = {}
    for row, target in zip(rows, targets):
        for name, value in row.get("params", {}).items():
            item = by_param.setdefault(str(name), {"values": [], "targets": []})
            item["values"].append(value)
            item["targets"].append(target)

    raw: dict[str, float] = {}
    for name, data in by_param.items():
        values = data["values"]
        ys = data["targets"]
        if len(values) < 2:
            continue
        if all(isinstance(v, (int, float, np.integer, np.floating)) for v in values):
            score = _numeric_importance([float(v) for v in values], ys)
        else:
            score = _categorical_importance([str(v) for v in values], ys)
        if score > 0.0 and np.isfinite(score):
            raw[name] = float(score)

    total = float(sum(raw.values()))
    if total <= 0.0:
        return {}
    return {name: score / total for name, score in sorted(raw.items(), key=lambda x: x[1], reverse=True)}


__all__ = [
    "_rankdata",
    "_numeric_importance",
    "_categorical_importance",
    "_compute_param_importance",
]
