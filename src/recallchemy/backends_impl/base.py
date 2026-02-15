from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

import numpy as np
import optuna
from numpy.typing import NDArray

from .utils import _bounds, _ensure_k
from ..metrics import canonical_metric, mrr_at_k, ndcg_at_k, recall_at_k
from ..types import EvaluationResult


class VectorBackend(ABC):
    name: str
    module_name: str

    def __init__(self, overrides: dict[str, Any] | None = None):
        self.overrides = dict(overrides or {})

    @classmethod
    def availability(cls) -> tuple[bool, str | None]:
        try:
            importlib.import_module(cls.module_name)
            return True, None
        except Exception as exc:  # pragma: no cover - depends on environment
            return False, f"{cls.module_name} import failed: {exc}"

    @abstractmethod
    def suggest_params(self, trial: optuna.Trial, n_train: int, dim: int, top_k: int) -> dict[str, Any]:
        raise NotImplementedError

    def suggest_local_params(
        self,
        trial: optuna.Trial,
        n_train: int,
        dim: int,
        top_k: int,
        anchor_params: dict[str, Any],
    ) -> dict[str, Any]:
        del anchor_params
        return self.suggest_params(trial=trial, n_train=n_train, dim=dim, top_k=top_k)

    @abstractmethod
    def build_index(self, train: NDArray[np.float32], metric: str, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def query_index(
        self,
        index: Any,
        query: NDArray[np.float32],
        top_k: int,
        metric: str,
        params: dict[str, Any],
    ) -> NDArray[np.int64]:
        raise NotImplementedError

    def evaluate(
        self,
        train: NDArray[np.float32],
        queries: NDArray[np.float32],
        ground_truth: NDArray[np.int64],
        metric: str,
        top_k: int,
        params: dict[str, Any],
    ) -> EvaluationResult:
        metric = canonical_metric(metric)

        build_start = perf_counter()
        index = self.build_index(train, metric, params)
        build_time = perf_counter() - build_start

        predictions = np.empty((queries.shape[0], top_k), dtype=np.int64)
        latencies = np.empty((queries.shape[0],), dtype=np.float64)

        for i, q in enumerate(queries):
            query_start = perf_counter()
            ids = self.query_index(index, q, top_k, metric, params)
            latencies[i] = (perf_counter() - query_start) * 1000.0
            predictions[i] = _ensure_k(np.asarray(ids, dtype=np.int64), top_k)

        return EvaluationResult(
            backend=self.name,
            params=params,
            recall=recall_at_k(predictions, ground_truth, top_k),
            mean_query_ms=float(np.mean(latencies)),
            p95_query_ms=float(np.percentile(latencies, 95)),
            build_time_s=float(build_time),
            ndcg_at_k=ndcg_at_k(predictions, ground_truth, top_k),
            mrr_at_k=mrr_at_k(predictions, ground_truth, top_k),
        )

    def _override_range(self, key: str, default: tuple[int, int]) -> tuple[int, int]:
        value = self.overrides.get(key)
        if value is None:
            return default
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{self.name}: override '{key}' must be [low, high]")
        low, high = int(value[0]), int(value[1])
        return _bounds(low, high)

    def _override_int_choices(self, key: str, default: list[int]) -> list[int]:
        value = self.overrides.get(key)
        if value is None:
            return sorted(set(int(v) for v in default))
        if not isinstance(value, list) or not value:
            raise ValueError(f"{self.name}: override '{key}' must be a non-empty list")
        return sorted(set(int(v) for v in value))

    def _override_str_choices(self, key: str, default: list[str]) -> list[str]:
        value = self.overrides.get(key)
        if value is None:
            return sorted(set(default))
        if not isinstance(value, list) or not value:
            raise ValueError(f"{self.name}: override '{key}' must be a non-empty list")
        return sorted(set(str(v) for v in value))


__all__ = ["VectorBackend"]
