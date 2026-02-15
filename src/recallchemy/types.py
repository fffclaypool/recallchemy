from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class DatasetBundle:
    train: NDArray[np.float32]
    queries: NDArray[np.float32]
    metric: str
    ground_truth: NDArray[np.int64] | None = None


@dataclass(slots=True)
class EvaluationResult:
    backend: str
    params: dict[str, Any]
    recall: float
    mean_query_ms: float
    p95_query_ms: float
    build_time_s: float
    ndcg_at_k: float = 0.0
    mrr_at_k: float = 0.0
