from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BackendRecommendation:
    backend: str
    rationale: str
    metrics: dict[str, float]
    params: dict[str, Any]
    top_trials: list[dict[str, Any]]
    trial_history: list[dict[str, Any]] | None = None


__all__ = ["BackendRecommendation"]
