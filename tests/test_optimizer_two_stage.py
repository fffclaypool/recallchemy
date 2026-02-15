import numpy as np
import optuna

from recallchemy.backends import VectorBackend
from recallchemy.optimizer import optimize_backend
from recallchemy.types import DatasetBundle, EvaluationResult


class DummyTwoStageBackend(VectorBackend):
    name = "dummy-two-stage"
    module_name = "math"

    def suggest_params(self, trial: optuna.Trial, n_train: int, dim: int, top_k: int):
        del n_train, dim, top_k
        return {"x": int(trial.suggest_int("x_global", 0, 100))}

    def suggest_local_params(self, trial: optuna.Trial, n_train: int, dim: int, top_k: int, anchor_params):
        del n_train, dim, top_k
        anchor = int(anchor_params["x"])
        low = max(0, anchor - 8)
        high = min(100, anchor + 8)
        return {"x": int(trial.suggest_int("x_local", low, high))}

    def build_index(self, train, metric, params):
        del train, metric
        return params["x"]

    def query_index(self, index, query, top_k, metric, params):
        del query, top_k, metric, params
        return np.array([index], dtype=np.int64)

    def evaluate(self, train, queries, ground_truth, metric, top_k, params):
        del train, queries, ground_truth, metric, top_k
        x = int(params["x"])
        distance = abs(x - 37)
        recall = max(0.0, 1.0 - (distance / 100.0))
        return EvaluationResult(
            backend=self.name,
            params=params,
            recall=recall,
            mean_query_ms=float(distance),
            p95_query_ms=float(distance),
            build_time_s=0.01,
        )


def _dummy_dataset() -> DatasetBundle:
    train = np.zeros((8, 4), dtype=np.float32)
    queries = np.zeros((4, 4), dtype=np.float32)
    ground_truth = np.zeros((4, 1), dtype=np.int64)
    return DatasetBundle(train=train, queries=queries, metric="euclidean", ground_truth=ground_truth)


def test_optimize_backend_runs_two_stage_and_returns_local_trials():
    rec = optimize_backend(
        backend=DummyTwoStageBackend(),
        dataset=_dummy_dataset(),
        top_k=1,
        n_trials=10,
        target_recall=0.8,
        seed=11,
        local_refine=True,
        stage1_ratio=0.6,
    )
    assert 0 <= rec.params["x"] <= 100
    assert any(row.get("stage") == "local" for row in rec.top_trials)


class DummyFallbackBackend(VectorBackend):
    name = "dummy-fallback"
    module_name = "math"

    def suggest_params(self, trial: optuna.Trial, n_train: int, dim: int, top_k: int):
        del n_train, dim, top_k
        # Stage-1 must miss target so stage-2 switches to global fallback.
        return {"x": int(trial.suggest_int("x_global", 0, 80))}

    def suggest_local_params(self, trial: optuna.Trial, n_train: int, dim: int, top_k: int, anchor_params):
        del n_train, dim, top_k, anchor_params
        return {"x": int(trial.suggest_int("x_local", 0, 5))}

    def build_index(self, train, metric, params):
        del train, metric
        return params["x"]

    def query_index(self, index, query, top_k, metric, params):
        del query, top_k, metric, params
        return np.array([index], dtype=np.int64)

    def evaluate(self, train, queries, ground_truth, metric, top_k, params):
        del train, queries, ground_truth, metric, top_k
        x = int(params["x"])
        # Reach target only when x is high. A narrow local range around low anchors
        # would fail; global fallback should recover by exploring wide range again.
        recall = 0.6 if x < 90 else 1.0
        p95 = float(max(0, 120 - x))
        return EvaluationResult(
            backend=self.name,
            params=params,
            recall=recall,
            mean_query_ms=p95,
            p95_query_ms=p95,
            build_time_s=0.01,
        )


def test_optimize_backend_uses_global_fallback_when_stage1_misses_target():
    rec = optimize_backend(
        backend=DummyFallbackBackend(),
        dataset=_dummy_dataset(),
        top_k=1,
        n_trials=6,
        target_recall=0.95,
        seed=1,
        local_refine=True,
        stage1_ratio=0.6,
        record_history=True,
    )
    assert rec.trial_history is not None
    assert any(row.get("stage") == "global_fallback" for row in rec.trial_history)
