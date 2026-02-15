import numpy as np
import optuna

from recallchemy.backends import VectorBackend
from recallchemy.comparison_analysis import _compare_final_metrics, baseline_params_for_backend, build_comparison_analysis
from recallchemy.optimizer import optimize_backend
from recallchemy.types import DatasetBundle, EvaluationResult


class DummyBackend(VectorBackend):
    name = "annoy"
    module_name = "math"

    def suggest_params(self, trial: optuna.Trial, n_train: int, dim: int, top_k: int):
        del n_train, dim, top_k
        return {"x": int(trial.suggest_int("x", 0, 20))}

    def build_index(self, train, metric, params):
        del train, metric
        return params["x"]

    def query_index(self, index, query, top_k, metric, params):
        del query, top_k, metric, params
        return np.array([index], dtype=np.int64)

    def evaluate(self, train, queries, ground_truth, metric, top_k, params):
        del train, queries, ground_truth, metric, top_k
        x = int(params.get("x", 10))
        recall = min(1.0, 0.5 + (x / 40.0))
        p95 = max(0.001, 1.0 - (x / 40.0))
        return EvaluationResult(
            backend=self.name,
            params=params,
            recall=recall,
            mean_query_ms=p95,
            p95_query_ms=p95,
            build_time_s=0.01,
        )


def _dataset() -> DatasetBundle:
    train = np.zeros((16, 8), dtype=np.float32)
    queries = np.zeros((8, 8), dtype=np.float32)
    gt = np.zeros((8, 1), dtype=np.int64)
    return DatasetBundle(train=train, queries=queries, metric="euclidean", ground_truth=gt)


def test_baseline_params_for_backend_defaults():
    params = baseline_params_for_backend(backend_name="annoy", n_train=1200, dim=32, top_k=10)
    assert params["n_trees"] == 100
    assert params["search_k"] == -1


def test_build_comparison_analysis_shapes():
    backend = DummyBackend()
    dataset = _dataset()

    main_rec = optimize_backend(
        backend=backend,
        dataset=dataset,
        top_k=1,
        n_trials=4,
        target_recall=0.8,
        seed=42,
        local_refine=False,
        sampler="tpe",
        record_history=True,
    )
    analysis = build_comparison_analysis(
        backends=[backend],
        dataset=dataset,
        top_k=1,
        n_trials=4,
        target_recall=0.8,
        timeout=None,
        top_n_trials=3,
        local_refine=False,
        stage1_ratio=0.6,
        constraints=None,
        seed_start=42,
        seed_count=2,
        runtime_seed=42,
        main_optuna_recommendations={backend.name: main_rec},
    )
    assert analysis["seed_count"] == 2
    assert analysis["seeds"] == [42, 43]
    assert "annoy" in analysis["by_backend"]
    assert "anytime" in analysis["by_backend"]["annoy"]
    assert "impact" in analysis["by_backend"]["annoy"]
    assert "distribution" in analysis["by_backend"]["annoy"]
    assert "win_rate" in analysis["by_backend"]["annoy"]

    impact = analysis["by_backend"]["annoy"]["impact"]
    assert "p95_reduction_vs_baseline_pct" in impact
    assert "time_to_target" in impact
    assert "optuna" in impact["time_to_target"]
    assert "random" in impact["time_to_target"]


def test_build_comparison_analysis_impact_metrics_have_expected_direction():
    backend = DummyBackend()
    dataset = _dataset()
    analysis = build_comparison_analysis(
        backends=[backend],
        dataset=dataset,
        top_k=1,
        n_trials=6,
        target_recall=0.8,
        timeout=None,
        top_n_trials=3,
        local_refine=False,
        stage1_ratio=0.6,
        constraints=None,
        seed_start=50,
        seed_count=2,
        runtime_seed=50,
        main_optuna_recommendations=None,
    )
    impact = analysis["by_backend"]["annoy"]["impact"]
    assert impact["baseline_p95_query_ms"] > 0.0
    assert impact["optuna_p95_query_ms_mean"] > 0.0
    assert impact["p95_speedup_vs_baseline_x"] > 0.0
    assert -100.0 <= impact["p95_reduction_vs_baseline_pct"] <= 100.0
    assert impact["time_to_target"]["optuna"]["stats"]["n"] >= 1


def test_compare_final_metrics_uses_latency_when_both_meet_target():
    result = _compare_final_metrics(
        optuna_metrics={"recall": 0.991, "p95_query_ms": 0.7},
        random_metrics={"recall": 0.995, "p95_query_ms": 1.0},
        target_recall=0.99,
    )
    assert result == "win"


def test_compare_final_metrics_ties_when_latency_gap_is_small():
    result = _compare_final_metrics(
        optuna_metrics={"recall": 0.995, "p95_query_ms": 1.05},
        random_metrics={"recall": 0.995, "p95_query_ms": 1.11},
        target_recall=0.99,
        p95_tolerance_ms=0.1,
    )
    assert result == "tie"
