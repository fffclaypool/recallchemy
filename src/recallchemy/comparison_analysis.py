from __future__ import annotations

from typing import Any

import numpy as np

from .backends import VectorBackend
from .optimizer import BackendRecommendation, optimize_backend, select_recommendation
from .types import DatasetBundle


def baseline_params_for_backend(
    *,
    backend_name: str,
    n_train: int,
    dim: int,
    top_k: int,
) -> dict[str, Any]:
    if backend_name == "annoy":
        return {
            "space_profile": "baseline",
            "n_trees": 100,
            "search_k": -1,
            "search_k_mode": "auto",
        }
    if backend_name == "hnswlib":
        return {
            "space_profile": "baseline",
            "M": 16,
            "ef_construction": 100,
            "ef_search": max(64, 2 * int(top_k)),
        }
    if backend_name == "faiss-ivf":
        n_list = max(1, int(round(np.sqrt(max(1, n_train)))))
        n_probe = max(8, n_list // 8)
        n_probe = min(n_list, n_probe)
        return {
            "space_profile": "baseline",
            "index_type": "ivf_flat",
            "n_list": int(n_list),
            "n_probe": int(n_probe),
            "rerank_k_factor": 1,
        }
    raise ValueError(f"unsupported backend for baseline params: {backend_name!r} (dim={dim})")


def _metric_stats(
    values: list[float],
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> dict[str, float | int]:
    if not values:
        nan = float("nan")
        return {
            "n": 0,
            "mean": nan,
            "std": nan,
            "ci_low": nan,
            "ci_high": nan,
        }
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    margin = 1.96 * std / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else 0.0
    ci_low = float(mean - margin)
    ci_high = float(mean + margin)
    if min_value is not None:
        ci_low = max(min_value, ci_low)
        ci_high = max(min_value, ci_high)
    if max_value is not None:
        ci_low = min(max_value, ci_low)
        ci_high = min(max_value, ci_high)
    return {
        "n": int(arr.shape[0]),
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _distribution_summary(metrics_rows: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    return {
        "recall": _metric_stats([row["recall"] for row in metrics_rows], min_value=0.0, max_value=1.0),
        "p95_query_ms": _metric_stats([row["p95_query_ms"] for row in metrics_rows], min_value=0.0),
        "build_time_s": _metric_stats([row["build_time_s"] for row in metrics_rows], min_value=0.0),
    }


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _trial_stats(values: list[int]) -> dict[str, float | int]:
    if not values:
        nan = float("nan")
        return {"n": 0, "mean": nan, "median": nan, "p90": nan}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def _safe_pct_reduction(before: float, after: float) -> float:
    if not np.isfinite(before) or not np.isfinite(after) or before <= 0.0:
        return float("nan")
    return float(((before - after) / before) * 100.0)


def _safe_speedup(before: float, after: float) -> float:
    if not np.isfinite(before) or not np.isfinite(after) or after <= 0.0:
        return float("nan")
    return float(before / after)


def _best_so_far_sequence(
    *,
    trial_history: list[dict[str, Any]],
    target_recall: float,
    constraints: dict[str, float] | None,
) -> list[dict[str, float]]:
    rows = sorted(trial_history, key=lambda row: int(row["trial"]))
    if not rows:
        return []
    sequence: list[dict[str, float]] = []
    for i in range(1, len(rows) + 1):
        selected, _ = select_recommendation(rows[:i], target_recall=target_recall, constraints=constraints)
        sequence.append(
            {
                "trial": int(i),
                "recall": float(selected["recall"]),
                "p95_query_ms": float(selected["p95_query_ms"]),
            }
        )
    return sequence


def _anytime_summary(
    *,
    optuna_histories: list[list[dict[str, Any]]],
    random_histories: list[list[dict[str, Any]]],
    target_recall: float,
    constraints: dict[str, float] | None,
) -> list[dict[str, Any]]:
    optuna_sequences = [
        _best_so_far_sequence(trial_history=rows, target_recall=target_recall, constraints=constraints)
        for rows in optuna_histories
    ]
    random_sequences = [
        _best_so_far_sequence(trial_history=rows, target_recall=target_recall, constraints=constraints)
        for rows in random_histories
    ]

    max_steps = max(
        [len(seq) for seq in optuna_sequences] + [len(seq) for seq in random_sequences] + [0]
    )
    summary: list[dict[str, Any]] = []
    for step in range(max_steps):
        opt_recall = [seq[step]["recall"] for seq in optuna_sequences if len(seq) > step]
        opt_p95 = [seq[step]["p95_query_ms"] for seq in optuna_sequences if len(seq) > step]
        rnd_recall = [seq[step]["recall"] for seq in random_sequences if len(seq) > step]
        rnd_p95 = [seq[step]["p95_query_ms"] for seq in random_sequences if len(seq) > step]
        summary.append(
            {
                "trial": step + 1,
                "optuna": {
                    "recall": _metric_stats(opt_recall, min_value=0.0, max_value=1.0),
                    "p95_query_ms": _metric_stats(opt_p95, min_value=0.0),
                },
                "random": {
                    "recall": _metric_stats(rnd_recall, min_value=0.0, max_value=1.0),
                    "p95_query_ms": _metric_stats(rnd_p95, min_value=0.0),
                },
            }
        )
    return summary


def _first_trial_reaching_target(
    *,
    trial_history: list[dict[str, Any]],
    target_recall: float,
    constraints: dict[str, float] | None,
) -> int | None:
    sequence = _best_so_far_sequence(
        trial_history=trial_history,
        target_recall=target_recall,
        constraints=constraints,
    )
    for row in sequence:
        if float(row["recall"]) >= target_recall:
            return int(row["trial"])
    return None


def _compare_final_metrics(
    *,
    optuna_metrics: dict[str, float],
    random_metrics: dict[str, float],
    recall_tolerance: float = 1e-3,
) -> str:
    opt_recall = float(optuna_metrics["recall"])
    rnd_recall = float(random_metrics["recall"])
    if opt_recall > rnd_recall + recall_tolerance:
        return "win"
    if rnd_recall > opt_recall + recall_tolerance:
        return "loss"
    opt_p95 = float(optuna_metrics["p95_query_ms"])
    rnd_p95 = float(random_metrics["p95_query_ms"])
    if opt_p95 + 1e-12 < rnd_p95:
        return "win"
    if rnd_p95 + 1e-12 < opt_p95:
        return "loss"
    return "tie"


def _impact_summary(
    *,
    baseline_metrics: dict[str, float] | None,
    optuna_by_seed: dict[int, BackendRecommendation],
    random_by_seed: dict[int, BackendRecommendation],
    target_recall: float,
    constraints: dict[str, float] | None,
) -> dict[str, Any]:
    opt_metrics = [rec.metrics for rec in optuna_by_seed.values()]
    rnd_metrics = [rec.metrics for rec in random_by_seed.values()]

    baseline_recall = float("nan")
    baseline_p95 = float("nan")
    if baseline_metrics:
        baseline_recall = float(baseline_metrics.get("recall", float("nan")))
        baseline_p95 = float(baseline_metrics.get("p95_query_ms", float("nan")))

    opt_recall_mean = _mean_or_nan([float(m.get("recall", float("nan"))) for m in opt_metrics])
    opt_p95_mean = _mean_or_nan([float(m.get("p95_query_ms", float("nan"))) for m in opt_metrics])
    rnd_recall_mean = _mean_or_nan([float(m.get("recall", float("nan"))) for m in rnd_metrics])
    rnd_p95_mean = _mean_or_nan([float(m.get("p95_query_ms", float("nan"))) for m in rnd_metrics])

    opt_ttt = [
        t
        for t in (
            _first_trial_reaching_target(
                trial_history=rec.trial_history or [],
                target_recall=target_recall,
                constraints=constraints,
            )
            for rec in optuna_by_seed.values()
        )
        if t is not None
    ]
    rnd_ttt = [
        t
        for t in (
            _first_trial_reaching_target(
                trial_history=rec.trial_history or [],
                target_recall=target_recall,
                constraints=constraints,
            )
            for rec in random_by_seed.values()
        )
        if t is not None
    ]

    return {
        "target_recall": float(target_recall),
        "optuna_recall_mean": opt_recall_mean,
        "random_recall_mean": rnd_recall_mean,
        "baseline_recall": baseline_recall,
        "optuna_p95_query_ms_mean": opt_p95_mean,
        "random_p95_query_ms_mean": rnd_p95_mean,
        "baseline_p95_query_ms": baseline_p95,
        "recall_gain_vs_baseline": (
            float(opt_recall_mean - baseline_recall)
            if np.isfinite(opt_recall_mean) and np.isfinite(baseline_recall)
            else float("nan")
        ),
        "recall_gain_vs_random": (
            float(opt_recall_mean - rnd_recall_mean)
            if np.isfinite(opt_recall_mean) and np.isfinite(rnd_recall_mean)
            else float("nan")
        ),
        "p95_reduction_vs_baseline_pct": _safe_pct_reduction(baseline_p95, opt_p95_mean),
        "p95_reduction_vs_random_pct": _safe_pct_reduction(rnd_p95_mean, opt_p95_mean),
        "p95_speedup_vs_baseline_x": _safe_speedup(baseline_p95, opt_p95_mean),
        "p95_speedup_vs_random_x": _safe_speedup(rnd_p95_mean, opt_p95_mean),
        "time_to_target": {
            "optuna": {
                "reach_rate": (
                    float(len(opt_ttt) / len(optuna_by_seed)) if optuna_by_seed else float("nan")
                ),
                "stats": _trial_stats(opt_ttt),
            },
            "random": {
                "reach_rate": (
                    float(len(rnd_ttt) / len(random_by_seed)) if random_by_seed else float("nan")
                ),
                "stats": _trial_stats(rnd_ttt),
            },
        },
    }


def build_comparison_analysis(
    *,
    backends: list[VectorBackend],
    dataset: DatasetBundle,
    top_k: int,
    n_trials: int,
    target_recall: float,
    timeout: int | None,
    top_n_trials: int,
    local_refine: bool,
    stage1_ratio: float,
    constraints: dict[str, float] | None,
    seed_start: int,
    seed_count: int,
    runtime_seed: int,
    main_optuna_recommendations: dict[str, BackendRecommendation] | None = None,
) -> dict[str, Any]:
    seeds = [int(seed_start + i) for i in range(max(0, int(seed_count)))]
    analysis: dict[str, Any] = {
        "seed_start": int(seed_start),
        "seed_count": int(seed_count),
        "seeds": seeds,
        "by_backend": {},
    }
    if not seeds:
        return analysis

    main_map = main_optuna_recommendations or {}

    for backend in backends:
        backend_data: dict[str, Any] = {
            "baseline": None,
            "distribution": {},
            "anytime": [],
            "impact": {},
            "win_rate": {"wins": 0, "ties": 0, "losses": 0, "win_rate": 0.0, "n_pairs": 0},
            "errors": [],
        }
        try:
            base_params = baseline_params_for_backend(
                backend_name=backend.name,
                n_train=int(dataset.train.shape[0]),
                dim=int(dataset.train.shape[1]),
                top_k=int(top_k),
            )
            base_result = backend.evaluate(
                train=dataset.train,
                queries=dataset.queries,
                ground_truth=dataset.ground_truth,
                metric=dataset.metric,
                top_k=top_k,
                params=base_params,
            )
            backend_data["baseline"] = {
                "params": base_params,
                "metrics": {
                    "recall": float(base_result.recall),
                    "p95_query_ms": float(base_result.p95_query_ms),
                    "build_time_s": float(base_result.build_time_s),
                },
            }
        except Exception as exc:
            backend_data["errors"].append(f"baseline evaluation failed: {exc}")

        optuna_by_seed: dict[int, BackendRecommendation] = {}
        random_by_seed: dict[int, BackendRecommendation] = {}

        for seed in seeds:
            try:
                reused = None
                if seed == runtime_seed:
                    reused = main_map.get(backend.name)
                if reused is not None and reused.trial_history:
                    opt_rec = reused
                else:
                    opt_rec = optimize_backend(
                        backend=backend,
                        dataset=dataset,
                        top_k=top_k,
                        n_trials=n_trials,
                        target_recall=target_recall,
                        timeout=timeout,
                        seed=seed,
                        top_n_trials=top_n_trials,
                        local_refine=local_refine,
                        stage1_ratio=stage1_ratio,
                        constraints=constraints,
                        tracking_sink=None,
                        sampler="tpe",
                        record_history=True,
                    )
                optuna_by_seed[seed] = opt_rec
            except Exception as exc:
                backend_data["errors"].append(f"optuna run failed (seed={seed}): {exc}")

            try:
                rnd_rec = optimize_backend(
                    backend=backend,
                    dataset=dataset,
                    top_k=top_k,
                    n_trials=n_trials,
                    target_recall=target_recall,
                    timeout=timeout,
                    seed=seed,
                    top_n_trials=top_n_trials,
                    local_refine=local_refine,
                    stage1_ratio=stage1_ratio,
                    constraints=constraints,
                    tracking_sink=None,
                    sampler="random",
                    record_history=True,
                )
                random_by_seed[seed] = rnd_rec
            except Exception as exc:
                backend_data["errors"].append(f"random run failed (seed={seed}): {exc}")

        opt_metrics = [rec.metrics for rec in optuna_by_seed.values()]
        rnd_metrics = [rec.metrics for rec in random_by_seed.values()]

        distribution: dict[str, Any] = {}
        baseline = backend_data.get("baseline")
        if baseline:
            distribution["baseline-fixed"] = _distribution_summary([baseline["metrics"]])
        if rnd_metrics:
            distribution["random-search"] = _distribution_summary(rnd_metrics)
        if opt_metrics:
            distribution["optuna"] = _distribution_summary(opt_metrics)
        backend_data["distribution"] = distribution

        opt_histories = [rec.trial_history or [] for rec in optuna_by_seed.values()]
        rnd_histories = [rec.trial_history or [] for rec in random_by_seed.values()]
        backend_data["anytime"] = _anytime_summary(
            optuna_histories=opt_histories,
            random_histories=rnd_histories,
            target_recall=target_recall,
            constraints=constraints,
        )
        backend_data["impact"] = _impact_summary(
            baseline_metrics=(baseline["metrics"] if baseline else None),
            optuna_by_seed=optuna_by_seed,
            random_by_seed=random_by_seed,
            target_recall=target_recall,
            constraints=constraints,
        )

        wins = 0
        ties = 0
        losses = 0
        pair_count = 0
        for seed in seeds:
            if seed not in optuna_by_seed or seed not in random_by_seed:
                continue
            pair_count += 1
            result = _compare_final_metrics(
                optuna_metrics=optuna_by_seed[seed].metrics,
                random_metrics=random_by_seed[seed].metrics,
            )
            if result == "win":
                wins += 1
            elif result == "loss":
                losses += 1
            else:
                ties += 1
        backend_data["win_rate"] = {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "n_pairs": pair_count,
            "win_rate": float(wins / pair_count) if pair_count else 0.0,
        }
        analysis["by_backend"][backend.name] = backend_data

    return analysis
