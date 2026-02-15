from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna

from .backends import VectorBackend
from .tracking import TrackingSink
from .types import DatasetBundle, EvaluationResult


@dataclass(slots=True)
class BackendRecommendation:
    backend: str
    rationale: str
    metrics: dict[str, float]
    params: dict[str, Any]
    top_trials: list[dict[str, Any]]
    trial_history: list[dict[str, Any]] | None = None


def _trial_to_row(
    trial: optuna.trial.FrozenTrial,
    *,
    stage: str,
    trial_offset: int = 0,
) -> dict[str, Any]:
    assert trial.values is not None
    recall, p95_ms, build_time = trial.values
    return {
        "trial": trial.number + trial_offset,
        "stage": stage,
        "recall": float(recall),
        "p95_query_ms": float(p95_ms),
        "build_time_s": float(build_time),
        "mean_query_ms": float(trial.user_attrs.get("mean_query_ms", 0.0)),
        "ndcg_at_k": float(trial.user_attrs.get("ndcg_at_k", 0.0)),
        "mrr_at_k": float(trial.user_attrs.get("mrr_at_k", 0.0)),
        "params": dict(trial.user_attrs.get("params", {})),
    }


def _set_trial_attrs(trial: optuna.Trial, result: EvaluationResult) -> None:
    trial.set_user_attr("params", result.params)
    trial.set_user_attr("mean_query_ms", result.mean_query_ms)
    trial.set_user_attr("p95_query_ms", result.p95_query_ms)
    trial.set_user_attr("build_time_s", result.build_time_s)
    trial.set_user_attr("ndcg_at_k", result.ndcg_at_k)
    trial.set_user_attr("mrr_at_k", result.mrr_at_k)


def _evaluate_with_backend(
    *,
    backend: VectorBackend,
    dataset: DatasetBundle,
    top_k: int,
    params: dict[str, Any],
) -> EvaluationResult:
    return backend.evaluate(
        train=dataset.train,
        queries=dataset.queries,
        ground_truth=dataset.ground_truth,
        metric=dataset.metric,
        top_k=top_k,
        params=params,
    )


def _split_trials(n_trials: int, local_refine: bool, stage1_ratio: float) -> tuple[int, int]:
    if n_trials <= 0:
        return 0, 0
    if not local_refine or n_trials < 6:
        return n_trials, 0
    ratio = min(0.9, max(0.5, stage1_ratio))
    stage1 = int(round(n_trials * ratio))
    if n_trials <= 12:
        # Keep most budget in stage-1 for small budgets so TPE can learn globally.
        stage1 = max(stage1, n_trials - 2)
    stage1 = min(n_trials - 1, max(2, stage1))
    return stage1, n_trials - stage1


def _tpe_startup_trials(n_trials: int) -> int:
    if n_trials <= 2:
        return 1
    return max(2, min(10, int(np.ceil(n_trials * 0.25))))


def _build_sampler(sampler: str, *, seed: int, n_trials: int) -> optuna.samplers.BaseSampler:
    if sampler == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=_tpe_startup_trials(n_trials),
        )
    if sampler == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"unsupported sampler={sampler!r}; expected 'tpe' or 'random'")


def _passes_constraints(row: dict[str, Any], constraints: dict[str, float] | None) -> bool:
    if not constraints:
        return True

    min_recall = constraints.get("min_recall")
    if min_recall is not None and row["recall"] < float(min_recall):
        return False

    max_p95 = constraints.get("max_p95_query_ms")
    if max_p95 is None:
        max_p95 = constraints.get("max_p95_ms")
    if max_p95 is not None and row["p95_query_ms"] > float(max_p95):
        return False

    max_mean = constraints.get("max_mean_query_ms")
    if max_mean is None:
        max_mean = constraints.get("max_mean_ms")
    if max_mean is not None and row["mean_query_ms"] > float(max_mean):
        return False

    max_build = constraints.get("max_build_time_s")
    if max_build is not None and row["build_time_s"] > float(max_build):
        return False

    return True


def select_recommendation(
    rows: list[dict[str, Any]],
    target_recall: float,
    constraints: dict[str, float] | None = None,
) -> tuple[dict[str, Any], str]:
    if not rows:
        raise ValueError("No completed rows to select from")

    constrained_rows = [row for row in rows if _passes_constraints(row, constraints)]
    constraints_applied = bool(constraints)
    if not constrained_rows:
        constrained_rows = rows

    satisfied = [row for row in constrained_rows if row["recall"] >= target_recall]
    if satisfied:
        best = min(satisfied, key=lambda x: (x["p95_query_ms"], x["build_time_s"], -x["recall"]))
        rationale = f"Selected lowest p95 latency among trials with recall >= {target_recall:.3f}"
        if constraints_applied:
            if len(constrained_rows) < len(rows):
                rationale += " while satisfying scenario constraints"
            else:
                rationale += " (all trials satisfied constraints)"
        return best, rationale

    best = max(constrained_rows, key=lambda x: (x["recall"], -x["p95_query_ms"], -x["build_time_s"]))
    rationale = f"No trial reached recall >= {target_recall:.3f}; selected the highest-recall trial"
    if constraints_applied and len(constrained_rows) == len(rows):
        rationale += " under constraints"
    elif constraints_applied:
        rationale += " under relaxed constraints (no trial satisfied all constraints)"
    return best, rationale


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


def optimize_backend(
    backend: VectorBackend,
    dataset: DatasetBundle,
    top_k: int,
    n_trials: int,
    target_recall: float,
    timeout: int | None = None,
    seed: int = 42,
    top_n_trials: int = 5,
    local_refine: bool = True,
    stage1_ratio: float = 0.6,
    constraints: dict[str, float] | None = None,
    tracking_sink: TrackingSink | None = None,
    sampler: str = "tpe",
    record_history: bool = False,
) -> BackendRecommendation:
    stage1_trials, stage2_trials = _split_trials(
        n_trials=n_trials,
        local_refine=local_refine,
        stage1_ratio=stage1_ratio,
    )
    if timeout is None:
        stage1_timeout = None
        stage2_timeout = None
    else:
        stage1_timeout = max(1, int(timeout * (stage1_trials / max(1, n_trials))))
        stage2_timeout = max(1, timeout - stage1_timeout) if stage2_trials > 0 else None

    stage1_sampler = _build_sampler(sampler, seed=seed, n_trials=stage1_trials)
    stage1_study = optuna.create_study(
        directions=["maximize", "minimize", "minimize"],
        sampler=stage1_sampler,
        study_name=f"{backend.name}-optimization-stage1",
    )

    def stage1_objective(trial: optuna.Trial) -> tuple[float, float, float]:
        params = backend.suggest_params(
            trial=trial,
            n_train=dataset.train.shape[0],
            dim=dataset.train.shape[1],
            top_k=top_k,
        )
        try:
            result = _evaluate_with_backend(
                backend=backend,
                dataset=dataset,
                top_k=top_k,
                params=params,
            )
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            if tracking_sink is not None:
                tracking_sink.log_trial(
                    backend=backend.name,
                    stage="global",
                    trial=trial.number,
                    state="pruned",
                    params=params,
                    error=str(exc),
                )
            raise optuna.TrialPruned(str(exc)) from exc

        _set_trial_attrs(trial, result)
        if tracking_sink is not None:
            tracking_sink.log_trial(
                backend=backend.name,
                stage="global",
                trial=trial.number,
                state="complete",
                params=result.params,
                recall=result.recall,
                mean_query_ms=result.mean_query_ms,
                p95_query_ms=result.p95_query_ms,
                build_time_s=result.build_time_s,
                ndcg_at_k=result.ndcg_at_k,
                mrr_at_k=result.mrr_at_k,
            )
        return (result.recall, result.p95_query_ms, result.build_time_s)

    stage1_study.optimize(
        stage1_objective,
        n_trials=stage1_trials,
        timeout=stage1_timeout,
        show_progress_bar=False,
    )

    stage1_rows = [
        _trial_to_row(t, stage="global", trial_offset=0)
        for t in stage1_study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
    ]
    if not stage1_rows:
        raise RuntimeError(f"{backend.name}: no successful trials")

    if tracking_sink is not None:
        for metric in ("recall", "p95_query_ms", "build_time_s"):
            tracking_sink.log_param_importance(
                backend=backend.name,
                stage="global",
                metric=metric,
                importances=_compute_param_importance(stage1_rows, metric_key=metric),
            )

    all_rows = list(stage1_rows)

    if stage2_trials > 0:
        anchor, _ = select_recommendation(
            stage1_rows,
            target_recall=target_recall,
            constraints=constraints,
        )
        anchor_params = anchor["params"]

        stage2_sampler = _build_sampler(sampler, seed=seed + 1, n_trials=stage2_trials)
        stage2_study = optuna.create_study(
            directions=["maximize", "minimize", "minimize"],
            sampler=stage2_sampler,
            study_name=f"{backend.name}-optimization-stage2",
        )

        def stage2_objective(trial: optuna.Trial) -> tuple[float, float, float]:
            params = backend.suggest_local_params(
                trial=trial,
                n_train=dataset.train.shape[0],
                dim=dataset.train.shape[1],
                top_k=top_k,
                anchor_params=anchor_params,
            )
            try:
                result = _evaluate_with_backend(
                    backend=backend,
                    dataset=dataset,
                    top_k=top_k,
                    params=params,
                )
            except Exception as exc:
                trial.set_user_attr("error", str(exc))
                if tracking_sink is not None:
                    tracking_sink.log_trial(
                        backend=backend.name,
                        stage="local",
                        trial=trial.number + stage1_trials,
                        state="pruned",
                        params=params,
                        error=str(exc),
                    )
                raise optuna.TrialPruned(str(exc)) from exc
            _set_trial_attrs(trial, result)
            if tracking_sink is not None:
                tracking_sink.log_trial(
                    backend=backend.name,
                    stage="local",
                    trial=trial.number + stage1_trials,
                    state="complete",
                    params=result.params,
                    recall=result.recall,
                    mean_query_ms=result.mean_query_ms,
                    p95_query_ms=result.p95_query_ms,
                    build_time_s=result.build_time_s,
                    ndcg_at_k=result.ndcg_at_k,
                    mrr_at_k=result.mrr_at_k,
                )
            return (result.recall, result.p95_query_ms, result.build_time_s)

        stage2_study.optimize(
            stage2_objective,
            n_trials=stage2_trials,
            timeout=stage2_timeout,
            show_progress_bar=False,
        )
        stage2_rows = [
            _trial_to_row(t, stage="local", trial_offset=stage1_trials)
            for t in stage2_study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
        ]
        all_rows.extend(stage2_rows)
        if tracking_sink is not None and stage2_rows:
            for metric in ("recall", "p95_query_ms", "build_time_s"):
                tracking_sink.log_param_importance(
                    backend=backend.name,
                    stage="local",
                    metric=metric,
                    importances=_compute_param_importance(stage2_rows, metric_key=metric),
                )

    selected, rationale = select_recommendation(
        all_rows,
        target_recall=target_recall,
        constraints=constraints,
    )
    ranked = sorted(
        all_rows,
        key=lambda row: (-row["recall"], row["p95_query_ms"], row["build_time_s"]),
    )

    recommendation = BackendRecommendation(
        backend=backend.name,
        rationale=rationale,
        metrics={
            "recall": float(selected["recall"]),
            "mean_query_ms": float(selected["mean_query_ms"]),
            "p95_query_ms": float(selected["p95_query_ms"]),
            "build_time_s": float(selected["build_time_s"]),
            "ndcg_at_k": float(selected.get("ndcg_at_k", 0.0)),
            "mrr_at_k": float(selected.get("mrr_at_k", 0.0)),
        },
        params=selected["params"],
        top_trials=ranked[:top_n_trials],
        trial_history=sorted(all_rows, key=lambda row: row["trial"]) if record_history else None,
    )
    if tracking_sink is not None:
        tracking_sink.log_recommendation(
            backend=backend.name,
            params=recommendation.params,
            metrics=recommendation.metrics,
            rationale=recommendation.rationale,
        )
    return recommendation
