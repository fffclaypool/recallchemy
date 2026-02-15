from __future__ import annotations

from typing import Any

import optuna

from ..backends import VectorBackend
from ..tracking import TrackingSink
from ..types import DatasetBundle, EvaluationResult
from .importance import _compute_param_importance
from .models import BackendRecommendation
from .sampler import _build_sampler, _split_trials
from .selection import _stage_has_target_hit, select_recommendation


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
        stage2_mode = "local"
        anchor_params: dict[str, Any] | None = None
        if _stage_has_target_hit(stage1_rows, target_recall=target_recall, constraints=constraints):
            anchor, _ = select_recommendation(
                stage1_rows,
                target_recall=target_recall,
                constraints=constraints,
            )
            anchor_params = anchor["params"]
        else:
            # Local refinement can get trapped around low-recall anchors when stage-1
            # never reaches the recall target. Continue global exploration instead.
            stage2_mode = "global_fallback"

        stage2_sampler = _build_sampler(sampler, seed=seed + 1, n_trials=stage2_trials)
        stage2_study = optuna.create_study(
            directions=["maximize", "minimize", "minimize"],
            sampler=stage2_sampler,
            study_name=f"{backend.name}-optimization-stage2",
        )

        def stage2_objective(trial: optuna.Trial) -> tuple[float, float, float]:
            if stage2_mode == "local":
                assert anchor_params is not None
                params = backend.suggest_local_params(
                    trial=trial,
                    n_train=dataset.train.shape[0],
                    dim=dataset.train.shape[1],
                    top_k=top_k,
                    anchor_params=anchor_params,
                )
            else:
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
                        stage=stage2_mode,
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
                    stage=stage2_mode,
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
            _trial_to_row(t, stage=stage2_mode, trial_offset=stage1_trials)
            for t in stage2_study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.values is not None
        ]
        all_rows.extend(stage2_rows)
        if tracking_sink is not None and stage2_rows:
            for metric in ("recall", "p95_query_ms", "build_time_s"):
                tracking_sink.log_param_importance(
                    backend=backend.name,
                    stage=stage2_mode,
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


__all__ = ["optimize_backend"]
