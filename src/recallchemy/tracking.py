from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, prefix=full_key))
        else:
            flattened[full_key] = value
    return flattened


class TrackingSink:
    def log_trial(
        self,
        *,
        backend: str,
        stage: str,
        trial: int,
        state: str,
        params: dict[str, Any] | None = None,
        recall: float | None = None,
        mean_query_ms: float | None = None,
        p95_query_ms: float | None = None,
        build_time_s: float | None = None,
        error: str | None = None,
    ) -> None:
        del backend, stage, trial, state, params, recall, mean_query_ms, p95_query_ms, build_time_s, error

    def log_recommendation(
        self,
        *,
        backend: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        rationale: str,
    ) -> None:
        del backend, params, metrics, rationale

    def log_run_summary(self, *, metadata: dict[str, Any]) -> None:
        del metadata

    def log_param_importance(
        self,
        *,
        backend: str,
        stage: str,
        metric: str,
        importances: dict[str, float],
    ) -> None:
        del backend, stage, metric, importances

    def finish(self) -> None:
        return


class NullTrackingSink(TrackingSink):
    pass


@dataclass(slots=True)
class WandbConfig:
    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    job_type: str | None = None
    tags: list[str] | None = None
    mode: str | None = None


class WandbTrackingSink(TrackingSink):
    def __init__(
        self,
        *,
        config: WandbConfig,
        runtime: dict[str, Any],
        dataset_meta: dict[str, Any],
    ):
        try:
            import wandb
        except Exception as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "WandB is enabled but 'wandb' is not installed. "
                "Install with: pip install -e '.[wandb]'"
            ) from exc

        if not config.project:
            raise ValueError("WandB is enabled but project is missing")

        self._wandb = wandb
        run_config = {
            "runtime": runtime,
            "dataset": dataset_meta,
        }
        self._run = wandb.init(
            project=config.project,
            entity=config.entity,
            name=config.run_name,
            group=config.group,
            job_type=config.job_type,
            tags=config.tags,
            mode=config.mode,
            config=run_config,
        )

    def log_trial(
        self,
        *,
        backend: str,
        stage: str,
        trial: int,
        state: str,
        params: dict[str, Any] | None = None,
        recall: float | None = None,
        mean_query_ms: float | None = None,
        p95_query_ms: float | None = None,
        build_time_s: float | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "backend": backend,
            "stage": stage,
            "trial": trial,
            "state": state,
        }
        if recall is not None:
            payload[f"{backend}/{stage}/recall"] = recall
        if mean_query_ms is not None:
            payload[f"{backend}/{stage}/mean_query_ms"] = mean_query_ms
        if p95_query_ms is not None:
            payload[f"{backend}/{stage}/p95_query_ms"] = p95_query_ms
        if build_time_s is not None:
            payload[f"{backend}/{stage}/build_time_s"] = build_time_s
        if error:
            payload[f"{backend}/{stage}/error"] = error
        if params:
            flat = _flatten_dict(params, prefix=f"{backend}/{stage}/params")
            payload.update(flat)
        self._wandb.log(payload)

    def log_recommendation(
        self,
        *,
        backend: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        rationale: str,
    ) -> None:
        payload = {
            f"{backend}/recommended_recall": metrics.get("recall"),
            f"{backend}/recommended_mean_query_ms": metrics.get("mean_query_ms"),
            f"{backend}/recommended_p95_query_ms": metrics.get("p95_query_ms"),
            f"{backend}/recommended_build_time_s": metrics.get("build_time_s"),
        }
        payload.update(_flatten_dict(params, prefix=f"{backend}/recommended_params"))
        self._wandb.log(payload)
        self._run.summary[f"{backend}_rationale"] = rationale
        self._run.summary[f"{backend}_recommended_params_json"] = json.dumps(params, ensure_ascii=False)

    def log_run_summary(self, *, metadata: dict[str, Any]) -> None:
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                self._run.summary[f"run_{key}"] = value
            else:
                self._run.summary[f"run_{key}_json"] = json.dumps(value, ensure_ascii=False)

    def log_param_importance(
        self,
        *,
        backend: str,
        stage: str,
        metric: str,
        importances: dict[str, float],
    ) -> None:
        if not importances:
            return

        rows = [[name, float(score)] for name, score in importances.items()]
        table = self._wandb.Table(columns=["parameter", "importance"], data=rows)
        plot = self._wandb.plot.bar(
            table,
            "parameter",
            "importance",
            title=f"{backend} {stage} importance ({metric})",
        )
        payload: dict[str, Any] = {
            f"{backend}/{stage}/importance/{metric}_table": table,
            f"{backend}/{stage}/importance/{metric}_bar": plot,
        }
        for name, score in importances.items():
            payload[f"{backend}/{stage}/importance/{metric}/{name}"] = float(score)
        self._wandb.log(payload)
        self._run.summary[f"{backend}_{stage}_{metric}_importance_json"] = json.dumps(importances, ensure_ascii=False)

    def finish(self) -> None:
        self._run.finish()


def build_tracking_sink(
    *,
    runtime: dict[str, Any],
    dataset_meta: dict[str, Any],
) -> TrackingSink:
    wandb_cfg_raw = dict(runtime.get("wandb", {}))
    config = WandbConfig(
        enabled=bool(wandb_cfg_raw.get("enabled", False)),
        project=wandb_cfg_raw.get("project"),
        entity=wandb_cfg_raw.get("entity"),
        run_name=wandb_cfg_raw.get("run_name"),
        group=wandb_cfg_raw.get("group"),
        job_type=wandb_cfg_raw.get("job_type"),
        tags=list(wandb_cfg_raw.get("tags", [])) if wandb_cfg_raw.get("tags") else None,
        mode=wandb_cfg_raw.get("mode"),
    )
    if not config.enabled:
        return NullTrackingSink()
    return WandbTrackingSink(config=config, runtime=runtime, dataset_meta=dataset_meta)
