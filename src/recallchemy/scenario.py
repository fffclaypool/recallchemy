from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_RUNTIME: dict[str, Any] = {
    "metric": None,
    "backends": ["all"],
    "top_k": 10,
    "trials": 30,
    "timeout": None,
    "target_recall": 0.95,
    "query_fraction": 0.1,
    "max_train": None,
    "max_queries": 1000,
    "gt_batch_size": 64,
    "seed": 42,
    "local_refine": True,
    "stage1_ratio": 0.6,
    "output": "recommendations.json",
    "top_n_trials": 5,
    "compare_seeds": 5,
    "compare_seed_start": None,
    "constraints": {},
    "backend_overrides": {},
    "wandb": {
        "enabled": False,
        "project": None,
        "entity": None,
        "run_name": None,
        "group": None,
        "job_type": None,
        "tags": [],
        "mode": None,
    },
}


def _as_dict(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"scenario: '{name}' must be a mapping")
    return dict(value)


def _as_list(value: Any, *, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"scenario: '{name}' must be a list")
    return list(value)


def _normalize_constraints(raw: dict[str, Any]) -> dict[str, float]:
    alias_to_canonical = {
        "max_p95_ms": "max_p95_query_ms",
        "max_mean_ms": "max_mean_query_ms",
    }
    constraints: dict[str, float] = {}
    for key, value in raw.items():
        if value is None:
            continue
        normalized_key = alias_to_canonical.get(str(key), str(key))
        # Keep explicit canonical values if both alias and canonical are provided.
        if str(key) in alias_to_canonical and normalized_key in constraints:
            continue
        constraints[normalized_key] = float(value)
    return constraints


def _normalize_wandb(raw: dict[str, Any]) -> dict[str, Any]:
    cfg = {
        "enabled": bool(raw.get("enabled", False)),
        "project": raw.get("project"),
        "entity": raw.get("entity"),
        "run_name": raw.get("run_name"),
        "group": raw.get("group"),
        "job_type": raw.get("job_type"),
        "mode": raw.get("mode"),
    }
    tags = raw.get("tags")
    if tags is None:
        cfg["tags"] = []
    elif isinstance(tags, list):
        cfg["tags"] = [str(x) for x in tags]
    else:
        raise ValueError("scenario: 'wandb.tags' must be a list")
    return cfg


def load_scenario(path: str | Path) -> dict[str, Any]:
    scenario_path = Path(path)
    raw_loaded = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    raw = _as_dict(raw_loaded, name="root")

    dataset = _as_dict(raw.get("dataset"), name="dataset")
    if "path" not in dataset:
        raise ValueError("scenario: dataset.path is required")

    evaluation = _as_dict(raw.get("evaluation"), name="evaluation")
    optimization = _as_dict(raw.get("optimization"), name="optimization")
    analysis = _as_dict(raw.get("analysis"), name="analysis")
    backends = _as_dict(raw.get("backends"), name="backends")
    output = _as_dict(raw.get("output"), name="output")
    constraints = _as_dict(raw.get("constraints"), name="constraints")
    backend_overrides = _as_dict(raw.get("backend_overrides"), name="backend_overrides")
    wandb = _as_dict(raw.get("wandb"), name="wandb")

    include = _as_list(backends.get("include"), name="backends.include")
    if not include:
        include = list(DEFAULT_RUNTIME["backends"])

    cfg = dict(DEFAULT_RUNTIME)
    cfg.update(
        {
            "dataset": str(dataset["path"]),
            "metric": dataset.get("metric", cfg["metric"]),
            "query_fraction": float(dataset.get("query_fraction", cfg["query_fraction"])),
            "max_train": dataset.get("max_train", cfg["max_train"]),
            "max_queries": dataset.get("max_queries", cfg["max_queries"]),
            "seed": int(dataset.get("seed", optimization.get("seed", cfg["seed"]))),
            "backends": [str(x) for x in include],
            "top_k": int(evaluation.get("top_k", cfg["top_k"])),
            "target_recall": float(evaluation.get("target_recall", cfg["target_recall"])),
            "gt_batch_size": int(evaluation.get("gt_batch_size", cfg["gt_batch_size"])),
            "trials": int(optimization.get("trials", cfg["trials"])),
            "timeout": optimization.get("timeout", cfg["timeout"]),
            "local_refine": bool(optimization.get("local_refine", cfg["local_refine"])),
            "stage1_ratio": float(optimization.get("stage1_ratio", cfg["stage1_ratio"])),
            "output": str(output.get("path", cfg["output"])),
            "top_n_trials": int(output.get("save_top_trials", cfg["top_n_trials"])),
            "compare_seeds": int(analysis.get("compare_seeds", cfg["compare_seeds"])),
            "compare_seed_start": (
                int(analysis["seed_start"])
                if analysis.get("seed_start") is not None
                else cfg["compare_seed_start"]
            ),
            "constraints": _normalize_constraints(constraints),
            "backend_overrides": backend_overrides,
            "wandb": _normalize_wandb(wandb),
            "scenario_path": str(scenario_path.resolve()),
            "scenario_name": str(raw.get("name", scenario_path.stem)),
            "scenario_version": int(raw.get("version", 1)),
        }
    )
    return cfg
