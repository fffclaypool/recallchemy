from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .backends import resolve_backends
from .comparison_analysis import build_comparison_analysis
from .dataset import load_dataset
from .metrics import compute_ground_truth
from .optimizer import BackendRecommendation, optimize_backend
from .report import serialize_recommendations_payload, write_comparison_reports
from .scenario import DEFAULT_RUNTIME, load_scenario
from .tracking import build_tracking_sink
from .types import DatasetBundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="recallchemy-tune",
        description="Tune ANN/vector-index parameters with Optuna for a given dataset",
    )
    parser.add_argument("--scenario", default=None, help="Scenario YAML file path")
    parser.add_argument("--dataset", default=None, help="Input dataset path (.hdf5/.h5/.npz/.npy)")
    parser.add_argument(
        "--metric",
        default=None,
        choices=["euclidean", "angular", "cosine", "dot", "l2", "ip", "inner_product"],
        help="Distance metric. If omitted, inferred from dataset when possible.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        help="Backends to tune: all, hnswlib, annoy, faiss-ivf",
    )
    parser.add_argument("--top-k", type=int, default=None, help="k for nearest-neighbor search")
    parser.add_argument("--trials", type=int, default=None, help="Optuna trials per backend")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout seconds per backend")
    parser.add_argument(
        "--target-recall",
        type=float,
        default=None,
        help="Selection threshold for recommended config",
    )
    parser.add_argument("--min-recall", type=float, default=None, help="Hard lower bound for recall")
    parser.add_argument("--max-p95-ms", type=float, default=None, help="Hard upper bound for p95 query latency (ms)")
    parser.add_argument("--max-mean-ms", type=float, default=None, help="Hard upper bound for mean query latency (ms)")
    parser.add_argument("--max-build-time-s", type=float, default=None, help="Hard upper bound for build time (s)")
    parser.add_argument(
        "--query-fraction",
        type=float,
        default=None,
        help="If dataset has no query split, keep this fraction as query set",
    )
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap for train vectors")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap for query vectors")
    parser.add_argument("--gt-batch-size", type=int, default=None, help="Batch size for exact ground truth")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--disable-local-refine",
        action="store_true",
        help="Disable stage-2 local refinement and run only global search",
    )
    parser.add_argument(
        "--stage1-ratio",
        type=float,
        default=None,
        help="Fraction of trials used for stage-1 global search when local refinement is enabled",
    )
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--top-n-trials", type=int, default=None, help="Keep top-N trial summaries per backend")
    parser.add_argument(
        "--compare-seeds",
        type=int,
        default=None,
        help="Enable comparison analysis with N seeds (Optuna vs random + baseline)",
    )
    parser.add_argument(
        "--compare-seed-start",
        type=int,
        default=None,
        help="Starting seed for comparison analysis; defaults to --seed",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases tracking")
    parser.add_argument("--wandb-project", default=None, help="WandB project name")
    parser.add_argument("--wandb-entity", default=None, help="WandB entity/team")
    parser.add_argument("--wandb-run-name", default=None, help="WandB run name")
    parser.add_argument("--wandb-group", default=None, help="WandB run group")
    parser.add_argument("--wandb-job-type", default=None, help="WandB job type")
    parser.add_argument("--wandb-mode", default=None, help="WandB mode (online/offline/disabled)")
    parser.add_argument("--wandb-tags", nargs="+", default=None, help="WandB tags")
    return parser.parse_args()


def _dataset_with_ground_truth(dataset: DatasetBundle, top_k: int, batch_size: int) -> DatasetBundle:
    if dataset.ground_truth is not None and dataset.ground_truth.shape[1] >= top_k:
        return dataset
    ground_truth = compute_ground_truth(
        train=dataset.train,
        queries=dataset.queries,
        k=top_k,
        metric=dataset.metric,
        batch_size=batch_size,
    )
    return DatasetBundle(
        train=dataset.train,
        queries=dataset.queries,
        metric=dataset.metric,
        ground_truth=ground_truth,
    )


def _print_summary(recommendations: list[BackendRecommendation]) -> None:
    print("")
    print("Recommended configurations:")
    for rec in recommendations:
        metrics = rec.metrics
        print(
            f"- {rec.backend}: recall={metrics['recall']:.4f}, "
            f"ndcg@k={metrics.get('ndcg_at_k', 0.0):.4f}, "
            f"mrr@k={metrics.get('mrr_at_k', 0.0):.4f}, "
            f"p95={metrics['p95_query_ms']:.3f}ms, "
            f"build={metrics['build_time_s']:.3f}s, params={rec.params}"
        )


def _collect_constraints_from_args(args: argparse.Namespace) -> dict[str, float]:
    constraints: dict[str, float] = {}
    if args.min_recall is not None:
        constraints["min_recall"] = float(args.min_recall)
    if args.max_p95_ms is not None:
        constraints["max_p95_query_ms"] = float(args.max_p95_ms)
    if args.max_mean_ms is not None:
        constraints["max_mean_query_ms"] = float(args.max_mean_ms)
    if args.max_build_time_s is not None:
        constraints["max_build_time_s"] = float(args.max_build_time_s)
    return constraints


def _build_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.scenario:
        runtime = load_scenario(args.scenario)
    else:
        runtime = dict(DEFAULT_RUNTIME)

    if args.dataset is not None:
        runtime["dataset"] = args.dataset
    if args.metric is not None:
        runtime["metric"] = args.metric
    if args.backends is not None:
        runtime["backends"] = list(args.backends)
    if args.top_k is not None:
        runtime["top_k"] = int(args.top_k)
    if args.trials is not None:
        runtime["trials"] = int(args.trials)
    if args.timeout is not None:
        runtime["timeout"] = int(args.timeout)
    if args.target_recall is not None:
        runtime["target_recall"] = float(args.target_recall)
    if args.query_fraction is not None:
        runtime["query_fraction"] = float(args.query_fraction)
    if args.max_train is not None:
        runtime["max_train"] = int(args.max_train)
    if args.max_queries is not None:
        runtime["max_queries"] = int(args.max_queries)
    if args.gt_batch_size is not None:
        runtime["gt_batch_size"] = int(args.gt_batch_size)
    if args.seed is not None:
        runtime["seed"] = int(args.seed)
    if args.stage1_ratio is not None:
        runtime["stage1_ratio"] = float(args.stage1_ratio)
    if args.output is not None:
        runtime["output"] = str(args.output)
    if args.top_n_trials is not None:
        runtime["top_n_trials"] = int(args.top_n_trials)
    if args.compare_seeds is not None:
        runtime["compare_seeds"] = max(0, int(args.compare_seeds))
    if args.compare_seed_start is not None:
        runtime["compare_seed_start"] = int(args.compare_seed_start)

    if args.disable_local_refine:
        runtime["local_refine"] = False
    elif "local_refine" not in runtime:
        runtime["local_refine"] = bool(DEFAULT_RUNTIME["local_refine"])

    merged_constraints = dict(runtime.get("constraints", {}))
    merged_constraints.update(_collect_constraints_from_args(args))
    runtime["constraints"] = merged_constraints

    wandb_cfg = dict(runtime.get("wandb", {}))
    if args.wandb:
        wandb_cfg["enabled"] = True
    if args.wandb_project is not None:
        wandb_cfg["project"] = args.wandb_project
    if args.wandb_entity is not None:
        wandb_cfg["entity"] = args.wandb_entity
    if args.wandb_run_name is not None:
        wandb_cfg["run_name"] = args.wandb_run_name
    if args.wandb_group is not None:
        wandb_cfg["group"] = args.wandb_group
    if args.wandb_job_type is not None:
        wandb_cfg["job_type"] = args.wandb_job_type
    if args.wandb_mode is not None:
        wandb_cfg["mode"] = args.wandb_mode
    if args.wandb_tags is not None:
        wandb_cfg["tags"] = [str(x) for x in args.wandb_tags]
    runtime["wandb"] = wandb_cfg

    if not runtime.get("dataset"):
        raise ValueError("dataset is required. Set --dataset or dataset.path in --scenario.")
    return runtime


def main() -> None:
    args = parse_args()
    runtime = _build_runtime_config(args)

    dataset = load_dataset(
        path=runtime["dataset"],
        metric=runtime["metric"],
        query_fraction=runtime["query_fraction"],
        seed=runtime["seed"],
        max_train=runtime["max_train"],
        max_queries=runtime["max_queries"],
    )

    print(
        f"dataset loaded: train={dataset.train.shape}, queries={dataset.queries.shape}, "
        f"metric={dataset.metric}"
    )
    dataset = _dataset_with_ground_truth(
        dataset=dataset,
        top_k=runtime["top_k"],
        batch_size=runtime["gt_batch_size"],
    )
    print(f"ground truth ready: shape={dataset.ground_truth.shape}")

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(Path(runtime["dataset"]).resolve()),
        "metric": dataset.metric,
        "train_size": int(dataset.train.shape[0]),
        "query_size": int(dataset.queries.shape[0]),
        "dimension": int(dataset.train.shape[1]),
        "top_k": int(runtime["top_k"]),
        "trials_per_backend": int(runtime["trials"]),
        "target_recall": float(runtime["target_recall"]),
        "local_refine": bool(runtime["local_refine"]),
        "stage1_ratio": float(runtime["stage1_ratio"]),
        "compare_seeds": int(runtime.get("compare_seeds", 0)),
        "compare_seed_start": runtime.get("compare_seed_start"),
        "constraints": runtime.get("constraints", {}),
        "backend_overrides": runtime.get("backend_overrides", {}),
        "scenario_path": runtime.get("scenario_path"),
        "scenario_name": runtime.get("scenario_name"),
        "scenario_version": runtime.get("scenario_version"),
        "wandb": runtime.get("wandb", {}),
    }
    tracking_sink = build_tracking_sink(runtime=runtime, dataset_meta=metadata)
    try:
        backends, skipped = resolve_backends(
            runtime["backends"],
            backend_overrides=runtime.get("backend_overrides"),
        )
        for name, reason in skipped.items():
            print(f"skip backend '{name}': {reason}")
        if not backends:
            raise RuntimeError("No available backends. Install optional deps or adjust --backends.")

        analysis_enabled = int(runtime.get("compare_seeds", 0)) > 0
        recommendations: list[BackendRecommendation] = []
        for backend in backends:
            print(f"optimizing: {backend.name} ({runtime['trials']} trials)")
            recommendation = optimize_backend(
                backend=backend,
                dataset=dataset,
                top_k=runtime["top_k"],
                n_trials=runtime["trials"],
                timeout=runtime["timeout"],
                target_recall=runtime["target_recall"],
                seed=runtime["seed"],
                top_n_trials=runtime["top_n_trials"],
                local_refine=runtime["local_refine"],
                stage1_ratio=runtime["stage1_ratio"],
                constraints=runtime.get("constraints"),
                tracking_sink=tracking_sink,
                sampler="tpe",
                record_history=analysis_enabled,
            )
            recommendations.append(recommendation)
            print(
                f"best {backend.name}: recall={recommendation.metrics['recall']:.4f}, "
                f"p95={recommendation.metrics['p95_query_ms']:.3f}ms"
            )

        comparison_analysis: dict[str, Any] | None = None
        if analysis_enabled:
            compare_seed_start = runtime.get("compare_seed_start")
            if compare_seed_start is None:
                compare_seed_start = int(runtime["seed"])
            print(
                f"comparison analysis: seeds={runtime['compare_seeds']} "
                f"(start={compare_seed_start})"
            )
            comparison_analysis = build_comparison_analysis(
                backends=backends,
                dataset=dataset,
                top_k=runtime["top_k"],
                n_trials=runtime["trials"],
                target_recall=runtime["target_recall"],
                timeout=runtime["timeout"],
                top_n_trials=runtime["top_n_trials"],
                local_refine=runtime["local_refine"],
                stage1_ratio=runtime["stage1_ratio"],
                constraints=runtime.get("constraints"),
                seed_start=int(compare_seed_start),
                seed_count=int(runtime["compare_seeds"]),
                runtime_seed=int(runtime["seed"]),
                main_optuna_recommendations={rec.backend: rec for rec in recommendations},
            )

        metadata["skipped_backends"] = skipped
        payload = serialize_recommendations_payload(
            recommendations,
            metadata,
            comparison_analysis=comparison_analysis,
        )
        output = Path(runtime["output"])
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        report_md, report_html = write_comparison_reports(
            output_json_path=output,
            recommendations=recommendations,
            metadata=metadata,
            comparison_analysis=comparison_analysis,
        )

        _print_summary(recommendations)
        print(f"\nresults written: {output.resolve()}")
        print(f"comparison report (markdown): {report_md.resolve()}")
        print(f"comparison report (html): {report_html.resolve()}")
        tracking_sink.log_run_summary(metadata=metadata)
    finally:
        tracking_sink.finish()


if __name__ == "__main__":
    main()
