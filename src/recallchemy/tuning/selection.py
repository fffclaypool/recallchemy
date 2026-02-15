from __future__ import annotations

from typing import Any


def recommendation_order_key(
    *,
    recall: float,
    p95_query_ms: float,
    build_time_s: float,
    target_recall: float,
) -> tuple[float, float, float, float]:
    if recall >= target_recall:
        # Unified decision rule: once target is met, lower p95 wins.
        return (0.0, p95_query_ms, build_time_s, -recall)
    return (1.0, -recall, p95_query_ms, build_time_s)


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


def _meets_target_under_constraints(
    row: dict[str, Any],
    *,
    target_recall: float,
    constraints: dict[str, float] | None,
) -> bool:
    if not _passes_constraints(row, constraints):
        return False
    return float(row["recall"]) >= float(target_recall)


def _stage_has_target_hit(
    rows: list[dict[str, Any]],
    *,
    target_recall: float,
    constraints: dict[str, float] | None,
) -> bool:
    return any(
        _meets_target_under_constraints(row, target_recall=target_recall, constraints=constraints) for row in rows
    )


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

    best = min(
        constrained_rows,
        key=lambda row: recommendation_order_key(
            recall=float(row["recall"]),
            p95_query_ms=float(row["p95_query_ms"]),
            build_time_s=float(row["build_time_s"]),
            target_recall=float(target_recall),
        ),
    )
    if float(best["recall"]) >= target_recall:
        rationale = f"Selected lowest p95 latency among trials with recall >= {target_recall:.3f}"
        if constraints_applied:
            if len(constrained_rows) < len(rows):
                rationale += " while satisfying scenario constraints"
            else:
                rationale += " (all trials satisfied constraints)"
    else:
        rationale = f"No trial reached recall >= {target_recall:.3f}; selected the highest-recall trial"
        if constraints_applied and len(constrained_rows) == len(rows):
            rationale += " under constraints"
        elif constraints_applied:
            rationale += " under relaxed constraints (no trial satisfied all constraints)"
    return best, rationale


__all__ = [
    "recommendation_order_key",
    "_passes_constraints",
    "_meets_target_under_constraints",
    "_stage_has_target_hit",
    "select_recommendation",
]
