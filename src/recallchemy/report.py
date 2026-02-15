from __future__ import annotations

from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Any

from .optimizer import BackendRecommendation


def _recommendation_rows(recommendations: list[BackendRecommendation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in recommendations:
        metrics = rec.metrics
        rows.append(
            {
                "backend": rec.backend,
                "recall": float(metrics.get("recall", 0.0)),
                "p95_query_ms": float(metrics.get("p95_query_ms", float("inf"))),
                "mean_query_ms": float(metrics.get("mean_query_ms", float("inf"))),
                "build_time_s": float(metrics.get("build_time_s", float("inf"))),
                "ndcg_at_k": float(metrics.get("ndcg_at_k", 0.0)),
                "mrr_at_k": float(metrics.get("mrr_at_k", 0.0)),
                "rationale": rec.rationale,
            }
        )
    rows.sort(key=lambda row: (-row["recall"], row["p95_query_ms"], row["build_time_s"]))
    return rows


def _pareto_front_indices(rows: list[dict[str, Any]]) -> list[int]:
    # Objectives: maximize recall, minimize p95 latency.
    frontier: list[int] = []
    for i, cand in enumerate(rows):
        dominated = False
        for j, other in enumerate(rows):
            if i == j:
                continue
            recall_ok = other["recall"] >= cand["recall"]
            latency_ok = other["p95_query_ms"] <= cand["p95_query_ms"]
            strictly_better = other["recall"] > cand["recall"] or other["p95_query_ms"] < cand["p95_query_ms"]
            if recall_ok and latency_ok and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(i)
    return frontier


def _is_finite_number(value: Any) -> bool:
    try:
        x = float(value)
    except Exception:
        return False
    return x == x and x not in (float("inf"), float("-inf"))


def _fmt_number(value: Any, decimals: int = 4) -> str:
    if not _is_finite_number(value):
        return "-"
    return f"{float(value):.{decimals}f}"


def _fmt_mean_std(stats: dict[str, Any], decimals: int = 4) -> str:
    n = int(stats.get("n", 0) or 0)
    if n <= 0:
        return "-"
    mean = stats.get("mean")
    std = stats.get("std")
    if n == 1:
        return _fmt_number(mean, decimals=decimals)
    return f"{_fmt_number(mean, decimals=decimals)}±{_fmt_number(std, decimals=decimals)}"


def _fmt_ci(stats: dict[str, Any], decimals: int = 4) -> str:
    n = int(stats.get("n", 0) or 0)
    if n <= 0:
        return "-"
    return f"[{_fmt_number(stats.get('ci_low'), decimals=decimals)}, {_fmt_number(stats.get('ci_high'), decimals=decimals)}]"


def _fmt_pct(value: Any, decimals: int = 1) -> str:
    if not _is_finite_number(value):
        return "-"
    return f"{float(value):.{decimals}f}%"


def _fmt_multiplier(value: Any, decimals: int = 2) -> str:
    if not _is_finite_number(value):
        return "-"
    return f"{float(value):.{decimals}f}x"


def _append_analysis_markdown(lines: list[str], comparison_analysis: dict[str, Any]) -> None:
    by_backend = comparison_analysis.get("by_backend", {})
    if not by_backend:
        return

    seeds = comparison_analysis.get("seeds", [])
    lines.extend(
        [
            "",
            "## analysis setup",
            "",
            f"- comparison_seeds: {len(seeds)}",
            f"- seeds: {seeds}",
            "",
            "## 1. Anytime (best-so-far by trial)",
            "",
        ]
    )

    for backend in sorted(by_backend.keys()):
        section = by_backend[backend]
        lines.extend(
            [
                f"### backend: {backend}",
                "",
                "| trial | optuna best recall (mean±std) | optuna best p95_ms (mean±std) | random best recall (mean±std) | random best p95_ms (mean±std) |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        anytime_rows = section.get("anytime", [])
        if not anytime_rows:
            lines.append("| - | - | - | - | - |")
        else:
            for row in anytime_rows:
                optuna = row.get("optuna", {})
                random = row.get("random", {})
                lines.append(
                    "| "
                    f"{int(row.get('trial', 0))} | "
                    f"{_fmt_mean_std(optuna.get('recall', {}), decimals=4)} | "
                    f"{_fmt_mean_std(optuna.get('p95_query_ms', {}), decimals=4)} | "
                    f"{_fmt_mean_std(random.get('recall', {}), decimals=4)} | "
                    f"{_fmt_mean_std(random.get('p95_query_ms', {}), decimals=4)} |"
                )
        lines.append("")

    lines.extend(
        [
            "## 2. Impact summary (baseline vs optimization)",
            "",
            "| backend | recall gain vs baseline | p95 reduction vs baseline | p95 speedup vs baseline | p95 reduction vs random | p95 speedup vs random | optuna time-to-target (mean trial) | random time-to-target (mean trial) | optuna reach_rate | random reach_rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for backend in sorted(by_backend.keys()):
        impact = by_backend[backend].get("impact", {})
        ttt = impact.get("time_to_target", {})
        opt_ttt = ttt.get("optuna", {})
        rnd_ttt = ttt.get("random", {})
        lines.append(
            "| "
            f"{backend} | "
            f"{_fmt_number(impact.get('recall_gain_vs_baseline'), decimals=4)} | "
            f"{_fmt_pct(impact.get('p95_reduction_vs_baseline_pct'), decimals=1)} | "
            f"{_fmt_multiplier(impact.get('p95_speedup_vs_baseline_x'), decimals=2)} | "
            f"{_fmt_pct(impact.get('p95_reduction_vs_random_pct'), decimals=1)} | "
            f"{_fmt_multiplier(impact.get('p95_speedup_vs_random_x'), decimals=2)} | "
            f"{_fmt_number(opt_ttt.get('stats', {}).get('mean'), decimals=2)} | "
            f"{_fmt_number(rnd_ttt.get('stats', {}).get('mean'), decimals=2)} | "
            f"{_fmt_pct(float(opt_ttt.get('reach_rate', float('nan'))) * 100.0, decimals=1)} | "
            f"{_fmt_pct(float(rnd_ttt.get('reach_rate', float('nan'))) * 100.0, decimals=1)} |"
        )
    lines.append("")

    lines.extend(
        [
            "## 3. Distribution (final best across seeds)",
            "",
            "| backend | method | recall mean±std | recall 95% CI | ndcg mean±std | ndcg 95% CI | mrr mean±std | mrr 95% CI | p95_ms mean±std | p95_ms 95% CI | build_s mean±std |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for backend in sorted(by_backend.keys()):
        dist = by_backend[backend].get("distribution", {})
        for method in ("baseline-fixed", "random-search", "optuna"):
            item = dist.get(method)
            if not item:
                continue
            recall_stats = item.get("recall", {})
            ndcg_stats = item.get("ndcg_at_k", {})
            mrr_stats = item.get("mrr_at_k", {})
            p95_stats = item.get("p95_query_ms", {})
            build_stats = item.get("build_time_s", {})
            lines.append(
                "| "
                f"{backend} | {method} | "
                f"{_fmt_mean_std(recall_stats, decimals=4)} | "
                f"{_fmt_ci(recall_stats, decimals=4)} | "
                f"{_fmt_mean_std(ndcg_stats, decimals=4)} | "
                f"{_fmt_ci(ndcg_stats, decimals=4)} | "
                f"{_fmt_mean_std(mrr_stats, decimals=4)} | "
                f"{_fmt_ci(mrr_stats, decimals=4)} | "
                f"{_fmt_mean_std(p95_stats, decimals=4)} | "
                f"{_fmt_ci(p95_stats, decimals=4)} | "
                f"{_fmt_mean_std(build_stats, decimals=4)} |"
            )

    lines.extend(
        [
            "",
            "## 4. Win-rate (Optuna vs Random, same budget)",
            "",
            "| backend | wins | ties | losses | win_rate |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for backend in sorted(by_backend.keys()):
        wr = by_backend[backend].get("win_rate", {})
        lines.append(
            "| "
            f"{backend} | {int(wr.get('wins', 0))} | {int(wr.get('ties', 0))} | "
            f"{int(wr.get('losses', 0))} | {float(wr.get('win_rate', 0.0)) * 100.0:.1f}% |"
        )

    error_lines: list[str] = []
    for backend in sorted(by_backend.keys()):
        errors = by_backend[backend].get("errors", [])
        for err in errors:
            error_lines.append(f"- {backend}: {err}")
    if error_lines:
        lines.extend(["", "## analysis errors", ""])
        lines.extend(error_lines)


def _build_markdown(
    rows: list[dict[str, Any]],
    pareto_indices: list[int],
    metadata: dict[str, Any],
    comparison_analysis: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# recallchemy backend comparison",
        "",
        f"- generated_at: {metadata.get('generated_at')}",
        f"- dataset: {metadata.get('dataset_path')}",
        f"- metric: {metadata.get('metric')}",
        f"- target_recall: {metadata.get('target_recall')}",
        "",
        "## comparison table",
        "",
        "| backend | recall | ndcg_at_k | mrr_at_k | p95_query_ms | mean_query_ms | build_time_s | pareto |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    pareto_set = set(pareto_indices)
    for i, row in enumerate(rows):
        lines.append(
            "| "
            f"{row['backend']} | {row['recall']:.4f} | {row['ndcg_at_k']:.4f} | {row['mrr_at_k']:.4f} | "
            f"{row['p95_query_ms']:.4f} | {row['mean_query_ms']:.4f} | {row['build_time_s']:.4f} | "
            f"{'yes' if i in pareto_set else 'no'} |"
        )

    lines.extend(["", "## target recall qualified (ranked by p95)", ""])
    target_recall = metadata.get("target_recall")
    if _is_finite_number(target_recall):
        qualified = [row for row in rows if row["recall"] >= float(target_recall)]
        lines.extend(
            [
                "| rank | backend | recall | ndcg_at_k | mrr_at_k | p95_query_ms |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        if qualified:
            for rank, row in enumerate(
                sorted(qualified, key=lambda r: (r["p95_query_ms"], r["build_time_s"], -r["recall"])),
                start=1,
            ):
                lines.append(
                    "| "
                    f"{rank} | {row['backend']} | {row['recall']:.4f} | {row['ndcg_at_k']:.4f} | "
                    f"{row['mrr_at_k']:.4f} | {row['p95_query_ms']:.4f} |"
                )
        else:
            lines.append("| - | - | - | - | - | - |")
    else:
        lines.append("- target_recall is not set")

    lines.extend(["", "## pareto front (recall up, p95 down)", ""])
    if pareto_indices:
        for i in pareto_indices:
            row = rows[i]
            lines.append(f"- {row['backend']}: recall={row['recall']:.4f}, p95={row['p95_query_ms']:.4f}ms")
    else:
        lines.append("- none")
    lines.extend(["", "## recommendation rationale", ""])
    for row in rows:
        lines.append(f"- {row['backend']}: {row['rationale']}")
    lines.append("")

    if comparison_analysis:
        _append_analysis_markdown(lines, comparison_analysis)
        lines.append("")
    return "\n".join(lines)


def _build_svg(rows: list[dict[str, Any]], pareto_indices: list[int]) -> str:
    width = 900
    height = 420
    margin_left = 80
    margin_right = 40
    margin_top = 30
    margin_bottom = 70

    if not rows:
        return "<svg width='900' height='420' xmlns='http://www.w3.org/2000/svg'></svg>"

    x_values = [row["p95_query_ms"] for row in rows]
    y_values = [row["recall"] for row in rows]
    x_min = min(0.0, min(x_values))
    x_max = max(x_values)
    if x_max <= x_min:
        x_max = x_min + 1.0
    y_min = min(0.0, min(y_values))
    y_max = max(1.0, max(y_values))
    if y_max <= y_min:
        y_max = y_min + 1.0

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    def x_px(value: float) -> float:
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_w

    def y_px(value: float) -> float:
        return margin_top + (1.0 - ((value - y_min) / (y_max - y_min))) * plot_h

    pareto_set = set(pareto_indices)
    parts: list[str] = [
        f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>",
        "<rect x='0' y='0' width='100%' height='100%' fill='#f8fafc'/>",
        f"<line x1='{margin_left}' y1='{height-margin_bottom}' x2='{width-margin_right}' y2='{height-margin_bottom}' stroke='#334155' stroke-width='1'/>",
        f"<line x1='{margin_left}' y1='{margin_top}' x2='{margin_left}' y2='{height-margin_bottom}' stroke='#334155' stroke-width='1'/>",
        f"<text x='{width/2}' y='{height-20}' text-anchor='middle' font-size='13' fill='#0f172a'>p95 query latency (ms) lower is better</text>",
        f"<text x='18' y='{height/2}' text-anchor='middle' transform='rotate(-90 18 {height/2})' font-size='13' fill='#0f172a'>recall higher is better</text>",
    ]
    for i, row in enumerate(rows):
        cx = x_px(row["p95_query_ms"])
        cy = y_px(row["recall"])
        fill = "#dc2626" if i in pareto_set else "#334155"
        radius = 7 if i in pareto_set else 5
        name = escape(str(row["backend"]))
        parts.append(f"<circle cx='{cx:.2f}' cy='{cy:.2f}' r='{radius}' fill='{fill}'/>")
        parts.append(
            f"<text x='{cx+8:.2f}' y='{cy-8:.2f}' font-size='12' fill='#0f172a'>{name}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _build_html(rows: list[dict[str, Any]], pareto_indices: list[int], metadata: dict[str, Any]) -> str:
    table_rows: list[str] = []
    pareto_set = set(pareto_indices)
    for i, row in enumerate(rows):
        table_rows.append(
            "<tr>"
            f"<td>{escape(str(row['backend']))}</td>"
            f"<td>{row['recall']:.4f}</td>"
            f"<td>{row['ndcg_at_k']:.4f}</td>"
            f"<td>{row['mrr_at_k']:.4f}</td>"
            f"<td>{row['p95_query_ms']:.4f}</td>"
            f"<td>{row['mean_query_ms']:.4f}</td>"
            f"<td>{row['build_time_s']:.4f}</td>"
            f"<td>{'yes' if i in pareto_set else 'no'}</td>"
            "</tr>"
        )
    svg = _build_svg(rows, pareto_indices)
    return (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>recallchemy comparison report</title>"
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,sans-serif;background:#eef2ff;color:#111827;margin:0;padding:24px;}"
        ".card{background:#fff;border-radius:12px;padding:16px 18px;margin-bottom:16px;box-shadow:0 6px 20px rgba(15,23,42,0.08);}"
        "table{border-collapse:collapse;width:100%;font-size:14px;}"
        "th,td{padding:8px 10px;border-bottom:1px solid #e5e7eb;text-align:left;}"
        "th{background:#f8fafc;}"
        "h1,h2{margin:0 0 12px 0;}"
        "</style></head><body>"
        "<div class='card'>"
        "<h1>recallchemy backend comparison</h1>"
        f"<div>dataset: {escape(str(metadata.get('dataset_path')))}</div>"
        f"<div>metric: {escape(str(metadata.get('metric')))} | target_recall: {escape(str(metadata.get('target_recall')))}</div>"
        f"<div>generated_at: {escape(str(metadata.get('generated_at')))}</div>"
        "</div>"
        "<div class='card'>"
        "<h2>pareto scatter</h2>"
        f"{svg}"
        "</div>"
        "<div class='card'>"
        "<h2>comparison table</h2>"
        "<table><thead><tr>"
        "<th>backend</th><th>recall</th><th>ndcg_at_k</th><th>mrr_at_k</th><th>p95_query_ms</th><th>mean_query_ms</th><th>build_time_s</th><th>pareto</th>"
        "</tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table></div></body></html>"
    )


def write_comparison_reports(
    *,
    output_json_path: Path,
    recommendations: list[BackendRecommendation],
    metadata: dict[str, Any],
    comparison_analysis: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    rows = _recommendation_rows(recommendations)
    pareto_indices = _pareto_front_indices(rows)

    md_path = output_json_path.with_suffix(".comparison.md")
    html_path = output_json_path.with_suffix(".comparison.html")
    md_path.write_text(
        _build_markdown(rows, pareto_indices, metadata, comparison_analysis=comparison_analysis),
        encoding="utf-8",
    )
    html_path.write_text(_build_html(rows, pareto_indices, metadata), encoding="utf-8")
    return md_path, html_path


def serialize_recommendations_payload(
    recommendations: list[BackendRecommendation],
    metadata: dict[str, Any],
    comparison_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "metadata": metadata,
        "recommendations": [asdict(r) for r in recommendations],
    }
    if comparison_analysis is not None:
        payload["comparison_analysis"] = comparison_analysis
    return payload
