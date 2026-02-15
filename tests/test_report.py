from pathlib import Path

from recallchemy.optimizer import BackendRecommendation
from recallchemy.report import serialize_recommendations_payload, write_comparison_reports


def _recs() -> list[BackendRecommendation]:
    return [
        BackendRecommendation(
            backend="annoy",
            rationale="r1",
            metrics={"recall": 0.96, "mean_query_ms": 0.08, "p95_query_ms": 0.10, "build_time_s": 0.5},
            params={"n_trees": 100},
            top_trials=[],
        ),
        BackendRecommendation(
            backend="hnswlib",
            rationale="r2",
            metrics={"recall": 0.98, "mean_query_ms": 0.12, "p95_query_ms": 0.20, "build_time_s": 0.4},
            params={"M": 16},
            top_trials=[],
        ),
        BackendRecommendation(
            backend="faiss-ivf",
            rationale="r3",
            metrics={"recall": 0.95, "mean_query_ms": 0.18, "p95_query_ms": 0.30, "build_time_s": 0.6},
            params={"n_list": 64},
            top_trials=[],
        ),
    ]


def test_serialize_recommendations_payload():
    payload = serialize_recommendations_payload(_recs(), {"metric": "angular"})
    assert payload["metadata"]["metric"] == "angular"
    assert payload["recommendations"][0]["backend"] == "annoy"
    assert len(payload["recommendations"]) == 3


def test_write_comparison_reports(tmp_path: Path):
    output_json = tmp_path / "recommendations.json"
    output_json.write_text("{}", encoding="utf-8")

    md_path, html_path = write_comparison_reports(
        output_json_path=output_json,
        recommendations=_recs(),
        metadata={
            "generated_at": "2026-02-14T00:00:00+00:00",
            "dataset_path": "/tmp/data.npz",
            "metric": "angular",
            "target_recall": 0.95,
        },
    )
    assert md_path.exists()
    assert html_path.exists()

    md = md_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "## comparison table" in md
    assert "| hnswlib | 0.9800 | 0.2000 |" in md
    assert "- annoy: recall=0.9600, p95=0.1000ms" in md
    assert "- hnswlib: recall=0.9800, p95=0.2000ms" in md
    assert "pareto scatter" in html
    assert "faiss-ivf" in html


def test_write_comparison_reports_with_analysis_sections(tmp_path: Path):
    output_json = tmp_path / "recommendations.json"
    output_json.write_text("{}", encoding="utf-8")

    analysis = {
        "seed_start": 42,
        "seed_count": 2,
        "seeds": [42, 43],
        "by_backend": {
            "annoy": {
                "anytime": [
                    {
                        "trial": 1,
                        "optuna": {
                            "recall": {"n": 2, "mean": 0.91, "std": 0.01},
                            "p95_query_ms": {"n": 2, "mean": 0.12, "std": 0.01},
                        },
                        "random": {
                            "recall": {"n": 2, "mean": 0.88, "std": 0.02},
                            "p95_query_ms": {"n": 2, "mean": 0.18, "std": 0.03},
                        },
                    }
                ],
                "distribution": {
                    "baseline-fixed": {
                        "recall": {"n": 1, "mean": 0.9, "std": 0.0, "ci_low": 0.9, "ci_high": 0.9},
                        "p95_query_ms": {"n": 1, "mean": 0.2, "std": 0.0, "ci_low": 0.2, "ci_high": 0.2},
                        "build_time_s": {"n": 1, "mean": 0.05, "std": 0.0, "ci_low": 0.05, "ci_high": 0.05},
                    },
                    "random-search": {
                        "recall": {"n": 2, "mean": 0.92, "std": 0.01, "ci_low": 0.91, "ci_high": 0.93},
                        "p95_query_ms": {"n": 2, "mean": 0.14, "std": 0.02, "ci_low": 0.12, "ci_high": 0.16},
                        "build_time_s": {"n": 2, "mean": 0.04, "std": 0.01, "ci_low": 0.03, "ci_high": 0.05},
                    },
                    "optuna": {
                        "recall": {"n": 2, "mean": 0.95, "std": 0.01, "ci_low": 0.94, "ci_high": 0.96},
                        "p95_query_ms": {"n": 2, "mean": 0.1, "std": 0.01, "ci_low": 0.09, "ci_high": 0.11},
                        "build_time_s": {"n": 2, "mean": 0.04, "std": 0.0, "ci_low": 0.04, "ci_high": 0.04},
                    },
                },
                "win_rate": {"wins": 2, "ties": 0, "losses": 0, "win_rate": 1.0},
                "errors": [],
            }
        },
    }

    md_path, _ = write_comparison_reports(
        output_json_path=output_json,
        recommendations=_recs(),
        metadata={
            "generated_at": "2026-02-14T00:00:00+00:00",
            "dataset_path": "/tmp/data.npz",
            "metric": "angular",
            "target_recall": 0.95,
        },
        comparison_analysis=analysis,
    )
    md = md_path.read_text(encoding="utf-8")
    assert "## 1. Anytime (best-so-far by trial)" in md
    assert "## 3. Distribution (final best across seeds)" in md
    assert "## 4. Win-rate (Optuna vs Random, same budget)" in md
    assert "| annoy | 2 | 0 | 0 | 100.0% |" in md
