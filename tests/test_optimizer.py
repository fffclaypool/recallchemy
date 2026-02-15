from recallchemy.optimizer import _compute_param_importance, _split_trials, _tpe_startup_trials, select_recommendation


def test_select_recommendation_prefers_low_latency_when_target_met():
    rows = [
        {"trial": 0, "recall": 0.95, "p95_query_ms": 5.0, "build_time_s": 1.0, "mean_query_ms": 4.0, "params": {"a": 1}},
        {"trial": 1, "recall": 0.96, "p95_query_ms": 4.0, "build_time_s": 1.5, "mean_query_ms": 3.8, "params": {"a": 2}},
    ]
    selected, rationale = select_recommendation(rows, target_recall=0.94)
    assert selected["trial"] == 1
    assert "recall >=" in rationale


def test_select_recommendation_falls_back_to_best_recall():
    rows = [
        {"trial": 0, "recall": 0.70, "p95_query_ms": 2.0, "build_time_s": 1.0, "mean_query_ms": 1.5, "params": {"a": 1}},
        {"trial": 1, "recall": 0.75, "p95_query_ms": 4.0, "build_time_s": 0.5, "mean_query_ms": 2.0, "params": {"a": 2}},
    ]
    selected, rationale = select_recommendation(rows, target_recall=0.9)
    assert selected["trial"] == 1
    assert "No trial reached" in rationale


def test_select_recommendation_respects_constraints():
    rows = [
        {"trial": 0, "recall": 0.99, "p95_query_ms": 20.0, "build_time_s": 1.0, "mean_query_ms": 10.0, "params": {"a": 1}},
        {"trial": 1, "recall": 0.96, "p95_query_ms": 5.0, "build_time_s": 1.2, "mean_query_ms": 3.0, "params": {"a": 2}},
    ]
    selected, rationale = select_recommendation(
        rows,
        target_recall=0.95,
        constraints={"max_p95_query_ms": 10.0},
    )
    assert selected["trial"] == 1
    assert "constraints" in rationale


def test_compute_param_importance_returns_empty_on_no_trials():
    got = _compute_param_importance([], metric_key="recall")
    assert got == {}


def test_compute_param_importance_uses_row_params():
    rows = [
        {"recall": 0.90, "p95_query_ms": 10.0, "build_time_s": 1.1, "params": {"n_trees": 50, "search_k": 300}},
        {"recall": 0.95, "p95_query_ms": 8.0, "build_time_s": 1.3, "params": {"n_trees": 120, "search_k": 800}},
        {"recall": 0.99, "p95_query_ms": 6.0, "build_time_s": 1.6, "params": {"n_trees": 300, "search_k": 2000}},
    ]
    got = _compute_param_importance(rows, metric_key="recall")
    assert "n_trees" in got
    assert "search_k" in got


def test_split_trials_biases_stage1_for_small_budget():
    stage1, stage2 = _split_trials(n_trials=8, local_refine=True, stage1_ratio=0.6)
    assert stage1 == 6
    assert stage2 == 2


def test_split_trials_disables_two_stage_when_too_few_trials():
    stage1, stage2 = _split_trials(n_trials=5, local_refine=True, stage1_ratio=0.6)
    assert stage1 == 5
    assert stage2 == 0


def test_tpe_startup_trials_scales_with_budget():
    assert _tpe_startup_trials(2) == 1
    assert _tpe_startup_trials(8) == 2
    assert _tpe_startup_trials(40) == 10
