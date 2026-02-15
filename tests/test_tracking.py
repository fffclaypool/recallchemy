from recallchemy.tracking import build_tracking_sink


def test_build_tracking_sink_returns_null_when_disabled():
    sink = build_tracking_sink(
        runtime={"wandb": {"enabled": False}},
        dataset_meta={"dataset_path": "/tmp/data.npy"},
    )
    sink.log_trial(
        backend="annoy",
        stage="global",
        trial=0,
        state="complete",
        params={"a": 1},
        recall=0.9,
    )
    sink.log_recommendation(
        backend="annoy",
        params={"a": 1},
        metrics={"recall": 0.9},
        rationale="ok",
    )
    sink.log_run_summary(metadata={"x": 1})
    sink.log_param_importance(
        backend="annoy",
        stage="global",
        metric="p95_query_ms",
        importances={"n_trees": 0.7, "search_k": 0.3},
    )
    sink.finish()
