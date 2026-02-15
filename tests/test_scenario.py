from pathlib import Path

from recallchemy.scenario import DEFAULT_RUNTIME, load_scenario


def test_load_scenario_parses_sections(tmp_path: Path):
    scenario_file = tmp_path / "scenario.yaml"
    scenario_file.write_text(
        """
version: 1
name: smoke
dataset:
  path: /tmp/data.npz
  metric: euclidean
  max_train: 2000
  max_queries: 300
evaluation:
  top_k: 20
  target_recall: 0.98
optimization:
  trials: 40
  timeout: 120
  local_refine: false
  stage1_ratio: 0.7
  seed: 7
analysis:
  compare_seeds: 4
  seed_start: 100
backends:
  include: [annoy]
constraints:
  min_recall: 0.95
  max_p95_ms: 12.5
backend_overrides:
  annoy:
    n_trees_range: [50, 600]
output:
  path: /tmp/out.json
  save_top_trials: 8
wandb:
  enabled: true
  project: recallchemy
  run_name: smoke-run
  tags: [test, smoke]
""".strip(),
        encoding="utf-8",
    )

    cfg = load_scenario(scenario_file)
    assert cfg["dataset"] == "/tmp/data.npz"
    assert cfg["metric"] == "euclidean"
    assert cfg["max_train"] == 2000
    assert cfg["max_queries"] == 300
    assert cfg["top_k"] == 20
    assert cfg["target_recall"] == 0.98
    assert cfg["trials"] == 40
    assert cfg["timeout"] == 120
    assert cfg["local_refine"] is False
    assert cfg["stage1_ratio"] == 0.7
    assert cfg["seed"] == 7
    assert cfg["compare_seeds"] == 4
    assert cfg["compare_seed_start"] == 100
    assert cfg["backends"] == ["annoy"]
    assert cfg["constraints"]["min_recall"] == 0.95
    assert cfg["constraints"]["max_p95_query_ms"] == 12.5
    assert "max_p95_ms" not in cfg["constraints"]
    assert cfg["backend_overrides"]["annoy"]["n_trees_range"] == [50, 600]
    assert cfg["output"] == "/tmp/out.json"
    assert cfg["top_n_trials"] == 8
    assert cfg["wandb"]["enabled"] is True
    assert cfg["wandb"]["project"] == "recallchemy"
    assert cfg["wandb"]["run_name"] == "smoke-run"
    assert cfg["wandb"]["tags"] == ["test", "smoke"]
    assert cfg["scenario_name"] == "smoke"


def test_default_runtime_compare_seeds_is_robust():
    assert int(DEFAULT_RUNTIME["compare_seeds"]) >= 5
