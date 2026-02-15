# recallchemy

CLI tool that tunes ANN/vector-index parameters with Optuna for a given dataset.
It uses candidate values inspired by `ann-benchmarks` as search-space priors.

## Features

- Tune from `hdf5 / npz / npy` datasets
- Per-backend optimization for `hnswlib`, `annoy`, and `faiss-ivf`
- Hybrid search spaces: ann-benchmarks priors + log-scale integer ranges
- Conditional parameters (example: FAISS `ivf_flat` vs `ivf_pq`)
- Two-stage optimization: global search then local refinement near the best config
- Metrics: `recall@k`, `p95 query latency`, `build time`
- Recommendation policy: lowest p95 latency among configs meeting `target_recall`
- JSON output with recommendation and top trials
- Auto-generated comparison reports (`.comparison.md` and `.comparison.html`) with backend table and Pareto view
- Impact-oriented analysis: baseline-vs-optuna improvement summary and time-to-target trial counts

## Install

```bash
pip install -e .
```

To run backend benchmarks:

```bash
pip install -e ".[backends]"
```

To enable Weights & Biases tracking:

```bash
pip install -e ".[wandb]"
```

To run tests:

```bash
pip install -e ".[dev]"
pytest -q
```

## Usage

```bash
recallchemy-tune \
  --dataset /path/to/glove-100-angular.hdf5 \
  --metric angular \
  --backends hnswlib annoy faiss-ivf \
  --trials 40 \
  --stage1-ratio 0.6 \
  --top-k 10 \
  --target-recall 0.95 \
  --max-queries 500 \
  --output recommendations.json
```

When `--output recommendations.json` is used, these files are also generated automatically:
- `recommendations.comparison.md`
- `recommendations.comparison.html`

To include robustness analysis in the markdown report
(`1. Anytime`, `2. Impact summary`, `3. Distribution`, `4. Win-rate`), run with comparison seeds:

```bash
recallchemy-tune \
  --dataset /path/to/glove-100-angular.hdf5 \
  --metric angular \
  --backends hnswlib annoy faiss-ivf \
  --trials 30 \
  --compare-seeds 5 \
  --compare-seed-start 100 \
  --output recommendations.json
```

Or:

```bash
python -m recallchemy \
  --dataset /path/to/data.npz \
  --backends all
```

## How to Read Results

When `compare_seeds > 0`, the markdown report includes:
- `1. Anytime`
- `2. Impact summary`
- `3. Distribution`
- `4. Win-rate`

### Impact summary (section 2)

- `recall gain vs baseline`: recall improvement over fixed baseline params
- `p95 reduction vs baseline`: latency reduction percentage vs baseline
  - positive is faster
  - negative means slower than baseline
- `p95 reduction vs random`: latency reduction percentage vs random search (same budget)
- `time-to-target`: average trial index where `target_recall` is first reached
- `reach_rate`: fraction of seeds that reached `target_recall`

Recommended interpretation:
- first, verify both methods have high `reach_rate`
- if both reach `target_recall`, compare `p95` and prefer lower latency
- if one side does not reach target reliably, recall feasibility is the bottleneck

### Win-rate (section 4)

Win-rate compares Optuna and random search seed-by-seed under equal budget.

- `win`: Optuna better on that seed
- `loss`: random better
- `tie`: equivalent
- `win_rate = wins / n_pairs`

Decision rule:
- if both meet `target_recall`, lower `p95_query_ms` wins
- if only one meets `target_recall`, that one wins
- otherwise, higher recall wins (then lower p95 as tie-breaker)

## GitHub Actions + Git LFS dataset

This repository includes a manual workflow:
- `.github/workflows/tune-from-lfs-dataset.yml`

It generates:
- `recommendations.json`
- `recommendations.comparison.md`
- `recommendations.comparison.html`

as GitHub Actions artifacts.

### 1. Track datasets with Git LFS

The repository is configured to track `datasets/**` with Git LFS
(`.gitattributes`).

```bash
git lfs install
mkdir -p datasets
cp /path/to/glove-100-angular.hdf5 datasets/
git add .gitattributes datasets/glove-100-angular.hdf5
git commit -m "chore: add dataset via git lfs"
git push
```

### 2. Run from Actions UI

Open Actions -> `Tune From LFS Dataset` -> `Run workflow`, then set:
- `scenario_name` (primary): choose the scenario (example: `ambiguous-hard`)
- optional `dataset_path`: repository-relative dataset override
  (empty means `dataset.path` from the selected scenario)
- optional tuning overrides (`backends`, `trials`, `top_k`, `target_recall`, etc.)
- optional `trial_budgets`: space-separated budgets (example: `10 30 80`) to run budget sweep in one job

The workflow resolves the selected scenario, validates that the resolved dataset is LFS-tracked,
pulls the LFS object, runs `recallchemy`, and uploads report artifacts (`artifacts/**`,
including per-budget outputs).

If `trials` is empty, scenario default `optimization.trials` is used.

Scenario-driven run:

```bash
recallchemy-tune --scenario scenarios/example.yaml
```

Enable WandB from CLI:

```bash
recallchemy-tune \
  --scenario scenarios/example.yaml \
  --wandb \
  --wandb-project recallchemy \
  --wandb-run-name exp-01 \
  --wandb-tags annbench stage2
```

WandB visualization tips:
- Track locally only (offline mode):
  ```bash
  recallchemy-tune --scenario scenarios/example.yaml --wandb --wandb-mode offline
  ```
- Sync offline run to cloud later:
  ```bash
  wandb sync ./wandb/offline-run-*
  ```
- Logged metrics include:
  - per trial: `"{backend}/{stage}/recall"`, `"{backend}/{stage}/p95_query_ms"`, `"{backend}/{stage}/build_time_s"`
  - recommended config: `"{backend}/recommended_*"`
  - run metadata in `wandb.config` (dataset, target recall, constraints, scenario info)

CLI overrides can be mixed with scenario:

```bash
recallchemy-tune \
  --scenario scenarios/example.yaml \
  --trials 30 \
  --output /tmp/override_result.json
```

## Scenario YAML

`scenario.yaml` lets you define constraints and backend-specific search-space limits in one file.

Top-level sections:
- `version`, `name`
- `dataset`: `path`, `metric`, `max_train`, `max_queries`, `query_fraction`, `seed`
- `evaluation`: `top_k`, `target_recall`, `gt_batch_size`
- `optimization`: `trials`, `timeout`, `local_refine`, `stage1_ratio`, `seed`
- `analysis`: `compare_seeds`, `seed_start`
- `backends`: `include` (for example: `[annoy, hnswlib]`)
- `constraints`: `min_recall`, `max_p95_ms`, `max_mean_ms`, `max_build_time_s`
- `backend_overrides`: backend-specific ranges/choices
- `output`: `path`, `save_top_trials`
- `wandb`: `enabled`, `project`, `entity`, `run_name`, `group`, `job_type`, `tags`, `mode`

Override examples:
- `annoy`: `n_trees_range`, `search_k_range`, `n_trees_prior_choices`, `search_k_prior_choices`
- `hnswlib`: `M_range`, `ef_construction_range`, `ef_search_range`, and `*_prior_choices`
- `faiss-ivf`: `n_list_range`, `n_probe_range`, `index_types`, `pq_m_choices`, `pq_nbits_choices`, `coarse_hnsw_m_choices`, `rerank_k_factor_choices`

## Input formats

- `.hdf5/.h5`:
  - Required: `train`
  - Optional: `test` or `queries`
  - Optional: `neighbors` or `ground_truth`
  - Optional: `attrs["distance"]`
- `.npz`:
  - Required: `train`
  - Optional: `test` / `queries` / `neighbors` / `ground_truth` / `distance`
- `.npy`:
  - 2D array (auto split into train/query with `query_fraction`)

## Hard dataset recipe (ambiguous / crowded neighbors)

To create a harder angular dataset where nearest neighbors are more ambiguous:

```bash
python scripts/generate_ambiguous_dataset.py \
  --output datasets/ambiguous-hard-angular.npz \
  --train-size 120000 \
  --query-size 8000 \
  --dim 256 \
  --latent-dim 48 \
  --n-centers 512 \
  --cluster-noise 0.22 \
  --duplicate-fraction 0.10
```

Then run:

```bash
recallchemy-tune --scenario scenarios/ambiguous-hard.yaml
```

## Search-space priors from ann-benchmarks

Candidate values are seeded from these files (checked on 2026-02-14):

- `ann_benchmarks/algorithms/hnswlib/config.yml`
- `ann_benchmarks/algorithms/annoy/config.yml`
- `ann_benchmarks/algorithms/faiss/config.yml`

These priors are combined with extended search ranges:
- `hnswlib`: `M`, `ef_construction`, `ef_search` via prior-guided or log-int search
- `annoy`: `n_trees`, `search_k` via prior-guided or extended modes (`absolute`, `tree_scaled`, `auto`)
- `faiss`: `n_list`, `n_probe`, conditional `index_type` (`ivf_flat`, `ivf_hnsw_flat`, `ivf_pq`, `ivf_hnsw_pq`)
  plus `coarse_hnsw_m`, `pq_m`, `pq_nbits`, and query-time `rerank_k_factor`
- Stage-2 local refinement keeps index family fixed and narrows numeric ranges around stage-1 best values
  Use `--disable-local-refine` to fall back to single-stage search.

## Notes

- This MVP targets index libraries (`hnswlib`, `annoy`, `faiss`).
  Server-style vector DBs such as Qdrant/Milvus/Weaviate can be added with the same backend interface.
- For large datasets, exact ground-truth computation is expensive.
  Use `--max-train` and `--max-queries` to control runtime.
