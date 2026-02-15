import optuna

from recallchemy.backends import AnnoyBackend, FaissIVFBackend, HnswlibBackend, _pq_m_candidates


def _assert_profiles_share_study(backend, *, n_train: int, dim: int, top_k: int):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    study.enqueue_trial({"space_profile": "annbench"})
    study.enqueue_trial({"space_profile": "extended"})

    def objective(trial: optuna.Trial) -> float:
        backend.suggest_params(trial=trial, n_train=n_train, dim=dim, top_k=top_k)
        return 0.0

    study.optimize(objective, n_trials=2)
    assert len(study.trials) == 2


def test_hnsw_extended_space_fixed_trial():
    trial = optuna.trial.FixedTrial(
        {
            "space_profile": "extended",
            "M_ext": 16,
            "ef_construction_ext": 320,
            "ef_search_ext": 80,
        }
    )
    params = HnswlibBackend().suggest_params(trial=trial, n_train=100000, dim=64, top_k=10)
    assert params["space_profile"] == "extended"
    assert params["M"] == 16
    assert params["ef_construction"] == 320
    assert params["ef_search"] == 80


def test_hnsw_local_space_fixed_trial():
    trial = optuna.trial.FixedTrial(
        {
            "M_local": 18,
            "ef_construction_local": 400,
            "ef_search_local": 120,
        }
    )
    params = HnswlibBackend().suggest_local_params(
        trial=trial,
        n_train=100000,
        dim=64,
        top_k=10,
        anchor_params={"M": 16, "ef_construction": 320, "ef_search": 100},
    )
    assert params["space_profile"] == "local_refine"
    assert params["M"] == 18
    assert params["ef_construction"] == 400
    assert params["ef_search"] == 120


def test_annoy_extended_tree_scaled_mode():
    trial = optuna.trial.FixedTrial(
        {
            "space_profile": "extended",
            "n_trees_ext": 300,
            "search_k_mode": "tree_scaled",
            "tree_scaled_factor": 10.0,
        }
    )
    params = AnnoyBackend().suggest_params(trial=trial, n_train=10000, dim=32, top_k=10)
    assert params["space_profile"] == "extended"
    assert params["n_trees"] == 300
    assert params["search_k_mode"] == "tree_scaled"
    assert params["search_k"] == 3000


def test_annoy_local_space_fixed_trial():
    trial = optuna.trial.FixedTrial(
        {
            "n_trees_local": 240,
            "search_k_mode_local": "absolute",
            "search_k_local": 2800,
        }
    )
    params = AnnoyBackend().suggest_local_params(
        trial=trial,
        n_train=20000,
        dim=32,
        top_k=10,
        anchor_params={"n_trees": 200, "search_k": 3200},
    )
    assert params["space_profile"] == "local_refine"
    assert params["n_trees"] == 240
    assert params["search_k"] == 2800


def test_annoy_override_ranges_are_applied():
    backend = AnnoyBackend(overrides={"n_trees_range": [300, 300], "search_k_range": [2000, 2000]})
    trial = optuna.trial.FixedTrial(
        {
            "space_profile": "extended",
            "n_trees_ext": 300,
            "search_k_mode": "absolute",
            "search_k_ext": 2000,
        }
    )
    params = backend.suggest_params(trial=trial, n_train=10000, dim=32, top_k=10)
    assert params["n_trees"] == 300
    assert params["search_k"] == 2000


def test_faiss_extended_ivf_pq_mode():
    trial = optuna.trial.FixedTrial(
        {
            "space_profile": "extended",
            "n_list_ext": 256,
            "n_probe_ext": 32,
            "index_type": "ivf_pq",
            "rerank_k_factor": 4,
            "pq_m": 16,
            "pq_nbits": 8,
        }
    )
    params = FaissIVFBackend().suggest_params(trial=trial, n_train=500000, dim=128, top_k=10)
    assert params["space_profile"] == "extended"
    assert params["index_type"] == "ivf_pq"
    assert params["n_list"] == 256
    assert params["n_probe"] == 32
    assert params["rerank_k_factor"] == 4
    assert params["pq_m"] == 16
    assert params["pq_nbits"] == 8


def test_pq_candidates_for_prime_dimension():
    assert _pq_m_candidates(97) == []


def test_faiss_extended_ivf_hnsw_pq_mode():
    trial = optuna.trial.FixedTrial(
        {
            "space_profile": "extended",
            "n_list_ext": 512,
            "n_probe_ext": 64,
            "index_type": "ivf_hnsw_pq",
            "coarse_hnsw_m": 32,
            "rerank_k_factor": 8,
            "pq_m": 16,
            "pq_nbits": 8,
        }
    )
    params = FaissIVFBackend().suggest_params(trial=trial, n_train=500000, dim=128, top_k=10)
    assert params["index_type"] == "ivf_hnsw_pq"
    assert params["coarse_hnsw_m"] == 32
    assert params["rerank_k_factor"] == 8


def test_faiss_local_space_fixed_trial():
    trial = optuna.trial.FixedTrial(
        {
            "n_list_local": 600,
            "n_probe_local": 50,
            "rerank_k_factor_local": 8,
            "coarse_hnsw_m_local": 24,
            "pq_m_local": 16,
            "pq_nbits_local": 8,
        }
    )
    params = FaissIVFBackend().suggest_local_params(
        trial=trial,
        n_train=500000,
        dim=128,
        top_k=10,
        anchor_params={
            "index_type": "ivf_hnsw_pq",
            "n_list": 512,
            "n_probe": 48,
            "rerank_k_factor": 4,
            "coarse_hnsw_m": 16,
            "pq_m": 16,
            "pq_nbits": 8,
        },
    )
    assert params["space_profile"] == "local_refine"
    assert params["index_type"] == "ivf_hnsw_pq"
    assert params["n_list"] == 600
    assert params["n_probe"] == 50
    assert params["rerank_k_factor"] == 8
    assert params["coarse_hnsw_m"] == 24
    assert params["pq_m"] == 16
    assert params["pq_nbits"] == 8


def test_hnsw_profiles_do_not_conflict_in_optuna():
    _assert_profiles_share_study(HnswlibBackend(), n_train=100000, dim=96, top_k=10)


def test_annoy_profiles_do_not_conflict_in_optuna():
    _assert_profiles_share_study(AnnoyBackend(), n_train=50000, dim=64, top_k=10)


def test_faiss_profiles_do_not_conflict_in_optuna():
    _assert_profiles_share_study(FaissIVFBackend(), n_train=200000, dim=96, top_k=10)
