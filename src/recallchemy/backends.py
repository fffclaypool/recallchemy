from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

import numpy as np
import optuna
from numpy.typing import NDArray

from .metrics import canonical_metric, recall_at_k
from .priors import (
    ANNOY_N_TREES,
    ANNOY_N_TREES_RANGE,
    ANNOY_SEARCH_K,
    ANNOY_SEARCH_K_RANGE,
    FAISS_COARSE_HNSW_M,
    FAISS_IVF_NLIST,
    FAISS_IVF_NLIST_RANGE,
    FAISS_IVF_NPROBE,
    FAISS_IVF_NPROBE_RANGE,
    FAISS_PQ_M,
    FAISS_PQ_NBITS,
    FAISS_RERANK_K_FACTOR,
    HNSWLIB_EF_CONSTRUCTION,
    HNSWLIB_EF_CONSTRUCTION_RANGE,
    HNSWLIB_EF_SEARCH,
    HNSWLIB_EF_SEARCH_RANGE,
    HNSWLIB_M,
    HNSWLIB_M_RANGE,
)
from .types import EvaluationResult


def _ensure_k(ids: NDArray[np.int64], k: int) -> NDArray[np.int64]:
    if ids.shape[0] >= k:
        return ids[:k]
    padded = np.full((k,), -1, dtype=np.int64)
    padded[: ids.shape[0]] = ids
    return padded


def _bounds(low: int, high: int) -> tuple[int, int]:
    if high < low:
        return low, low
    return low, high


def _local_int_window(
    center: int,
    low: int,
    high: int,
    *,
    scale: float = 2.0,
    min_span: int = 2,
) -> tuple[int, int]:
    if center <= 0:
        center = low
    raw_low = int(center / scale)
    raw_high = int(center * scale)
    window_low = max(low, raw_low)
    window_high = min(high, raw_high)
    if window_high - window_low < min_span:
        half = max(1, min_span // 2)
        window_low = max(low, center - half)
        window_high = min(high, center + half)
    if window_high < window_low:
        return window_low, window_low
    return window_low, window_high


def _suggest_int_log(trial: optuna.Trial, name: str, low: int, high: int) -> int:
    low, high = _bounds(low, high)
    if low == high:
        return low
    if low <= 1:
        return trial.suggest_int(name, low, high)
    return trial.suggest_int(name, low, high, log=True)


def _pq_m_candidates(dim: int, preferred_values: list[int] | None = None) -> list[int]:
    base_values = preferred_values if preferred_values is not None else FAISS_PQ_M
    preferred = [m for m in base_values if m <= dim and dim % m == 0]
    if preferred:
        return preferred
    return [m for m in range(2, min(dim - 1, 128) + 1) if dim % m == 0 and m in set(base_values)]


def _neighbor_choices(values: list[int], anchor: int, radius: int = 1) -> list[int]:
    uniq = sorted(set(values))
    if not uniq:
        return []
    if anchor not in uniq:
        return uniq
    idx = uniq.index(anchor)
    lo = max(0, idx - radius)
    hi = min(len(uniq), idx + radius + 1)
    return uniq[lo:hi]


def _rerank_ids(
    query: NDArray[np.float32],
    candidate_ids: NDArray[np.int64],
    train: NDArray[np.float32],
    metric: str,
    top_k: int,
) -> NDArray[np.int64]:
    valid_ids = candidate_ids[candidate_ids >= 0]
    if valid_ids.size <= top_k:
        return valid_ids[:top_k]

    vectors = train[valid_ids]
    q = query.reshape(1, -1)
    if metric == "euclidean":
        distances = np.sum((vectors - q) * (vectors - q), axis=1)
        order = np.argsort(distances)
    else:
        # For angular metric, train/query are already normalized before this call.
        scores = (vectors @ q.T).reshape(-1)
        order = np.argsort(-scores)
    return valid_ids[order[:top_k]]


class VectorBackend(ABC):
    name: str
    module_name: str

    def __init__(self, overrides: dict[str, Any] | None = None):
        self.overrides = dict(overrides or {})

    @classmethod
    def availability(cls) -> tuple[bool, str | None]:
        try:
            importlib.import_module(cls.module_name)
            return True, None
        except Exception as exc:  # pragma: no cover - depends on environment
            return False, f"{cls.module_name} import failed: {exc}"

    @abstractmethod
    def suggest_params(self, trial: optuna.Trial, n_train: int, dim: int, top_k: int) -> dict[str, Any]:
        raise NotImplementedError

    def suggest_local_params(
        self,
        trial: optuna.Trial,
        n_train: int,
        dim: int,
        top_k: int,
        anchor_params: dict[str, Any],
    ) -> dict[str, Any]:
        del anchor_params
        return self.suggest_params(trial=trial, n_train=n_train, dim=dim, top_k=top_k)

    @abstractmethod
    def build_index(self, train: NDArray[np.float32], metric: str, params: dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def query_index(
        self,
        index: Any,
        query: NDArray[np.float32],
        top_k: int,
        metric: str,
        params: dict[str, Any],
    ) -> NDArray[np.int64]:
        raise NotImplementedError

    def evaluate(
        self,
        train: NDArray[np.float32],
        queries: NDArray[np.float32],
        ground_truth: NDArray[np.int64],
        metric: str,
        top_k: int,
        params: dict[str, Any],
    ) -> EvaluationResult:
        metric = canonical_metric(metric)

        build_start = perf_counter()
        index = self.build_index(train, metric, params)
        build_time = perf_counter() - build_start

        predictions = np.empty((queries.shape[0], top_k), dtype=np.int64)
        latencies = np.empty((queries.shape[0],), dtype=np.float64)

        for i, q in enumerate(queries):
            query_start = perf_counter()
            ids = self.query_index(index, q, top_k, metric, params)
            latencies[i] = (perf_counter() - query_start) * 1000.0
            predictions[i] = _ensure_k(np.asarray(ids, dtype=np.int64), top_k)

        return EvaluationResult(
            backend=self.name,
            params=params,
            recall=recall_at_k(predictions, ground_truth, top_k),
            mean_query_ms=float(np.mean(latencies)),
            p95_query_ms=float(np.percentile(latencies, 95)),
            build_time_s=float(build_time),
        )

    def _override_range(self, key: str, default: tuple[int, int]) -> tuple[int, int]:
        value = self.overrides.get(key)
        if value is None:
            return default
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{self.name}: override '{key}' must be [low, high]")
        low, high = int(value[0]), int(value[1])
        return _bounds(low, high)

    def _override_int_choices(self, key: str, default: list[int]) -> list[int]:
        value = self.overrides.get(key)
        if value is None:
            return sorted(set(int(v) for v in default))
        if not isinstance(value, list) or not value:
            raise ValueError(f"{self.name}: override '{key}' must be a non-empty list")
        return sorted(set(int(v) for v in value))

    def _override_str_choices(self, key: str, default: list[str]) -> list[str]:
        value = self.overrides.get(key)
        if value is None:
            return sorted(set(default))
        if not isinstance(value, list) or not value:
            raise ValueError(f"{self.name}: override '{key}' must be a non-empty list")
        return sorted(set(str(v) for v in value))


class HnswlibBackend(VectorBackend):
    name = "hnswlib"
    module_name = "hnswlib"

    def suggest_params(self, trial: optuna.Trial, n_train: int, dim: int, top_k: int) -> dict[str, Any]:
        del dim
        m_low, m_high_global = self._override_range("M_range", HNSWLIB_M_RANGE)
        m_prior_choices = self._override_int_choices("M_prior_choices", HNSWLIB_M)
        efc_prior_choices = self._override_int_choices("ef_construction_prior_choices", HNSWLIB_EF_CONSTRUCTION)
        efs_prior_choices = self._override_int_choices("ef_search_prior_choices", HNSWLIB_EF_SEARCH)
        efc_range = self._override_range("ef_construction_range", HNSWLIB_EF_CONSTRUCTION_RANGE)
        efs_range = self._override_range("ef_search_range", HNSWLIB_EF_SEARCH_RANGE)

        max_m = max(m_low, min(m_high_global, int(np.sqrt(n_train))))
        m_candidates = [m for m in m_prior_choices if m <= max_m and m >= m_low]
        if not m_candidates:
            m_candidates = [m_low, min(max_m, 16)]

        ef_search_candidates = [ef for ef in efs_prior_choices if ef >= top_k and ef <= max(top_k, n_train)]
        if not ef_search_candidates:
            ef_search_candidates = [max(top_k, efs_range[0])]

        profile = trial.suggest_categorical("space_profile", ["annbench", "extended"])
        if profile == "annbench":
            m = trial.suggest_categorical("M_prior", sorted(set(m_candidates)))
            ef_construction_candidates = [v for v in efc_prior_choices if v >= (2 * int(m))]
            if not ef_construction_candidates:
                ef_construction_candidates = [max(2 * int(m), efc_range[0])]
            return {
                "space_profile": profile,
                "M": int(m),
                "ef_construction": int(
                    trial.suggest_categorical("ef_construction_prior", ef_construction_candidates)
                ),
                "ef_search": int(trial.suggest_categorical("ef_search_prior", sorted(set(ef_search_candidates)))),
            }

        m = _suggest_int_log(trial, "M_ext", m_low, max_m)
        efc_low = max(efc_range[0], 2 * m)
        efc_high = min(efc_range[1], max(efc_low, n_train))
        efs_low = max(efs_range[0], top_k)
        efs_high = min(efs_range[1], max(efs_low, max(200, n_train // 10)))

        return {
            "space_profile": profile,
            "M": int(m),
            "ef_construction": int(_suggest_int_log(trial, "ef_construction_ext", efc_low, efc_high)),
            "ef_search": int(_suggest_int_log(trial, "ef_search_ext", efs_low, efs_high)),
        }

    def suggest_local_params(
        self,
        trial: optuna.Trial,
        n_train: int,
        dim: int,
        top_k: int,
        anchor_params: dict[str, Any],
    ) -> dict[str, Any]:
        del dim
        m_low, m_high_global = self._override_range("M_range", HNSWLIB_M_RANGE)
        efc_range = self._override_range("ef_construction_range", HNSWLIB_EF_CONSTRUCTION_RANGE)
        efs_range = self._override_range("ef_search_range", HNSWLIB_EF_SEARCH_RANGE)
        m_high = max(m_low, min(m_high_global, int(np.sqrt(n_train))))
        anchor_m = int(anchor_params.get("M", max(m_low, min(16, m_high))))
        m_l, m_h = _local_int_window(anchor_m, m_low, m_high, scale=1.8, min_span=4)
        m = _suggest_int_log(trial, "M_local", m_l, m_h)

        anchor_efc = int(anchor_params.get("ef_construction", max(efc_range[0], 2 * m)))
        efc_l, efc_h = _local_int_window(anchor_efc, max(efc_range[0], 2 * m), efc_range[1], scale=2.0)

        anchor_efs = int(anchor_params.get("ef_search", max(top_k, efs_range[0])))
        efs_l, efs_h = _local_int_window(anchor_efs, max(top_k, efs_range[0]), efs_range[1], scale=2.0)

        return {
            "space_profile": "local_refine",
            "M": int(m),
            "ef_construction": int(_suggest_int_log(trial, "ef_construction_local", efc_l, efc_h)),
            "ef_search": int(_suggest_int_log(trial, "ef_search_local", efs_l, efs_h)),
        }

    def build_index(self, train: NDArray[np.float32], metric: str, params: dict[str, Any]) -> Any:
        import hnswlib

        space = {"euclidean": "l2", "angular": "cosine", "dot": "ip"}.get(metric)
        if space is None:
            raise ValueError(f"{self.name} does not support metric={metric}")

        index = hnswlib.Index(space=space, dim=train.shape[1])
        index.init_index(
            max_elements=train.shape[0],
            ef_construction=int(params["ef_construction"]),
            M=int(params["M"]),
        )
        index.add_items(train, np.arange(train.shape[0], dtype=np.int64))
        index.set_num_threads(1)
        index.set_ef(int(params["ef_search"]))
        return index

    def query_index(
        self,
        index: Any,
        query: NDArray[np.float32],
        top_k: int,
        metric: str,
        params: dict[str, Any],
    ) -> NDArray[np.int64]:
        del metric, params
        labels, _ = index.knn_query(query.reshape(1, -1), k=top_k)
        return labels[0]


class AnnoyBackend(VectorBackend):
    name = "annoy"
    module_name = "annoy"

    def suggest_params(self, trial: optuna.Trial, n_train: int, dim: int, top_k: int) -> dict[str, Any]:
        del dim
        profile = trial.suggest_categorical("space_profile", ["annbench", "extended"])

        trees_low, trees_high_global = self._override_range("n_trees_range", ANNOY_N_TREES_RANGE)
        search_low_override, search_high_override = self._override_range("search_k_range", ANNOY_SEARCH_K_RANGE)
        n_trees_prior_choices = self._override_int_choices("n_trees_prior_choices", ANNOY_N_TREES)
        search_k_prior_choices = self._override_int_choices("search_k_prior_choices", ANNOY_SEARCH_K)

        max_search_k = max(top_k, min(search_high_override, n_train * 4))
        search_k_candidates = [v for v in search_k_prior_choices if top_k <= v <= max_search_k]
        if not search_k_candidates:
            search_k_candidates = [max(top_k, 100)]
        n_trees_candidates = [t for t in n_trees_prior_choices if t <= max(800, n_train // 2) and t >= trees_low]
        if not n_trees_candidates:
            n_trees_candidates = [max(trees_low, 50), max(trees_low, 100)]

        if profile == "annbench":
            return {
                "space_profile": profile,
                "n_trees": int(trial.suggest_categorical("n_trees_prior", sorted(set(n_trees_candidates)))),
                "search_k": int(trial.suggest_categorical("search_k_prior", sorted(set(search_k_candidates)))),
            }

        trees_high = min(trees_high_global, max(trees_low, max(200, n_train // 10)))
        n_trees = int(_suggest_int_log(trial, "n_trees_ext", trees_low, trees_high))

        search_k_mode = trial.suggest_categorical("search_k_mode", ["absolute", "tree_scaled", "auto"])
        search_k_low = max(top_k, search_low_override)
        search_k_high = min(search_high_override, max(search_k_low, n_train * 32))

        if search_k_mode == "auto":
            search_k = -1
        elif search_k_mode == "tree_scaled":
            factor = trial.suggest_float("tree_scaled_factor", 1.0, 256.0, log=True)
            search_k = int(round(n_trees * factor))
            search_k = int(min(search_k_high, max(search_k_low, search_k)))
        else:
            search_k = int(_suggest_int_log(trial, "search_k_ext", search_k_low, search_k_high))

        return {
            "space_profile": profile,
            "search_k_mode": search_k_mode,
            "n_trees": n_trees,
            "search_k": search_k,
        }

    def suggest_local_params(
        self,
        trial: optuna.Trial,
        n_train: int,
        dim: int,
        top_k: int,
        anchor_params: dict[str, Any],
    ) -> dict[str, Any]:
        del dim
        trees_low, trees_high = self._override_range("n_trees_range", ANNOY_N_TREES_RANGE)
        search_low_override, search_high_override = self._override_range("search_k_range", ANNOY_SEARCH_K_RANGE)
        trees_high = min(trees_high, max(trees_low, n_train))
        anchor_trees = int(anchor_params.get("n_trees", max(trees_low, 100)))
        trees_l, trees_h = _local_int_window(anchor_trees, trees_low, trees_high, scale=1.8, min_span=8)
        n_trees = int(_suggest_int_log(trial, "n_trees_local", trees_l, trees_h))

        search_low, search_high = search_low_override, search_high_override
        search_low = max(search_low, top_k)
        search_high = min(search_high, max(search_low, n_train * 64))
        anchor_search = int(anchor_params.get("search_k", max(search_low, 1000)))

        if anchor_search <= 0:
            mode = trial.suggest_categorical("search_k_mode_local", ["auto", "absolute", "tree_scaled"])
        else:
            mode = trial.suggest_categorical("search_k_mode_local", ["absolute", "tree_scaled", "auto"])

        if mode == "auto":
            search_k = -1
        elif mode == "tree_scaled":
            anchor_factor = max(1.0, (anchor_search / max(1, anchor_trees)) if anchor_search > 0 else 8.0)
            f_l, f_h = anchor_factor / 2.0, anchor_factor * 2.0
            factor = trial.suggest_float("tree_scaled_factor_local", max(1.0, f_l), min(512.0, f_h), log=True)
            search_k = int(round(n_trees * factor))
            search_k = int(min(search_high, max(search_low, search_k)))
        else:
            search_anchor = anchor_search if anchor_search > 0 else max(search_low, n_trees)
            s_l, s_h = _local_int_window(search_anchor, search_low, search_high, scale=2.0, min_span=16)
            search_k = int(_suggest_int_log(trial, "search_k_local", s_l, s_h))

        return {
            "space_profile": "local_refine",
            "search_k_mode": mode,
            "n_trees": n_trees,
            "search_k": search_k,
        }

    def build_index(self, train: NDArray[np.float32], metric: str, params: dict[str, Any]) -> Any:
        from annoy import AnnoyIndex

        mapped = {"angular": "angular", "euclidean": "euclidean", "dot": "dot"}.get(metric)
        if mapped is None:
            raise ValueError(f"{self.name} does not support metric={metric}")
        index = AnnoyIndex(train.shape[1], mapped)
        for i, vector in enumerate(train):
            index.add_item(i, vector.tolist())
        index.build(int(params["n_trees"]))
        return index

    def query_index(
        self,
        index: Any,
        query: NDArray[np.float32],
        top_k: int,
        metric: str,
        params: dict[str, Any],
    ) -> NDArray[np.int64]:
        del metric
        ids = index.get_nns_by_vector(query.tolist(), top_k, search_k=int(params["search_k"]))
        return np.asarray(ids, dtype=np.int64)


class FaissIVFBackend(VectorBackend):
    name = "faiss-ivf"
    module_name = "faiss"

    def suggest_params(self, trial: optuna.Trial, n_train: int, dim: int, top_k: int) -> dict[str, Any]:
        del top_k
        profile = trial.suggest_categorical("space_profile", ["annbench", "extended"])
        nlist_range = self._override_range("n_list_range", FAISS_IVF_NLIST_RANGE)
        nprobe_range = self._override_range("n_probe_range", FAISS_IVF_NPROBE_RANGE)
        nlist_prior_choices = self._override_int_choices("n_list_prior_choices", FAISS_IVF_NLIST)
        nprobe_prior_choices = self._override_int_choices("n_probe_prior_choices", FAISS_IVF_NPROBE)
        pq_m_base_choices = self._override_int_choices("pq_m_choices", FAISS_PQ_M)
        pq_nbits_choices = self._override_int_choices("pq_nbits_choices", FAISS_PQ_NBITS)
        coarse_hnsw_choices = self._override_int_choices("coarse_hnsw_m_choices", FAISS_COARSE_HNSW_M)
        rerank_choices_all = self._override_int_choices("rerank_k_factor_choices", FAISS_RERANK_K_FACTOR)

        if profile == "annbench":
            max_nlist = max(1, n_train // 39)
            nlist_candidates = [n for n in nlist_prior_choices if n <= max_nlist]
            if not nlist_candidates:
                nlist_candidates = [max(1, min(32, n_train // 5))]
            n_list = int(trial.suggest_categorical("n_list_prior", sorted(set(nlist_candidates))))
            nprobe_candidates = [p for p in nprobe_prior_choices if 1 <= p <= n_list]
            if not nprobe_candidates:
                nprobe_candidates = [1]
            return {
                "space_profile": profile,
                "index_type": "ivf_flat",
                "n_list": n_list,
                "n_probe": int(trial.suggest_categorical("n_probe_prior", sorted(set(nprobe_candidates)))),
                "rerank_k_factor": 1,
            }

        nlist_low = min(nlist_range[0], max(1, n_train))
        nlist_high_global = nlist_range[1]
        nlist_high = min(nlist_high_global, max(nlist_low, n_train // 8))
        n_list = int(_suggest_int_log(trial, "n_list_ext", nlist_low, nlist_high))

        nprobe_low, nprobe_high_global = nprobe_range
        nprobe_high = min(nprobe_high_global, max(nprobe_low, n_list))
        n_probe = int(_suggest_int_log(trial, "n_probe_ext", nprobe_low, nprobe_high))

        pq_candidates = _pq_m_candidates(dim, preferred_values=pq_m_base_choices)
        index_type_choices = ["ivf_flat", "ivf_hnsw_flat"]
        if pq_candidates:
            index_type_choices.extend(["ivf_pq", "ivf_hnsw_pq"])
        index_type_choices = self._override_str_choices("index_types", index_type_choices)
        if not pq_candidates:
            index_type_choices = [x for x in index_type_choices if x not in {"ivf_pq", "ivf_hnsw_pq"}]
        if not index_type_choices:
            index_type_choices = ["ivf_flat"]
        index_type = trial.suggest_categorical("index_type", index_type_choices)

        params: dict[str, Any] = {
            "space_profile": profile,
            "index_type": index_type,
            "n_list": n_list,
            "n_probe": min(n_probe, n_list),
            "rerank_k_factor": int(trial.suggest_categorical("rerank_k_factor", rerank_choices_all)),
        }
        if "hnsw" in index_type:
            params["coarse_hnsw_m"] = int(trial.suggest_categorical("coarse_hnsw_m", coarse_hnsw_choices))
        if index_type == "ivf_pq":
            params["pq_m"] = int(trial.suggest_categorical("pq_m", pq_candidates))
            params["pq_nbits"] = int(trial.suggest_categorical("pq_nbits", pq_nbits_choices))
        elif index_type == "ivf_hnsw_pq":
            params["pq_m"] = int(trial.suggest_categorical("pq_m", pq_candidates))
            params["pq_nbits"] = int(trial.suggest_categorical("pq_nbits", pq_nbits_choices))
        return params

    def suggest_local_params(
        self,
        trial: optuna.Trial,
        n_train: int,
        dim: int,
        top_k: int,
        anchor_params: dict[str, Any],
    ) -> dict[str, Any]:
        del top_k
        index_type = str(anchor_params.get("index_type", "ivf_flat"))
        nlist_range = self._override_range("n_list_range", FAISS_IVF_NLIST_RANGE)
        nprobe_range = self._override_range("n_probe_range", FAISS_IVF_NPROBE_RANGE)
        pq_m_base_choices = self._override_int_choices("pq_m_choices", FAISS_PQ_M)
        pq_nbits_base_choices = self._override_int_choices("pq_nbits_choices", FAISS_PQ_NBITS)
        coarse_hnsw_base_choices = self._override_int_choices("coarse_hnsw_m_choices", FAISS_COARSE_HNSW_M)
        rerank_base_choices = self._override_int_choices("rerank_k_factor_choices", FAISS_RERANK_K_FACTOR)

        nlist_low, nlist_high_global = nlist_range
        nlist_low = min(nlist_low, max(1, n_train))
        nlist_high = min(nlist_high_global, max(nlist_low, n_train // 4))
        anchor_nlist = int(anchor_params.get("n_list", max(nlist_low, 64)))
        nlist_l, nlist_h = _local_int_window(anchor_nlist, nlist_low, nlist_high, scale=1.8, min_span=16)
        n_list = int(_suggest_int_log(trial, "n_list_local", nlist_l, nlist_h))

        nprobe_low, nprobe_high_global = nprobe_range
        nprobe_high = min(nprobe_high_global, max(nprobe_low, n_list))
        anchor_nprobe = int(anchor_params.get("n_probe", min(10, n_list)))
        nprobe_l, nprobe_h = _local_int_window(anchor_nprobe, nprobe_low, nprobe_high, scale=2.0, min_span=2)
        n_probe = int(_suggest_int_log(trial, "n_probe_local", nprobe_l, nprobe_h))

        rerank_choices = _neighbor_choices(
            rerank_base_choices,
            int(anchor_params.get("rerank_k_factor", 1)),
            radius=1,
        )
        if not rerank_choices:
            rerank_choices = rerank_base_choices

        params: dict[str, Any] = {
            "space_profile": "local_refine",
            "index_type": index_type,
            "n_list": n_list,
            "n_probe": min(n_probe, n_list),
            "rerank_k_factor": int(trial.suggest_categorical("rerank_k_factor_local", rerank_choices)),
        }

        if "hnsw" in index_type:
            coarse_choices = _neighbor_choices(
                coarse_hnsw_base_choices,
                int(anchor_params.get("coarse_hnsw_m", 16)),
                radius=1,
            )
            params["coarse_hnsw_m"] = int(trial.suggest_categorical("coarse_hnsw_m_local", coarse_choices))

        if "pq" in index_type:
            pq_candidates = _pq_m_candidates(dim, preferred_values=pq_m_base_choices)
            if not pq_candidates:
                raise ValueError("PQ index_type selected but no valid pq_m for given dimension")
            pq_m_anchor = int(anchor_params.get("pq_m", pq_candidates[0]))
            pq_m_choices = _neighbor_choices(pq_candidates, pq_m_anchor, radius=1)
            pq_nbits_anchor = int(anchor_params.get("pq_nbits", pq_nbits_base_choices[-1]))
            pq_nbits_choices = _neighbor_choices(pq_nbits_base_choices, pq_nbits_anchor, radius=1)
            params["pq_m"] = int(trial.suggest_categorical("pq_m_local", pq_m_choices))
            params["pq_nbits"] = int(trial.suggest_categorical("pq_nbits_local", pq_nbits_choices))

        return params

    def build_index(self, train: NDArray[np.float32], metric: str, params: dict[str, Any]) -> Any:
        import faiss

        if metric not in {"euclidean", "angular", "dot"}:
            raise ValueError(f"{self.name} does not support metric={metric}")

        faiss.omp_set_num_threads(1)
        data = np.asarray(train, dtype=np.float32).copy()
        if metric == "angular":
            norms = np.linalg.norm(data, axis=1, keepdims=True)
            np.divide(data, norms, out=data, where=norms > 0)

        if metric in {"euclidean", "angular"}:
            base = faiss.IndexFlatL2(data.shape[1])
            faiss_metric = faiss.METRIC_L2
        else:
            base = faiss.IndexFlatIP(data.shape[1])
            faiss_metric = faiss.METRIC_INNER_PRODUCT

        index_type = str(params.get("index_type", "ivf_flat"))
        if index_type == "ivf_pq":
            pq_m = int(params["pq_m"])
            if data.shape[1] % pq_m != 0:
                raise ValueError(f"pq_m must divide dimension: dim={data.shape[1]}, pq_m={pq_m}")
            index = faiss.IndexIVFPQ(
                base,
                data.shape[1],
                int(params["n_list"]),
                pq_m,
                int(params["pq_nbits"]),
                faiss_metric,
            )
        elif index_type == "ivf_hnsw_flat":
            coarse_hnsw_m = int(params["coarse_hnsw_m"])
            factory = f"IVF{int(params['n_list'])}_HNSW{coarse_hnsw_m},Flat"
            index = faiss.index_factory(data.shape[1], factory, faiss_metric)
        elif index_type == "ivf_hnsw_pq":
            pq_m = int(params["pq_m"])
            if data.shape[1] % pq_m != 0:
                raise ValueError(f"pq_m must divide dimension: dim={data.shape[1]}, pq_m={pq_m}")
            coarse_hnsw_m = int(params["coarse_hnsw_m"])
            factory = (
                f"IVF{int(params['n_list'])}_HNSW{coarse_hnsw_m},"
                f"PQ{pq_m}x{int(params['pq_nbits'])}"
            )
            index = faiss.index_factory(data.shape[1], factory, faiss_metric)
        else:
            index = faiss.IndexIVFFlat(base, data.shape[1], int(params["n_list"]), faiss_metric)

        index.train(data)
        index.add(data)
        index.nprobe = int(params["n_probe"])
        return {"index": index, "metric": metric, "train": data}

    def query_index(
        self,
        index: Any,
        query: NDArray[np.float32],
        top_k: int,
        metric: str,
        params: dict[str, Any],
    ) -> NDArray[np.int64]:
        faiss_index = index["index"]
        metric = index["metric"]
        rerank_factor = max(1, int(params.get("rerank_k_factor", 1)))
        candidate_k = min(faiss_index.ntotal, max(top_k, top_k * rerank_factor))
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if metric == "angular":
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            np.divide(q, norms, out=q, where=norms > 0)
        _, ids = faiss_index.search(q, candidate_k)
        base = ids[0]
        if rerank_factor <= 1:
            return base[:top_k]
        return _rerank_ids(
            query=q[0],
            candidate_ids=base,
            train=index["train"],
            metric=metric,
            top_k=top_k,
        )


BACKENDS: dict[str, type[VectorBackend]] = {
    HnswlibBackend.name: HnswlibBackend,
    AnnoyBackend.name: AnnoyBackend,
    FaissIVFBackend.name: FaissIVFBackend,
}


def resolve_backends(
    requested: list[str],
    backend_overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[VectorBackend], dict[str, str]]:
    names = requested
    if "all" in names:
        names = list(BACKENDS.keys())

    backend_overrides = backend_overrides or {}
    selected: list[VectorBackend] = []
    skipped: dict[str, str] = {}
    for name in names:
        backend_cls = BACKENDS.get(name)
        if backend_cls is None:
            skipped[name] = "unknown backend name"
            continue
        ok, reason = backend_cls.availability()
        if not ok:
            skipped[name] = reason or "not available"
            continue
        selected.append(backend_cls(overrides=backend_overrides.get(name, {})))
    return selected, skipped
