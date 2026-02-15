from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from numpy.typing import NDArray

from .base import VectorBackend
from .utils import (
    _local_int_window,
    _neighbor_choices,
    _pq_m_candidates,
    _rerank_ids,
    _suggest_int_log,
)
from ..priors import (
    FAISS_COARSE_HNSW_M,
    FAISS_IVF_NLIST,
    FAISS_IVF_NLIST_RANGE,
    FAISS_IVF_NPROBE,
    FAISS_IVF_NPROBE_RANGE,
    FAISS_PQ_M,
    FAISS_PQ_NBITS,
    FAISS_RERANK_K_FACTOR,
)


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
            nprobe_candidates = [p for p in nprobe_prior_choices if p >= 1]
            if not nprobe_candidates:
                nprobe_candidates = [1]
            n_probe_prior = int(trial.suggest_categorical("n_probe_prior", sorted(set(nprobe_candidates))))
            return {
                "space_profile": profile,
                "index_type": "ivf_flat",
                "n_list": n_list,
                # Keep categorical choices static across trials for Optuna compatibility.
                "n_probe": min(int(n_list), max(1, int(n_probe_prior))),
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
            factory = f"IVF{int(params['n_list'])}_HNSW{coarse_hnsw_m},PQ{pq_m}x{int(params['pq_nbits'])}"
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
        del metric
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


__all__ = ["FaissIVFBackend"]
