from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from numpy.typing import NDArray

from .base import VectorBackend
from .utils import _local_int_window, _suggest_int_log
from ..priors import (
    HNSWLIB_EF_CONSTRUCTION,
    HNSWLIB_EF_CONSTRUCTION_RANGE,
    HNSWLIB_EF_SEARCH,
    HNSWLIB_EF_SEARCH_RANGE,
    HNSWLIB_M,
    HNSWLIB_M_RANGE,
)


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

        ef_construction_candidates = [ef for ef in efc_prior_choices if ef >= efc_range[0]]
        if not ef_construction_candidates:
            ef_construction_candidates = [max(efc_range[0], 2 * m_low)]

        ef_search_candidates = [ef for ef in efs_prior_choices if ef >= top_k and ef <= max(top_k, n_train)]
        if not ef_search_candidates:
            ef_search_candidates = [max(top_k, efs_range[0])]

        profile = trial.suggest_categorical("space_profile", ["annbench", "extended"])
        if profile == "annbench":
            m = trial.suggest_categorical("M_prior", sorted(set(m_candidates)))
            ef_construction_prior = int(
                trial.suggest_categorical("ef_construction_prior", sorted(set(ef_construction_candidates)))
            )
            return {
                "space_profile": profile,
                "M": int(m),
                # Keep categorical choices static across trials for Optuna compatibility.
                "ef_construction": max(int(ef_construction_prior), max(efc_range[0], 2 * int(m))),
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


__all__ = ["HnswlibBackend"]
