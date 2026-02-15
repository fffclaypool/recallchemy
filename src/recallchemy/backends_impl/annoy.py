from __future__ import annotations

from typing import Any

import numpy as np
import optuna
from numpy.typing import NDArray

from .base import VectorBackend
from .utils import _local_int_window, _suggest_int_log
from ..priors import ANNOY_N_TREES, ANNOY_N_TREES_RANGE, ANNOY_SEARCH_K, ANNOY_SEARCH_K_RANGE


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


__all__ = ["AnnoyBackend"]
