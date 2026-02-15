from __future__ import annotations

import numpy as np
import optuna


def _split_trials(n_trials: int, local_refine: bool, stage1_ratio: float) -> tuple[int, int]:
    if n_trials <= 0:
        return 0, 0
    if not local_refine or n_trials < 6:
        return n_trials, 0
    ratio = min(0.9, max(0.5, stage1_ratio))
    stage1 = int(round(n_trials * ratio))
    if n_trials <= 12:
        # Keep most budget in stage-1 for small budgets so TPE can learn globally.
        stage1 = max(stage1, n_trials - 2)
    stage1 = min(n_trials - 1, max(2, stage1))
    return stage1, n_trials - stage1


def _tpe_startup_trials(n_trials: int) -> int:
    if n_trials <= 2:
        return 1
    return max(2, min(10, int(np.ceil(n_trials * 0.25))))


def _build_sampler(sampler: str, *, seed: int, n_trials: int) -> optuna.samplers.BaseSampler:
    if sampler == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            n_startup_trials=_tpe_startup_trials(n_trials),
        )
    if sampler == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    raise ValueError(f"unsupported sampler={sampler!r}; expected 'tpe' or 'random'")


__all__ = ["_split_trials", "_tpe_startup_trials", "_build_sampler"]
