"""Bounded hyperparameter search: grid, random and successive halving.

The search runs on cross-validation folds built from the training rows only. Every trial is a full pipeline fit
(preprocessing is re-learned inside each fold), so tuning cannot leak validation or test information into
preprocessing. Searches are limited by a trial cap and a time budget and can be cancelled between fits.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler

METHODS = {"none": "No tuning (use default or chosen settings)", "grid": "Grid search (every combination)",
           "random": "Random search (a sample of combinations)", "halving": "Successive halving (many combinations on few rows, the best on more)"}
MAX_TRIALS = 60
MAX_BUDGET_S = 3600


class TuningError(ValueError):
    """The requested search cannot run. The message is safe to show to the user."""


@dataclass
class TuningConfig:
    method: str = "none"
    n_iter: int = 10                 # random / halving: how many combinations to start from
    max_trials: int = 30             # hard cap on combinations evaluated
    time_budget_s: float = 300.0     # per model; the search stops after the trial that crosses it
    factor: int = 3                  # halving: keep 1/factor of combinations each round

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.method not in METHODS:
            raise TuningError(f"Unknown tuning method '{self.method}'.")
        if not 1 <= int(self.max_trials) <= MAX_TRIALS:
            raise TuningError(f"The trial limit must be between 1 and {MAX_TRIALS}.")
        if not 1 <= int(self.n_iter) <= MAX_TRIALS:
            raise TuningError(f"The number of sampled combinations must be between 1 and {MAX_TRIALS}.")
        if not 1 <= float(self.time_budget_s) <= MAX_BUDGET_S:
            raise TuningError(f"The time budget must be between 1 second and {MAX_BUDGET_S // 60} minutes.")
        if not 2 <= int(self.factor) <= 5:
            raise TuningError("The halving factor must be between 2 and 5.")


@dataclass
class TuningResult:
    method: str
    space: dict[str, list[Any]]
    trials: list[dict[str, Any]] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    best_std: float | None = None
    stopped: str = ""                # why the search ended early, if it did
    seconds: float = 0.0
    grid_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_params(space: dict[str, list[Any]], cfg: TuningConfig, seed: int) -> tuple[list[dict[str, Any]], int]:
    """The combinations a search would try, and the full grid size."""
    if not space:
        return [], 0
    grid = list(ParameterGrid(space))
    if cfg.method == "grid":
        if len(grid) > cfg.max_trials:
            raise TuningError(f"The grid has {len(grid)} combinations, more than the limit of {cfg.max_trials}. "
                              "Use random search or successive halving, or raise the limit.")
        return grid, len(grid)
    n = min(int(cfg.n_iter) if cfg.method == "random" else max(int(cfg.n_iter), 2 * cfg.factor), len(grid), cfg.max_trials)
    return list(ParameterSampler(space, n_iter=n, random_state=seed)), len(grid)


def _evaluate(params, folds, make_pipe, score_fn, X, y, check, row_cap: int | None, rng, passthrough=()) -> tuple[float | None, float | None, list[float], str]:
    try:
        return _evaluate_inner(params, folds, make_pipe, score_fn, X, y, check, row_cap, rng)
    except passthrough:
        raise
    except Exception as e:  # one bad combination must not end the search
        return None, None, [], f"{type(e).__name__}: {e}"


def _evaluate_inner(params, folds, make_pipe, score_fn, X, y, check, row_cap, rng):
    scores: list[float] = []
    for a, b in folds:
        check()
        if row_cap is not None and len(a) > row_cap:
            a = np.sort(rng.choice(a, row_cap, replace=False))
        pipe = make_pipe(params).fit(X.iloc[a], y.iloc[a])
        s = score_fn(pipe, X.iloc[b], y.iloc[b])
        if s is None:
            return None, None, scores, "The metric could not be computed in every fold."
        scores.append(float(s))
    return float(np.mean(scores)), float(np.std(scores)), scores, ""


def tune(space: dict[str, list[Any]], cfg: TuningConfig, *, make_pipe: Callable[[dict[str, Any]], Any], score_fn: Callable[[Any, Any, Any], float | None],
         X, y, folds: list[tuple[np.ndarray, np.ndarray]], higher_is_better: bool, seed: int = 42, check: Callable[[], None] = lambda: None,
         progress: Callable[[str], None] | None = None, passthrough: tuple = ()) -> TuningResult:
    """Search ``space`` (names as the estimator sees them) and return every trial with its cross-validated score."""
    cfg.validate()
    cands, grid_size = candidate_params(space, cfg, seed)
    res = TuningResult(cfg.method, {k: list(v) for k, v in space.items()}, grid_size=grid_size)
    if not cands:
        res.stopped = "This model has no settings to tune."
        return res
    t0 = time.time()
    rng = np.random.RandomState(seed)
    sign = 1.0 if higher_is_better else -1.0

    def over_budget() -> bool:
        return time.time() - t0 > cfg.time_budget_s

    if cfg.method in ("grid", "random"):
        for i, p in enumerate(cands):
            if over_budget():
                res.stopped = f"Stopped after {i} of {len(cands)} combinations: the {cfg.time_budget_s:g}-second budget was reached."
                break
            if progress:
                progress(f"combination {i + 1} of {len(cands)}")
            t1 = time.time()
            m, sd, sc, err = _evaluate(p, folds, make_pipe, score_fn, X, y, check, None, rng, passthrough)
            res.trials.append({"params": p, "score": m, "std": sd, "fold_scores": sc, "rows_per_fold": int(np.mean([len(a) for a, _ in folds])),
                               "seconds": round(time.time() - t1, 3), "error": err, "round": 1})
    else:  # successive halving over the number of training rows per fold
        n_full = int(min(len(a) for a, _ in folds))
        rounds = max(1, int(math.ceil(math.log(len(cands), cfg.factor)))) if len(cands) > 1 else 1
        rows = max(30, int(n_full / (cfg.factor ** (rounds - 1))))
        alive = list(cands)
        r = 1
        while alive:
            cap = None if rows >= n_full else rows
            scored = []
            for p in alive:
                if over_budget():
                    res.stopped = f"Stopped in round {r}: the {cfg.time_budget_s:g}-second budget was reached."
                    break
                if progress:
                    progress(f"round {r}: {len(alive)} combinations on {min(rows, n_full):,} rows")
                t1 = time.time()
                m, sd, sc, err = _evaluate(p, folds, make_pipe, score_fn, X, y, check, cap, rng, passthrough)
                res.trials.append({"params": p, "score": m, "std": sd, "fold_scores": sc, "rows_per_fold": int(min(rows, n_full)),
                                   "seconds": round(time.time() - t1, 3), "error": err, "round": r})
                if m is not None:
                    scored.append((sign * m, p))
            if res.stopped or cap is None or len(scored) <= 1:
                break
            scored.sort(key=lambda t: -t[0])
            alive = [p for _, p in scored[:max(1, len(scored) // cfg.factor)]]
            rows = min(n_full, rows * cfg.factor)
            r += 1
    final_round = max((t["round"] for t in res.trials), default=1)
    pool = [t for t in res.trials if t["score"] is not None and t["round"] == final_round] or [t for t in res.trials if t["score"] is not None]
    if not pool:
        raise TuningError("No combination could be scored: " + (res.trials[0]["error"] if res.trials else res.stopped))
    best = max(pool, key=lambda t: sign * t["score"])
    res.best_params, res.best_score, res.best_std = dict(best["params"]), best["score"], best["std"]
    res.seconds = round(time.time() - t0, 2)
    return res
