"""Cross-validation fold construction on the training rows only.

All folds are positional indices into the training part. Test rows never appear here, and validation rows are
never mixed into folds. Time-ordered data uses forward-chaining folds, so a model is never validated on the past
after being trained on the future.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (GroupKFold, KFold, RepeatedKFold, RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold,
                                     TimeSeriesSplit)

METHODS = {
    "auto": "Automatic (matches the split strategy)",
    "kfold": "K-fold (shuffled)",
    "stratified": "Stratified k-fold (keeps class shares in every fold)",
    "repeated": "Repeated k-fold (several shuffles, more stable estimate)",
    "group": "Group k-fold (each entity stays in one fold)",
    "timeseries": "Time-series (forward chaining: train on earlier rows, validate on later ones)",
}
MAX_FOLDS = 20
MAX_REPEATS = 10
MAX_FITS = 100


class CVError(ValueError):
    """The requested cross-validation cannot be done. The message is safe to show to the user."""


@dataclass
class CVConfig:
    method: str = "auto"
    folds: int = 5
    repeats: int = 1
    gap: int = 0                     # time series: rows left out between training and validation to avoid look-ahead

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.method not in METHODS:
            raise CVError(f"Unknown cross-validation method '{self.method}'.")
        if not 2 <= int(self.folds) <= MAX_FOLDS:
            raise CVError(f"Choose between 2 and {MAX_FOLDS} folds.")
        if not 1 <= int(self.repeats) <= MAX_REPEATS:
            raise CVError(f"Choose between 1 and {MAX_REPEATS} repeats.")
        if int(self.gap) < 0:
            raise CVError("The time-series gap cannot be negative.")
        if self.folds * self.repeats > MAX_FITS:
            raise CVError(f"{self.folds} folds x {self.repeats} repeats is more than {MAX_FITS} fits per model.")


@dataclass
class Folds:
    folds: list[tuple[np.ndarray, np.ndarray]]
    method: str
    description: str
    warnings: list[str]

    def summary(self) -> list[dict[str, int]]:
        return [{"fold": i + 1, "train rows": int(len(a)), "validation rows": int(len(b))} for i, (a, b) in enumerate(self.folds)]


def make_folds(n: int, cfg: CVConfig, *, task: str, split_strategy: str, y: pd.Series | None = None, groups: pd.Series | None = None,
               times: pd.Series | None = None, seed: int = 42) -> Folds:
    """Build folds over ``n`` training rows (positions 0..n-1)."""
    cfg.validate()
    warns: list[str] = []
    k = int(cfg.folds)
    method = cfg.method
    if method == "auto":
        method = {"chronological": "timeseries", "group": "group"}.get(split_strategy, "stratified" if task == "classification" else "kfold")
        if cfg.repeats > 1 and method in ("stratified", "kfold"):
            method = "repeated"
    if split_strategy == "chronological" and method != "timeseries":
        warns.append("The data is split by time but cross-validation shuffles rows, so models are validated partly on the past. "
                      "Time-series folds are safer here.")
    if split_strategy == "group" and method not in ("group",):
        warns.append("The data is split by entity but these folds can put the same entity in training and validation, which flatters scores.")
    idx = np.arange(n)
    yv = None if y is None else pd.Series(y).reset_index(drop=True)

    if method == "timeseries":
        if times is None:
            raise CVError("Time-series cross-validation needs the date column that orders the rows.")
        t = pd.to_datetime(pd.Series(times).reset_index(drop=True), errors="coerce")
        if t.isna().any():
            raise CVError("The date column has missing or invalid dates, so rows cannot be ordered for time-series folds.")
        order = np.argsort(t.values, kind="stable")
        k = min(k, n - 1)
        if n // (k + 1) < 2:
            raise CVError(f"{n} training rows are too few for {k} time-series folds.")
        tss = TimeSeriesSplit(n_splits=k, gap=int(cfg.gap))
        folds = [(order[a], order[b]) for a, b in tss.split(order)]
        desc = f"{k} forward-chaining time-series folds" + (f" with a gap of {cfg.gap} rows" if cfg.gap else "")
        if cfg.repeats > 1:
            warns.append("Repeats do not apply to time-series folds (the order is fixed); one pass was used.")
        return Folds(folds, method, desc, warns)

    if method == "group":
        if groups is None:
            raise CVError("Group cross-validation needs the column that identifies the entity.")
        g = pd.Series(groups).reset_index(drop=True)
        ng = g.nunique()
        if ng < 2:
            raise CVError("Group cross-validation needs at least 2 distinct groups in the training rows.")
        k = min(k, ng)
        if task == "classification" and yv is not None and yv.value_counts().min() >= k:
            folds = list(StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(idx, yv, g))
            desc = f"{k}-fold stratified group cross-validation"
        else:
            folds = list(GroupKFold(n_splits=k).split(idx, groups=g))
            desc = f"{k}-fold group cross-validation"
        return Folds(folds, method, desc, warns)

    stratify = task == "classification" and yv is not None and method in ("stratified", "repeated")
    if stratify and yv.value_counts().min() < k:
        warns.append(f"The smallest class has only {int(yv.value_counts().min())} training rows, fewer than {k} folds, so folds were not stratified.")
        stratify = False
    if method == "stratified" and not stratify:
        method = "kfold"
    k = min(k, n)
    reps = max(int(cfg.repeats), 2) if method == "repeated" else int(cfg.repeats)
    if reps > 1:
        method = "repeated"
        splitter = RepeatedStratifiedKFold(n_splits=k, n_repeats=reps, random_state=seed) if stratify else RepeatedKFold(n_splits=k, n_repeats=reps, random_state=seed)
        desc = f"{k}-fold cross-validation repeated {reps} times" + (" (stratified)" if stratify else "")
        folds = list(splitter.split(idx, yv if stratify else None))
        return Folds(folds, method, desc, warns)
    if stratify:
        folds = list(StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, yv))
        desc = f"{k}-fold stratified cross-validation"
    else:
        folds = list(KFold(k, shuffle=True, random_state=seed).split(idx))
        desc = f"{k}-fold cross-validation"
    return Folds(folds, method, desc, warns)
