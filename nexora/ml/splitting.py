"""Train / validation / test splitting with leakage-aware strategies. Splits are positional and reproducible."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

STRATEGIES = {"random": "Random", "stratified": "Stratified (keeps class shares)", "group": "Group-aware (keeps each entity on one side)",
              "chronological": "Chronological (train on the past, test on the future)"}


class SplitError(ValueError):
    """The requested split cannot be made. The message is safe to show to the user."""


@dataclass
class SplitConfig:
    strategy: str = "random"
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 42
    group_col: str | None = None
    time_col: str | None = None

    def validate(self) -> None:
        if self.strategy not in STRATEGIES:
            raise SplitError(f"Unknown split strategy '{self.strategy}'.")
        if min(self.train, self.val, self.test) < 0 or self.train <= 0 or self.test <= 0:
            raise SplitError("Training and test shares must be above zero; the validation share cannot be negative.")
        if abs(self.train + self.val + self.test - 1.0) > 1e-6:
            raise SplitError("Train, validation and test shares must add up to 100%.")
        if self.strategy == "group" and not self.group_col:
            raise SplitError("Choose the column that identifies the entity (for example customer) for a group-aware split.")
        if self.strategy == "chronological" and not self.time_col:
            raise SplitError("Choose the date column that orders the rows for a chronological split.")


@dataclass
class SplitResult:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    config: SplitConfig
    effective_strategy: str
    warnings: list[str] = field(default_factory=list)

    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def to_dict(self) -> dict[str, Any]:
        return {"config": asdict(self.config), "effective_strategy": self.effective_strategy, "sizes": self.sizes(), "warnings": self.warnings}


def recommend_split(n_rows: int) -> tuple[SplitConfig, str]:
    """Pick shares from dataset size. Validation is skipped on small data; cross-validation on the training part replaces it."""
    if n_rows >= 1000:
        return SplitConfig(train=0.7, val=0.15, test=0.15), (
            f"{n_rows:,} rows is enough to hold out separate validation and test sets (70/15/15). Models are compared on the validation set.")
    return SplitConfig(train=0.8, val=0.0, test=0.2), (
        f"With {n_rows:,} rows, a separate validation set would be too small to be reliable, so 80/20 is used and models are compared with "
        "cross-validation inside the training part. The test set stays untouched.")


def _two_way(idx: np.ndarray, frac_second: float, seed: int, stratify: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    a, b = train_test_split(idx, test_size=frac_second, random_state=seed, shuffle=True, stratify=stratify)
    return np.sort(a), np.sort(b)


def make_split(n_rows: int, cfg: SplitConfig, *, y: pd.Series | None = None, groups: pd.Series | None = None,
               times: pd.Series | None = None) -> SplitResult:
    cfg.validate()
    if n_rows < 10:
        raise SplitError(f"{n_rows} rows is too few to split into training and test sets (at least 10 are needed).")
    idx = np.arange(n_rows)
    warns: list[str] = []
    strategy = cfg.strategy
    val_frac_of_rest = cfg.val / (cfg.train + cfg.val) if cfg.val > 0 else 0.0

    if strategy == "chronological":
        t = pd.to_datetime(pd.Series(times).reset_index(drop=True), errors="coerce")
        if t.isna().any():
            raise SplitError("The date column has missing or invalid dates; remove or fix them before a chronological split.")
        order = np.argsort(t.values, kind="stable")
        n_test = max(1, int(round(n_rows * cfg.test)))
        n_val = int(round(n_rows * cfg.val))
        n_train = n_rows - n_test - n_val
        if n_train < 1:
            raise SplitError("The shares leave no rows for training.")
        return SplitResult(np.sort(order[:n_train]), np.sort(order[n_train:n_train + n_val]), np.sort(order[n_train + n_val:]), cfg, strategy, warns)

    if strategy == "group":
        g = pd.Series(groups).reset_index(drop=True)
        if g.isna().any():
            raise SplitError("The group column has missing values; fill or remove them first.")
        if g.nunique() < 3:
            raise SplitError("A group-aware split needs at least 3 distinct groups.")
        outer = GroupShuffleSplit(n_splits=1, test_size=cfg.test, random_state=cfg.seed)
        rest, test = next(outer.split(idx, groups=g))
        if cfg.val > 0:
            inner = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_rest, random_state=cfg.seed)
            tr_i, va_i = next(inner.split(rest, groups=g.iloc[rest]))
            return SplitResult(np.sort(rest[tr_i]), np.sort(rest[va_i]), np.sort(test), cfg, strategy, warns)
        return SplitResult(np.sort(rest), np.array([], dtype=int), np.sort(test), cfg, strategy, warns)

    strat = None
    if strategy == "stratified":
        if y is None:
            raise SplitError("Stratified splitting needs a categorical target.")
        counts = pd.Series(y).value_counts()
        if counts.min() < 3 or (cfg.val > 0 and counts.min() < 4):
            warns.append(f"The smallest class has only {int(counts.min())} row(s), too few to keep its share in every part. A plain random split was used instead.")
            strategy = "random"
        else:
            strat = pd.Series(y).reset_index(drop=True).values
    rest, test = _two_way(idx, cfg.test, cfg.seed, strat)
    if cfg.val > 0:
        tr, va = _two_way(rest, val_frac_of_rest, cfg.seed, strat[rest] if strat is not None else None)
        return SplitResult(tr, va, test, cfg, strategy, warns)
    return SplitResult(rest, np.array([], dtype=int), test, cfg, strategy, warns)
