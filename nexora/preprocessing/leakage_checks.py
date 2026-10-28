"""Warnings about leakage and unsafe splitting. These are screening checks: they warn, the user decides."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ..eda import profiling as P

POST_OUTCOME = re.compile(r"(cancel|refund|outcome|result|resolution|after|final|churn|convert|chargeback|settled|closed|actual|post_|_post|reason_for)", re.I)


@dataclass
class LeakageWarning:
    level: str            # warning | notice
    kind: str             # target_leakage | post_outcome_name | duplicate_leakage | time_split | group_split
    columns: list[str]
    message: str
    suggest_exclude: bool = False


def _purity(feature: pd.Series, y: pd.Series) -> float:
    d = pd.DataFrame({"f": feature.astype(object).where(feature.notna(), "(missing)").astype(str), "y": np.asarray(y)})
    return float(d.groupby("f")["y"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(d))


def check_target_leakage(df: pd.DataFrame, target: str, task: str, features: list[str], max_features: int = 60) -> list[LeakageWarning]:
    out: list[LeakageWarning] = []
    y_all = df[target]
    ok = y_all.notna()
    if ok.sum() < 20:
        return out
    y = y_all[ok].astype(str) if task == "classification" else y_all[ok]
    base = float(y.value_counts(normalize=True).iloc[0]) if task == "classification" else None
    for c in features[:max_features]:
        s = df.loc[ok, c]
        role = P.column_role(s, c)
        if role in ("empty", "identifier", "text") or (role == "constant" and not s.isna().any()):
            continue          # a single value plus blanks is NOT constant for leakage purposes: "filled in" can itself give the answer away
        reason = None
        if task == "classification":
            if role in ("categorical", "boolean"):
                pur = _purity(s, y)
                if pur >= 0.98 and pur - base >= 0.03:
                    reason = f"it identifies the outcome in {100 * pur:.1f}% of rows by itself (a guess without it would be right {100 * base:.1f}% of the time)"
            elif role == "numeric":
                x = pd.to_numeric(s, errors="coerce").astype(float)
                x_f = x.fillna(x.median() if x.notna().any() else 0).values.reshape(-1, 1)
                acc = float(DecisionTreeClassifier(max_depth=3, random_state=0).fit(x_f, y).score(x_f, y))
                if acc >= 0.98 and acc - base >= 0.03:
                    reason = f"a few thresholds on it separate the outcome in {100 * acc:.1f}% of rows"
            if reason is None and s.isna().any() and s.isna().mean() < 0.99:
                pur = _purity(pd.Series(np.where(s.isna(), "m", "p"), index=s.index), y)
                if pur >= 0.98 and pur - base >= 0.03:
                    reason = f"whether it is filled in predicts the outcome in {100 * pur:.1f}% of rows (typical of information recorded after the outcome)"
        else:
            yn = pd.to_numeric(y, errors="coerce")
            if role == "numeric":
                r = float(pd.to_numeric(s, errors="coerce").astype(float).corr(yn))
                if np.isfinite(r) and abs(r) >= 0.98:
                    reason = f"it is almost a copy of the target (correlation {r:.3f})"
            elif role in ("categorical", "boolean"):
                g = pd.DataFrame({"f": s.astype(str), "y": yn}).dropna()
                if len(g) and g["y"].var() > 0:
                    eta2 = 1 - g.groupby("f")["y"].apply(lambda v: ((v - v.mean()) ** 2).sum()).sum() / ((g["y"] - g["y"].mean()) ** 2).sum()
                    if eta2 >= 0.98 and g["f"].nunique() < 0.5 * len(g):
                        reason = f"its categories explain {100 * eta2:.1f}% of the variation in the target"
        if reason:
            out.append(LeakageWarning("warning", "target_leakage", [c], f"'{c}' may leak the answer: {reason}. Exclude it unless it is known before the prediction is made.", True))
    return out


def check_post_outcome_names(features: list[str], target: str) -> list[LeakageWarning]:
    hits = [c for c in features if POST_OUTCOME.search(c)]
    if not hits:
        return []
    return [LeakageWarning("notice", "post_outcome_name", hits,
                           f"Column name(s) {', '.join(repr(c) for c in hits)} suggest information recorded after the outcome of '{target}'. "
                           "A model must only use what is known at prediction time.", False)]


def check_duplicates(df: pd.DataFrame, features: list[str], target: str | None) -> list[LeakageWarning]:
    cols = [c for c in features + ([target] if target else []) if c in df.columns]
    n = int(df[cols].astype(str).duplicated().sum()) if cols else 0
    if not n:
        return []
    return [LeakageWarning("notice", "duplicate_leakage", [], f"{n:,} row(s) repeat another row exactly. If copies land in both training and test sets, scores look better than they are.")]


def duplicates_across_splits(X_train: pd.DataFrame, X_other: pd.DataFrame) -> int:
    a = set(map(tuple, X_train.astype(str).values))
    return int(sum(1 for r in map(tuple, X_other.astype(str).values) if r in a))


def check_split_strategy(df: pd.DataFrame, strategy: str, time_cols: list[str], group_candidates: list[str]) -> list[LeakageWarning]:
    out: list[LeakageWarning] = []
    if strategy in ("random", "stratified") and time_cols:
        out.append(LeakageWarning("notice", "time_split", time_cols,
                                  f"The data has a date column ({', '.join(time_cols)}). If you want to predict the future, a random split lets the model peek at later periods; "
                                  "choose a chronological split."))
    if strategy in ("random", "stratified") and group_candidates:
        out.append(LeakageWarning("notice", "group_split", group_candidates,
                                  f"'{group_candidates[0]}' repeats across rows. If rows from the same entity are in both training and test sets, scores are optimistic; "
                                  "a group-aware split keeps each entity on one side."))
    return out


def group_candidates(df: pd.DataFrame) -> list[str]:
    out = []
    n = len(df)
    for c in df.columns:
        s = df[c]
        if P._is_numeric(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        k = s.nunique()
        if 2 < k < 0.9 * n and re.search(r"(id|key|code|account|customer|patient|user|device)", c, re.I):
            out.append(c)
    return out
