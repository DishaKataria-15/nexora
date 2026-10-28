"""Suggest likely targets and tasks from the data. These are suggestions: the user always confirms the target."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ..eda import profiling as P

HINT = re.compile(r"(target|label|class|churn|outcome|price|amount|sales|revenue|score|status|default|fraud|converted|demand|units_sold|^y$)", re.I)


def task_for_column(s: pd.Series, name: str = "") -> tuple[str | None, str]:
    """('classification' | 'regression' | None, reason)"""
    role = P.column_role(s, name)
    if role in ("empty", "constant", "identifier", "datetime", "text"):
        return None, {"identifier": "identifier columns are not outcomes", "datetime": "dates are not predicted directly", "constant": "same value in every row",
                      "empty": "no values", "text": "free text"}[role]
    k = s.nunique()
    if role == "numeric":
        if k > 15 or (not pd.api.types.is_integer_dtype(s) and k > 10):
            return "regression", f"numeric with {k} distinct values"
        return "classification", f"numeric but only {k} distinct values, so it behaves like categories"
    if k <= 30:
        return "classification", f"{k} distinct categories"
    return None, f"{k} distinct values is too many categories to predict"


def recommend_targets(df: pd.DataFrame, task: str | None = None, limit: int = 6) -> list[dict[str, Any]]:
    out = []
    n = len(df)
    for i, c in enumerate(df.columns):
        t, why = task_for_column(df[c], c)
        if t is None or (task and t != task):
            continue
        score = (2.0 if HINT.search(c) else 0.0) + (0.5 if i == len(df.columns) - 1 else 0.0) - 1.5 * float(df[c].isna().mean())
        out.append({"column": c, "task": t, "reason": why + (", name suggests an outcome" if HINT.search(c) else ""), "score": score,
                    "missing %": round(100 * df[c].isna().sum() / max(n, 1), 1)})
    out.sort(key=lambda r: -r["score"])
    return out[:limit]
