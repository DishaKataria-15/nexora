"""Suggest likely targets and tasks from the data. These are suggestions: the user always confirms the target."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ..eda import profiling as P

CLASS_OUTCOME = re.compile(r"(^|_)(target|label|class|churn|churned|outcome|fraud|is_fraud|default|defaulted|converted|status|survived|approved|attrition|y)($|_)", re.I)
VALUE_OUTCOME = re.compile(r"(^|_)(price|prices|sales|revenue|demand|units_sold|sold|profit)($|_)", re.I)
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


def suggest_goal(df: pd.DataFrame) -> dict[str, Any]:
    """A suggested goal for the Goal step: {"task", "column" (or None), "reason"}.

    A column whose name sounds like an outcome (churn, price, status...) suggests predicting it. Without one, the
    suggestion is to discover groups, with a note that the user may still know of an outcome column.
    """
    best: tuple[float, str, str] | None = None
    for i, c in enumerate(df.columns):
        t, _ = task_for_column(df[c], c)
        if t is None:
            continue
        name = c.lower()
        numeric = pd.api.types.is_numeric_dtype(df[c])
        if CLASS_OUTCOME.search(name):
            task = "classification"
        elif VALUE_OUTCOME.search(name) and numeric:
            task = "regression"
        else:
            continue
        if t != task:                                 # the Target step must accept the suggested column for this goal
            continue
        score = 2.0 + (0.5 if i == len(df.columns) - 1 else 0.0) - 2.0 * float(df[c].isna().mean())
        if best is None or score > best[0]:
            best = (score, c, task)
    if best and best[0] >= 1.0:
        _, c, task = best
        kind = "a category (such as yes/no)" if task == "classification" else "a number"
        return {"task": task, "column": c, "reason": f"'{c}' sounds like an outcome, so predicting it as {kind} is a natural start. You can pick any other column later."}
    usable = [c for c in df.columns if P.column_role(df[c], c) == "numeric"]
    if len(usable) >= 2:
        return {"task": "clustering", "column": None,
                "reason": "No column clearly sounds like an outcome, so finding groups of similar rows is a reasonable start. "
                          "If you do have a column you want to predict, choose one of the prediction goals."}
    return {"task": "explore", "column": None,
            "reason": "No column clearly sounds like an outcome and there are few numeric columns, so charts and summaries are a good first look."}
