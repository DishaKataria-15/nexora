"""Exploratory data analysis: roles, summaries and computed (never invented) observations."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

ID_NAME = re.compile(r"(^|_|\b)(id|uuid|key|ref|code|no|number)($|_|\b)", re.I)
HIGH_CARDINALITY = 50


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def column_role(s: pd.Series, name: str = "") -> str:
    """One of: empty, constant, datetime, boolean, identifier, numeric, categorical, text."""
    n = int(s.notna().sum())
    if n == 0:
        return "empty"
    if s.dropna().nunique() <= 1:
        return "constant"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
        return "boolean"
    uniq = s.dropna().nunique()
    if _is_numeric(s):
        if pd.api.types.is_integer_dtype(s) and uniq >= 0.98 * n and n > 20 and ID_NAME.search(name):
            return "identifier"
        return "numeric"
    txt = s.dropna().astype(str)
    if uniq >= 0.95 * n and n > 20:
        return "text" if txt.str.len().mean() > 40 else "identifier"
    if uniq <= 2 and set(txt.str.lower().unique()) <= {"yes", "no", "y", "n", "true", "false", "t", "f", "0", "1"}:
        return "boolean"
    if txt.str.len().mean() > 60 and uniq > 0.5 * n:
        return "text"
    return "categorical"


def roles(df: pd.DataFrame) -> dict[str, str]:
    return {c: column_role(df[c], c) for c in df.columns}


def dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    r = roles(df)
    counts: dict[str, int] = {}
    for v in r.values():
        counts[v] = counts.get(v, 0) + 1
    cells = max(df.shape[0] * df.shape[1], 1)
    miss = int(df.isna().sum().sum())
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1]), "missing_cells": miss, "missing_pct": round(100 * miss / cells, 2),
            "duplicate_rows": int(df.astype(str).duplicated().sum()) if len(df) else 0,
            "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024 / 1024, 2), "role_counts": counts}


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for c in df.columns:
        s = df[c]
        role = column_role(s, c)
        row: dict[str, Any] = {"column": c, "role": role, "type": str(s.dtype), "missing": int(s.isna().sum()),
                               "missing %": round(100 * s.isna().sum() / n, 1), "unique": int(s.nunique(dropna=True))}
        nn = s.dropna()
        if _is_numeric(s) and len(nn):
            row.update({"min": float(nn.min()), "median": float(nn.median()), "mean": float(nn.mean()), "max": float(nn.max()),
                        "std": float(nn.std()) if len(nn) > 1 else 0.0})
        elif pd.api.types.is_datetime64_any_dtype(s) and len(nn):
            row.update({"earliest": str(nn.min())[:19], "latest": str(nn.max())[:19]})
        elif len(nn):
            vc = nn.astype(str).value_counts()
            row["top value"] = f"{vc.index[0]} ({int(vc.iloc[0])})"
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_columns(df: pd.DataFrame, usable_only: bool = True) -> list[str]:
    out = []
    for c in df.columns:
        if _is_numeric(df[c]):
            if usable_only and column_role(df[c], c) in ("empty", "constant", "identifier"):
                continue
            out.append(c)
    return out


def categorical_columns(df: pd.DataFrame, max_unique: int | None = None) -> list[str]:
    out = []
    for c in df.columns:
        if column_role(df[c], c) in ("categorical", "boolean"):
            if max_unique is None or df[c].nunique() <= max_unique:
                out.append(c)
    return out


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) and df[c].notna().sum() > 1]


def outlier_summary(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """IQR screening. Outliers are review signals, never errors."""
    rows = []
    for c in numeric_columns(df):
        s = df[c].dropna().astype(float)
        if s.nunique() < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if not iqr:
            continue
        lo, hi = q1 - k * iqr, q3 + k * iqr
        cnt = int(((s < lo) | (s > hi)).sum())
        rows.append({"column": c, "outliers": cnt, "share %": round(100 * cnt / len(s), 2), "lower fence": round(float(lo), 4),
                     "upper fence": round(float(hi), 4), "min": float(s.min()), "max": float(s.max())})
    return pd.DataFrame(rows, columns=["column", "outliers", "share %", "lower fence", "upper fence", "min", "max"])


def correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    cols = numeric_columns(df)
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].astype(float).corr(method=method, min_periods=3)


def top_correlations(corr: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    if corr.empty:
        return pd.DataFrame(columns=["column A", "column B", "correlation"])
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append((a, b, float(v)))
    pairs.sort(key=lambda t: -abs(t[2]))
    return pd.DataFrame(pairs[:n], columns=["column A", "column B", "correlation"])


def target_distribution(s: pd.Series) -> pd.DataFrame:
    vc = s.value_counts(dropna=False)
    total = int(vc.sum())
    return pd.DataFrame({"value": [("(missing)" if pd.isna(i) else str(i)) for i in vc.index], "count": vc.values.astype(int),
                         "share %": (100 * vc.values / max(total, 1)).round(1)})


@dataclass
class Observation:
    kind: str
    text: str
    columns: list[str]
    severity: str = "info"   # info | notice | warning


def observations(df: pd.DataFrame, max_items: int = 12) -> list[Observation]:
    """Plain-language statements, each derived from a number computed on this data."""
    out: list[Observation] = []
    n = len(df)
    if n == 0:
        return [Observation("empty", "The dataset has no rows.", [], "warning")]
    if n < 100:
        out.append(Observation("small", f"Only {n} rows: patterns may not be reliable.", [], "notice"))
    miss = df.isna().mean().sort_values(ascending=False)
    for c, v in miss[miss >= 0.2].head(3).items():
        out.append(Observation("missing", f"'{c}' is {100 * v:.0f}% missing.", [c], "warning" if v >= 0.5 else "notice"))
    for c in numeric_columns(df):
        s = df[c].dropna().astype(float)
        if len(s) > 20 and s.nunique() > 5:
            sk = float(s.skew())
            if abs(sk) >= 1.5:
                out.append(Observation("skew", f"'{c}' is strongly {'right' if sk > 0 else 'left'}-skewed (skewness {sk:.1f}); the mean is pulled towards the long tail.", [c]))
    osum = outlier_summary(df, 3.0)
    for _, r in osum[osum["outliers"] > 0].head(3).iterrows():
        out.append(Observation("outlier", f"'{r['column']}' has {int(r['outliers'])} value(s) beyond 3x the interquartile range. These are unusual, not necessarily wrong.", [r["column"]]))
    for c in df.columns:
        role = column_role(df[c], c)
        if role == "constant":
            out.append(Observation("constant", f"'{c}' has the same value in every row, so it cannot explain differences.", [c]))
        elif role == "categorical" and df[c].nunique() > HIGH_CARDINALITY:
            out.append(Observation("cardinality", f"'{c}' has {df[c].nunique()} distinct values; charts show the most frequent and group the rest.", [c]))
    tc = top_correlations(correlation_matrix(df), 3)
    for _, r in tc[tc["correlation"].abs() >= 0.7].iterrows():
        out.append(Observation("correlation", f"'{r['column A']}' and '{r['column B']}' move together (correlation {r['correlation']:.2f}). This shows association, not cause.",
                               [r["column A"], r["column B"]]))
    return out[:max_items]
