"""Before/after measurement and reconciliation for data preparation."""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from ..validation.profiler import profile_dataframe
from .model import LogEntry


def _fmt_float(v: Any) -> str:
    if v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
        return ""
    return format(float(v), ".15g")


def to_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Render a typed frame as text (blank = missing) the way it would be written to CSV.

    Used for validation of the prepared data and for export, so what is validated is what is exported.
    """
    out: dict[str, pd.Series] = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            nn = s.dropna()
            has_time = bool((nn.dt.normalize() != nn).any()) if len(nn) else False
            fmt = "%Y-%m-%dT%H:%M:%S" if has_time else "%Y-%m-%d"
            out[c] = s.dt.strftime(fmt).astype(object).where(s.notna(), "")
        elif pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
            out[c] = s.astype(object).map(lambda v: "" if pd.isna(v) else ("true" if bool(v) else "false")).astype(object)
        elif str(s.dtype) == "Int64" or pd.api.types.is_integer_dtype(s):
            out[c] = s.astype(object).map(lambda v: "" if pd.isna(v) else str(int(v))).astype(object)
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype(object).map(_fmt_float).astype(object)
        else:
            out[c] = s.astype(object).map(lambda v: "" if pd.isna(v) else str(v)).astype(object)
    return pd.DataFrame(out, index=df.index)


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a frame (values, column names and order), used to prove the raw data was not altered."""
    h = hashlib.sha256()
    h.update("\x1f".join(map(str, df.columns)).encode("utf-8"))
    h.update(str(df.shape).encode())
    h.update(pd.util.hash_pandas_object(df.astype(str), index=True).values.tobytes())
    return h.hexdigest()


def quality_snapshot(text_df: pd.DataFrame, null_tokens: list[str]) -> dict[str, Any]:
    """Domain-independent quality indicators computed from a text frame."""
    n_rows, n_cols = text_df.shape
    tokens = set(t.lower() for t in null_tokens)
    sp = profile_dataframe(text_df.reset_index(drop=True), list(tokens))
    missing_cells = int(sum(c.missing_count for c in sp.columns))
    cells = max(n_rows * n_cols, 1)
    dup_copies = int(text_df.reset_index(drop=True).duplicated(keep="first").sum()) if n_rows else 0
    issue_cols = [c.name for c in sp.columns if c.issues]
    types: dict[str, int] = {}
    for c in sp.columns:
        types[c.inferred_type] = types.get(c.inferred_type, 0) + 1
    return {
        "rows": n_rows, "columns": n_cols, "missing_cells": missing_cells,
        "missing_pct": round(100 * missing_cells / cells, 2),
        "exact_duplicate_rows": dup_copies,
        "columns_with_format_issues": len(issue_cols), "format_issue_columns": issue_cols,
        "column_types": types,
    }


def reconcile(raw_rows: int, raw_cols: list[str], prepared: pd.DataFrame, log: list[LogEntry],
              raw_hash_before: str, raw_hash_after: str, identifier_cols: list[str],
              raw_non_null: dict[str, int]) -> dict[str, Any]:
    """Check invariants that must hold after any preparation run. Every check is computed, none assumed."""
    applied = [e for e in log if e.status not in ("not_approved",)]
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    removed = sum(max(e.rows_in - e.rows_out, 0) for e in applied)
    add("Row counts reconcile", raw_rows - removed == len(prepared),
        f"{raw_rows:,} raw rows - {removed:,} removed = {raw_rows - removed:,}; prepared has {len(prepared):,}.")
    chain_ok, cur = True, raw_rows
    for e in applied:
        if e.rows_in != cur:
            chain_ok = False
        cur = e.rows_out
    add("Step-by-step row chain is consistent", chain_ok and cur == len(prepared), "Each step starts with the row count the previous step ended with.")
    add("No rows were invented", bool(prepared.index.is_unique) and bool(prepared.index.isin(range(raw_rows)).all()),
        "Every prepared row maps back to one raw row.")
    new_cols = [c for c in prepared.columns if c not in raw_cols]
    dropped_cols = [c for c in raw_cols if c not in prepared.columns]
    add("Columns accounted for", len(raw_cols) + len(new_cols) - len(dropped_cols) == prepared.shape[1],
        f"{len(raw_cols)} raw + {len(new_cols)} added - {len(dropped_cols)} dropped = {prepared.shape[1]}.")
    inv = [c for c in identifier_cols if c in prepared.columns and prepared[c].notna().sum() > raw_non_null.get(c, 0)]
    add("Identifiers were not invented", not inv, "No identifier column has more values than the raw data." if not inv else f"More values than raw in: {inv}")
    add("Raw data is unchanged", raw_hash_before == raw_hash_after, "Fingerprint of the raw data is identical before and after preparation.")
    failed = [e for e in applied if e.status == "failed"]
    add("No step failed", not failed, "All executed steps completed." if not failed else f"{len(failed)} step(s) failed: " + ", ".join(e.title for e in failed))
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "rows_raw": raw_rows, "rows_prepared": len(prepared),
            "rows_removed": removed}


def schema_of(df: pd.DataFrame) -> dict[str, str]:
    """Column -> kind used to round-trip a prepared CSV back into typed columns."""
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out[c] = "datetime"
        elif pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
            out[c] = "boolean"
        elif str(s.dtype) == "Int64" or pd.api.types.is_integer_dtype(s):
            out[c] = "integer"
        elif pd.api.types.is_numeric_dtype(s):
            out[c] = "float"
        else:
            out[c] = "text"
    return out


def typed_from_schema(text_df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Rebuild a typed frame from CSV text and a schema produced by :func:`schema_of`."""
    out = {}
    for c in text_df.columns:
        s = text_df[c].astype(object).replace("", np.nan)      # blank means missing, whatever string dtype pandas used
        kind = schema.get(c, "text")
        if kind == "datetime":
            out[c] = pd.to_datetime(s, errors="coerce", format="ISO8601")
        elif kind == "integer":
            out[c] = pd.to_numeric(s, errors="coerce").astype("Int64")
        elif kind == "float":
            out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
        elif kind == "boolean":
            out[c] = s.map({"true": True, "false": False}).astype("boolean")
        else:
            out[c] = s.astype(object)
    return pd.DataFrame(out, index=text_df.index)
