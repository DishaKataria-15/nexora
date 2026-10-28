"""Combine several datasets into one table: join on key columns, add per-key summaries, or stack rows.

Everything here works on the raw text tables exactly as uploaded, so nothing is re-typed or cleaned on the way.
Key values are compared after trimming surrounding spaces (and, optionally, ignoring case). Every step reports
what matched and what did not, and the result is saved as a new dataset; the source datasets are never changed.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

JOIN_TYPES = {
    "left": "Keep every row of the main table (add matching information where it exists)",
    "inner": "Keep only rows that match in both tables",
    "outer": "Keep every row from both tables",
}
MODES = {
    "join": "Join: add the other table's columns to each matching row",
    "summary": "Summarise: add per-key statistics from the other table (for example the average rating per movie)",
    "stack": "Stack: put the other table's rows underneath (same kind of rows, for example one file per month)",
}
AGGS = ["count", "mean", "sum", "min", "max", "median", "nunique"]
MAX_GROWTH = 20
_TRAILING_ZERO = re.compile(r"^-?\d+\.0+$")          # a join may not multiply the main table's rows by more than this


class CombineError(ValueError):
    """The datasets cannot be combined as configured. The message is safe to show to the user."""


@dataclass
class Step:
    mode: str                                  # join | summary | stack
    other: str                                 # label of the other dataset
    left_on: list[str] = field(default_factory=list)
    right_on: list[str] = field(default_factory=list)
    how: str = "left"
    ignore_case: bool = False
    columns: list[str] | None = None           # join: which columns of the other table to bring (None = all)
    aggs: dict[str, list[str]] = field(default_factory=dict)   # summary: {column: [count, mean, ...]}
    prefix: str = ""                           # added before brought-in column names that would clash

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StepReport:
    step: int
    mode: str
    other: str
    rows_before: int
    rows_other: int
    rows_after: int
    matched_rows: int | None = None
    unmatched_main: int | None = None
    unmatched_other_keys: int | None = None
    duplicate_other_keys: int | None = None
    columns_added: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    examples_unmatched: list[str] = field(default_factory=list)


def _norm_keys(df: pd.DataFrame, cols: list[str], ignore_case: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for i, c in enumerate(cols):
        s = df[c].astype(object).where(df[c].notna(), None).map(lambda v: None if v is None else str(v).strip())
        s = s.map(lambda v: None if v in (None, "") else (v.lower() if ignore_case else v))
        # "12" and "12.0" are the same key (a spreadsheet often adds ".0"); leading zeros such as "007" are kept as written
        s = s.map(lambda v: v if v is None else (v[: v.index(".")] if _TRAILING_ZERO.match(v) else v))
        out[f"__k{i}"] = s
    return out


def _check_cols(df: pd.DataFrame, cols: list[str], which: str) -> None:
    if not cols:
        raise CombineError(f"Choose the key column(s) in the {which} table.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise CombineError(f"The {which} table has no column {', '.join(repr(c) for c in missing)}.")


def _unique_name(name: str, taken: set[str], prefix: str) -> str:
    cand = name if name not in taken else f"{prefix}{name}" if prefix else name
    i = 2
    base = cand
    while cand in taken:
        cand = f"{base}_{i}"
        i += 1
    return cand


def apply_step(main: pd.DataFrame, other: pd.DataFrame, step: Step, n: int, max_rows: int) -> tuple[pd.DataFrame, StepReport]:
    if step.mode not in MODES:
        raise CombineError(f"Unknown combine mode '{step.mode}'.")
    rep = StepReport(n, step.mode, step.other, len(main), len(other), 0)
    prefix = step.prefix or (step.other.split(" (")[0].strip().replace(" ", "_")[:20] + "_")

    if step.mode == "stack":
        same = list(main.columns) == list(other.columns)
        extra, lacking = sorted(set(other.columns) - set(main.columns)), sorted(set(main.columns) - set(other.columns))
        if extra or lacking:
            rep.warnings.append("The tables do not have the same columns. "
                                + (f"Only in '{step.other}': {', '.join(extra[:8])}. " if extra else "")
                                + (f"Missing from '{step.other}': {', '.join(lacking[:8])}. " if lacking else "")
                                + "Missing values are left empty.")
        elif not same:
            rep.warnings.append("The same columns appear in a different order; they were matched by name.")
        out = pd.concat([main, other], ignore_index=True, sort=False)
        if len(out) > max_rows:
            raise CombineError(f"The stacked table would have {len(out):,} rows, above the limit of {max_rows:,}.")
        rep.rows_after = len(out)
        rep.columns_added = extra
        return out, rep

    _check_cols(main, step.left_on, "main")
    _check_cols(other, step.right_on, "other")
    if len(step.left_on) != len(step.right_on):
        raise CombineError("Choose the same number of key columns on both sides.")
    if step.how not in JOIN_TYPES:
        raise CombineError(f"Unknown join type '{step.how}'.")
    lk, rk = _norm_keys(main, step.left_on, step.ignore_case), _norm_keys(other, step.right_on, step.ignore_case)
    kcols = list(lk.columns)
    lmiss, rmiss = int(lk.isna().any(axis=1).sum()), int(rk.isna().any(axis=1).sum())
    if lmiss:
        rep.warnings.append(f"{lmiss:,} row(s) of the main table have an empty key; they cannot match anything.")
    if rmiss:
        rep.warnings.append(f"{rmiss:,} row(s) of '{step.other}' have an empty key and were ignored.")
    right = other.loc[rk.notna().all(axis=1)].copy()
    rkeys = rk.loc[right.index]
    taken = set(main.columns)

    if step.mode == "summary":
        if not step.aggs:
            raise CombineError("Choose at least one statistic to add (for example the count of rows or the mean of a column).")
        g = pd.concat([rkeys, right], axis=1)
        pieces = []
        for col, funcs in step.aggs.items():
            if col not in right.columns:
                raise CombineError(f"'{step.other}' has no column '{col}'.")
            bad = [f for f in funcs if f not in AGGS]
            if bad:
                raise CombineError(f"Unknown statistic(s): {', '.join(bad)}.")
            vals = pd.to_numeric(right[col], errors="coerce")
            numeric_ok = vals.notna().sum() >= 0.9 * max(right[col].notna().sum(), 1)
            for f in funcs:
                if f in ("mean", "sum", "min", "max", "median") and not numeric_ok:
                    raise CombineError(f"'{col}' is not numeric, so its {f} cannot be computed. Use count or nunique.")
                src = vals if f in ("mean", "sum", "min", "max", "median") else right[col]
                agg = pd.concat([g[kcols], src.rename("__v")], axis=1).groupby(kcols, dropna=True)["__v"].agg(f)
                name = _unique_name(f"{col}_{f}", taken, prefix)
                taken.add(name)
                pieces.append(agg.rename(name))
                rep.columns_added.append(name)
        summ = pd.concat(pieces, axis=1).reset_index()
        rep.duplicate_other_keys = 0
        left = pd.concat([main.reset_index(drop=True), lk.reset_index(drop=True)], axis=1)
        out = left.merge(summ, on=kcols, how="left" if step.how == "left" else step.how, indicator=True)
        _match_report(rep, out, step, left, summ, kcols)
        out = out.drop(columns=kcols + ["_merge"])
        rep.rows_after = len(out)
        return out, rep

    # join
    cols = [c for c in (step.columns if step.columns is not None else list(other.columns)) if c in other.columns and c not in step.right_on]
    if step.columns is not None and not cols:
        raise CombineError(f"Choose at least one column to bring in from '{step.other}'.")
    dup = int(rkeys.duplicated().sum())
    rep.duplicate_other_keys = dup
    if dup:
        rep.warnings.append(f"{dup:,} key value(s) appear more than once in '{step.other}', so a matching row of the main table is repeated "
                            "once per match. If the other table has several rows per key (for example many ratings per movie), "
                            "'Summarise' usually fits better.")
    renamed = {}
    for c in cols:
        new = _unique_name(c, taken, prefix)
        taken.add(new)
        renamed[c] = new
    r = pd.concat([rkeys, right[cols].rename(columns=renamed)], axis=1)
    left = pd.concat([main.reset_index(drop=True), lk.reset_index(drop=True)], axis=1)
    est = int(left[kcols].merge(r[kcols], on=kcols, how=step.how).shape[0])
    if est > max_rows:
        raise CombineError(f"The joined table would have {est:,} rows, above the limit of {max_rows:,}. Check the keys, or use 'Summarise'.")
    if len(main) and est > MAX_GROWTH * len(main):
        raise CombineError(f"The join would turn {len(main):,} rows into {est:,}. The keys are probably not the right ones; check them, or use 'Summarise'.")
    out = left.merge(r, on=kcols, how=step.how, indicator=True)
    _match_report(rep, out, step, left, r, kcols)
    if step.how == "outer":
        # rows that exist only in the other table: copy their key values into the main key columns so they are not blank
        only_r = out["_merge"] == "right_only"
        for lc, kc in zip(step.left_on, kcols):
            out.loc[only_r, lc] = out.loc[only_r, kc]
    out = out.drop(columns=kcols + ["_merge"])
    rep.columns_added = list(renamed.values())
    rep.rows_after = len(out)
    return out, rep


def _match_report(rep: StepReport, merged: pd.DataFrame, step: Step, left: pd.DataFrame, right: pd.DataFrame, kcols: list[str]) -> None:
    m = merged["_merge"]
    rep.matched_rows = int((m == "both").sum())
    lkeys = set(map(tuple, left[kcols].dropna().values.tolist()))
    rkeys = set(map(tuple, right[kcols].dropna().values.tolist()))
    unmatched_main = left[kcols].dropna().apply(tuple, axis=1).map(lambda k: k not in rkeys)
    rep.unmatched_main = int(unmatched_main.sum()) + int(left[kcols].isna().any(axis=1).sum())
    only_other = rkeys - lkeys
    rep.unmatched_other_keys = len(only_other)
    if rep.unmatched_main:
        ex = left.loc[unmatched_main[unmatched_main].index, step.left_on].astype(str).agg(" | ".join, axis=1).unique()[:5]
        rep.examples_unmatched = [str(x) for x in ex]
        share = 100 * rep.unmatched_main / max(len(left), 1)
        rep.warnings.append(f"{rep.unmatched_main:,} row(s) of the main table ({share:.1f}%) found no match in '{step.other}'"
                            + (" and were dropped (inner join)." if step.how == "inner" else "; their new columns are empty."))
    if only_other:
        rep.warnings.append(f"{len(only_other):,} key value(s) in '{step.other}' have no row in the main table"
                            + (" and were added as new rows (outer join)." if step.how == "outer" else " and were not used."))
    if rep.matched_rows == 0:
        rep.warnings.append("Nothing matched. Check that the key columns hold the same kind of value on both sides (for example IDs, not names).")


def combine(frames: dict[str, pd.DataFrame], base: str, steps: list[Step], max_rows: int) -> tuple[pd.DataFrame, list[StepReport]]:
    """Apply the steps in order, starting from ``frames[base]``. Inputs are never modified."""
    if base not in frames:
        raise CombineError("Choose the main table.")
    if not steps:
        raise CombineError("Add at least one other dataset to combine with.")
    out = frames[base].copy()
    reports = []
    for i, s in enumerate(steps, start=1):
        if s.other not in frames:
            raise CombineError(f"Dataset '{s.other}' is not available.")
        out, rep = apply_step(out, frames[s.other].copy(), s, i, max_rows)
        reports.append(rep)
    return out, reports


def suggest_keys(a: pd.DataFrame, b: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Column pairs that look like a shared key: (column in a, column in b, share of b's values found in a)."""
    def norm(c: str) -> str:
        return str(c).lower().replace("_", "").replace(" ", "").replace("-", "")
    sets = {ca: set(a[ca].dropna().astype(str).str.strip().head(20000)) for ca in a.columns if a[ca].nunique() > 1}
    out = []
    for cb in b.columns:
        vb = b[cb].dropna().astype(str).str.strip().head(20000)
        if vb.empty or b[cb].nunique() <= 1:
            continue
        for ca, va in sets.items():
            same_name = norm(ca) == norm(cb)
            if not same_name and len(out) >= 20:
                continue
            cover = float(vb.isin(va).mean())
            if (same_name and cover > 0) or cover >= 0.6:
                out.append((ca, cb, round(cover, 3), same_name))
    out.sort(key=lambda t: (not t[3], -t[2]))
    return [(x, y, c) for x, y, c, _ in out[:5]]
