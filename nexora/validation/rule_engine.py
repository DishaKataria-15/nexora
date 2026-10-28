"""Generic, configuration-driven rule engine.

The engine knows nothing about any industry. A rule is a *type* plus parameters
and logical field names; a column mapping connects logical fields to the real
dataset columns. The engine never changes the input DataFrame.

Each check returns ``Output(hits=[...])``. The runner turns hits into
``Finding`` objects and records a ``RuleResult`` for *every* rule, including
rules that were disabled or could not run (so nothing fails silently).
"""
from __future__ import annotations

import ast
import logging
import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from .findings import Finding, RuleResult
from .profiles import Profile, Rule, apply_overrides, expression_names
from . import scoring

log = logging.getLogger(__name__)

NS_UTC = "datetime64[ns, UTC]"


class SkipRule(Exception):
    """Raised by a check when it cannot run (missing reference data etc.)."""


@dataclass
class Hit:
    row: int | None
    fields: list[str]
    observed: str
    expected: str
    detail: str = ""
    category: str | None = None  # optional per-hit override (e.g. rollover -> Informational)
    severity: str | None = None


@dataclass
class Output:
    hits: list[Hit] = field(default_factory=list)
    rows_checked: int = 0
    cells_checked: int = 0
    note: str = ""


def _trunc(value: Any, n: int = 200) -> str:
    s = str(value)
    return s if len(s) <= n else s[: n - 3] + "..."


def _fmt_seconds(sec: float) -> str:
    sec = float(sec)
    if sec < 120:
        return f"{sec:.0f} s"
    if sec < 7200:
        return f"{sec / 60:.1f} min"
    if sec < 172800:
        return f"{sec / 3600:.1f} h"
    return f"{sec / 86400:.1f} days"


# ----------------------------------------------------------------------------
# Execution context: parsing helpers with caching
# ----------------------------------------------------------------------------

class Context:
    def __init__(self, df: pd.DataFrame, mapping: dict[str, str | None], as_of: pd.Timestamp,
                 null_tokens: list[str], datetime_formats: list[str],
                 reference_tables: dict[str, pd.DataFrame] | None = None):
        self.df = df
        self.mapping = {k: v for k, v in mapping.items() if v and v in df.columns}
        self.as_of = as_of
        self.null_tokens = set(null_tokens)
        self.datetime_formats = datetime_formats
        self.reference_tables = reference_tables or {}
        self._cache: dict[Any, Any] = {}

    def resolve(self, f: str) -> str | None:
        return self.mapping.get(f)

    def col(self, f: str) -> pd.Series:
        return self.df[self.mapping[f]]

    def raw(self, f: str) -> pd.Series:
        return self.col(f)

    def blank(self, f: str) -> pd.Series:
        key = ("blank", f)
        if key not in self._cache:
            s = self.col(f).str.strip().str.lower()
            self._cache[key] = s.isin(self.null_tokens)
        return self._cache[key]

    def numeric(self, f: str, thousands: bool = False) -> pd.Series:
        key = ("num", f, thousands)
        if key not in self._cache:
            s = self.col(f).str.strip()
            if thousands:
                s = s.str.replace(",", "", regex=False)
            num = pd.to_numeric(s, errors="coerce").astype(float)
            num = num.where(np.isfinite(num))
            self._cache[key] = num.where(~self.blank(f))
        return self._cache[key]

    def datetime(self, f: str, formats: list[str] | None = None) -> pd.Series:
        fmts = tuple(formats or self.datetime_formats)
        key = ("dt", f, fmts)
        if key not in self._cache:
            s = self.col(f).str.strip()
            out = pd.Series(pd.NaT, index=s.index, dtype=NS_UTC)
            for fmt in fmts:
                parsed = pd.to_datetime(s, format=fmt, errors="coerce", utc=True).astype(NS_UTC)
                out = out.fillna(parsed)
            self._cache[key] = out.where(~self.blank(f))
        return self._cache[key]

    def eligible(self, when: list[dict[str, Any]]) -> pd.Series:
        mask = pd.Series(True, index=self.df.index)
        for cond in when:
            f = cond["field"]
            norm = self.col(f).str.strip().str.lower()
            if "in" in cond:
                mask &= norm.isin({str(v).strip().lower() for v in cond["in"]})
            if "not_in" in cond:
                mask &= ~norm.isin({str(v).strip().lower() for v in cond["not_in"]})
            if cond.get("blank"):
                mask &= self.blank(f)
            if cond.get("not_blank"):
                mask &= ~self.blank(f)
        return mask

    def group_key(self, group_fields: list[str] | None) -> pd.Series:
        if not group_fields:
            return pd.Series("", index=self.df.index)
        key = self.col(group_fields[0]).str.strip()
        for g in group_fields[1:]:
            key = key + "|" + self.col(g).str.strip()
        return key

    def display(self, f: str, row: int) -> str:
        return _trunc(self.col(f).iat[row])


def ref(row: int) -> int:
    """Spreadsheet-style row number (header is row 1)."""
    return int(row) + 2


# ----------------------------------------------------------------------------
# Safe arithmetic expression evaluation (numbers + - * / only)
# ----------------------------------------------------------------------------

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def eval_expression(expr: str, values: dict[str, pd.Series]) -> pd.Series:
    def walk(node: ast.AST):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.UnaryOp):
            v = walk(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            return _BIN[type(node.op)](walk(node.left), walk(node.right))
        raise ValueError("unsupported expression")  # unreachable: validated at load time

    result = walk(ast.parse(expr, mode="eval"))
    if not isinstance(result, pd.Series):
        raise ValueError("expression must reference at least one field")
    return result.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------

def chk_required(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    out = Output(rows_checked=int(elig.sum()))
    for f in rule.fields:
        bad = ctx.blank(f) & elig
        out.cells_checked += int(elig.sum())
        for i in bad[bad].index:
            out.hits.append(Hit(int(i), [f], "(blank)", "A non-empty value"))
    return out


def chk_unique(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    if rule.fields == ["*"]:
        cols = list(ctx.df.columns)
        names = cols
        usable = elig
    else:
        cols = [ctx.resolve(f) for f in rule.fields]
        names = list(rule.fields)
        usable = elig.copy()
        for f in rule.fields:
            usable &= ~ctx.blank(f)  # records with blank key parts are the 'required' rule's job
    sub = ctx.df.loc[usable, cols]
    out = Output(rows_checked=int(usable.sum()))
    if sub.empty:
        return out
    dup = sub.duplicated(keep=False)
    dup_rows = sub[dup]
    if dup_rows.empty:
        return out
    keys = dup_rows.apply(lambda r: tuple(r.values), axis=1)
    groups: dict[tuple, list[int]] = {}
    for idx, k in keys.items():
        groups.setdefault(k, []).append(int(idx))
    for k, idxs in groups.items():
        refs = ", ".join(str(ref(i)) for i in idxs[:5]) + (", ..." if len(idxs) > 5 else "")
        shown = " | ".join(_trunc(v, 60) for v in k[:4]) + (" ..." if len(k) > 4 else "")
        for i in idxs:
            out.hits.append(Hit(i, names if rule.fields != ["*"] else ["(all columns)"], shown,
                                f"Each combination of {', '.join(names[:4])} to appear once",
                                f"The same value appears {len(idxs)} times (rows {refs})."))
    return out


def chk_numeric(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    out = Output()
    thousands = bool(rule.params.get("allow_thousands", False))
    for f in rule.fields:
        nb = elig & ~ctx.blank(f)
        out.cells_checked += int(nb.sum())
        bad = nb & ctx.numeric(f, thousands).isna()
        for i in bad[bad].index:
            out.hits.append(Hit(int(i), [f], ctx.display(f, i), "A plain number (no text or units)"))
    out.rows_checked = out.cells_checked
    return out


def chk_datetime(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    out = Output()
    fmts = rule.params.get("formats")
    shown = ", ".join(fmts or ctx.datetime_formats)
    for f in rule.fields:
        nb = elig & ~ctx.blank(f)
        out.cells_checked += int(nb.sum())
        bad = nb & ctx.datetime(f, fmts).isna()
        for i in bad[bad].index:
            out.hits.append(Hit(int(i), [f], ctx.display(f, i), f"A valid date/time in format: {shown}"))
    out.rows_checked = out.cells_checked
    return out


def chk_regex(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    out = Output()
    pattern = rule.params["pattern"]
    flags = 0
    if rule.params.get("ignore_case", False):
        import re
        flags = re.IGNORECASE
    for f in rule.fields:
        nb = elig & ~ctx.blank(f)
        out.cells_checked += int(nb.sum())
        s = ctx.col(f).str.strip().str.slice(0, 500)
        ok = s.str.fullmatch(pattern, flags=flags)
        bad = nb & ~ok.astype(bool)
        for i in bad[bad].index:
            out.hits.append(Hit(int(i), [f], ctx.display(f, i), f"Format matching {_trunc(pattern, 80)}",
                                rule.params.get("format_hint", "")))
    out.rows_checked = out.cells_checked
    return out


def chk_allowed_values(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    out = Output()
    cs = bool(rule.params.get("case_sensitive", False))
    allowed = {str(v).strip() if cs else str(v).strip().lower() for v in rule.params["values"]}
    shown = ", ".join(str(v) for v in rule.params["values"][:12]) + (", ..." if len(rule.params["values"]) > 12 else "")
    for f in rule.fields:
        nb = elig & ~ctx.blank(f)
        out.cells_checked += int(nb.sum())
        s = ctx.col(f).str.strip()
        s = s if cs else s.str.lower()
        bad = nb & ~s.isin(allowed)
        for i in bad[bad].index:
            out.hits.append(Hit(int(i), [f], ctx.display(f, i), f"One of: {shown}"))
    out.rows_checked = out.cells_checked
    return out


def _bounds_text(lo, hi) -> str:
    if lo is not None and hi is not None:
        return f"Between {lo} and {hi}"
    return f">= {lo}" if lo is not None else f"<= {hi}"


def _range_one(rule: Rule, ctx: Context, elig: pd.Series, f: str, out: Output) -> None:
    p = rule.params
    num = ctx.numeric(f, bool(p.get("allow_thousands", False)))
    valid = elig & num.notna()
    checked = int(valid.sum())
    if p.get("ranges_by_unit"):
        units = ctx.col(p["unit_field"]).str.strip().str.lower()
        table = {str(k).strip().lower(): v for k, v in p["ranges_by_unit"].items()}
        for i in valid[valid].index:
            r = table.get(units.at[i])
            if r is None:
                checked -= 1  # no configured range for this unit: not checked
                continue
            lo, hi, v = r.get("min"), r.get("max"), num.at[i]
            if (lo is not None and v < lo) or (hi is not None and v > hi):
                out.hits.append(Hit(int(i), [f, p["unit_field"]], f"{ctx.display(f, i)} {units.at[i]}",
                                    f"{_bounds_text(lo, hi)} {units.at[i]} (configured)",
                                    "This is a configured plausibility limit, not proof of an error."))
    else:
        lo, hi = p.get("min"), p.get("max")
        no = pd.Series(False, index=num.index)
        bad = valid & (((num < lo) if lo is not None else no) | ((num > hi) if hi is not None else no))
        for i in bad[bad].index:
            out.hits.append(Hit(int(i), [f], ctx.display(f, i), f"{_bounds_text(lo, hi)} (configured)",
                                "This is a configured plausibility limit, not proof of an error."))
    out.cells_checked += checked
    out.rows_checked = max(out.rows_checked, checked)


def chk_range(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    out = Output()
    for f in rule.fields:
        _range_one(rule, ctx, elig, f, out)
    return out


def _resolve_time_bound(value: Any, ctx: Context) -> pd.Timestamp:
    if str(value).strip().lower() == "as_of":
        return ctx.as_of
    return pd.Timestamp(value, tz="UTC") if pd.Timestamp(value).tzinfo is None else pd.Timestamp(value).tz_convert("UTC")


def chk_datetime_range(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    out = Output()
    p = rule.params
    lo = _resolve_time_bound(p["not_before"], ctx) if p.get("not_before") else None
    hi = _resolve_time_bound(p["not_after"], ctx) if p.get("not_after") else None
    exp = []
    if lo is not None:
        exp.append(f"on or after {lo:%Y-%m-%d}")
    if hi is not None:
        exp.append(f"on or before {hi:%Y-%m-%d}")
    for f in rule.fields:
        ts = ctx.datetime(f, p.get("formats"))
        valid = elig & ts.notna()
        out.cells_checked += int(valid.sum())
        no = pd.Series(False, index=ts.index)
        bad = valid & (((ts < lo) if lo is not None else no) | ((ts > hi) if hi is not None else no))
        for i in bad[bad].index:
            out.hits.append(Hit(int(i), [f], ctx.display(f, i), "A date " + " and ".join(exp)))
    out.rows_checked = out.cells_checked
    return out


_CMP = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq, "!=": operator.ne}


def chk_compare(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    p = rule.params
    a, b = rule.fields
    if p["kind"] == "numeric":
        va, vb = ctx.numeric(a), ctx.numeric(b)
    elif p["kind"] == "text":
        va = ctx.col(a).str.strip().str.lower().where(~ctx.blank(a))
        vb = ctx.col(b).str.strip().str.lower().where(~ctx.blank(b))
    else:
        va, vb = ctx.datetime(a, p.get("formats")), ctx.datetime(b, p.get("formats"))
    valid = elig & va.notna() & vb.notna()
    bad = valid & ~_CMP[p["op"]](va, vb).astype(bool)
    out = Output(rows_checked=int(valid.sum()), cells_checked=int(valid.sum()))
    for i in bad[bad].index:
        out.hits.append(Hit(int(i), [a, b], f"{ctx.display(a, i)} vs {ctx.display(b, i)}",
                            f"{a} {p['op']} {b}"))
    return out


def chk_date_order(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    start, end = rule.fields
    allow_equal = bool(rule.params.get("allow_equal", True))
    ts, te = ctx.datetime(start, rule.params.get("formats")), ctx.datetime(end, rule.params.get("formats"))
    valid = elig & ts.notna() & te.notna()
    bad = valid & ((te < ts) if allow_equal else (te <= ts))
    out = Output(rows_checked=int(valid.sum()), cells_checked=int(valid.sum()))
    for i in bad[bad].index:
        out.hits.append(Hit(int(i), [start, end], f"{start}={ctx.display(start, i)}; {end}={ctx.display(end, i)}",
                            f"{end} should not be earlier than {start}"))
    return out


def chk_expression(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    p = rule.params
    target = p["target"]
    names = expression_names(p["expression"])
    nums = {n: ctx.numeric(n) for n in names}
    for n in p.get("blank_as_zero", []) or []:  # e.g. an empty discount means "no discount"
        if n in nums:
            nums[n] = nums[n].where(~ctx.blank(n), 0.0)
    tgt = ctx.numeric(target)
    expected = eval_expression(p["expression"], nums)
    valid = elig & tgt.notna() & expected.notna()
    tol_abs, tol_pct = float(p.get("tolerance_abs", 0.01)), float(p.get("tolerance_pct", 0.0))
    allowed = np.maximum(tol_abs, expected.abs() * tol_pct / 100.0)
    bad = valid & ((tgt - expected).abs() > allowed)
    out = Output(rows_checked=int(valid.sum()), cells_checked=int(valid.sum()))
    for i in bad[bad].index:
        out.hits.append(Hit(int(i), [target] + [n for n in names if n != target],
                            f"{target}={ctx.display(target, i)}",
                            f"{target} = {p['expression']} (here {expected.at[i]:.6g})",
                            f"Difference is {abs(tgt.at[i] - expected.at[i]):.6g}; tolerance is {allowed.at[i]:.6g}."))
    return out


def chk_referential(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    p = rule.params
    f = rule.fields[0]
    if p.get("reference_field"):
        rf = p["reference_field"]
        allowed = set(ctx.col(rf)[~ctx.blank(rf)].str.strip())
        where = f"values of {rf} in this dataset"
    elif p.get("reference_table"):
        tbl = ctx.reference_tables.get(p["reference_table"])
        if tbl is None:
            raise SkipRule(f"Reference table '{p['reference_table']}' was not provided.")
        if p["reference_column"] not in tbl.columns:
            raise SkipRule(f"Reference table '{p['reference_table']}' has no column '{p['reference_column']}'.")
        allowed = set(tbl[p["reference_column"]].astype(str).str.strip())
        where = f"{p['reference_table']}.{p['reference_column']}"
    else:
        allowed = {str(v).strip() for v in p["reference_values"]}
        where = "the configured code list"
    nb = elig & ~ctx.blank(f)
    bad = nb & ~ctx.col(f).str.strip().isin(allowed)
    out = Output(rows_checked=int(nb.sum()), cells_checked=int(nb.sum()))
    for i in bad[bad].index:
        out.hits.append(Hit(int(i), [f], ctx.display(f, i), f"A value found in {where}"))
    return out


def _sorted_frame(ctx: Context, elig: pd.Series, value: pd.Series, group_fields: list[str] | None,
                  extra: pd.Series | None = None) -> pd.DataFrame:
    valid = elig & value.notna()
    d = pd.DataFrame({"g": ctx.group_key(group_fields)[valid], "v": value[valid]})
    if extra is not None:
        d["o"] = extra[valid]
    return d


def chk_timestamp_gap(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    p = rule.params
    f = rule.fields[0]
    ts = ctx.datetime(f, p.get("formats"))
    d = _sorted_frame(ctx, elig, ts, p.get("group_by")).sort_values(["g", "v"], kind="stable")
    prev_t = d.groupby("g")["v"].shift(1)
    prev_i = pd.Series(d.index, index=d.index).groupby(d["g"]).shift(1)
    gap = d["v"] - prev_t
    limit = pd.to_timedelta(float(p["max_gap_seconds"]), unit="s")
    bad = gap > limit
    out = Output(rows_checked=len(d), cells_checked=len(d))
    for i in bad[bad].index:
        grp = f" for '{d.at[i, 'g']}'" if p.get("group_by") else ""
        out.hits.append(Hit(int(i), [f], ctx.display(f, i),
                            f"No more than {_fmt_seconds(limit.total_seconds())} between consecutive records",
                            f"Gap of {_fmt_seconds(gap.at[i].total_seconds())} since row {ref(int(prev_i.at[i]))}{grp}. "
                            "A gap can be a real outage, a sensor off period or a data-loss problem."))
    return out


def chk_timestamp_order(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    f = rule.fields[0]
    ts = ctx.datetime(f, rule.params.get("formats"))
    d = _sorted_frame(ctx, elig, ts, rule.params.get("group_by")).sort_values("g", kind="stable")
    prev_t = d.groupby("g")["v"].shift(1)
    prev_i = pd.Series(d.index, index=d.index).groupby(d["g"]).shift(1)
    bad = (d["v"] - prev_t) < pd.to_timedelta(0, unit="s")
    out = Output(rows_checked=len(d), cells_checked=len(d))
    for i in bad[bad].index:
        out.hits.append(Hit(int(i), [f], ctx.display(f, i), "Timestamps that do not go backwards in file order",
                            f"This timestamp is earlier than the one on row {ref(int(prev_i.at[i]))}."))
    return out


def chk_sequence(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    p = rule.params
    f = rule.fields[0]
    step = float(p.get("step", 1))
    d = _sorted_frame(ctx, elig, ctx.numeric(f), p.get("group_by")).sort_values(["g", "v"], kind="stable")
    diff = d.groupby("g")["v"].diff()
    prev_v = d.groupby("g")["v"].shift(1)
    bad = diff > step + 1e-9
    out = Output(rows_checked=len(d), cells_checked=len(d))
    for i in bad[bad].index:
        missing = int(round((diff.at[i] / step) - 1))
        out.hits.append(Hit(int(i), [f], ctx.display(f, i), f"Consecutive values (step {step:g})",
                            f"Jumped from {prev_v.at[i]:g} to {d.at[i, 'v']:g}; about {missing} value(s) missing."))
    return out


def chk_monotonic(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    p = rule.params
    f = rule.fields[0]
    extra = ctx.datetime(p["order_by"]) if p.get("order_by") else None
    d = _sorted_frame(ctx, elig, ctx.numeric(f), p.get("group_by"), extra)
    d = d.sort_values(["g", "o"], kind="stable") if extra is not None else d.sort_values("g", kind="stable")
    prev_v = d.groupby("g")["v"].shift(1)
    prev_i = pd.Series(d.index, index=d.index).groupby(d["g"]).shift(1)
    bad = d["v"] < prev_v
    out = Output(rows_checked=len(d), cells_checked=len(d))
    roll = p.get("rollover_max")
    for i in bad[bad].index:
        detail = f"Value dropped from {prev_v.at[i]:g} to {d.at[i, 'v']:g} (previous reading on row {ref(int(prev_i.at[i]))})."
        if roll is not None and prev_v.at[i] >= 0.9 * float(roll) and d.at[i, "v"] <= 0.1 * float(roll):
            out.hits.append(Hit(int(i), [f], ctx.display(f, i), "Cumulative reading that does not decrease",
                                detail + f" This looks like a meter rollover (maximum {roll}); confirm with the meter record.",
                                category="Informational", severity="Info"))
        else:
            out.hits.append(Hit(int(i), [f], ctx.display(f, i), "Cumulative reading that does not decrease",
                                detail + " Possible meter reset, replacement or data error."))
    return out


def chk_unit_consistency(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    p = rule.params
    f = rule.fields[0]
    nb = elig & ~ctx.blank(f)
    units = ctx.col(f).str.strip()
    flagged: dict[int, Hit] = {}
    if p.get("expected_units"):
        allowed = {str(u).strip().lower() for u in p["expected_units"]}
        shown = ", ".join(str(u) for u in p["expected_units"])
        for i in nb[nb & ~units.str.lower().isin(allowed)].index:
            flagged[int(i)] = Hit(int(i), [f], units.at[i], f"A unit from: {shown}")
    if p.get("group_by") or p.get("dataset_wide"):
        key = ctx.group_key(p.get("group_by"))
        d = pd.DataFrame({"g": key[nb], "u": units[nb]})
        for g, sub in d.groupby("g"):
            counts = sub["u"].str.lower().value_counts()
            if len(counts) > 1:
                main = counts.index[0]
                for i in sub.index[sub["u"].str.lower() != main]:
                    where = f"'{g}'" if p.get("group_by") else "this dataset"
                    flagged.setdefault(int(i), Hit(int(i), [f] + list(p.get("group_by") or []), units.at[i],
                                                   f"One consistent unit within {where} (most common is '{main}')",
                                                   f"Mixed units found within {where}: {', '.join(sub['u'].unique()[:5])}."))
    out = Output(rows_checked=int(nb.sum()), cells_checked=int(nb.sum()))
    out.hits = list(flagged.values())
    return out


def chk_freshness(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    p = rule.params
    f = rule.fields[0]
    hours = float(p["max_age_hours"]) if p.get("max_age_hours") is not None else float(p["max_age_days"]) * 24
    ts = ctx.datetime(f, p.get("formats"))[elig].dropna()
    out = Output(rows_checked=1, cells_checked=1)
    if ts.empty:
        out.hits.append(Hit(None, [f], "(no valid timestamps)", f"Newest record within {hours:g} hours of {ctx.as_of:%Y-%m-%d %H:%M} UTC"))
        return out
    newest = ts.max()
    age_h = (ctx.as_of - newest).total_seconds() / 3600
    if age_h > hours:
        out.hits.append(Hit(None, [f], f"newest={newest:%Y-%m-%d %H:%M} UTC",
                            f"Newest record no older than {hours:g} hours at {ctx.as_of:%Y-%m-%d %H:%M} UTC",
                            f"The newest record is {_fmt_seconds(age_h * 3600)} old."))
    return out


def chk_statistical(rule: Rule, ctx: Context, elig: pd.Series) -> Output:
    p = rule.params
    method = p.get("method", "iqr")
    excluded = set(p.get("exclude_fields", []) or [])
    fields = [f for f in rule.fields if f not in excluded]
    if not fields:
        raise SkipRule("All fields are excluded from statistical screening by the profile.")
    min_n = int(p.get("min_samples", 10))
    out = Output()
    notes: list[str] = []
    disclaimer = " A statistical anomaly is a screening signal only; the value may be genuine."
    if method == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:  # pragma: no cover
            raise SkipRule("scikit-learn is not installed.") from exc
        feats = pd.concat([ctx.numeric(f) for f in fields], axis=1)
        valid = elig & feats.notna().all(axis=1)
        out.rows_checked = int(valid.sum())
        if valid.sum() < min_n:
            raise SkipRule(f"Only {int(valid.sum())} complete rows; at least {min_n} are needed.")
        model = IsolationForest(contamination=float(p.get("contamination", 0.02)), random_state=42, n_estimators=100)
        X = feats[valid].to_numpy()
        pred = model.fit_predict(X)
        score = model.decision_function(X)
        for pos, i in enumerate(feats[valid].index):
            if pred[pos] == -1:
                out.hits.append(Hit(int(i), list(fields), "; ".join(f"{f}={ctx.display(f, i)}" for f in fields),
                                    "A combination typical of the rest of the data",
                                    f"Isolation Forest score {score[pos]:.3f} (lower = more isolated; contamination="
                                    f"{p.get('contamination', 0.02)}, random_state=42)." + disclaimer))
        return out
    group_fields = p.get("group_by")
    keys = ctx.group_key(group_fields)
    for f in fields:
        num = ctx.numeric(f)
        valid = elig & num.notna()
        out.rows_checked += int(valid.sum())
        for g, idx in keys[valid].groupby(keys[valid]).groups.items():
            vals = num[idx]
            label = f" in group '{g}'" if group_fields else ""
            if len(vals) < min_n:
                notes.append(f"{f}{label}: only {len(vals)} values (need {min_n}); skipped")
                continue
            if method == "iqr":
                k = float(p.get("k", 1.5))
                q1, q3 = np.percentile(vals, [25, 75])
                iqr = q3 - q1
                if iqr == 0:
                    notes.append(f"{f}{label}: IQR is zero; method not applicable")
                    continue
                lo, hi = q1 - k * iqr, q3 + k * iqr
                flagged = vals[(vals < lo) | (vals > hi)]
                expected = f"Within [{lo:.4g}, {hi:.4g}] (IQR fences, k={k:g})"
                detail = f"Outside the typical range computed from {len(vals)} values (Q1={q1:.4g}, Q3={q3:.4g}){label}."
            else:
                thr = float(p.get("threshold", 3.5))
                med = float(np.median(vals))
                mad = float(np.median(np.abs(vals - med)))
                if mad == 0:
                    notes.append(f"{f}{label}: MAD is zero; method not applicable")
                    continue
                z = 0.6745 * (vals - med) / mad
                flagged = vals[z.abs() > thr]
                expected = f"Robust z-score within +/-{thr:g} (median={med:.4g}, MAD={mad:.4g})"
                detail = f"Robust z-score is far from the median of {len(vals)} values{label}."
            for i, v in flagged.items():
                out.hits.append(Hit(int(i), [f], ctx.display(f, i), expected, detail + disclaimer))
    out.note = "; ".join(notes[:5])
    return out


# Rule types that check each listed field independently: they run on whichever fields are mapped.
PER_FIELD_TYPES = {"required", "numeric", "datetime", "regex", "allowed_values", "datetime_range"}

CHECKS: dict[str, Callable[[Rule, Context, pd.Series], Output]] = {
    "required": chk_required, "unique": chk_unique, "duplicate_rows": chk_unique, "numeric": chk_numeric,
    "datetime": chk_datetime, "regex": chk_regex, "allowed_values": chk_allowed_values, "range": chk_range,
    "datetime_range": chk_datetime_range, "compare": chk_compare, "date_order": chk_date_order,
    "expression_equals": chk_expression, "referential": chk_referential, "timestamp_gap": chk_timestamp_gap,
    "timestamp_order": chk_timestamp_order, "sequence": chk_sequence, "monotonic": chk_monotonic,
    "unit_consistency": chk_unit_consistency, "freshness": chk_freshness, "statistical_anomaly": chk_statistical,
}

DEFAULT_RECOMMENDATION = {
    "Invalid": "Check the source record and correct it at the source system if it is wrong.",
    "Inconsistent": "Compare the related fields or records and confirm which value is correct.",
    "Suspicious": "Review with someone who knows the context before changing anything.",
    "Statistical anomaly": "Review; extreme values can be genuine. Do not delete without evidence.",
    "Informational": "No action may be needed; use as context for the data owner.",
}


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------

@dataclass
class ValidationOutcome:
    findings: list[Finding]
    rule_results: list[RuleResult]
    mapping: dict[str, str | None]
    started_at: str
    completed_at: str
    as_of: str
    n_rows: int
    summary: dict[str, Any]


def suggest_effective_mapping(df: pd.DataFrame, profile: Profile, mapping: dict[str, str | None]) -> dict[str, str | None]:
    return {f: (mapping.get(f) if mapping.get(f) in df.columns else None) for f in profile.fields}


def run_validation(
    df: pd.DataFrame,
    profile: Profile,
    mapping: dict[str, str | None],
    *,
    dataset_id: str,
    run_id: str,
    as_of: datetime | pd.Timestamp | str | None = None,
    rule_overrides: dict[str, dict[str, Any]] | None = None,
    anomaly_screening: bool | None = None,
    reference_tables: dict[str, pd.DataFrame] | None = None,
    max_findings_per_rule: int = 2000,
    progress: Callable[[float, str], None] | None = None,
) -> ValidationOutcome:
    """Run all enabled rules of ``profile`` against ``df`` (never modifies ``df``).

    ``anomaly_screening``: None = follow profile defaults, False = switch every
    statistical rule off, True = switch every statistical rule on.
    ``as_of`` fixes "now" for freshness/date-range rules so runs are reproducible.
    """
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    work = df.reset_index(drop=True).fillna("").astype(str)
    as_of_ts = pd.Timestamp(as_of if as_of is not None else datetime.now(timezone.utc))
    as_of_ts = as_of_ts.tz_localize("UTC") if as_of_ts.tzinfo is None else as_of_ts.tz_convert("UTC")
    eff_map = suggest_effective_mapping(work, profile, mapping)
    ctx = Context(work, eff_map, as_of_ts, profile.null_tokens, profile.datetime_formats, reference_tables)

    rules = apply_overrides(profile, rule_overrides)
    if anomaly_screening is not None:
        for r in rules:
            if r.rule_type == "statistical_anomaly":
                r.enabled = bool(anomaly_screening)

    findings: list[Finding] = []
    results: list[RuleResult] = []
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{run_id}-F{counter:05d}"

    # Schema-level check: expected (required) fields that are not mapped
    schema_res = RuleResult("SCHEMA-001", "Expected fields are present", "schema", "Informational", "Medium",
                            profile.version, None, "passed", rows_checked=len(work))
    for fname, spec in profile.fields.items():
        if spec.get("required") and fname not in ctx.mapping:
            schema_res.status = "failed"
            schema_res.findings_total += 1
            schema_res.findings_stored += 1
            findings.append(Finding(
                next_id(), run_id, dataset_id, None, None, [], "SCHEMA-001", profile.version,
                schema_res.name, "Informational", "Medium", "(no column mapped)",
                f"A column for '{spec['label']}' ({fname})",
                f"The profile expects a field '{spec['label']}' but no dataset column is mapped to it. "
                "Rules that depend on it cannot run.",
                "Map the field to the right column on the Schema page, or ignore this if the field does not apply."))
    results.append(schema_res)

    total = max(len(rules), 1)
    for n, rule in enumerate(rules, 1):
        if progress:
            progress(n / total, f"Running {rule.rule_id}: {rule.name}")
        res = RuleResult(rule.rule_id, rule.name, rule.rule_type, rule.category, rule.severity, rule.version,
                         rule.dimension, "passed")
        results.append(res)
        if not rule.enabled:
            res.status, res.message = "disabled", "Disabled (profile default or user choice)."
            continue
        partial = ""
        per_field = rule.rule_type in PER_FIELD_TYPES or (
            rule.rule_type == "statistical_anomaly" and rule.params.get("method", "iqr") != "isolation_forest")
        if per_field and rule.fields:
            mapped = [f for f in rule.fields if f in ctx.mapping]
            if mapped and len(mapped) < len(rule.fields):
                partial = "Not checked (no column mapped): " + ", ".join(f for f in rule.fields if f not in ctx.mapping) + "."
                rule.fields = mapped
        missing = [f for f in rule.required_fields() if f not in ctx.mapping]
        if missing:
            res.status = "skipped"
            res.message = "Cannot run: no dataset column is mapped for " + ", ".join(missing) + "."
            continue
        try:
            elig = ctx.eligible(rule.when)
            out = CHECKS[rule.rule_type](rule, ctx, elig)
        except SkipRule as exc:
            res.status, res.message = "skipped", str(exc)
            continue
        except Exception as exc:  # a broken rule must not abort the whole run
            log.exception("Rule %s failed", rule.rule_id)
            res.status, res.message = "error", f"Rule failed unexpectedly: {type(exc).__name__}: {exc}"
            continue
        res.rows_checked, res.cells_checked = out.rows_checked, out.cells_checked
        res.message = " ".join(x for x in (partial, out.note) if x)
        res.findings_total = len(out.hits)
        res.flagged_rows = {h.row for h in out.hits if h.row is not None
                            and (h.category or rule.category) not in ("Informational",)}
        res.rows_flagged = len({h.row for h in out.hits if h.row is not None})
        res.status = "failed" if out.hits else "passed"
        hits = sorted(out.hits, key=lambda h: (-1 if h.row is None else h.row, ",".join(h.fields)))
        if len(hits) > max_findings_per_rule:
            res.message = (res.message + " " if res.message else "") + \
                f"Showing the first {max_findings_per_rule} of {len(hits)} findings."
            hits = hits[:max_findings_per_rule]
        res.findings_stored = len(hits)
        for h in hits:
            cols = [ctx.resolve(f) or f for f in h.fields]
            findings.append(Finding(
                next_id(), run_id, dataset_id, h.row, None if h.row is None else ref(h.row), cols,
                rule.rule_id, rule.version, rule.name, h.category or rule.category, h.severity or rule.severity,
                _trunc(h.observed), _trunc(h.expected, 300),
                (rule.explanation + (" " + h.detail if h.detail else "")).strip(),
                rule.recommendation or DEFAULT_RECOMMENDATION[rule.category]))
    completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    summary = scoring.compute_quality_summary(results, findings, len(work), profile.scoring_weights)
    return ValidationOutcome(findings, results, eff_map, started, completed, as_of_ts.isoformat(), len(work), summary)
