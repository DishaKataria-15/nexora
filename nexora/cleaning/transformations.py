"""Transformation executors for the preparation engine.

Design rules
------------
* Executors never modify the frame they receive; they return a new one.
* Each executor computes all changes first and assigns them last, so a failure
  cannot leave a half-changed frame behind.
* The working frame keeps the *original 0-based row position* as its index, so
  every logged row number can be traced back to the raw file.
* Nothing is guessed: ambiguous values are reported and left to the user.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from .model import Action, LogEntry, OpResult

log = logging.getLogger(__name__)

MAX_LISTED_ROWS = 5000
MAX_EXAMPLES = 5
CURRENCY = re.compile(r"[€$£₹¥]")
BOOL_TRUE = {"true", "yes", "y", "t"}
BOOL_FALSE = {"false", "no", "n", "f"}


def row_ref(index_value: int) -> int:
    """Spreadsheet-style row number (header is row 1) for a 0-based original position."""
    return int(index_value) + 2


@dataclass
class ExecContext:
    null_tokens: set[str]
    as_of: str = ""
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Parsing helpers (pure functions, unit tested directly)
# ----------------------------------------------------------------------------

def detect_decimal_comma(values: pd.Series) -> bool:
    """True if the column consistently uses a decimal comma ("12,50")."""
    v = values.astype(str).str.strip()
    comma_dec = v.str.fullmatch(r"[+-]?\d+,\d{1,2}").sum()
    point_dec = v.str.contains(r"\d\.\d", regex=True).sum()
    return bool(comma_dec > 0 and point_dec == 0)


def parse_number(text: Any, decimal_comma: bool = False) -> float | None:
    """Parse a number written as plain text; None if it is not unambiguously a number."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    t = str(text).strip()
    if not t:
        return None
    negative = False
    if t.startswith("(") and t.endswith(")"):
        negative, t = True, t[1:-1].strip()
    t = CURRENCY.sub("", t).strip()
    if t.startswith("+"):
        t = t[1:]
    if t.startswith("-"):
        negative, t = (not negative), t[1:]
    if re.fullmatch(r"\d+(\.\d+)?", t) or re.fullmatch(r"\.\d+", t):
        val = float(t)
    elif re.fullmatch(r"\d+(\.\d+)?[eE][+-]?\d+", t):
        val = float(t)
    elif re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", t) and not decimal_comma:
        val = float(t.replace(",", ""))
    elif decimal_comma and re.fullmatch(r"\d{1,3},\d{3}", t):
        return None  # "1,000" is ambiguous (one thousand or one point zero): never guess
    elif decimal_comma and re.fullmatch(r"\d{1,3}(\.\d{3})*,\d+|\d+,\d+", t):
        val = float(t.replace(".", "").replace(",", "."))
    else:
        return None
    if not np.isfinite(val):
        return None
    return -val if negative else val


ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?(Z|[+-]\d{2}:?\d{2})?")
NUMERIC_DATE_RE = re.compile(r"(\d{1,4})[/.-](\d{1,2})[/.-](\d{1,4})([ T]\d{1,2}:\d{2}(:\d{2})?)?")
NAMED_DATE_RE = re.compile(
    r"(\d{1,2}[ -](jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[ ,-]+\d{4}|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.? \d{1,2},? \d{4})([ T]\d{1,2}:\d{2}(:\d{2})?)?",
    re.I)
TZ_RE = re.compile(r"(Z|[+-]\d{2}:?\d{2})$")


@dataclass
class DateAnalysis:
    order: str                 # none | dmy | mdy | ambiguous | conflict
    n_iso: int = 0
    n_numeric_other: int = 0
    n_named: int = 0
    n_unrecognised: int = 0
    n_dmy_evidence: int = 0
    n_mdy_evidence: int = 0
    n_ambiguous: int = 0
    n_two_digit_year: int = 0
    tz_mix: str = "none"       # none | all | mixed


def analyze_dates(values: pd.Series) -> DateAnalysis:
    """Classify the written date layouts in a column of non-missing text values."""
    iso = named = other = unrec = dmy = mdy = amb = two = 0
    tz_flags = []
    for raw in values.astype(str).str.strip():
        if ISO_RE.fullmatch(raw):
            iso += 1
            tz_flags.append(bool(TZ_RE.search(raw)) and len(raw) > 10)
            continue
        m = NUMERIC_DATE_RE.fullmatch(raw)
        if m:
            a, b, c = m.group(1), m.group(2), m.group(3)
            if len(a) == 4:
                other += 1  # year first, written Y/M/D: unambiguous
                continue
            if len(c) != 4:
                two += 1
                unrec += 1
                continue
            other += 1
            ia, ib = int(a), int(b)
            if ia > 12 and ib <= 12:
                dmy += 1
            elif ib > 12 and ia <= 12:
                mdy += 1
            elif ia <= 12 and ib <= 12 and ia != ib:
                amb += 1
            continue
        if NAMED_DATE_RE.fullmatch(raw):
            named += 1
            continue
        unrec += 1
    if dmy and mdy:
        order = "conflict"
    elif dmy:
        order = "dmy"
    elif mdy:
        order = "mdy"
    elif amb:
        order = "ambiguous"
    else:
        order = "none"
    tz_mix = "none"
    if tz_flags:
        tz_mix = "all" if all(tz_flags) else ("mixed" if any(tz_flags) else "none")
    return DateAnalysis(order, iso, other, named, unrec, dmy, mdy, amb, two, tz_mix)


def _recognised_layout(v: str) -> bool:
    """True for layouts whose meaning is unambiguous once day/month order is known (4-digit years only)."""
    m = NUMERIC_DATE_RE.fullmatch(v)
    if m:
        return len(m.group(1)) == 4 or len(m.group(3)) == 4
    return bool(NAMED_DATE_RE.fullmatch(v))


def parse_dates(values: pd.Series, order: str = "auto") -> tuple[pd.Series, pd.Series]:
    """Parse text dates. Returns (datetimes with NaT where unreadable, mask of unreadable values).

    ``values`` must already have missing values as NaN. ``order`` is dmy or mdy for
    ambiguous numeric dates; ISO and written-out month names never need it.
    """
    s = values.astype(object)
    present = s.notna()
    text = s.where(present, "").astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    iso_mask = text.map(lambda v: bool(ISO_RE.fullmatch(v))) & present
    if iso_mask.any():
        has_tz = text[iso_mask].map(lambda v: len(v) > 10 and bool(TZ_RE.search(v)))
        if has_tz.all():
            parsed = pd.to_datetime(text[iso_mask], format="ISO8601", errors="coerce", utc=True).dt.tz_localize(None)
        else:
            parsed = pd.to_datetime(text[iso_mask], format="ISO8601", errors="coerce")
            if getattr(parsed.dt, "tz", None) is not None:
                parsed = parsed.dt.tz_localize(None)
        out.loc[iso_mask] = parsed.astype("datetime64[ns]")
    other = present & ~iso_mask
    if other.any():
        dayfirst = order == "dmy"
        sub = text[other]
        parsed = pd.Series(pd.NaT, index=sub.index, dtype="datetime64[ns]")
        ymd = sub.map(lambda v: bool(re.fullmatch(r"\d{4}[/.-]\d{1,2}[/.-]\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?", v)))
        for mask, kwargs in ((ymd, {"dayfirst": False}), (~ymd, {"dayfirst": dayfirst})):
            if mask.any():
                with np.errstate(all="ignore"):
                    p = pd.to_datetime(sub[mask], format="mixed", errors="coerce", **kwargs)
                parsed.loc[mask] = p.astype("datetime64[ns]")
        recognised = sub.map(_recognised_layout)
        parsed = parsed.where(recognised)
        out.loc[other] = parsed
    unreadable = present & out.isna()
    return out, unreadable


def _cap(rows: list[int]) -> list[int]:
    return rows[:MAX_LISTED_ROWS]


def _examples(before: pd.Series, after: pd.Series, mask: pd.Series, n: int = MAX_EXAMPLES) -> list[dict[str, Any]]:
    out = []
    for idx in list(mask[mask].index[:n]):
        b, a = before.loc[idx], after.loc[idx]
        out.append({"row": row_ref(idx), "before": "" if pd.isna(b) else str(b),
                    "after": "" if pd.isna(a) else str(a)})
    return out


# ----------------------------------------------------------------------------
# Executors
# ----------------------------------------------------------------------------

Executor = Callable[[pd.DataFrame, dict[str, Any], ExecContext], tuple[pd.DataFrame, OpResult]]


def op_standardize_nulls(df, p, cx):
    cols = [c for c in (p.get("columns") or list(df.columns)) if c in df.columns]
    updates, changed, examples = {}, 0, []
    for c in cols:
        s = df[c]
        txt = s.where(s.notna(), "").astype(str).str.strip().str.lower()
        mask = s.notna() & txt.isin(cx.null_tokens)
        if mask.any():
            new = s.astype(object).where(~mask, np.nan)
            updates[c] = new
            changed += int(mask.sum())
            if len(examples) < MAX_EXAMPLES:
                examples += _examples(s, new, mask, 2)
    out = df.copy()
    for c, s in updates.items():
        out[c] = s
    return out, OpResult(len(df), len(out), changed, examples=examples[:MAX_EXAMPLES],
                         notes=f"{len(updates)} column(s) contained empty or placeholder values.")


def op_trim_whitespace(df, p, cx):
    cols = [c for c in (p.get("columns") or list(df.columns)) if c in df.columns]
    updates, changed, examples = {}, 0, []
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        txt = s.astype(object)
        stripped = txt.map(lambda v: v.strip() if isinstance(v, str) else v)
        mask = txt.notna() & (txt != stripped)
        if mask.any():
            updates[c] = stripped
            changed += int(mask.sum())
            examples += _examples(txt, stripped, mask, 2)
    out = df.copy()
    for c, s in updates.items():
        out[c] = s
    return out, OpResult(len(df), len(out), changed, examples=examples[:MAX_EXAMPLES])


def op_normalize_case(df, p, cx):
    """p['mappings'] = {col: {'mode': 'upper'|'lower'|'title'|None, 'map': {variant: canonical}}}"""
    updates, changed, examples = {}, 0, []
    for c, spec in (p.get("mappings") or {}).items():
        if c not in df.columns:
            continue
        s = df[c].astype(object)
        new = s.copy()
        mode = spec.get("mode")
        if mode in ("upper", "lower", "title"):
            fn = {"upper": str.upper, "lower": str.lower, "title": str.title}[mode]
            new = new.map(lambda v: fn(v) if isinstance(v, str) else v)
        mp = spec.get("map") or {}
        if mp:
            new = new.map(lambda v: mp.get(v, v) if isinstance(v, str) else v)
        mask = s.notna() & (s != new)
        if mask.any():
            updates[c] = new
            changed += int(mask.sum())
            examples += _examples(s, new, mask, 2)
    out = df.copy()
    for c, s in updates.items():
        out[c] = s
    return out, OpResult(len(df), len(out), changed, examples=examples[:MAX_EXAMPLES])


def _unparseable_policy(p: dict[str, Any], col: str) -> str:
    return (p.get("on_unparseable") or {}).get(col, "leave")


def op_convert_numeric(df, p, cx):
    updates, changed, examples = {}, 0, []
    skipped_rows: list[int] = []
    notes: list[str] = []
    for c in [c for c in p.get("columns", []) if c in df.columns]:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
        present = s.notna()
        txt = s.astype(object).where(present, "").astype(str)
        dec = detect_decimal_comma(txt[present]) if present.any() else False
        parsed = txt.map(lambda v: parse_number(v, dec)) if present.any() else pd.Series(np.nan, index=s.index)
        parsed = pd.to_numeric(parsed, errors="coerce")
        bad = present & parsed.isna()
        if bad.any():
            policy = _unparseable_policy(p, c)
            rows = [row_ref(i) for i in bad[bad].index]
            skipped_rows += rows
            if policy != "set_missing":
                notes.append(f"'{c}': {int(bad.sum())} value(s) are not numbers; column left as text (no change).")
                continue
            notes.append(f"'{c}': {int(bad.sum())} unreadable value(s) set to missing as you chose.")
        ok = present & parsed.notna()
        as_int = bool(ok.any()) and bool((parsed[ok] % 1 == 0).all()) and not dec \
            and not txt[ok].str.contains(r"[.eE]", regex=True).any()
        new = parsed.astype("Int64") if as_int else parsed.astype("float64")
        updates[c] = new
        changed += int(ok.sum())
        examples += _examples(s, new, ok & (s.astype(object).astype(str) != new.astype(object).astype(str)), 2)
    out = df.copy()
    for c, s in updates.items():
        out[c] = s
    status = "applied" if updates else "skipped"
    return out, OpResult(len(df), len(out), changed, skipped_rows=_cap(skipped_rows), skipped_count=len(skipped_rows),
                         notes=" ".join(notes) or f"{len(updates)} column(s) converted to numbers.",
                         examples=examples[:MAX_EXAMPLES], status=status)


def op_parse_dates(df, p, cx):
    updates, changed, examples = {}, 0, []
    skipped_rows: list[int] = []
    notes: list[str] = []
    for c in [c for c in p.get("columns", []) if c in df.columns]:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        present = s.notna()
        if not present.any():
            continue
        info = analyze_dates(s[present])
        order = (p.get("order") or {}).get(c) or ("auto" if info.order == "none" else info.order)
        if info.order in ("ambiguous", "conflict") and order not in ("dmy", "mdy"):
            notes.append(f"'{c}': day/month order is {info.order}; column left as text until you choose.")
            skipped_rows += [row_ref(i) for i in present[present].index]
            continue
        if info.tz_mix == "mixed":
            notes.append(f"'{c}': some dates have a time zone and some do not; column left as text.")
            skipped_rows += [row_ref(i) for i in present[present].index]
            continue
        parsed, bad = parse_dates(s, order)
        if bad.any():
            rows = [row_ref(i) for i in bad[bad].index]
            skipped_rows += rows
            if _unparseable_policy(p, c) != "set_missing":
                notes.append(f"'{c}': {int(bad.sum())} value(s) are not valid dates; column left as text (no change).")
                continue
            notes.append(f"'{c}': {int(bad.sum())} invalid date(s) set to missing as you chose.")
        if info.tz_mix == "all":
            notes.append(f"'{c}': time zones were converted to UTC.")
        updates[c] = parsed
        changed += int((present & parsed.notna()).sum())
        examples += _examples(s, parsed, present & parsed.notna(), 2)
    out = df.copy()
    for c, s in updates.items():
        out[c] = s
    return out, OpResult(len(df), len(out), changed, skipped_rows=_cap(skipped_rows), skipped_count=len(skipped_rows),
                         notes=" ".join(notes) or f"{len(updates)} column(s) converted to dates.",
                         examples=examples[:MAX_EXAMPLES], status="applied" if updates else "skipped")


def op_convert_boolean(df, p, cx):
    updates, changed = {}, 0
    for c in [c for c in p.get("columns", []) if c in df.columns]:
        s = df[c]
        present = s.notna()
        low = s.astype(object).where(present, "").astype(str).str.strip().str.lower()
        known = low.isin(BOOL_TRUE | BOOL_FALSE)
        if present.any() and not (known | ~present).all():
            continue
        new = pd.Series(pd.NA, index=s.index, dtype="boolean")
        new[present & low.isin(BOOL_TRUE)] = True
        new[present & low.isin(BOOL_FALSE)] = False
        updates[c] = new
        changed += int(present.sum())
    out = df.copy()
    for c, s in updates.items():
        out[c] = s
    return out, OpResult(len(df), len(out), changed, status="applied" if updates else "skipped")


def _norm_unit(v: Any) -> str:
    return str(v).strip().replace("°", "").upper()


def op_standardize_units(df, p, cx):
    """p: value_col, unit_col, target_unit, conversions {UNIT: [scale, offset]} meaning y = x*scale + offset.
    Optional p['also']: further value columns measured in the same unit column (converted with the same factors)."""
    vc, uc, target = p["value_col"], p["unit_col"], p["target_unit"]
    also = [c for c in (p.get("also") or []) if c in df.columns and c != vc]
    conv = {str(k).upper(): v for k, v in p["conversions"].items()}
    if vc not in df.columns or uc not in df.columns:
        return df.copy(), OpResult(len(df), len(df), notes="Unit or value column not available.", status="skipped")
    if not pd.api.types.is_numeric_dtype(df[vc]):
        return df.copy(), OpResult(len(df), len(df), notes=f"'{vc}' is not numeric yet, so units were not converted.", status="skipped")
    also = [c for c in also if pd.api.types.is_numeric_dtype(df[c])]
    values = df[vc].astype("float64")
    units = df[uc].map(lambda v: _norm_unit(v) if pd.notna(v) else None)
    known = units.isin(conv.keys())
    has_value = values.notna()
    for c in also:
        has_value = has_value | df[c].notna()
    todo = known & values.notna() & (units != target.upper())
    out = df.copy()
    new_vals, new_units = values.copy(), df[uc].astype(object).copy()
    for u, (a, b) in conv.items():
        m = todo & (units == u)
        if m.any():
            new_vals[m] = (values[m] * a + b).round(6)
    for c in also:
        cv = df[c].astype("float64")
        nv = cv.copy()
        for u, (a, b) in conv.items():
            m = known & cv.notna() & (units == u) & (units != target.upper())
            if m.any():
                nv[m] = (cv[m] * a + b).round(6)
        out[c] = nv.astype("float64")
    new_units[known & has_value] = target
    unknown = df[uc].notna() & has_value & ~known
    missing_unit = df[uc].isna() & has_value
    skipped = [row_ref(i) for i in (unknown | missing_unit)[(unknown | missing_unit)].index]
    out[vc] = new_vals.astype("float64")
    out[uc] = new_units
    changed = int(todo.sum())
    note = f"Converted {changed} reading(s) to {target}" + (f" (also {', '.join(also)})." if also else ".")
    if skipped:
        note += f" {len(skipped)} reading(s) have a missing or unrecognised unit and were left unchanged."
    return out, OpResult(len(df), len(out), changed, skipped_rows=_cap(skipped), skipped_count=len(skipped), notes=note,
                         examples=_examples(values, new_vals, todo, 3), status="applied")


def op_fill_constant(df, p, cx):
    updates, changed = {}, 0
    for c, val in (p.get("fills") or {}).items():
        if c not in df.columns:
            continue
        s = df[c]
        mask = s.isna()
        if not mask.any():
            continue
        if pd.api.types.is_numeric_dtype(s):
            num = parse_number(val)
            if num is None:
                continue
            fill = int(num) if str(s.dtype) == "Int64" and num == int(num) else num
        else:
            fill = val
        new = s.copy()
        new[mask] = fill
        updates[c] = new
        changed += int(mask.sum())
    out = df.copy()
    for c, s in updates.items():
        out[c] = s
    return out, OpResult(len(df), len(out), changed, status="applied" if updates else "no_change")


def op_remove_exact_duplicates(df, p, cx):
    key = [c for c in (p.get("key") or []) if c in df.columns]
    if p.get("policy") == "keyed" and not key:
        return df.copy(), OpResult(len(df), len(df), notes="Key columns not available; nothing removed.", status="skipped")
    cmp = df.astype(object).where(df.notna(), None)
    dup = cmp.duplicated(keep="first")
    if p.get("policy") == "keyed":
        dup = dup & df[key].notna().all(axis=1)
    removed = [row_ref(i) for i in dup[dup].index]
    out = df.loc[~dup].copy()
    return out, OpResult(len(df), len(out), 0, removed_rows=_cap(removed), skipped_count=0,
                         notes=f"Removed {len(removed)} repeated copy(ies); the first occurrence of each record was kept.",
                         status="applied" if removed else "no_change")


def _eligible(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def op_impute(df, p, cx):
    """Optional: fill missing values with a column statistic (analytics only; never for identifiers/targets)."""
    updates, changed, notes = {}, 0, []
    for c in _eligible(df, p.get("numeric_columns", [])):
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or not s.isna().any():
            continue
        stat = s.median() if p.get("numeric_strategy", "median") == "median" else s.mean()
        if pd.isna(stat):
            continue
        new = s.copy()
        fill = int(round(stat)) if str(s.dtype) == "Int64" else float(stat)
        new[s.isna()] = fill
        updates[c] = new
        changed += int(s.isna().sum())
        notes.append(f"{c}: {int(s.isna().sum())} filled with {p.get('numeric_strategy', 'median')} {fill:g}")
    for c in _eligible(df, p.get("categorical_columns", [])):
        s = df[c]
        if not s.isna().any():
            continue
        if p.get("categorical_strategy", "mode") == "unknown":
            fill = "Unknown"
        else:
            m = s.dropna().mode()
            if m.empty:
                continue
            fill = m.iloc[0]
        new = s.astype(object).copy()
        new[s.isna()] = fill
        updates[c] = new
        changed += int(s.isna().sum())
        notes.append(f"{c}: {int(s.isna().sum())} filled with '{fill}'")
    out = df.copy()
    for c, s in updates.items():
        out[c] = s
    return out, OpResult(len(df), len(out), changed, notes="; ".join(notes[:10]), status="applied" if updates else "no_change")


def op_drop_empty_columns(df, p, cx):
    cols = [c for c in _eligible(df, p.get("columns", [])) if df[c].isna().all()]
    out = df.drop(columns=cols)
    return out, OpResult(len(df), len(out), 0, columns_dropped=cols, status="applied" if cols else "no_change",
                         notes=f"Dropped {len(cols)} column(s) with no values.")


def op_drop_rows_missing(df, p, cx):
    cols = _eligible(df, p.get("columns", []))
    if not cols:
        return df.copy(), OpResult(len(df), len(df), status="skipped", notes="Columns not available.")
    drop = df[cols].isna().any(axis=1)
    removed = [row_ref(i) for i in drop[drop].index]
    out = df.loc[~drop].copy()
    return out, OpResult(len(df), len(out), 0, removed_rows=_cap(removed),
                         notes=f"Removed {len(removed)} row(s) missing {', '.join(cols)}. Identifiers are never invented.",
                         status="applied" if removed else "no_change")


def op_flag_outliers(df, p, cx):
    k = float(p.get("k", 3.0))
    added, changed = [], 0
    out = df.copy()
    for c in _eligible(df, p.get("columns", [])):
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) or s.nunique() < 5:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if not iqr or pd.isna(iqr):
            continue
        flag = ((s < q1 - k * iqr) | (s > q3 + k * iqr)).fillna(False)
        name = f"{c}__outlier_flag"
        out[name] = flag.astype(bool)
        added.append(name)
        changed += int(flag.sum())
    return out, OpResult(len(df), len(out), changed, columns_added=added, status="applied" if added else "no_change",
                         notes="Flag columns were added. No value was changed or removed.")


def op_treat_outliers(df, p, cx):
    """Detect outliers with the configured method and apply the chosen treatment (never on categorical columns)."""
    from ..preprocessing.outliers import OutlierConfig, apply_treatment, detect
    cfg = OutlierConfig.from_dict(p.get("config") or {})
    treatment = p.get("treatment") or "report"
    report = detect(df, cfg)
    out, summ = apply_treatment(df, report, cfg, treatment)
    removed = [row_ref(i) for i in summ["rows_removed"]]
    changed = int(sum(summ["values_modified"].values()))
    per_col = ", ".join(f"{c}: {n}" for c, n in report.per_column.items() if n) or ("rows scored together" if report.multivariate else "none")
    notes = (f"{report.n_flagged:,} row(s) ({report.pct_flagged}%) flagged by {cfg.method}; per column: {per_col}. "
             f"Treatment: {treatment}." + (" " + " ".join(summ["warnings"]) if summ["warnings"] else ""))
    status = "applied" if (removed or changed or summ["columns_added"]) else "no_change"
    return out, OpResult(len(df), len(out), changed, columns_added=summ["columns_added"], removed_rows=_cap(removed), notes=notes, status=status)


EXECUTORS: dict[str, Executor] = {
    "standardize_nulls": op_standardize_nulls,
    "trim_whitespace": op_trim_whitespace,
    "normalize_case": op_normalize_case,
    "convert_numeric": op_convert_numeric,
    "parse_dates": op_parse_dates,
    "convert_boolean": op_convert_boolean,
    "standardize_units": op_standardize_units,
    "fill_constant": op_fill_constant,
    "remove_exact_duplicates": op_remove_exact_duplicates,
    "impute": op_impute,
    "drop_empty_columns": op_drop_empty_columns,
    "drop_rows_missing": op_drop_rows_missing,
    "flag_outliers": op_flag_outliers,
    "treat_outliers": op_treat_outliers,
}
# Execution order matters: text clean-up first, then typing, then value-level fixes, then removals.
ORDER = ["standardize_nulls", "trim_whitespace", "normalize_case", "convert_numeric", "parse_dates", "convert_boolean",
         "standardize_units", "fill_constant", "remove_exact_duplicates", "drop_rows_missing", "treat_outliers", "impute",
         "drop_empty_columns", "flag_outliers"]


def run_actions(df: pd.DataFrame, actions: list[Action], cx: ExecContext,
                progress: Callable[[float, str], None] | None = None,
                not_approved: list[Action] | None = None) -> tuple[pd.DataFrame, list[LogEntry]]:
    """Execute approved actions in a fixed, safe order. A failing action is logged and skipped."""
    ordered = sorted(actions, key=lambda a: (ORDER.index(a.op) if a.op in ORDER else 99, a.action_id))
    entries: list[LogEntry] = []
    cur = df.copy()
    for n, act in enumerate(ordered, 1):
        if progress:
            progress(n / max(len(ordered), 1), act.title)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fn = EXECUTORS.get(act.op)
        if fn is None:
            res, new = OpResult(len(cur), len(cur), status="failed", notes=f"Unknown operation '{act.op}'."), cur
        else:
            try:
                new, res = fn(cur, act.params, cx)
            except Exception as exc:  # isolate: one broken step must not abort the run
                log.exception("Action %s failed", act.action_id)
                new, res = cur, OpResult(len(cur), len(cur), status="failed", notes=f"Step failed: {type(exc).__name__}: {exc}")
        cur = new
        entries.append(LogEntry(
            step=n, action_id=act.action_id, operation=act.op, title=act.title, columns=list(act.columns),
            reason=act.rationale, parameters=_jsonable(act.params), rows_in=res.rows_in, rows_out=res.rows_out,
            values_changed=res.values_changed, status=res.status, skipped_count=res.skipped_count,
            skipped_rows=res.skipped_rows, removed_rows=res.removed_rows, notes=res.notes, timestamp=stamp))
    for act in (not_approved or []):
        entries.append(LogEntry(
            step=len(entries) + 1, action_id=act.action_id, operation=act.op, title=act.title, columns=list(act.columns),
            reason=act.rationale, parameters=_jsonable(act.params), rows_in=len(cur), rows_out=len(cur), values_changed=0,
            status="not_approved", skipped_count=0, skipped_rows=[], removed_rows=[],
            notes="Not applied (not approved).", timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return cur, entries


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj
