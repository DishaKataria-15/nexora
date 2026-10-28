"""Build a short, explainable preparation plan from the raw data, a domain profile and the user's goal.

The planner never changes the raw frame. Safe steps are dry-run on a scratch copy so that every count shown
to the user is computed (not estimated by a formula), and later proposals see the effect of earlier ones.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ..validation.profiler import ColumnProfile, SchemaProfile, profile_dataframe, suggest_mapping
from ..validation.profiles import Profile
from ..validation.rule_engine import run_validation
from .model import DECISION, OPTIONAL, REVIEW, SAFE, Action, Choice, Plan
from .transformations import ExecContext, analyze_dates, detect_decimal_comma, parse_dates, parse_number, row_ref, run_actions
from .validation import quality_snapshot, to_text_frame

ID_NAME = re.compile(r"(^|_|\b)(id|uuid|key|code|zip|postal|phone|ssn|ref)($|_|\b)", re.I)
MAX_CATEGORY_UNIQUE = 50


def cleaning_config(profile: Profile) -> dict[str, Any]:
    return dict((profile.settings or {}).get("cleaning") or {})


def domain_bounds(profile: Profile, mapping: dict[str, str | None], columns: list[str]) -> dict[str, tuple[float | None, float | None]]:
    """Unconditional numeric limits from the profile's range rules, keyed by dataset column."""
    out: dict[str, tuple[float | None, float | None]] = {}
    for r in profile.rules:
        if r.rule_type != "range" or r.when or r.params.get("ranges_by_unit") or not r.enabled:
            continue
        lo, hi = r.params.get("min"), r.params.get("max")
        if lo is None and hi is None:
            continue
        for f in r.fields:
            col = mapping.get(f)
            if col in columns:
                plo, phi = out.get(col, (None, None))
                out[col] = (lo if plo is None else max(plo, lo) if lo is not None else plo, hi if phi is None else min(phi, hi) if hi is not None else phi)
    return out


def default_mapping(columns: list[str], profile: Profile) -> dict[str, str | None]:
    return {s.field_name: s.column for s in suggest_mapping(columns, profile)}


class _Ids:
    def __init__(self) -> None:
        self.n = 0

    def next(self) -> str:
        self.n += 1
        return f"A{self.n:02d}"


def _names(cols: list[str], limit: int = 4) -> str:
    q = [f"'{c}'" for c in cols]
    return ", ".join(q[:limit]) + (f" and {len(q) - limit} more" if len(q) > limit else "")


def _plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _is_identifier(col: str, field_type: str | None, cp: ColumnProfile | None) -> bool:
    if field_type == "identifier":
        return True
    if cp is not None and cp.inferred_type == "text" and any("leading zeros" in i for i in cp.issues):
        return True
    return bool(ID_NAME.search(col)) and (cp is None or cp.unique_pct >= 0.5)


def _numeric_scan(s: pd.Series) -> tuple[float, int, bool]:
    """(share of non-missing values that parse, count that do not, decimal comma)"""
    present = s.dropna().astype(str)
    if present.empty:
        return 0.0, 0, False
    dec = detect_decimal_comma(present)
    ok = present.map(lambda v: parse_number(v, dec) is not None)
    return float(ok.mean()), int((~ok).sum()), dec


def _row_preview(raw_text: pd.DataFrame, i: int, cols: list[str]) -> list[Any]:
    return [raw_text.loc[i, c] for c in cols if c in raw_text.columns]


def _decision_numeric(act_id: str, col: str, n_bad: int, rows: list[int], examples: list[str], n_present: int) -> Action:
    pct = 100 * n_bad / max(n_present, 1)
    return Action(
        action_id=act_id, op="convert_numeric", kind=DECISION, columns=[col], default_enabled=False,
        title=f"'{col}': {_plural(n_bad, 'value')} cannot be read as numbers",
        summary="The rest of the column looks numeric. Choose how to handle the unreadable entries.",
        params={"columns": [col], "on_unparseable": {}},
        rationale="A numeric column cannot hold text. Converting would need those entries to become missing, which discards what was written.",
        expected_impact=f"{n_bad:,} of {n_present:,} values ({pct:.1f}%) are affected. Rows: {', '.join(map(str, rows[:8]))}{' ...' if len(rows) > 8 else ''}.",
        affected_rows=n_bad, affected_values=n_bad,
        alternatives=["Leave the column as text (nothing changes, but it cannot be charted or modelled as a number)",
                      "Fix the values in the source file and upload again"],
        examples=[{"row": r, "before": e, "after": "(missing)"} for r, e in zip(rows[:5], examples[:5])],
        choices=[Choice("leave", "Leave the column as text"), Choice("set_missing", f"Set the {n_bad} unreadable value(s) to missing")],
        choice_params={"leave": None, "set_missing": {"on_unparseable": {col: "set_missing"}}},
        default_choice="leave", reversible=True)


def build_plan(raw: pd.DataFrame, profile: Profile, mapping: dict[str, str | None], goal: str, *,
               dataset_name: str = "dataset", reference_tables: dict[str, pd.DataFrame] | None = None,
               as_of: str | None = None) -> Plan:
    """Analyse ``raw`` (text frame from ingestion) and propose preparation actions."""
    ids = _Ids()
    cfg = cleaning_config(profile)
    tokens = set(profile.null_tokens)
    cx = ExecContext(null_tokens=tokens)
    raw_text = raw.reset_index(drop=True)
    schema: SchemaProfile = profile_dataframe(raw_text, list(tokens))
    cprof = {c.name: c for c in schema.columns}
    field_of = {col: f for f, col in mapping.items() if col in raw_text.columns}
    ftype = {col: profile.fields[f].get("type") for col, f in field_of.items()}
    actions: list[Action] = []
    warnings: list[str] = []
    scratch = raw_text.astype(object).copy()

    def apply_safe(act: Action) -> None:
        nonlocal scratch
        scratch, entries = run_actions(scratch, [act], cx)
        e = entries[0]
        act.affected_values = e.values_changed
        act.affected_rows = max(e.values_changed, len(e.removed_rows))

    # --- 1. placeholders and blanks --------------------------------------------------------------------
    placeholder_cols = []
    for c in raw_text.columns:
        s = raw_text[c].astype(str).str.strip().str.lower()
        if s.isin(tokens).any():
            placeholder_cols.append(c)
    if placeholder_cols:
        a = Action(ids.next(), "standardize_nulls", SAFE, "Treat empty and placeholder entries as missing",
                   "Blank cells and entries like N/A or null are all recorded as 'missing' so they are counted and handled the same way.",
                   columns=placeholder_cols, params={"columns": placeholder_cols},
                   rationale=f"Placeholder words ({', '.join(sorted(t for t in tokens if t)[:6])}) are not real values. Treating them as missing keeps them out of averages and charts.",
                   alternatives=["Keep placeholders as text (they would count as real categories)"],
                   rule_ref="standardize_nulls")
        apply_safe(a)
        a.expected_impact = f"{a.affected_values:,} cell(s) in {len(placeholder_cols)} column(s)."
        actions.append(a)

    # --- 2. whitespace ----------------------------------------------------------------------------------
    ws_cols = [c for c in scratch.columns if scratch[c].dropna().map(lambda v: isinstance(v, str) and v != v.strip()).any()]
    if ws_cols:
        a = Action(ids.next(), "trim_whitespace", SAFE, "Remove extra spaces around values",
                   "Leading and trailing spaces make identical values look different ('A ' vs 'A').",
                   columns=ws_cols, params={"columns": ws_cols},
                   rationale="Spaces at the start or end of a value are almost always accidental and break matching and grouping.",
                   alternatives=["Keep values exactly as typed"], rule_ref="trim_whitespace")
        apply_safe(a)
        a.expected_impact = f"{a.affected_values:,} value(s) in {len(ws_cols)} column(s)."
        a.examples = [{"row": row_ref(i), "before": repr(raw_text.loc[i, ws_cols[0]]), "after": repr(str(raw_text.loc[i, ws_cols[0]]).strip())}
                      for i in raw_text.index[raw_text[ws_cols[0]].astype(str).map(lambda v: v != v.strip())][:3]]
        actions.append(a)

    # --- 3. spelling and case ---------------------------------------------------------------------------
    case_map: dict[str, dict[str, Any]] = {}
    for f, mode in (cfg.get("normalize_case") or {}).items():
        col = mapping.get(f)
        if col in scratch.columns:
            if isinstance(mode, dict):           # {"mode": "upper", "map": {"IN TRANSIT": "IN_TRANSIT"}}: domain spellings
                case_map[col] = {"mode": mode.get("mode"), "map": {str(k): str(v) for k, v in (mode.get("map") or {}).items()}}
            else:
                case_map[col] = {"mode": mode, "map": {}}
    for c in scratch.columns:
        if c in case_map or _is_identifier(c, ftype.get(c), cprof.get(c)):
            continue
        s = scratch[c].dropna()
        if s.empty or not s.map(lambda v: isinstance(v, str)).all():
            continue
        if s.nunique() > MAX_CATEGORY_UNIQUE or s.nunique() > 0.2 * max(len(s), 1):
            continue
        if pd.to_numeric(s.astype(str), errors="coerce").notna().mean() > 0.5:
            continue
        key = s.astype(str).str.lower().str.replace(r"\s+", "", regex=True)
        groups = s.groupby(key).agg(lambda x: x.value_counts().to_dict())
        mp = {}
        for _, counts in groups.items():
            if len(counts) > 1:
                canon = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                for variant in counts:
                    if variant != canon:
                        mp[variant] = canon
        if mp:
            case_map[c] = {"mode": None, "map": mp}
    if case_map:
        cols = list(case_map)
        a = Action(ids.next(), "normalize_case", SAFE, "Make spelling and capitalisation consistent",
                   "The same category written in different ways (e.g. 'sale', 'SALE', 'Sale') is merged into one form.",
                   columns=cols, params={"mappings": case_map},
                   rationale="Variants that differ only by capital letters or spacing are the same value; leaving them split distorts counts. "
                             "Configured fields use the domain's required form; others use the most frequent spelling.",
                   alternatives=["Keep every spelling as-is", "Choose the canonical spelling yourself in the source file"],
                   rule_ref="normalize_case")
        before = scratch.copy()
        apply_safe(a)
        a.expected_impact = f"{a.affected_values:,} value(s) in {len(cols)} column(s)."
        ex = []
        for c in cols:
            diff = before[c].notna() & (before[c] != scratch[c])
            ex += [{"row": row_ref(i), "before": str(before.loc[i, c]), "after": str(scratch.loc[i, c])} for i in diff[diff].index[:2]]
        a.examples = ex[:5]
        if a.affected_values:
            actions.append(a)
        else:
            ids.n -= 1

    # --- 4. numbers -------------------------------------------------------------------------------------
    safe_num: list[str] = []
    for c in scratch.columns:
        ft = ftype.get(c)
        cp = cprof.get(c)
        declared_numeric = ft in ("number", "integer")
        if ft in ("identifier", "datetime", "boolean", "category", "text") and not declared_numeric:
            continue
        s = scratch[c]
        if s.notna().sum() == 0:
            continue
        share, n_bad, _ = _numeric_scan(s)
        looks_numeric = share >= 0.8 and (cp is None or cp.inferred_type not in ("datetime",)) and not (
            ft is None and _is_identifier(c, ft, cp))
        if not (declared_numeric or looks_numeric):
            continue
        if n_bad == 0:
            safe_num.append(c)
        else:
            present = s.dropna().astype(str)
            dec = detect_decimal_comma(present)
            bad = present[~present.map(lambda v: parse_number(v, dec) is not None)]
            actions.append(_decision_numeric(ids.next(), c, n_bad, [row_ref(i) for i in bad.index], list(bad.values), len(present)))
    to_convert = [c for c in safe_num if not pd.api.types.is_numeric_dtype(scratch[c])]
    if to_convert:
        a = Action(ids.next(), "convert_numeric", SAFE, f"Read {_names(to_convert, 3)} as numbers",
                   "These columns are stored as text but every value is a plain number, so they are converted for sorting, charts and models.",
                   columns=to_convert, params={"columns": to_convert},
                   rationale="Numbers stored as text cannot be summed, plotted or modelled. Only columns where every value parses cleanly are converted. "
                             "Thousand separators and currency symbols are removed; ambiguous formats are never guessed.",
                   alternatives=["Keep as text"], rule_ref="convert_numeric")
        before = scratch.copy()
        apply_safe(a)
        a.expected_impact = f"{len(to_convert)} column(s) converted; {a.affected_values:,} value(s)."
        ex = []
        for c in to_convert:
            diff = before[c].notna() & (before[c].astype(str) != scratch[c].astype(str))
            ex += [{"row": row_ref(i), "before": str(before.loc[i, c]), "after": str(scratch.loc[i, c])} for i in diff[diff].index[:2]]
        a.examples = ex[:5]
        actions.append(a)

    # --- 5. dates ---------------------------------------------------------------------------------------
    safe_dates: list[str] = []
    for c in scratch.columns:
        ft, cp = ftype.get(c), cprof.get(c)
        if not (ft == "datetime" or (ft is None and cp is not None and cp.inferred_type == "datetime")):
            continue
        s = scratch[c]
        if pd.api.types.is_datetime64_any_dtype(s) or s.notna().sum() == 0:
            continue
        present = s.dropna()
        info = analyze_dates(present)
        _, bad_mask = parse_dates(s, info.order if info.order in ("dmy", "mdy") else "auto")
        n_bad = int(bad_mask.sum())
        bad_rows = [row_ref(i) for i in bad_mask[bad_mask].index]
        bad_examples = [str(v) for v in s[bad_mask].head(5)]
        if info.tz_mix == "mixed":
            warnings.append(f"'{c}' mixes dates with and without a time zone; it was left as text.")
            continue
        if info.order in ("ambiguous", "conflict"):
            if info.order == "conflict":
                why = "Some values can only be day-first (e.g. 25/12/2026) and others only month-first (e.g. 12/25/2026)."
                choices = [Choice("leave", "Leave the column as text")]
                cp_map: dict[str, dict[str, Any] | None] = {"leave": None}
            else:
                why = f"{info.n_ambiguous:,} value(s) such as 03/04/2026 could be read as day/month or month/day."
                choices = [Choice("dmy", "Day first (31/12/2026)"), Choice("mdy", "Month first (12/31/2026)"), Choice("leave", "Leave the column as text")]
                cp_map = {"leave": None}
                for k in ("dmy", "mdy"):
                    cp_map[k] = {"order": {c: k}, "on_unparseable": {c: "set_missing"}} if n_bad else {"order": {c: k}}
            actions.append(Action(
                ids.next(), "parse_dates", DECISION, f"'{c}': is the date written day-first or month-first?",
                "The order of day and month cannot be determined from the data. Nexora will not guess.",
                columns=[c], params={"columns": [c]}, default_enabled=False,
                rationale=why + " A wrong guess silently swaps dates, so you decide.",
                expected_impact=f"{info.n_ambiguous:,} ambiguous value(s) of {len(present):,}." + (f" {n_bad} value(s) are not valid dates and will become missing if you pick an order." if n_bad else ""),
                affected_rows=info.n_ambiguous, affected_values=len(present),
                alternatives=["Leave as text", "Re-export the source with ISO dates (YYYY-MM-DD)"],
                examples=[{"row": row_ref(i), "before": str(s.loc[i]), "after": "?"}
                          for i in present[present.astype(str).str.fullmatch(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{4}")].index[:4]],
                choices=choices, choice_params=cp_map, default_choice="leave"))
        elif n_bad:
            actions.append(Action(
                ids.next(), "parse_dates", DECISION, f"'{c}': {_plural(n_bad, 'value')} are not valid dates",
                "Most of the column is dates, but some entries cannot be read as a real date (for example 2026-02-30).",
                columns=[c], params={"columns": [c], "order": {c: info.order} if info.order in ("dmy", "mdy") else {}}, default_enabled=False,
                rationale="A date column cannot hold impossible dates. Setting them to missing discards what was written.",
                expected_impact=f"{n_bad:,} value(s). Rows: {', '.join(map(str, bad_rows[:8]))}{' ...' if len(bad_rows) > 8 else ''}.",
                affected_rows=n_bad, affected_values=n_bad,
                alternatives=["Leave as text", "Correct the dates in the source system"],
                examples=[{"row": r, "before": e, "after": "(missing)"} for r, e in zip(bad_rows[:5], bad_examples)],
                choices=[Choice("leave", "Leave the column as text"), Choice("set_missing", f"Set the {n_bad} invalid date(s) to missing")],
                choice_params={"leave": None, "set_missing": {"on_unparseable": {c: "set_missing"}}}, default_choice="leave"))
        else:
            safe_dates.append(c)
    if safe_dates:
        orders = {c: analyze_dates(scratch[c].dropna()).order for c in safe_dates}
        a = Action(ids.next(), "parse_dates", SAFE, f"Standardize dates in {_names(safe_dates, 3)}",
                   "Dates written in different layouts are converted to one real date type (YYYY-MM-DD).",
                   columns=safe_dates, params={"columns": safe_dates, "order": {c: o for c, o in orders.items() if o in ("dmy", "mdy")}},
                   rationale="Real dates can be sorted, compared and plotted over time. Only unambiguous layouts are converted automatically; "
                             "where day/month order is proven by values like 25/12/2026, that order is used.",
                   alternatives=["Keep as text"], rule_ref="parse_dates")
        before = scratch.copy()
        apply_safe(a)
        a.expected_impact = f"{len(safe_dates)} column(s); {a.affected_values:,} value(s) read as dates."
        ex = []
        for c in safe_dates:
            if not pd.api.types.is_datetime64_any_dtype(scratch[c]):
                continue
            diff = before[c].notna() & (before[c].astype(str) != scratch[c].dt.strftime("%Y-%m-%d").astype(object).where(scratch[c].notna(), ""))
            ex += [{"row": row_ref(i), "before": str(before.loc[i, c]), "after": str(scratch.loc[i, c])[:19]} for i in diff[diff].index[:2]]
        a.examples = ex[:5]
        actions.append(a)

    # --- 6. yes/no columns --------------------------------------------------------------------------------
    bool_cols = [c for c in scratch.columns if ftype.get(c) in (None, "boolean") and cprof.get(c) is not None
                 and cprof[c].inferred_type == "boolean" and not pd.api.types.is_numeric_dtype(scratch[c])]
    if bool_cols:
        a = Action(ids.next(), "convert_boolean", SAFE, f"Read {_names(bool_cols, 3)} as yes/no flags",
                   "Columns that only contain yes/no or true/false are converted to a true/false type.",
                   columns=bool_cols, params={"columns": bool_cols}, rationale="Boolean columns are handled correctly by charts and models.",
                   alternatives=["Keep as text"], rule_ref="convert_boolean")
        apply_safe(a)
        a.expected_impact = f"{len(bool_cols)} column(s)."
        actions.append(a)

    # --- 7. units -------------------------------------------------------------------------------------------
    for u in cfg.get("units", []) or []:
        vc, uc = mapping.get(u["value_field"]), mapping.get(u["unit_field"])
        if vc in scratch.columns and uc in scratch.columns:
            a = Action(ids.next(), "standardize_units", SAFE,
                       f"Convert {u.get('label', vc)} to one unit ({u['target_unit']})",
                       f"Readings recorded in different units are converted to {u['target_unit']} so they can be compared.",
                       columns=[vc, uc] + [mapping[f] for f in u.get("also_fields", []) if mapping.get(f) in scratch.columns],
                       params={"value_col": vc, "unit_col": uc, "target_unit": u["target_unit"], "conversions": u["conversions"],
                               "also": [mapping[f] for f in u.get("also_fields", []) if mapping.get(f) in scratch.columns]},
                       rationale=u.get("rationale", "Conversion factors are defined in the domain profile; readings with a missing or unknown unit are left untouched."),
                       alternatives=["Keep original units (comparisons across units would be wrong)"], rule_ref="standardize_units")
            before = scratch.copy()
            apply_safe(a)
            if a.affected_values:
                diff = before[uc].astype(str) != scratch[uc].astype(str)
                a.examples = [{"row": row_ref(i), "before": f"{before.loc[i, vc]} {before.loc[i, uc]}", "after": f"{scratch.loc[i, vc]} {scratch.loc[i, uc]}"}
                              for i in diff[diff].index[:4]]
                a.expected_impact = f"{a.affected_values:,} reading(s) converted."
                actions.append(a)
            else:
                ids.n -= 1

    # --- 8. configured missing-value fills ---------------------------------------------------------------------
    fills = {}
    for f, spec in (cfg.get("missing") or {}).items():
        col = mapping.get(f)
        if col in scratch.columns and spec.get("strategy") == "constant" and scratch[col].isna().any():
            fills[col] = spec["value"]
    if fills:
        a = Action(ids.next(), "fill_constant", SAFE, f"Fill known blanks in {_names(list(fills), 3)}",
                   "In this domain a blank in these fields has a defined meaning, so it is filled with that value.",
                   columns=list(fills), params={"fills": fills},
                   rationale="; ".join(f"{(cfg['missing'][f]).get('rationale', '')}" for f in (cfg.get("missing") or {}) if mapping.get(f) in fills).strip() or "Defined in the domain profile.",
                   alternatives=["Leave blank (arithmetic checks would treat them as missing)"], rule_ref="fill_constant")
        apply_safe(a)
        a.expected_impact = f"{a.affected_values:,} blank(s) filled."
        actions.append(a)

    # --- 9. duplicates ------------------------------------------------------------------------------------------
    dcfg = cfg.get("duplicates") or {}
    key_cols = [mapping.get(f) for f in dcfg.get("key", [])]
    key_ok = bool(key_cols) and all(k in scratch.columns for k in key_cols)
    cmp = scratch.astype(object).where(scratch.notna(), None)
    dup_mask = cmp.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if key_ok:
        keyed_dup = dup_mask & scratch[key_cols].notna().all(axis=1)
        n_keyed = int(keyed_dup.sum())
        if n_keyed:
            a = Action(ids.next(), "remove_exact_duplicates", SAFE, "Remove repeated copies of identical records",
                       f"{n_keyed:,} row(s) are exact copies of an earlier row (same {_names(key_cols, 3)} and every other value).",
                       columns=key_cols, params={"key": key_cols, "policy": "keyed"},
                       rationale="A row identical in every column, including the record key, is the same record loaded twice. "
                                 "Rows that share a key but differ in any value are NOT removed; they are listed for review.",
                       alternatives=["Keep all copies (totals would be double-counted)", "Remove duplicates yourself after checking the source"],
                       reversible=True, rule_ref="remove_exact_duplicates")
            apply_safe(a)
            a.affected_rows = n_keyed
            a.expected_impact = f"{n_keyed:,} row(s) removed; the first occurrence of each is kept."
            a.examples = [{"row": row_ref(i), "before": " | ".join(str(v) for v in _row_preview(raw_text, i, key_cols)), "after": "(removed)"}
                          for i in keyed_dup[keyed_dup].index[:3]]
            actions.append(a)
        near = scratch.loc[~dup_mask & scratch[key_cols].notna().all(axis=1)]
        n_near = int(near.duplicated(subset=key_cols, keep=False).sum())
        if n_near:
            actions.append(Action(ids.next(), "review_near_duplicates", REVIEW, f"{n_near:,} rows share a record key but differ in other values",
                                  "These could be corrections, partial refunds or genuine repeated events, so they are kept and listed for review.",
                                  columns=key_cols, rationale="Similar-looking rows are not deleted. Only identical copies are.",
                                  expected_impact="No change will be made.", affected_rows=n_near, default_enabled=False))
    elif n_dup:
        actions.append(Action(
            ids.next(), "remove_exact_duplicates", DECISION, f"{n_dup:,} row(s) are identical to an earlier row",
            "Without a record key Nexora cannot tell a loading mistake from a genuine repeated event.",
            columns=[], params={"policy": "all"}, default_enabled=False,
            rationale="Two identical rows can be a duplicated load or two real events (for example two identical purchases). You decide.",
            expected_impact=f"{n_dup:,} row(s) would be removed.", affected_rows=n_dup,
            alternatives=["Keep all rows", "Add a record key in a domain profile so repeats can be judged automatically"],
            examples=[{"row": row_ref(i), "before": " | ".join(str(v) for v in _row_preview(raw_text, i, list(raw_text.columns)[:3])), "after": "(removed)"}
                      for i in dup_mask[dup_mask].index[:3]],
            choices=[Choice("keep", "Keep all rows"), Choice("remove", f"Remove the {n_dup:,} repeated row(s)")],
            choice_params={"keep": None, "remove": {}}, default_choice="keep", reversible=True))

    # --- 10. missing required identifiers --------------------------------------------------------------------------
    id_required = [mapping[f] for f, sp in profile.fields.items() if sp.get("type") == "identifier" and sp.get("required") and mapping.get(f) in scratch.columns]
    id_missing_cols = [c for c in id_required if scratch[c].isna().any()]
    if id_missing_cols:
        mask = scratch[id_missing_cols].isna().any(axis=1)
        n = int(mask.sum())
        actions.append(Action(
            ids.next(), "drop_rows_missing", DECISION, f"{n:,} row(s) have no {_names(id_missing_cols, 3)}",
            "Identifiers are never invented. These rows cannot be linked to a real record.",
            columns=id_missing_cols, params={"columns": id_missing_cols}, default_enabled=False,
            rationale="A record without its required identifier cannot be matched, de-duplicated or traced. Removing them is a business decision.",
            expected_impact=f"{n:,} row(s) would be removed. Rows: {', '.join(str(row_ref(i)) for i in mask[mask].index[:8])}.",
            affected_rows=n, alternatives=["Keep the rows and fix them in the source system"],
            choices=[Choice("keep", "Keep the rows"), Choice("drop", f"Remove the {n} row(s)")],
            choice_params={"keep": None, "drop": {}}, default_choice="keep", reversible=True))

    # --- 11. optional actions ----------------------------------------------------------------------------------------
    empty_cols = [c for c in scratch.columns if scratch[c].isna().all()]
    if empty_cols:
        actions.append(Action(ids.next(), "drop_empty_columns", OPTIONAL, f"Drop {len(empty_cols)} column(s) that contain no values",
                              "These columns have nothing in them.", columns=empty_cols, params={"columns": empty_cols}, default_enabled=False,
                              rationale="An empty column carries no information. It is kept by default because the profile may expect it.",
                              expected_impact=f"{len(empty_cols)} column(s) removed: {_names(empty_cols)}.", alternatives=["Keep the columns"]))
    num_cols = [c for c in scratch.columns if pd.api.types.is_numeric_dtype(scratch[c]) and not pd.api.types.is_bool_dtype(scratch[c])]
    if goal == "analytics":
        never = {c for c in scratch.columns if _is_identifier(c, ftype.get(c), cprof.get(c))}
        n_cols_imp = [c for c in num_cols if c not in never and 0 < scratch[c].isna().mean() <= 0.4]
        c_cols = [c for c in scratch.columns if c not in num_cols and c not in never and not pd.api.types.is_datetime64_any_dtype(scratch[c])
                  and 0 < scratch[c].isna().mean() <= 0.4 and scratch[c].nunique() <= MAX_CATEGORY_UNIQUE]
        if n_cols_imp or c_cols:
            total = int(sum(scratch[c].isna().sum() for c in n_cols_imp + c_cols))
            actions.append(Action(ids.next(), "impute", OPTIONAL, f"Fill {total:,} missing value(s) with typical values",
                                  "Numbers get the column median, categories get the most common value. Off by default because it invents values.",
                                  columns=n_cols_imp + c_cols, default_enabled=False,
                                  params={"numeric_columns": n_cols_imp, "numeric_strategy": "median", "categorical_columns": c_cols, "categorical_strategy": "mode"},
                                  rationale="Filled values are estimates, not observations. Identifiers, dates and columns with more than 40% missing are never filled.",
                                  expected_impact=f"{total:,} value(s) in {len(n_cols_imp) + len(c_cols)} column(s).",
                                  alternatives=["Leave blanks (charts and totals ignore them)", "Fill using a model later"]))
    elif goal in ("supervised", "unsupervised"):
        if scratch.isna().any().any():
            warnings.append("Missing values are not filled here. In ML Studio they are filled inside the model pipeline using the training data only, "
                            "which prevents information from the test data leaking into the model.")
    out_cols = [c for c in num_cols if scratch[c].nunique() >= 5 and not _is_identifier(c, ftype.get(c), cprof.get(c))]
    if out_cols:
        from ..preprocessing.outliers import OutlierConfig, detect
        ocfg = cleaning_config(profile).get("outliers") or {}
        bounds = domain_bounds(profile, mapping, out_cols)
        default_cfg = OutlierConfig(method="iqr", columns=out_cols, iqr_k=3.0, bounds=bounds)
        rep = detect(scratch, default_cfg)
        if rep.n_flagged:
            policy = ocfg.get("policy", "review")
            why = ("Unusual values can be errors, rare real events or exactly the records that matter, so nothing changes unless you choose a treatment. "
                   + (ocfg.get("rationale", "") + " " if ocfg.get("rationale") else "")
                   + ("This domain profile recommends reporting only." if policy == "report_only" else ""))
            per = ", ".join(f"{c} ({n})" for c, n in rep.per_column.items() if n)
            choices = [Choice("report", "Report only (no change)"), Choice("flag", "Add a flag column (values unchanged)"),
                       Choice("clip", "Clip to the detection bounds"), Choice("winsorize", "Winsorize to percentile limits"),
                       Choice("median", "Replace extreme values with the median"), Choice("remove", f"Remove flagged rows")]
            actions.append(Action(
                ids.next(), "treat_outliers", DECISION, f"{rep.n_flagged:,} row(s) contain unusual values in {sum(1 for n in rep.per_column.values() if n)} column(s)",
                "Detected with the IQR rule (3 x IQR) by default; you can change the method and thresholds and preview the effect before confirming.",
                columns=[c for c, n in rep.per_column.items() if n] or out_cols,
                params={"config": default_cfg.to_dict(), "treatment": "report", "domain_bounds": {k: list(v) for k, v in bounds.items()}, "policy": policy},
                rationale=why.strip(), expected_impact=f"{rep.n_flagged:,} row(s) ({rep.pct_flagged}%). Per column: {per}.",
                affected_rows=rep.n_flagged, default_enabled=False,
                alternatives=["Review them in Visualize & Dashboard", "Treat them inside the model pipeline (learned on training rows) in ML Studio"],
                choices=choices, choice_params={"report": None, **{k.key: {"treatment": k.key} for k in choices if k.key != "report"}},
                default_choice="report", rule_ref="treat_outliers"))

    # --- 12. what will remain (validation on the scratch result) ---------------------------------------------------------
    after_text = to_text_frame(scratch)
    kw = dict(dataset_id="plan", run_id="plan", as_of=as_of, reference_tables=reference_tables)
    out_before = run_validation(raw_text, profile, mapping, **kw)
    out_after = run_validation(after_text, profile, mapping, **kw)
    fixed = {r.rule_id for r in out_before.rule_results if r.status == "failed"} - {r.rule_id for r in out_after.rule_results if r.status == "failed"}
    remaining = [r for r in out_after.rule_results if r.status == "failed"]
    sev_order = ["Critical", "High", "Medium", "Low", "Info"]
    for r in sorted(remaining, key=lambda r: (sev_order.index(r.severity) if r.severity in sev_order else 9, r.rule_id)):
        ex_rows = sorted({f.row_ref for f in out_after.findings if f.rule_id == r.rule_id and f.row_ref})[:6]
        sample = next((f for f in out_after.findings if f.rule_id == r.rule_id), None)
        actions.append(Action(
            ids.next(), "review_rule", REVIEW, f"{r.name}: {r.rows_flagged:,} row(s)" if r.rows_flagged else f"{r.name}",
            sample.explanation if sample else r.message, columns=sample.columns if sample else [], params={"rule_id": r.rule_id},
            rationale=sample.recommendation if sample else "",
            expected_impact="Nexora will not change these. Rows: " + ", ".join(map(str, ex_rows)) + (" ..." if r.rows_flagged > 6 else ""),
            affected_rows=r.rows_flagged, default_enabled=False, rule_ref=f"{r.rule_id} ({r.severity}, {r.category})"))

    snap = quality_snapshot(raw_text, list(tokens))
    kind_counts = {k: sum(1 for a in actions if a.kind == k) for k in (SAFE, OPTIONAL, DECISION, REVIEW)}
    summary = {
        **{k: snap[k] for k in ("rows", "columns", "missing_cells", "missing_pct", "exact_duplicate_rows", "columns_with_format_issues")},
        "issues_found_before": sum(1 for r in out_before.rule_results if r.status == "failed"),
        "issues_expected_after": len(remaining), "issues_expected_fixed": len(fixed),
        "n_safe": kind_counts[SAFE], "n_decisions": kind_counts[DECISION] + kind_counts[OPTIONAL], "n_review": kind_counts[REVIEW],
        "candidate_ids": schema.candidate_ids, "unmapped_required": [f for f, sp in profile.fields.items() if sp.get("required") and not mapping.get(f)],
    }
    if summary["unmapped_required"]:
        warnings.append("Some fields the domain profile expects are not mapped to a column, so rules that depend on them cannot run: "
                        + ", ".join(summary["unmapped_required"]) + ".")
    return Plan(dataset_name=dataset_name, profile_id=profile.profile_id, profile_version=profile.version, goal=goal,
                mapping=dict(mapping), actions=actions, summary=summary, warnings=warnings, preview=scratch)
