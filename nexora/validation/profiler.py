"""Schema profiler: describes a dataset and suggests column mappings.

Everything here is a *suggestion*. Inferred types, units and mappings are shown
to the user for confirmation and are never applied silently.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .profiles import DEFAULT_NULL_TOKENS, Profile

BOOL_VALUES = {"true", "false", "yes", "no", "y", "n", "t", "f"}
ID_HINT = re.compile(r"(^|_|\b)(id|uuid|key|code|ref|no|number|num)($|_|\b)", re.I)
UNIT_SUFFIX = re.compile(r"^\s*[-+]?\d+([.,]\d+)?\s*[A-Za-z%°/]+\s*$")


@dataclass
class ColumnProfile:
    name: str
    inferred_type: str
    missing_count: int
    missing_pct: float
    unique_count: int
    unique_pct: float
    samples: list[str]
    stats: dict[str, float] = field(default_factory=dict)
    top_values: dict[str, int] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


@dataclass
class SchemaProfile:
    n_rows: int
    n_cols: int
    duplicate_rows: int
    columns: list[ColumnProfile]
    candidate_ids: list[str]
    candidate_dates: list[str]
    candidate_numeric: list[str]

    def column(self, name: str) -> ColumnProfile:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(name)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "column": c.name, "inferred type": c.inferred_type, "missing": c.missing_count,
            "missing %": round(c.missing_pct * 100, 1), "unique": c.unique_count,
            "unique %": round(c.unique_pct * 100, 1), "examples": ", ".join(c.samples[:3]),
            "notes": "; ".join(c.issues),
        } for c in self.columns])


def _shape(value: str) -> str:
    return re.sub(r"[A-Za-z]", "a", re.sub(r"\d", "9", value))


def profile_dataframe(df: pd.DataFrame, null_tokens: list[str] | None = None) -> SchemaProfile:
    """Profile a text DataFrame (as produced by ingestion)."""
    tokens = set(null_tokens or DEFAULT_NULL_TOKENS)
    n = len(df)
    cols: list[ColumnProfile] = []
    for name in df.columns:
        s = df[name].astype(str)
        blank = s.str.strip().str.lower().isin(tokens)
        vals = s[~blank].str.strip()
        non_null = len(vals)
        uniq = int(vals.nunique())
        issues: list[str] = []
        stats: dict[str, float] = {}
        itype = "empty" if non_null == 0 else "text"
        if non_null:
            low = vals.str.lower()
            num = pd.to_numeric(vals, errors="coerce")
            num_rate = float(num.notna().mean())
            iso = pd.to_datetime(vals[vals.str.contains(r"[-/:]", regex=True)], format="ISO8601", errors="coerce", utc=True)
            iso_rate = float(iso.notna().sum() / non_null)
            if set(low.unique()) <= BOOL_VALUES:
                itype = "boolean"
            elif num_rate >= 0.98 and (vals.str.fullmatch(r"[+-]?\d+").sum() == non_null):
                lead_zero = int(vals.str.fullmatch(r"0\d+").sum())
                if lead_zero:
                    itype = "text"
                    issues.append(f"{lead_zero} value(s) have leading zeros; treated as text/identifier, not a number")
                else:
                    itype = "integer"
            elif num_rate >= 0.98:
                itype = "float"
            elif num_rate >= 0.8:
                itype = "float"
                issues.append(f"{int((~num.notna()).sum())} value(s) are not numeric (mixed content)")
            elif iso_rate >= 0.9:
                itype = "datetime"
            elif iso_rate >= 0.3:
                itype = "datetime"
                issues.append(f"Only {iso_rate * 100:.0f}% of values are ISO-format dates; others may use other formats")
            else:
                mixed = pd.to_datetime(vals[vals.str.contains(r"[-/]", regex=True)].head(500), format="mixed",
                                       errors="coerce", dayfirst=False)
                cand = vals.str.contains(r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}", regex=True)
                if cand.mean() >= 0.8 and mixed.notna().mean() >= 0.8:
                    itype = "datetime"
                    issues.append("Dates are not in ISO format (YYYY-MM-DD); day/month order is ambiguous, verify before use")
                elif uniq <= max(20, int(0.05 * non_null)):
                    itype = "category"
            if itype in ("integer", "float"):
                nn = num.dropna()
                stats = {"min": float(nn.min()), "max": float(nn.max()), "mean": float(nn.mean()),
                         "median": float(nn.median()), "std": float(nn.std()) if len(nn) > 1 else 0.0}
            # conservative format/unit hints
            ws = int((s[~blank] != vals).sum())
            if ws:
                issues.append(f"{ws} value(s) have leading/trailing spaces")
            if itype in ("text", "category") and num_rate < 0.5:
                unit_like = float(vals.map(lambda v: bool(UNIT_SUFFIX.match(v))).mean())
                if unit_like >= 0.5:
                    issues.append("Values look like numbers with units attached (for example '12 kg'); check unit consistency")
            if itype in ("float",) and vals.str.contains(r"^\d+,\d{1,2}$", regex=True).mean() > 0.2:
                issues.append("Some values use a decimal comma")
            if itype in ("text", "category") and uniq > 1:
                groups = vals.groupby(low.str.replace(r"\s+", "", regex=True)).nunique()
                variants = int((groups > 1).sum())
                if variants:
                    issues.append(f"{variants} value(s) appear in more than one spelling/case")
            if itype == "datetime":
                shapes = vals.map(_shape).value_counts(normalize=True)
                if (shapes >= 0.02).sum() > 1:
                    issues.append("More than one date/time layout is present")
        top = vals.value_counts().head(5)
        cols.append(ColumnProfile(
            name=name, inferred_type=itype, missing_count=int(blank.sum()),
            missing_pct=float(blank.mean()) if n else 0.0, unique_count=uniq,
            unique_pct=float(uniq / non_null) if non_null else 0.0, samples=list(vals.head(5)),
            stats=stats, top_values={str(k): int(v) for k, v in top.items()}, issues=issues))
    dup = int(df.astype(str).duplicated(keep=False).sum()) if n else 0
    cand_ids = [c.name for c in cols if c.missing_count == 0 and c.unique_pct >= 0.98 and c.inferred_type not in ("float", "boolean")
                and (ID_HINT.search(c.name) or c.inferred_type in ("integer", "text")) and n > 1]
    return SchemaProfile(
        n_rows=n, n_cols=len(cols), duplicate_rows=dup, columns=cols, candidate_ids=cand_ids,
        candidate_dates=[c.name for c in cols if c.inferred_type == "datetime"],
        candidate_numeric=[c.name for c in cols if c.inferred_type in ("integer", "float")])


# -----------------------------------------------------------------------------
# Mapping suggestions
# -----------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@dataclass
class MappingSuggestion:
    field_name: str
    column: str | None
    confidence: str  # "exact", "alias", "fuzzy", "none"


def suggest_mapping(columns: list[str], profile: Profile) -> list[MappingSuggestion]:
    """Suggest which dataset column matches each logical profile field.

    Exact/alias matches are trusted; fuzzy matches are labelled so the user checks them.
    """
    by_norm: dict[str, str] = {}
    for c in columns:
        by_norm.setdefault(_norm(c), c)
    used: set[str] = set()
    out: dict[str, MappingSuggestion] = {}
    for fname, spec in profile.fields.items():  # pass 1: exact name, then aliases
        cand = [(_norm(fname), "exact")] + [(_norm(a), "alias") for a in spec.get("aliases", [])]
        for key, conf in cand:
            col = by_norm.get(key)
            if col and col not in used:
                out[fname] = MappingSuggestion(fname, col, conf)
                used.add(col)
                break
    for fname, spec in profile.fields.items():  # pass 2: fuzzy for the rest
        if fname in out:
            continue
        pool = {k: v for k, v in by_norm.items() if v not in used}
        keys = [_norm(fname)] + [_norm(a) for a in spec.get("aliases", [])]
        best, best_ratio = None, 0.0
        for k in keys:
            for m in difflib.get_close_matches(k, list(pool), n=1, cutoff=0.88):
                ratio = difflib.SequenceMatcher(None, k, m).ratio()
                if ratio > best_ratio:
                    best, best_ratio = pool[m], ratio
        if best:
            out[fname] = MappingSuggestion(fname, best, "fuzzy")
            used.add(best)
        else:
            out[fname] = MappingSuggestion(fname, None, "none")
    return [out[f] for f in profile.fields]


def mapping_warnings(mapping: dict[str, str | None], schema: SchemaProfile, profile: Profile) -> list[str]:
    """Plain-language warnings when a mapped column does not look like the expected field type."""
    warns: list[str] = []
    seen: dict[str, str] = {}
    for fname, col in mapping.items():
        if not col:
            continue
        if col in seen:
            warns.append(f"Column '{col}' is mapped to both '{seen[col]}' and '{fname}'.")
        seen[col] = fname
        spec = profile.fields.get(fname, {})
        try:
            cp = schema.column(col)
        except KeyError:
            warns.append(f"Column '{col}' does not exist in the dataset.")
            continue
        want = spec.get("type")
        if want in ("number", "integer") and cp.inferred_type not in ("integer", "float", "empty"):
            warns.append(f"'{fname}' expects a number but '{col}' looks like {cp.inferred_type}.")
        if want == "datetime" and cp.inferred_type not in ("datetime", "empty"):
            warns.append(f"'{fname}' expects a date/time but '{col}' looks like {cp.inferred_type}.")
    return warns
