"""Shared test helpers."""
from __future__ import annotations

import pandas as pd

from nexora.validation.profiles import profile_from_dict
from nexora.validation.rule_engine import run_validation

AS_OF = "2026-10-19T00:00:00+00:00"
DEFAULT_FIELDS = {n: {} for n in ["id", "a", "b", "c", "ts", "grp", "unit", "d1", "d2"]}


def prof(rules, fields=None, settings=None, weights=None):
    full = []
    for r in rules:
        r = dict(r)
        r.setdefault("name", r["id"])
        r.setdefault("severity", "Medium")
        r.setdefault("explanation", "explanation")
        r.setdefault("recommendation", "recommendation")
        full.append(r)
    raw = {"profile_id": "test", "name": "T", "version": "1.0.0", "fields": fields or DEFAULT_FIELDS, "rules": full,
           "settings": settings or {}}
    if weights:
        raw["scoring_weights"] = weights
    return profile_from_dict(raw)


def run(df, profile, mapping=None, **kw):
    mapping = mapping if mapping is not None else {f: f for f in profile.fields}
    kw.setdefault("as_of", AS_OF)
    return run_validation(df, profile, mapping, dataset_id="D", run_id="R", **kw)


def rows_of(out, rule_id):
    return sorted(f.row_ref for f in out.findings if f.rule_id == rule_id and f.row_ref is not None)


def result_of(out, rule_id):
    return next(r for r in out.rule_results if r.rule_id == rule_id)


def frame(**cols):
    return pd.DataFrame({k: [str(x) for x in v] for k, v in cols.items()})


def identity_mapping(df, profile):
    return {f: (f if f in df.columns else None) for f in profile.fields}
