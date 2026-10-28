"""Domain-aware cleaning for every built-in profile that declares settings.cleaning.

Each profile gets a synthetic dataset built from its own field list, with planted problems that its cleaning
settings must fix: repeated loads, inconsistent spellings and mixed units. Expected values are computed here from
the conversion factors, independently of the cleaning engine.
"""
import io

import numpy as np
import pandas as pd
import pytest

from nexora.cleaning import DECISION, SAFE
from nexora.config import get_settings
from nexora.validation.profiles import load_builtin_profiles

PROFILES = load_builtin_profiles(get_settings().profiles_dir)
FLAGSHIP = {"retail", "finance", "manufacturing"}
DOMAIN = sorted(p for p, prof in PROFILES.items() if p not in FLAGSHIP and p != "generic" and (prof.settings or {}).get("cleaning"))
N = 60
DUPES = 3


def test_all_fifteen_remaining_profiles_have_domain_cleaning():
    assert len(DOMAIN) == 15
    assert set(PROFILES) - set(DOMAIN) - FLAGSHIP == {"generic"}


def _allowed(prof, field):
    for r in prof.rules:
        if r.rule_type == "allowed_values" and field in r.fields:
            return list(r.params.get("values") or [])
    return None


def _make(prof):
    cl = prof.settings["cleaning"]
    rng = np.random.RandomState(abs(hash(prof.profile_id)) % 2**31)
    df = pd.DataFrame(index=range(N))
    for f, sp in prof.fields.items():
        t = sp.get("type")
        if t == "identifier":
            df[f] = [f"{f[:3].upper()}{i:04d}" for i in range(N)]
        elif t == "datetime":
            df[f] = pd.date_range("2025-01-01", periods=N, freq="D").strftime("%Y-%m-%d")
        elif t == "integer":
            df[f] = np.arange(1, N + 1)
        elif t == "number":
            df[f] = np.round(rng.uniform(10, 50, N), 3)
        elif t == "category":
            vals = _allowed(prof, f) or ["alpha", "beta"]
            df[f] = [vals[i % len(vals)] for i in range(N)]
        else:
            df[f] = [f"t{i % 5}" for i in range(N)]
    expected = {}
    # mixed units: odd rows are written in another convertible unit; the expected result is the even-row style value
    for u in cl.get("units") or []:
        conv = {k: v for k, v in u["conversions"].items()}
        target = u["target_unit"]
        others = [k for k in conv if str(k).upper() != str(target).upper()]
        vfields = [u["value_field"], *u.get("also_fields", [])]
        truth = {vf: df[vf].astype(float).copy() for vf in vfields}
        units = []
        for i in range(N):
            if i % 2:
                k = others[i % len(others)]
                a, b = conv[k]
                for vf in vfields:
                    df.loc[i, vf] = (truth[vf][i] - b) / a
                units.append(k)
            else:
                units.append(target)
        df[u["unit_field"]] = units
        expected[u["unit_field"]] = (target, truth)
    # inconsistent spellings for configured fields
    for f, mode in (cl.get("normalize_case") or {}).items():
        m = mode.get("mode") if isinstance(mode, dict) else mode
        mp = (mode.get("map") or {}) if isinstance(mode, dict) else {}
        flip = {"upper": str.lower, "lower": str.upper, "title": str.lower}[m]
        col = df[f].astype(str).tolist()
        for i in range(0, N, 3):
            col[i] = flip(col[i])
        for j, variant in enumerate(mp):
            col[(1 + 3 * j) % N] = variant.lower()
        df[f] = col
    base = df.copy()
    df = pd.concat([df, df.iloc[[5, 17, 33]]], ignore_index=True)          # repeated loads
    return df, base, expected


@pytest.mark.parametrize("pid", DOMAIN)
def test_domain_cleaning_fixes_planted_problems(ws, pid):
    prof = PROFILES[pid]
    cl = prof.settings["cleaning"]
    df, base, expected = _make(prof)
    h = ws.ingest(f"{pid}.csv", df.to_csv(index=False).encode())
    plan = ws.make_plan(h, pid, "analytics")
    ops = {a.op: a for a in plan.actions}

    # 1. the record key turns identical repeats into a SAFE removal (no decision needed)
    dup = ops["remove_exact_duplicates"]
    assert dup.kind == SAFE and dup.affected_rows == DUPES and dup.params["key"] == cl["duplicates"]["key"]
    res, _ = ws.run_preparation(h, plan, {}, {})
    p = res.prepared
    assert len(p) == N and res.reconciliation["ok"] and res.source_unchanged

    # 2. configured spellings
    for f, mode in (cl.get("normalize_case") or {}).items():
        m = mode.get("mode") if isinstance(mode, dict) else mode
        fn = {"upper": str.upper, "lower": str.lower, "title": str.title}[m]
        vals = p[f].dropna().astype(str)
        mp = (mode.get("map") or {}) if isinstance(mode, dict) else {}
        assert (vals == vals.map(fn)).all(), (f, vals[vals != vals.map(fn)].head())
        assert not set(vals) & set(mp), f                                   # domain variants mapped to the canonical code
        allowed = _allowed(prof, f)
        if allowed:
            assert set(vals) <= set(allowed), (f, set(vals) - set(allowed))

    # 3. units: every reading is in the target unit and has the independently computed value
    for ucol, (target, truth) in expected.items():
        assert (p[ucol] == target).all(), ucol
        for vf, want in truth.items():
            assert np.allclose(p[vf].astype(float).values, want.values, rtol=1e-6, atol=1e-6), vf
        assert any(a.op == "standardize_units" and a.kind == SAFE for a in plan.actions)

    # 4. outlier policy is carried into the (default: change nothing) decision
    pol = (cl.get("outliers") or {}).get("policy")
    if "treat_outliers" in ops:
        a = ops["treat_outliers"]
        assert a.kind == DECISION and a.default_choice == "report"
        assert a.params["policy"] == (pol or "review")
        if pol == "report_only":
            assert "recommends reporting only" in a.rationale


def test_profile_depth_reports_domain_cleaning(ws):
    assert "domain-aware cleaning" in ws.profile_depth(PROFILES["energy"])
    assert ws.profile_depth(PROFILES["generic"]).startswith("Generic")


@pytest.mark.parametrize("bad,msg", [({"units": [{"value_field": "x"}]}, "needs"), ({"duplicates": {"key": ["nope"]}}, "not a declared field"),
                                     ({"outliers": {"policy": "delete"}}, "policy"), ({"surprise": {}}, "unknown section"),
                                     ({"normalize_case": {"a": "shout"}}, "mode")])
def test_cleaning_section_is_validated(bad, msg):
    from nexora.validation.profiles import ProfileError, profile_from_dict
    raw = {"profile_id": "test_profile", "name": "T", "version": "1", "fields": {"a": {"type": "category"}}, "rules": [], "settings": {"cleaning": bad}}
    with pytest.raises(ProfileError, match=msg):
        profile_from_dict(raw)
