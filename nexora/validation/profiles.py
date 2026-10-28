"""Domain profiles: loading, validation, versioning and overrides.

A profile is plain data (YAML). It declares logical *fields* (with aliases used to
suggest column mappings) and *rules* (a rule type plus parameters). The engine in
rule_engine.py is generic; everything domain-specific lives in these files.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .findings import CATEGORIES, DIMENSIONS, SEVERITIES

# rule type -> (default category, default scoring dimension)
RULE_TYPE_DEFAULTS: dict[str, tuple[str, str | None]] = {
    "required": ("Invalid", "completeness"),
    "unique": ("Invalid", "uniqueness"),
    "duplicate_rows": ("Suspicious", "uniqueness"),
    "numeric": ("Invalid", "validity"),
    "datetime": ("Invalid", "validity"),
    "regex": ("Invalid", "validity"),
    "allowed_values": ("Invalid", "validity"),
    "range": ("Suspicious", None),
    "datetime_range": ("Suspicious", None),
    "compare": ("Inconsistent", "consistency"),
    "date_order": ("Inconsistent", "consistency"),
    "expression_equals": ("Inconsistent", "consistency"),
    "referential": ("Inconsistent", "consistency"),
    "timestamp_gap": ("Suspicious", None),
    "timestamp_order": ("Inconsistent", "consistency"),
    "sequence": ("Inconsistent", "consistency"),
    "monotonic": ("Inconsistent", "consistency"),
    "unit_consistency": ("Inconsistent", "consistency"),
    "freshness": ("Informational", "freshness"),
    "statistical_anomaly": ("Statistical anomaly", None),
}
RULE_TYPES = sorted(RULE_TYPE_DEFAULTS)

FIELD_TYPES = ["identifier", "text", "number", "integer", "datetime", "category", "boolean"]
COVERAGE_LEVELS = ["tested_end_to_end", "basic", "template"]
DEFAULT_NULL_TOKENS = ["", "null", "none", "nan", "n/a", "#n/a"]
SAFE_EXPR_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd,
)


class ProfileError(ValueError):
    """Raised when a profile file or definition is invalid."""


def expression_names(expr: str) -> list[str]:
    """Return variable names used in a safe arithmetic expression (validated)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ProfileError(f"Invalid expression '{expr}': {exc.msg}") from exc
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, SAFE_EXPR_NODES):
            raise ProfileError(f"Expression '{expr}' uses unsupported syntax ({type(node).__name__}). Only + - * / and numbers are allowed.")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ProfileError(f"Expression '{expr}' may only contain numeric constants.")
        if isinstance(node, ast.Name) and node.id not in names:
            names.append(node.id)
    return names


@dataclass
class Rule:
    rule_id: str
    name: str
    rule_type: str
    fields: list[str]
    severity: str
    category: str
    dimension: str | None
    explanation: str
    recommendation: str
    rationale: str
    params: dict[str, Any] = field(default_factory=dict)
    when: list[dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    version: str = "1.0.0"

    def required_fields(self) -> list[str]:
        """Every logical field that must be mapped for this rule to run."""
        out: list[str] = list(self.fields)
        p = self.params
        for key in ("reference_field", "target", "unit_field", "order_by"):
            if p.get(key):
                out.append(p[key])
        out.extend(p.get("group_by", []) or [])
        out.extend(c["field"] for c in self.when)
        if self.rule_type == "expression_equals":
            out.extend(expression_names(p["expression"]))
        seen: list[str] = []
        for f in out:
            if f != "*" and f not in seen:
                seen.append(f)
        return seen

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.rule_id, "name": self.name, "type": self.rule_type, "fields": self.fields,
            "severity": self.severity, "category": self.category, "dimension": self.dimension,
            "explanation": self.explanation, "recommendation": self.recommendation,
            "rationale": self.rationale, "params": self.params, "when": self.when,
            "enabled": self.enabled, "version": self.version,
        }
        return d


@dataclass
class Profile:
    profile_id: str
    name: str
    version: str
    owner: str
    description: str
    coverage: str
    guardrail: str
    settings: dict[str, Any]
    scoring_weights: dict[str, float]
    fields: dict[str, dict[str, Any]]
    rules: list[Rule]
    changelog: list[dict[str, str]] = field(default_factory=list)
    source: str = "builtin"  # builtin | custom

    @property
    def null_tokens(self) -> list[str]:
        return [str(t).lower() for t in self.settings.get("null_tokens", DEFAULT_NULL_TOKENS)]

    @property
    def datetime_formats(self) -> list[str]:
        return list(self.settings.get("datetime_formats", ["ISO8601"]))

    def rule(self, rule_id: str) -> Rule:
        for r in self.rules:
            if r.rule_id == rule_id:
                return r
        raise KeyError(rule_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id, "name": self.name, "version": self.version,
            "owner": self.owner, "description": self.description, "coverage": self.coverage,
            "guardrail": self.guardrail, "settings": self.settings,
            "scoring_weights": self.scoring_weights, "fields": self.fields,
            "rules": [r.to_dict() for r in self.rules], "changelog": self.changelog,
            "source": self.source,
        }

    @property
    def content_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()


# ----------------------------------------------------------------------------
# Parsing and validation
# ----------------------------------------------------------------------------

def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ProfileError(msg)


def _validate_params(rule_id: str, rtype: str, params: dict[str, Any], fields: list[str]) -> None:
    def need(*keys: str, any_of: bool = False) -> None:
        present = [k for k in keys if params.get(k) not in (None, [], {}, "")]
        if any_of:
            _require(bool(present), f"{rule_id}: type '{rtype}' needs one of {list(keys)} in params")
        else:
            missing = [k for k in keys if k not in present]
            _require(not missing, f"{rule_id}: type '{rtype}' is missing params {missing}")

    if rtype == "regex":
        need("pattern")
        pat = str(params["pattern"])
        _require(len(pat) <= 300, f"{rule_id}: regex pattern longer than 300 characters")
        try:
            re.compile(pat)
        except re.error as exc:
            raise ProfileError(f"{rule_id}: invalid regex ({exc})") from exc
    elif rtype == "allowed_values":
        need("values")
    elif rtype == "range":
        need("min", "max", "ranges_by_unit", any_of=True)
        if params.get("ranges_by_unit"):
            need("unit_field")
    elif rtype == "datetime_range":
        need("not_before", "not_after", any_of=True)
    elif rtype == "compare":
        need("op", "kind")
        _require(params["op"] in ("<", "<=", ">", ">=", "==", "!="), f"{rule_id}: bad compare op")
        _require(params["kind"] in ("numeric", "datetime", "text"), f"{rule_id}: compare kind must be numeric, datetime or text")
        if params["kind"] == "text":
            _require(params["op"] in ("==", "!="), f"{rule_id}: text comparison supports only == and !=")
        _require(len(fields) == 2, f"{rule_id}: compare needs exactly 2 fields")
    elif rtype == "date_order":
        _require(len(fields) == 2, f"{rule_id}: date_order needs [start_field, end_field]")
    elif rtype == "expression_equals":
        need("target", "expression")
        expression_names(params["expression"])
    elif rtype == "referential":
        need("reference_field", "reference_values", "reference_table", any_of=True)
        if params.get("reference_table"):
            need("reference_column")
    elif rtype == "timestamp_gap":
        need("max_gap_seconds")
    elif rtype == "freshness":
        need("max_age_hours", "max_age_days", any_of=True)
    elif rtype == "unit_consistency":
        need("expected_units", "group_by", "dataset_wide", any_of=True)
    elif rtype == "statistical_anomaly":
        _require(params.get("method", "iqr") in ("iqr", "robust_z", "isolation_forest"),
                 f"{rule_id}: method must be iqr, robust_z or isolation_forest")
    if rtype in ("sequence", "monotonic", "timestamp_gap", "timestamp_order", "freshness", "datetime", "numeric"):
        _require(len(fields) >= 1, f"{rule_id}: needs at least one field")


def rule_from_dict(raw: dict[str, Any], profile_version: str, known_fields: set[str]) -> Rule:
    for key in ("id", "name", "type", "severity", "explanation", "recommendation"):
        _require(key in raw, f"Rule is missing '{key}': {raw.get('id', raw)}")
    rid = str(raw["id"])
    rtype = raw["type"]
    _require(rtype in RULE_TYPE_DEFAULTS, f"{rid}: unknown rule type '{rtype}'. Known: {RULE_TYPES}")
    _require(raw["severity"] in SEVERITIES, f"{rid}: severity must be one of {SEVERITIES}")
    default_cat, default_dim = RULE_TYPE_DEFAULTS[rtype]
    category = raw.get("category", default_cat)
    _require(category in CATEGORIES, f"{rid}: category must be one of {CATEGORIES}")
    if rtype == "statistical_anomaly":
        category = "Statistical anomaly"  # statistical flags can never be labelled as errors
    else:
        _require(category != "Statistical anomaly", f"{rid}: only statistical_anomaly rules may use that category")
    dimension = raw.get("dimension", default_dim)
    if "dimension" not in raw and default_dim is None:
        dimension = {"Invalid": "validity", "Inconsistent": "consistency"}.get(category)
    if dimension is not None:
        _require(dimension in DIMENSIONS, f"{rid}: dimension must be one of {DIMENSIONS} or null")
    params = dict(raw.get("params") or {})
    fields = list(raw.get("fields") or [])
    if rtype == "expression_equals":
        params_fields = [params.get("target")] + expression_names(params.get("expression", "0")) if params.get("target") else []
        fields = fields or params_fields
    when = list(raw.get("when") or [])
    for cond in when:
        _require("field" in cond and any(k in cond for k in ("in", "not_in", "blank", "not_blank")),
                 f"{rid}: each 'when' condition needs 'field' and one of in/not_in/blank/not_blank")
    _validate_params(rid, rtype, params, fields)
    rule = Rule(
        rule_id=rid, name=str(raw["name"]), rule_type=rtype, fields=fields, severity=raw["severity"],
        category=category, dimension=dimension, explanation=str(raw["explanation"]).strip(),
        recommendation=str(raw["recommendation"]).strip(), rationale=str(raw.get("rationale", "")).strip(),
        params=params, when=when, enabled=bool(raw.get("enabled", True)),
        version=str(raw.get("version", profile_version)),
    )
    unknown = [f for f in rule.required_fields() if f not in known_fields]
    _require(not unknown, f"{rid}: references undeclared field(s) {unknown}. Declare them under 'fields'.")
    return rule


def profile_from_dict(raw: dict[str, Any], source: str = "builtin") -> Profile:
    for key in ("profile_id", "name", "version", "fields", "rules"):
        _require(key in raw, f"Profile is missing '{key}'")
    _require(re.fullmatch(r"[a-z][a-z0-9_]{1,40}", str(raw["profile_id"])) is not None,
             "profile_id must be lowercase letters, digits and underscores")
    fields = {}
    for fname, spec in raw["fields"].items():
        spec = dict(spec or {})
        spec.setdefault("label", fname.replace("_", " ").title())
        spec.setdefault("aliases", [])
        spec.setdefault("type", "text")
        spec.setdefault("required", False)
        _require(spec["type"] in FIELD_TYPES, f"Field {fname}: type must be one of {FIELD_TYPES}")
        fields[fname] = spec
    version = str(raw["version"])
    rules: list[Rule] = []
    seen: set[str] = set()
    for r in raw["rules"]:
        rule = rule_from_dict(r, version, set(fields))
        _require(rule.rule_id not in seen, f"Duplicate rule id {rule.rule_id}")
        seen.add(rule.rule_id)
        rules.append(rule)
    coverage = raw.get("coverage", "template")
    _require(coverage in COVERAGE_LEVELS, f"coverage must be one of {COVERAGE_LEVELS}")
    weights = {d: float(raw.get("scoring_weights", {}).get(d, 1.0)) for d in DIMENSIONS}
    _validate_cleaning((raw.get("settings") or {}).get("cleaning"), set(fields))
    return Profile(
        profile_id=str(raw["profile_id"]), name=str(raw["name"]), version=version,
        owner=str(raw.get("owner", "")), description=str(raw.get("description", "")).strip(),
        coverage=coverage, guardrail=str(raw.get("guardrail", "")).strip(),
        settings=dict(raw.get("settings") or {}), scoring_weights=weights, fields=fields,
        rules=rules, changelog=list(raw.get("changelog") or []), source=source,
    )


def _validate_cleaning(cl: Any, fields: set[str]) -> None:
    """Check the optional settings.cleaning section so a typo fails loudly instead of silently doing nothing."""
    if cl is None:
        return
    _require(isinstance(cl, dict), "settings.cleaning must be a mapping")
    known = {"duplicates", "normalize_case", "units", "missing", "outliers"}
    _require(not (set(cl) - known), f"settings.cleaning: unknown section(s) {sorted(set(cl) - known)}; allowed {sorted(known)}")
    dup = cl.get("duplicates") or {}
    for f in dup.get("key", []):
        _require(f in fields, f"settings.cleaning.duplicates: key field '{f}' is not a declared field")
    for f, mode in (cl.get("normalize_case") or {}).items():
        _require(f in fields, f"settings.cleaning.normalize_case: '{f}' is not a declared field")
        m = mode.get("mode") if isinstance(mode, dict) else mode
        _require(m in ("upper", "lower", "title", None), f"settings.cleaning.normalize_case.{f}: mode must be upper, lower or title")
    for u in cl.get("units") or []:
        for k in ("value_field", "unit_field", "target_unit", "conversions"):
            _require(k in u, f"settings.cleaning.units: each entry needs '{k}'")
        for f in [u["value_field"], u["unit_field"], *u.get("also_fields", [])]:
            _require(f in fields, f"settings.cleaning.units: '{f}' is not a declared field")
        for unit, pair in u["conversions"].items():
            _require(isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(x, (int, float)) for x in pair),
                     f"settings.cleaning.units: conversion for '{unit}' must be [scale, offset]")
        _require(str(u["target_unit"]).upper() in {str(k).upper() for k in u["conversions"]},
                 f"settings.cleaning.units: the target unit '{u['target_unit']}' needs a conversion entry (normally [1, 0])")
    for f, spec in (cl.get("missing") or {}).items():
        _require(f in fields, f"settings.cleaning.missing: '{f}' is not a declared field")
        _require(isinstance(spec, dict) and spec.get("strategy") == "constant" and "value" in spec,
                 f"settings.cleaning.missing.{f}: only {{strategy: constant, value: ...}} is supported")
    pol = (cl.get("outliers") or {}).get("policy", "review")
    _require(pol in ("review", "report_only"), "settings.cleaning.outliers.policy must be review or report_only")


def load_profile_file(path: str | Path) -> Profile:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ProfileError(f"{Path(path).name}: not valid YAML ({exc})") from exc
    _require(isinstance(raw, dict), f"{Path(path).name}: top level must be a mapping")
    try:
        return profile_from_dict(raw)
    except ProfileError as exc:
        raise ProfileError(f"{Path(path).name}: {exc}") from exc


def load_builtin_profiles(directory: str | Path) -> dict[str, Profile]:
    profiles: dict[str, Profile] = {}
    for path in sorted(Path(directory).glob("*.yaml")):
        p = load_profile_file(path)
        _require(p.profile_id not in profiles, f"Duplicate profile_id {p.profile_id}")
        profiles[p.profile_id] = p
    return profiles


# ----------------------------------------------------------------------------
# Overrides (enable/disable rules, change parameters/severity per run)
# ----------------------------------------------------------------------------

def apply_overrides(profile: Profile, overrides: dict[str, dict[str, Any]] | None) -> list[Rule]:
    """Return the effective rule list after user overrides. The profile is not mutated."""
    rules = [copy.deepcopy(r) for r in profile.rules]
    overrides = overrides or {}
    by_id = {r.rule_id: r for r in rules}
    for rid, ov in overrides.items():
        if rid not in by_id:
            raise ProfileError(f"Override refers to unknown rule {rid}")
        rule = by_id[rid]
        if "enabled" in ov:
            rule.enabled = bool(ov["enabled"])
        if "severity" in ov:
            _require(ov["severity"] in SEVERITIES, f"{rid}: bad severity override")
            rule.severity = ov["severity"]
        if ov.get("params"):
            rule.params = {**rule.params, **ov["params"]}
            _validate_params(rid, rule.rule_type, rule.params, rule.fields)
    return rules


def make_custom_profile(profile_id: str, name: str, fields: dict[str, dict[str, Any]], rules: list[dict[str, Any]],
                        description: str = "", owner: str = "user") -> Profile:
    """Build and validate a user-defined profile from simple dictionaries."""
    raw = {
        "profile_id": profile_id, "name": name, "version": "1.0.0", "owner": owner,
        "description": description or "User-defined custom profile.", "coverage": "template",
        "guardrail": "Custom rules are the responsibility of the person who wrote them.",
        "fields": fields, "rules": rules,
    }
    return profile_from_dict(raw, source="custom")
