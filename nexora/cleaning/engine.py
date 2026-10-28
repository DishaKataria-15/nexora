"""Execute an approved preparation plan and produce a verified before/after report."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from ..validation.profiles import Profile
from ..validation.rule_engine import run_validation
from .model import DECISION, OPTIONAL, REVIEW, SAFE, Action, Plan, PreparationResult
from .transformations import ExecContext, run_actions
from .validation import frame_fingerprint, quality_snapshot, reconcile, to_text_frame


def _merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_actions(plan: Plan, approvals: dict[str, bool] | None = None,
                    choices: dict[str, str] | None = None, overrides: dict[str, dict[str, Any]] | None = None) -> tuple[list[Action], list[Action]]:
    """Split plan actions into (to run, not approved), applying the user's decisions.

    Defaults are conservative: safe actions run, optional and decision actions do not unless chosen.
    ``overrides`` merges extra parameters (for example outlier method and thresholds) into chosen actions.
    """
    approvals, choices, overrides = approvals or {}, choices or {}, overrides or {}
    run, skipped = [], []
    for a in plan.actions:
        if a.kind == REVIEW:
            continue
        if a.kind == SAFE:
            (run if approvals.get(a.action_id, True) else skipped).append(a)
        elif a.kind == OPTIONAL:
            (run if approvals.get(a.action_id, False) else skipped).append(a)
        elif a.kind == DECISION:
            key = choices.get(a.action_id, a.default_choice)
            patch = a.choice_params.get(key) if key in a.choice_params else None
            if patch is None:
                skipped.append(a)
            else:
                b = copy.deepcopy(a)
                b.params = _merge(b.params, patch)
                if a.action_id in overrides:
                    b.params = _merge(b.params, overrides[a.action_id])
                run.append(b)
    return run, skipped


def _rule_rows(outcome) -> list[dict[str, Any]]:
    return [{"rule_id": r.rule_id, "name": r.name, "severity": r.severity, "category": r.category, "status": r.status,
             "rows_flagged": r.rows_flagged, "findings": r.findings_total, "message": r.message}
            for r in outcome.rule_results]


def prepare(raw: pd.DataFrame, profile: Profile, mapping: dict[str, str | None], plan: Plan, *,
            approvals: dict[str, bool] | None = None, choices: dict[str, str] | None = None,
            overrides: dict[str, dict[str, Any]] | None = None, reference_tables: dict[str, pd.DataFrame] | None = None, as_of: str | None = None, seed: int = 42,
            progress: Callable[[float, str], None] | None = None) -> PreparationResult:
    """Run the approved actions on a copy of ``raw``. ``raw`` itself is never modified."""
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw_text = raw.reset_index(drop=True)
    hash_before = frame_fingerprint(raw_text)
    tokens = list(profile.null_tokens)
    cx = ExecContext(null_tokens=set(tokens))
    run, skipped = resolve_actions(plan, approvals, choices, overrides)

    if progress:
        progress(0.05, "Checking the raw data")
    kw = dict(dataset_id="prep", run_id="prep", as_of=as_of, reference_tables=reference_tables)
    out_before = run_validation(raw_text, profile, mapping, **kw)
    snap_before = quality_snapshot(raw_text, tokens)
    raw_non_null = {c: int((~raw_text[c].astype(str).str.strip().str.lower().isin(set(tokens))).sum()) for c in raw_text.columns}

    def inner(frac: float, label: str) -> None:
        if progress:
            progress(0.1 + 0.6 * frac, label)

    prepared, log = run_actions(raw_text.astype(object).copy(), run, cx, progress=inner, not_approved=skipped)
    if progress:
        progress(0.75, "Validating the prepared data")
    after_text = to_text_frame(prepared)
    out_after = run_validation(after_text, profile, mapping, **kw)
    snap_after = quality_snapshot(after_text, tokens)
    hash_after = frame_fingerprint(raw_text)
    id_cols = [mapping[f] for f, sp in profile.fields.items() if sp.get("type") == "identifier" and mapping.get(f) in raw_text.columns]
    rec = reconcile(len(raw_text), list(raw_text.columns), prepared, log, hash_before, hash_after, id_cols, raw_non_null)

    unresolved: list[str] = []
    for r in out_after.rule_results:
        if r.status == "failed":
            unresolved.append(f"{r.name}: {r.rows_flagged:,} row(s) flagged ({r.severity}). Nexora does not change these; review them in the source system.")
    for e in log:
        if e.status == "not_approved" and e.operation not in ("flag_outliers", "impute", "drop_empty_columns"):
            unresolved.append(f"Not applied: {e.title}.")
        elif e.status == "applied" and e.skipped_count:
            unresolved.append(f"{e.title}: {e.skipped_count:,} value(s) could not be converted. {e.notes}".strip())
        elif e.status == "failed":
            unresolved.append(f"A step failed and was skipped: {e.title}. {e.notes}")
    if progress:
        progress(1.0, "Done")
    return PreparationResult(
        prepared=prepared, log=log, before=snap_before, after=snap_after, reconciliation=rec,
        findings_before=[f.to_dict() for f in out_before.findings], findings_after=[f.to_dict() for f in out_after.findings],
        rule_summary_before=_rule_rows(out_before), rule_summary_after=_rule_rows(out_after), unresolved=unresolved,
        source_unchanged=hash_before == hash_after, plan=plan,
        approved={"run": [a.action_id for a in run], "not_run": [a.action_id for a in skipped],
                  "choices": dict(choices or {}), "approvals": dict(approvals or {}), "overrides": copy.deepcopy(overrides or {})},
        seed=seed, started_at=started, completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
