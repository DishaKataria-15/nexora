"""Preparation, EDA and experiment reports (HTML + JSON/CSV)."""
from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import pandas as pd

from ..cleaning.model import PreparationResult
from . import html as H


def _delta(a: Any, b: Any) -> str:
    try:
        d = b - a
        return f"{d:+,.4g}" if d else "0"
    except TypeError:
        return ""


def preparation_report_json(res: PreparationResult, dataset_name: str, profile_id: str, profile_version: str, goal: str,
                            mapping: dict[str, Any], dataset_version: str = "") -> dict[str, Any]:
    return {"dataset": dataset_name, "dataset_version": dataset_version, "profile": {"id": profile_id, "version": profile_version}, "goal": goal,
            "mapping": mapping, "started_at": res.started_at, "completed_at": res.completed_at, "seed": res.seed,
            "before": res.before, "after": res.after, "reconciliation": res.reconciliation, "source_unchanged": res.source_unchanged,
            "approved": res.approved, "log": [e.to_dict() for e in res.log], "unresolved": res.unresolved,
            "rules_before": res.rule_summary_before, "rules_after": res.rule_summary_after}


def transformation_log_csv(res: PreparationResult) -> str:
    rows = []
    for e in res.log:
        rows.append({"step": e.step, "action_id": e.action_id, "operation": e.operation, "title": e.title, "columns": "; ".join(e.columns), "status": e.status,
                     "rows_in": e.rows_in, "rows_out": e.rows_out, "values_changed": e.values_changed, "skipped_records": e.skipped_count,
                     "removed_rows": "; ".join(map(str, e.removed_rows[:200])), "skipped_rows": "; ".join(map(str, e.skipped_rows[:200])),
                     "reason": e.reason, "parameters": json.dumps(e.parameters, default=str), "notes": e.notes, "timestamp": e.timestamp})
    return pd.DataFrame(rows).to_csv(index=False)


def preparation_report_html(res: PreparationResult, dataset_name: str, profile_name: str, profile_version: str, goal_label: str) -> str:
    b, a = res.before, res.after
    metrics = [("Rows", b["rows"], a["rows"]), ("Columns", b["columns"], a["columns"]), ("Missing cells", b["missing_cells"], a["missing_cells"]),
               ("Missing %", b["missing_pct"], a["missing_pct"]), ("Repeated identical rows", b["exact_duplicate_rows"], a["exact_duplicate_rows"]),
               ("Columns with format problems", b["columns_with_format_issues"], a["columns_with_format_issues"])]
    t1 = pd.DataFrame([{"Measure": m, "Before": x, "After": y, "Change": _delta(x, y)} for m, x, y in metrics])
    rec = res.reconciliation
    chk = "".join(f"<tr><td>{H.esc(c['check'])}</td><td class='{'ok' if c['ok'] else 'bad'}'>{'Pass' if c['ok'] else 'FAIL'}</td><td>{H.esc(c['detail'])}</td></tr>" for c in rec["checks"])
    steps = pd.DataFrame([{"Step": e.step, "Action": e.title, "Status": e.status.replace("_", " "), "Columns": ", ".join(e.columns[:6]) + ("..." if len(e.columns) > 6 else ""),
                           "Rows in": e.rows_in, "Rows out": e.rows_out, "Values changed": e.values_changed, "Skipped": e.skipped_count, "Notes": e.notes} for e in res.log])
    ra = {r["rule_id"]: r for r in res.rule_summary_after}
    rr = [{"Check": r["name"], "Severity": r["severity"], "Rows flagged before": r["rows_flagged"], "Rows flagged after": ra.get(r["rule_id"], {}).get("rows_flagged", 0)}
          for r in res.rule_summary_before if r["status"] == "failed" or ra.get(r["rule_id"], {}).get("status") == "failed"]
    unresolved = "".join(f"<li>{H.esc(u)}</li>" for u in res.unresolved) or "<li>None.</li>"
    body = (f"<h2>Before and after</h2>{H.table(t1)}"
            f"<h2>Reconciliation <span class='{'ok' if rec['ok'] else 'bad'}'>{'all checks passed' if rec['ok'] else 'ATTENTION: a check failed'}</span></h2>"
            f"<table><thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead><tbody>{chk}</tbody></table>"
            f"<h2>What was done</h2>{H.table(steps)}"
            f"<h2>Domain checks, before and after</h2>{H.table(pd.DataFrame(rr))}"
            f"<h2>Unresolved issues (not changed by Nexora)</h2><ul>{unresolved}</ul>"
            f"<p class='note'>The raw file was never modified. Row numbers refer to the original file (header = row 1). "
            f"The full step log, with removed and skipped row numbers, is available as CSV and JSON.</p>")
    return H.page(f"Preparation report: {dataset_name}", f"Domain profile: {profile_name} v{profile_version} | Goal: {goal_label} | Completed {res.completed_at}", body)


def eda_summary_html(name: str, overview: dict[str, Any], col_summary: pd.DataFrame, observations: list[str], outliers: pd.DataFrame, corr_pairs: pd.DataFrame) -> str:
    ov = pd.DataFrame([{"Rows": overview["rows"], "Columns": overview["columns"], "Missing cells": overview["missing_cells"],
                        "Missing %": overview["missing_pct"], "Repeated rows": overview["duplicate_rows"]}])
    obs = "".join(f"<li>{H.esc(o)}</li>" for o in observations) or "<li>No notable patterns were detected by the automatic checks.</li>"
    body = (f"<h2>Overview</h2>{H.table(ov)}<h2>What stands out</h2><ul>{obs}</ul><h2>Columns</h2>{H.table(col_summary)}"
            f"<h2>Unusual values (review, not errors)</h2>{H.table(outliers)}<h2>Strongest correlations</h2>{H.table(corr_pairs)}"
            "<p class='note'>Correlation shows that two columns move together; it does not show that one causes the other.</p>")
    return H.page(f"Data summary: {name}", "Computed from the dataset as loaded in Nexora", body)


def experiment_report_html(exp: dict[str, Any], dataset_name: str) -> str:
    cfg, parts = exp["config"], []
    task = cfg["task"]
    sp = exp.get("split_info", {})
    rows = [("Task", task), ("Dataset", dataset_name), ("Target", cfg.get("target") or "(none: clustering)"), ("Random seed", cfg["seed"]),
            ("Dataset fingerprint", (exp.get("dataset_fingerprint") or "")[:16]), ("Split strategy", sp.get("effective_strategy", "n/a")),
            ("Split sizes", sp.get("sizes", "n/a")), ("Model selection used", exp.get("selection_basis")), ("Primary metric", cfg.get("primary_metric")),
            ("Class balancing", cfg.get("imbalance")), ("Compute", exp.get("compute_note") or cfg.get("compute")), ("Preprocessing", cfg.get("preprocess")),
            ("Environment", exp.get("environment")), ("Duration (s)", exp.get("seconds"))]
    parts.append("<h2>Setup</h2>" + H.table(pd.DataFrame([{"Setting": k, "Value": json.dumps(v) if isinstance(v, (dict, list)) else v} for k, v in rows])))
    fs = exp.get("feature_set", {})
    parts.append("<h2>Features</h2>" + H.table(pd.DataFrame([{"Type": k, "Columns": ", ".join(fs.get(k, []))} for k in ("numeric", "categorical", "datetime", "text") if fs.get(k)])) +
                 ("<p class='meta'>Not used: " + H.esc("; ".join(f"{k} ({v})" for k, v in fs.get("excluded", {}).items())) + "</p>" if fs.get("excluded") else ""))
    if exp.get("warnings"):
        parts.append("<h2>Warnings</h2><ul>" + "".join(f"<li>{H.esc(w)}</li>" for w in exp["warnings"]) + "</ul>")
    if exp.get("leakage"):
        parts.append("<h2>Leakage and split checks</h2><ul>" + "".join(f"<li>{H.esc(w['message'])}</li>" for w in exp["leakage"]) + "</ul>")
    cand = pd.DataFrame([{"Model": c["name"], "Status": c["status"], f"{cfg.get('primary_metric')} (selection)": c["score"], "Spread": c.get("score_std"),
                          "Seconds": c["seconds"], "Note": c.get("error") or c.get("note", "")} for c in exp["candidates"]])
    parts.append(f"<h2>Model comparison (on {H.esc(exp.get('selection_basis'))})</h2>{H.table(cand)}")
    sel = next((c for c in exp["candidates"] if c["key"] == exp.get("selected_key")), None)
    if sel:
        parts.append(f"<p class='note'>Selected: <b>{H.esc(sel['name'])}</b>, the best {H.esc(cfg.get('primary_metric'))} under this protocol. "
                     "This is a statement about this data and split, not a general ranking of the algorithms.</p>")
    tr = exp.get("test_result")
    if tr:
        title = "Final results (held-out test set, used once)" if task != "clustering" else "Clustering quality (internal measures, all rows)"
        mrows = [{"Metric": k, "Value": v, "Baseline": (exp.get("test_baseline") or {}).get("metrics", {}).get(k)} for k, v in tr["metrics"].items()]
        parts.append(f"<h2>{title}</h2>{H.table(pd.DataFrame(mrows))}")
        if tr.get("unavailable"):
            parts.append("<p class='meta'>Not available: " + H.esc("; ".join(f"{k}: {v}" for k, v in tr["unavailable"].items())) + "</p>")
        if tr.get("confusion_matrix"):
            lab = tr["labels"]
            parts.append("<h3>Confusion matrix (rows = actual, columns = predicted)</h3>" + H.table(pd.DataFrame(tr["confusion_matrix"], index=lab, columns=lab).reset_index().rename(columns={"index": "actual"})))
        if tr.get("per_class") is not None and len(tr.get("per_class")):
            parts.append("<h3>By class</h3>" + H.table(tr["per_class"]))
        if exp.get("cluster_profile"):
            parts.append("<h3>Group profiles (original units)</h3>" + H.table(exp["cluster_profile"]))
        parts.append("<h2>Interpretation and caveats</h2><ul>" + "".join(f"<li>{H.esc(i)}</li>" for i in exp.get("interpretation", [])) + "</ul>")
    elif task != "clustering":
        parts.append("<p class='note'>The test set has not been used yet; no final performance is reported.</p>")
    if exp.get("importance"):
        parts.append("<h2>Feature importance (permutation, computed without the test set)</h2>" + H.table(exp["importance"]))
    if exp.get("prior_test_uses"):
        parts.append(f"<p class='note warn'>Note: {exp['prior_test_uses']} earlier experiment(s) already looked at this same test split. Repeated testing makes test scores optimistic.</p>")
    return H.page(f"Experiment report: {cfg.get('name') or task}", f"Created {exp.get('created_at')} | Status: {exp.get('status')}", "".join(parts))


def zip_bytes(files: dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, c in files.items():
            z.writestr(n, c if isinstance(c, bytes) else c.encode("utf-8"))
    return buf.getvalue()
