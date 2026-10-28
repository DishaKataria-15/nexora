"""Quality metrics.

Metrics are reported *separately*. A composite score is optional, clearly
labelled, and must not be used to compare unrelated domains.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .findings import CATEGORIES, DIMENSIONS, SEVERITIES, UNRESOLVED_STATUSES, Finding, RuleResult

FORMULAS = {
    "completeness": "1 - (blank cells in rule-checked required fields / cells checked by 'required' rules)",
    "validity": "1 - (rows with at least one Invalid-type finding in validity rules / all rows)",
    "uniqueness": "1 - (rows involved in a duplicate finding / all rows)",
    "consistency": "1 - (rows with at least one consistency finding / all rows)",
    "freshness": "Share of freshness rules that passed (1 = fresh, 0 = stale)",
}

COMPOSITE_EXPLANATION = (
    "Composite = weighted average of the dimension scores that could be measured "
    "(weights come from the profile; unmeasured dimensions are left out). It ignores severity, "
    "suspicious values and statistical anomalies, and rows that a rule did not apply to count as passing. "
    "Always read the unresolved high-severity count next to it. Do not compare composite scores between domains."
)


def compute_quality_summary(results: list[RuleResult], findings: list[Finding], n_rows: int,
                            weights: dict[str, float]) -> dict[str, Any]:
    dims: dict[str, float | None] = {}
    for dim in DIMENSIONS:
        rrs = [r for r in results if r.dimension == dim and r.status in ("passed", "failed")]
        score: float | None = None
        if rrs:
            if dim == "completeness":
                cells = sum(r.cells_checked for r in rrs)
                bad = sum(r.findings_total for r in rrs)
                score = (1 - bad / cells) if cells else None
            elif dim == "freshness":
                score = sum(r.status == "passed" for r in rrs) / len(rrs)
            else:
                rows: set[int] = set()
                for r in rrs:
                    rows |= r.flagged_rows
                score = (1 - len(rows) / n_rows) if n_rows else None
        dims[dim] = None if score is None else round(float(score), 4)
    measured = {d: s for d, s in dims.items() if s is not None}
    wsum = sum(weights.get(d, 1.0) for d in measured)
    composite = None
    if measured and wsum > 0:
        composite = round(sum(weights.get(d, 1.0) * s for d, s in measured.items()) / wsum, 4)
    status_counts = {s: sum(r.status == s for r in results) for s in ("passed", "failed", "skipped", "disabled", "error")}
    by_sev = {s: sum(f.severity == s for f in findings) for s in SEVERITIES}
    by_cat = {c: sum(f.category == c for f in findings) for c in CATEGORIES}
    return {
        "n_rows": n_rows,
        "dimensions": dims,
        "dimension_formulas": FORMULAS,
        "composite": composite,
        "composite_explanation": COMPOSITE_EXPLANATION,
        "weights_used": {d: weights.get(d, 1.0) for d in measured},
        "rule_status_counts": status_counts,
        "findings_total": len(findings),
        "findings_by_severity": by_sev,
        "findings_by_category": by_cat,
        "unresolved_high_severity": sum(f.severity in ("Critical", "High") and f.status in UNRESOLVED_STATUSES
                                        for f in findings),
    }


def live_counts(findings_df: pd.DataFrame) -> dict[str, Any]:
    """Counts recomputed from stored findings (reflect review decisions made after the run)."""
    if findings_df.empty:
        return {"total": 0, "unresolved": 0, "unresolved_high_severity": 0, "by_status": {}}
    unresolved = findings_df["status"].isin(UNRESOLVED_STATUSES)
    high = findings_df["severity"].isin(["Critical", "High"])
    return {
        "total": int(len(findings_df)),
        "unresolved": int(unresolved.sum()),
        "unresolved_high_severity": int((unresolved & high).sum()),
        "by_status": findings_df["status"].value_counts().to_dict(),
    }
