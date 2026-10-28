"""Shared vocabulary and data classes: severities, categories, findings, rule results."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

SEVERITIES = ["Critical", "High", "Medium", "Low", "Info"]
SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}  # lower number = more severe

CATEGORIES = ["Invalid", "Inconsistent", "Suspicious", "Statistical anomaly", "Informational"]

REVIEW_STATUSES = ["Open", "Under investigation", "False positive", "Reviewed and valid", "Resolved"]
UNRESOLVED_STATUSES = ("Open", "Under investigation")
# Statuses that need a written reason because they close or dismiss a finding.
REASON_REQUIRED_STATUSES = ("False positive", "Reviewed and valid", "Resolved")

DIMENSIONS = ["completeness", "validity", "uniqueness", "consistency", "freshness"]

RULE_STATUSES = ["passed", "failed", "skipped", "disabled", "error"]


@dataclass
class Finding:
    """One explainable observation about a row (or the whole dataset)."""

    finding_id: str
    run_id: str
    dataset_id: str
    row_index: int | None  # 0-based position in the original dataset; None = dataset level
    row_ref: int | None  # spreadsheet-style row number (header = row 1)
    columns: list[str]
    rule_id: str
    rule_version: str
    rule_name: str
    category: str
    severity: str
    observed_value: str
    expected: str
    explanation: str
    recommendation: str
    status: str = "Open"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleResult:
    """Execution record for one rule in one run (even if it was skipped)."""

    rule_id: str
    name: str
    rule_type: str
    category: str
    severity: str
    version: str
    dimension: str | None
    status: str  # passed | failed | skipped | disabled | error
    rows_checked: int = 0
    cells_checked: int = 0
    rows_flagged: int = 0
    findings_total: int = 0
    findings_stored: int = 0
    message: str = ""
    # Row indices flagged (kept in memory for scoring, never persisted).
    flagged_rows: set[int] = field(default_factory=set, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("flagged_rows", None)
        return d
