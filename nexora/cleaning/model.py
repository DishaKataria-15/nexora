"""Data classes shared by the preparation planner, executor and reports."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

GOALS = {
    "analytics": "Analytics and dashboards",
    "supervised": "Predict a value or category (supervised ML)",
    "unsupervised": "Discover groups (unsupervised ML)",
    "explore": "Explore and visualize only",
}

# Action kinds
SAFE = "safe"          # deterministic, meaning-preserving; runs after the single confirmation
OPTIONAL = "optional"  # changes data meaning or removes information; off by default
DECISION = "decision"  # the data is ambiguous; the user must choose how to proceed
REVIEW = "review"      # reported only; Nexora never changes these


@dataclass
class Choice:
    key: str
    label: str


@dataclass
class Action:
    action_id: str
    op: str
    kind: str
    title: str                     # plain-language name shown by default
    summary: str                   # one sentence
    columns: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_impact: str = ""
    affected_rows: int = 0
    affected_values: int = 0
    alternatives: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    choices: list[Choice] = field(default_factory=list)
    # choice key -> params patch to merge in; None means "do not run this action"
    choice_params: dict[str, dict[str, Any] | None] = field(default_factory=dict)
    default_choice: str | None = None
    default_enabled: bool = True
    reversible: bool = True
    rule_ref: str = ""             # technical reference, shown only under "View details"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OpResult:
    rows_in: int
    rows_out: int
    values_changed: int = 0
    columns_added: list[str] = field(default_factory=list)
    columns_dropped: list[str] = field(default_factory=list)
    skipped_rows: list[int] = field(default_factory=list)   # spreadsheet-style row numbers
    skipped_count: int = 0
    removed_rows: list[int] = field(default_factory=list)   # spreadsheet-style row numbers
    notes: str = ""
    examples: list[dict[str, Any]] = field(default_factory=list)
    status: str = "applied"        # applied | skipped | failed | not_approved | no_change


@dataclass
class LogEntry:
    step: int
    action_id: str
    operation: str
    title: str
    columns: list[str]
    reason: str
    parameters: dict[str, Any]
    rows_in: int
    rows_out: int
    values_changed: int
    status: str
    skipped_count: int
    skipped_rows: list[int]
    removed_rows: list[int]
    notes: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Plan:
    dataset_name: str
    profile_id: str
    profile_version: str
    goal: str
    mapping: dict[str, str | None]
    actions: list[Action]
    summary: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    preview: Any = field(default=None, repr=False)   # typed copy after the safe steps; used for live previews, never exported

    def by_kind(self, kind: str) -> list[Action]:
        return [a for a in self.actions if a.kind == kind]

    def action(self, op: str) -> Action | None:
        return next((a for a in self.actions if a.op == op), None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("preview", None)
        return d


@dataclass
class PreparationResult:
    prepared: Any                      # typed pandas DataFrame (index = original 0-based row position)
    log: list[LogEntry]
    before: dict[str, Any]
    after: dict[str, Any]
    reconciliation: dict[str, Any]
    findings_before: list[dict[str, Any]]
    findings_after: list[dict[str, Any]]
    rule_summary_before: list[dict[str, Any]]
    rule_summary_after: list[dict[str, Any]]
    unresolved: list[str]
    source_unchanged: bool
    plan: Plan | None = None
    approved: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    started_at: str = ""
    completed_at: str = ""
