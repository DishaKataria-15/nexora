"""Generic engine behaviour, independent of any industry profile."""
import copy

import pandas as pd
import pytest

from nexora.validation import rule_engine
from nexora.validation.profiles import ProfileError, profile_from_dict
from tests.helpers import DEFAULT_FIELDS, frame, prof, result_of, rows_of, run


def test_required_treats_blank_and_null_tokens_as_missing():
    p = prof([{"id": "R", "type": "required", "fields": ["a"]}])
    out = run(frame(a=["x", "", "  ", "N/A", "null", "ok"]), p)
    assert rows_of(out, "R") == [3, 4, 5, 6]


def test_every_finding_has_the_required_fields():
    p = prof([{"id": "R", "type": "required", "fields": ["a"], "severity": "High"}])
    f = run(frame(a=["x", ""]), p).findings[0]
    for attr in ("finding_id", "run_id", "dataset_id", "row_index", "row_ref", "columns", "rule_id", "rule_version",
                 "category", "severity", "observed_value", "expected", "explanation", "recommendation", "status"):
        assert getattr(f, attr) not in (None, ""), attr
    assert f.status == "Open" and f.row_ref == f.row_index + 2 and f.rule_version == "1.0.0"


def test_missing_column_skips_rule_with_clear_reason_and_schema_finding():
    fields = {"a": {"required": True}, "b": {}}
    p = prof([{"id": "R", "type": "required", "fields": ["a"]}, {"id": "N", "type": "numeric", "fields": ["b"]}], fields)
    out = run(frame(a=["1"]), p, mapping={"a": "a", "b": None})
    assert result_of(out, "N").status == "skipped" and "b" in result_of(out, "N").message
    out = run(frame(z=["1"]), p, mapping={"a": None, "b": None})
    assert result_of(out, "R").status == "skipped"
    assert [f.rule_id for f in out.findings] == ["SCHEMA-001"]


def test_per_field_rules_run_on_mapped_subset_and_report_the_rest():
    p = prof([{"id": "N", "type": "numeric", "fields": ["a", "b"]}])
    out = run(frame(a=["1", "x"]), p, mapping={"a": "a", "b": None})
    r = result_of(out, "N")
    assert r.status == "failed" and "b" in r.message and rows_of(out, "N") == [3]


def test_disabled_rule_does_not_run_and_is_reported():
    p = prof([{"id": "R", "type": "required", "fields": ["a"], "enabled": False}])
    out = run(frame(a=[""]), p)
    assert result_of(out, "R").status == "disabled" and out.findings == []
    out = run(frame(a=[""]), p, rule_overrides={"R": {"enabled": True}})
    assert len(out.findings) == 1


def test_overrides_change_severity_and_parameters_without_mutating_profile():
    p = prof([{"id": "G", "type": "range", "fields": ["a"], "params": {"max": 10}}])
    before = copy.deepcopy(p.to_dict())
    out = run(frame(a=["5", "20"]), p, rule_overrides={"G": {"params": {"max": 100}, "severity": "High"}})
    assert out.findings == []
    out = run(frame(a=["5", "20"]), p, rule_overrides={"G": {"severity": "High"}})
    assert out.findings[0].severity == "High"
    assert p.to_dict() == before
    with pytest.raises(ProfileError):
        run(frame(a=["1"]), p, rule_overrides={"NOPE": {"enabled": False}})


def test_unique_ignores_blanks_and_supports_composite_keys():
    p = prof([{"id": "U", "type": "unique", "fields": ["a", "b"]}])
    out = run(frame(a=["1", "1", "1", "", ""], b=["x", "x", "y", "z", "z"]), p)
    assert rows_of(out, "U") == [2, 3]
    assert "2 times" in out.findings[0].explanation


def test_duplicate_rows_all_columns():
    p = prof([{"id": "D", "type": "duplicate_rows", "fields": ["*"]}])
    out = run(frame(a=["1", "1", "2"], b=["x", "x", "x"]), p)
    assert rows_of(out, "D") == [2, 3]


def test_numeric_and_thousands_option():
    p = prof([{"id": "N", "type": "numeric", "fields": ["a"]}])
    assert rows_of(run(frame(a=["1", "1,000", "1e3", "x", "inf"]), p), "N") == [3, 5, 6]
    p2 = prof([{"id": "N", "type": "numeric", "fields": ["a"], "params": {"allow_thousands": True}}])
    assert rows_of(run(frame(a=["1", "1,000"]), p2), "N") == []


def test_datetime_formats_are_explicit():
    p = prof([{"id": "T", "type": "datetime", "fields": ["a"]}])
    assert rows_of(run(frame(a=["2026-01-31", "31/01/2026", "2026-13-01", "2026-01-31T10:00:00+02:00"]), p), "T") == [3, 4]
    p2 = prof([{"id": "T", "type": "datetime", "fields": ["a"], "params": {"formats": ["ISO8601", "%d/%m/%Y"]}}])
    assert rows_of(run(frame(a=["2026-01-31", "31/01/2026", "2026-13-01"]), p2), "T") == [4]


def test_regex_allowed_values_and_case_handling():
    p = prof([{"id": "X", "type": "regex", "fields": ["a"], "params": {"pattern": "[A-Z]{3}"}},
              {"id": "V", "type": "allowed_values", "fields": ["b"], "params": {"values": ["Red", "Blue"]}}])
    out = run(frame(a=["USD", "usd", "US"], b=["red", "BLUE", "green"]), p)
    assert rows_of(out, "X") == [3, 4] and rows_of(out, "V") == [4]
    p2 = prof([{"id": "V", "type": "allowed_values", "fields": ["b"], "params": {"values": ["Red"], "case_sensitive": True}}])
    assert rows_of(run(frame(b=["Red", "red"]), p2), "V") == [3]


def test_range_by_unit_and_missing_unit_config_not_checked():
    p = prof([{"id": "R", "type": "range", "fields": ["a"], "params": {
        "unit_field": "unit", "ranges_by_unit": {"C": {"min": 0, "max": 100}, "F": {"min": 32, "max": 212}}}}])
    out = run(frame(a=["50", "150", "150", "999"], unit=["C", "C", "F", "K"]), p)
    assert rows_of(out, "R") == [3]
    assert result_of(out, "R").rows_checked == 3  # the Kelvin row has no configured range


def test_when_conditions_restrict_rows_and_must_be_mapped():
    p = prof([{"id": "W", "type": "required", "fields": ["a"], "when": [{"field": "b", "in": ["x"]}]}])
    out = run(frame(a=["", "", "ok"], b=["x", "y", "x"]), p)
    assert rows_of(out, "W") == [2]
    assert result_of(run(frame(a=[""]), p, mapping={"a": "a", "b": None}), "W").status == "skipped"


def test_compare_numeric_datetime_and_text():
    p = prof([{"id": "C1", "type": "compare", "fields": ["a", "b"], "params": {"op": "<=", "kind": "numeric"}},
              {"id": "C2", "type": "compare", "fields": ["d1", "d2"], "params": {"op": "!=", "kind": "text"}},
              {"id": "O", "type": "date_order", "fields": ["ts", "c"], "params": {"allow_equal": False}}])
    df = frame(a=["1", "5", ""], b=["2", "3", "9"], d1=["x", "Y", "z"], d2=["q", "y ", "w"],
               ts=["2026-01-01", "2026-01-01", "2026-01-02"], c=["2026-01-02", "2026-01-01", "2026-01-01"])
    out = run(df, p)
    assert rows_of(out, "C1") == [3] and rows_of(out, "C2") == [3] and rows_of(out, "O") == [3, 4]


def test_expression_equals_tolerance_and_blank_as_zero():
    p = prof([{"id": "E", "type": "expression_equals",
               "params": {"target": "c", "expression": "a * b - d1", "tolerance_abs": 0.01, "blank_as_zero": ["d1"]}}])
    out = run(frame(a=["2", "2", "2", "2"], b=["10", "10", "10", "x"], d1=["", "5", "5", ""], c=["20", "15", "20", "20"]), p)
    assert rows_of(out, "E") == [4]
    assert "= a * b - d1" in out.findings[0].expected


def test_unsafe_expressions_are_rejected_at_load_time():
    for bad in ("__import__('os').system('x')", "a ** 2", "open('f')", "a if b else c", "'x' * a"):
        with pytest.raises(ProfileError):
            prof([{"id": "E", "type": "expression_equals", "params": {"target": "c", "expression": bad}}])


def test_referential_in_dataset_code_list_and_table():
    p = prof([{"id": "A", "type": "referential", "fields": ["a"], "params": {"reference_field": "id"}},
              {"id": "B", "type": "referential", "fields": ["b"], "params": {"reference_values": ["x", "y"]}},
              {"id": "C", "type": "referential", "fields": ["c"], "params": {"reference_table": "t", "reference_column": "k"}}])
    df = frame(id=["1", "2"], a=["1", "9"], b=["x", "z"], c=["k1", "k9"])
    out = run(df, p, reference_tables={"t": pd.DataFrame({"k": ["k1"]})})
    assert rows_of(out, "A") == [3] and rows_of(out, "B") == [3] and rows_of(out, "C") == [3]
    assert result_of(run(df, p), "C").status == "skipped"
    out = run(df, p, reference_tables={"t": pd.DataFrame({"zzz": ["k1"]})})
    assert result_of(out, "C").status == "skipped" and "no column" in result_of(out, "C").message


def test_timestamp_gap_order_and_sequence_by_group():
    p = prof([{"id": "G", "type": "timestamp_gap", "fields": ["ts"], "params": {"group_by": ["grp"], "max_gap_seconds": 60}},
              {"id": "O", "type": "timestamp_order", "fields": ["ts"], "params": {"group_by": ["grp"]}},
              {"id": "S", "type": "sequence", "fields": ["a"], "params": {"group_by": ["grp"]}}])
    df = frame(grp=["x", "x", "x", "y", "y"],
               ts=["2026-01-01T00:00:00", "2026-01-01T00:00:30", "2026-01-01T00:05:00", "2026-01-01T00:00:00", "2025-12-31T00:00:00"],
               a=["1", "2", "5", "1", "2"])
    out = run(df, p)
    assert rows_of(out, "G") == [4, 5] and rows_of(out, "O") == [6] and rows_of(out, "S") == [4]
    assert "4.5 min" in [f for f in out.findings if f.rule_id == "G"][0].explanation


def test_monotonic_flags_decrease_and_marks_rollover_informational():
    p = prof([{"id": "M", "type": "monotonic", "fields": ["a"], "params": {"group_by": ["grp"], "rollover_max": 1000}}])
    out = run(frame(grp=["m"] * 5, a=["100", "990", "5", "6", "2"]), p)
    cats = {f.row_ref: f.category for f in out.findings}
    assert cats == {4: "Informational", 6: "Inconsistent"}


def test_unit_consistency_group_expected_and_dataset_wide():
    p = prof([{"id": "U", "type": "unit_consistency", "fields": ["unit"], "params": {"expected_units": ["kg", "g"], "group_by": ["grp"]}},
              {"id": "W", "type": "unit_consistency", "fields": ["unit"], "params": {"dataset_wide": True}}])
    out = run(frame(grp=["a", "a", "a", "b"], unit=["kg", "kg", "g", "lb"]), p)
    assert rows_of(out, "U") == [4, 5]
    assert rows_of(out, "W") == [4, 5]  # 'kg' is the most common unit across the file (not 'lb' or 'g')


def test_freshness_and_datetime_range_use_as_of():
    p = prof([{"id": "F", "type": "freshness", "fields": ["ts"], "params": {"max_age_days": 2}},
              {"id": "R", "type": "datetime_range", "fields": ["ts"], "params": {"not_after": "as_of", "not_before": "2000-01-01"}}])
    df = frame(ts=["2026-10-18", "2026-10-30", "1999-01-01"])
    out = run(df, p, as_of="2026-10-19T00:00:00+00:00")
    assert rows_of(out, "R") == [3, 4]
    assert result_of(out, "F").status == "passed"  # newest record (30 Oct) is not older than as_of
    out = run(frame(ts=["2026-10-10"]), p, as_of="2026-10-19T00:00:00+00:00")
    assert result_of(out, "F").status == "failed" and out.findings[0].row_ref is None


# ---------------------------------------------------------------- outliers
VALUES = ["10", "11", "10", "12", "11", "10", "11", "12", "10", "11", "1000"]


def _stat_profile(**params):
    return prof([{"id": "S", "type": "statistical_anomaly", "fields": ["a"], "severity": "Low",
                  "params": {"min_samples": 5, **params}}])


def test_outlier_is_labelled_statistical_anomaly_never_invalid_and_data_is_preserved():
    df = frame(a=VALUES)
    snapshot = df.copy(deep=True)
    out = run(df, _stat_profile())
    assert [(f.row_ref, f.category) for f in out.findings] == [(12, "Statistical anomaly")]
    assert "not proof" in out.findings[0].explanation or "genuine" in out.findings[0].explanation
    assert out.findings[0].observed_value == "1000"          # original value preserved in the evidence
    pd.testing.assert_frame_equal(df, snapshot)               # the input frame is untouched, no rows removed
    assert len(df) == len(VALUES) and "1000" in df["a"].tolist()


def test_statistical_rule_cannot_be_given_an_error_category():
    p = prof([{"id": "S", "type": "statistical_anomaly", "fields": ["a"], "category": "Invalid", "params": {}}])
    assert p.rule("S").category == "Statistical anomaly"       # forced: a statistic is never an error label
    with pytest.raises(ProfileError):                           # and no other rule type may claim that category
        prof([{"id": "R", "type": "required", "fields": ["a"], "category": "Statistical anomaly"}])


def test_explicit_validation_failures_and_anomalies_are_distinct_categories():
    p = prof([{"id": "N", "type": "numeric", "fields": ["a"]},
              {"id": "S", "type": "statistical_anomaly", "fields": ["a"], "params": {"min_samples": 5}}])
    out = run(frame(a=VALUES + ["abc"]), p)
    cat = {f.rule_id: f.category for f in out.findings}
    assert cat == {"N": "Invalid", "S": "Statistical anomaly"}
    assert result_of(out, "S").rows_checked == 11  # 'abc' is not part of the statistics


def test_robust_z_and_isolation_forest_and_group_baseline():
    out = run(frame(a=VALUES), _stat_profile(method="robust_z", threshold=3.5))
    assert rows_of(out, "S") == [12]
    out = run(frame(a=VALUES + ["11"] * 10), _stat_profile(method="isolation_forest", contamination=0.05))
    assert 12 in rows_of(out, "S")
    p = prof([{"id": "S", "type": "statistical_anomaly", "fields": ["a"], "params": {"min_samples": 5, "group_by": ["grp"]}}])
    df = frame(a=["10", "11", "12", "10", "11", "1000", "1001", "1002", "1000", "1001"], grp=["x"] * 5 + ["y"] * 5)
    assert rows_of(run(df, p), "S") == []  # 1000 is normal for group y


def test_small_samples_and_constant_data_skip_the_method_with_a_note():
    out = run(frame(a=["1", "2", "3"]), _stat_profile(min_samples=10))
    assert out.findings == [] and "skipped" in result_of(out, "S").message
    out = run(frame(a=["5"] * 12 + ["6"]), _stat_profile())
    assert "IQR is zero" in result_of(out, "S").message


def test_anomaly_screening_switch_and_per_column_exclusion():
    p = prof([{"id": "S", "type": "statistical_anomaly", "fields": ["a", "b"], "enabled": False, "params": {"min_samples": 5}}])
    df = frame(a=VALUES, b=VALUES)
    assert result_of(run(df, p), "S").status == "disabled"
    assert set(rows_of(run(df, p, anomaly_screening=True), "S")) == {12}
    assert {f.columns[0] for f in run(df, p, anomaly_screening=True).findings} == {"a", "b"}
    out = run(df, p, anomaly_screening=True, rule_overrides={"S": {"params": {"exclude_fields": ["b"]}}})
    assert {f.columns[0] for f in out.findings} == {"a"}
    out = run(df, p, anomaly_screening=True, rule_overrides={"S": {"params": {"exclude_fields": ["a", "b"]}}})
    assert result_of(out, "S").status == "skipped"
    p_on = prof([{"id": "S", "type": "statistical_anomaly", "fields": ["a"], "params": {"min_samples": 5}}])
    assert result_of(run(df, p_on, anomaly_screening=False), "S").status == "disabled"


# ------------------------------------------------------------- robustness
def test_run_is_reproducible_and_does_not_modify_input():
    p = prof([{"id": "R", "type": "required", "fields": ["a"]}, {"id": "N", "type": "numeric", "fields": ["b"]}])
    df = frame(a=["", "x", ""], b=["1", "z", "3"])
    snap = df.copy(deep=True)
    a, b = run(df, p), run(df, p)
    key = lambda o: [(f.rule_id, f.row_index, f.columns, f.observed_value, f.expected, f.explanation) for f in o.findings]
    assert key(a) == key(b) and a.summary == b.summary
    pd.testing.assert_frame_equal(df, snap)


def test_a_failing_rule_does_not_abort_the_run(monkeypatch):
    p = prof([{"id": "R", "type": "required", "fields": ["a"]}, {"id": "N", "type": "numeric", "fields": ["a"]}])

    def boom(rule, ctx, elig):
        raise RuntimeError("kaput")
    monkeypatch.setitem(rule_engine.CHECKS, "numeric", boom)
    out = run(frame(a=[""]), p)
    assert result_of(out, "N").status == "error" and "kaput" in result_of(out, "N").message
    assert rows_of(out, "R") == [2]


def test_findings_per_rule_are_capped_with_a_visible_note():
    p = prof([{"id": "R", "type": "required", "fields": ["a"]}])
    out = run(frame(a=[""] * 50), p, max_findings_per_rule=10)
    r = result_of(out, "R")
    assert r.findings_total == 50 and r.findings_stored == 10 and "first 10 of 50" in r.message
    assert len(out.findings) == 10


def test_empty_dataset_runs_without_crashing():
    p = prof([{"id": "R", "type": "required", "fields": ["a"]}, {"id": "T", "type": "timestamp_gap", "fields": ["ts"], "params": {"max_gap_seconds": 1}}])
    out = run(pd.DataFrame({"a": [], "ts": []}, dtype=str), p)
    assert out.findings == [] and out.n_rows == 0


def test_profile_validation_rejects_bad_definitions():
    base = {"id": "X", "name": "x", "severity": "High", "explanation": "e", "recommendation": "r"}
    bad = [
        {**base, "type": "no_such_type", "fields": ["a"]},
        {**base, "type": "required", "fields": ["undeclared"]},
        {**base, "type": "regex", "fields": ["a"], "params": {"pattern": "("}},
        {**base, "type": "regex", "fields": ["a"], "params": {}},
        {**base, "type": "required", "fields": ["a"], "severity": "Catastrophic"},
        {**base, "type": "required", "fields": ["a"], "category": "Nonsense"},
        {**base, "type": "range", "fields": ["a"], "params": {}},
        {**base, "type": "compare", "fields": ["a"], "params": {"op": "<", "kind": "numeric"}},
    ]
    for r in bad:
        with pytest.raises(ProfileError):
            profile_from_dict({"profile_id": "test", "name": "T", "version": "1", "fields": DEFAULT_FIELDS, "rules": [r]})
    with pytest.raises(ProfileError):
        profile_from_dict({"profile_id": "Bad Id", "name": "T", "version": "1", "fields": {}, "rules": []})
    dup = {**base, "type": "required", "fields": ["a"]}
    with pytest.raises(ProfileError):
        profile_from_dict({"profile_id": "test", "name": "T", "version": "1", "fields": DEFAULT_FIELDS, "rules": [dup, dup]})


def test_profile_hash_changes_when_a_rule_changes():
    p1 = prof([{"id": "R", "type": "range", "fields": ["a"], "params": {"max": 10}}])
    p2 = prof([{"id": "R", "type": "range", "fields": ["a"], "params": {"max": 11}}])
    assert p1.content_hash != p2.content_hash and p1.content_hash == prof([{"id": "R", "type": "range", "fields": ["a"], "params": {"max": 10}}]).content_hash
