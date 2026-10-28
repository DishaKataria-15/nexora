"""Unit tests for every preparation rule: valid, invalid, boundary, missing and malformed inputs."""
import numpy as np
import pandas as pd
import pytest

from nexora.cleaning.model import SAFE, Action
from nexora.cleaning.transformations import (ExecContext, analyze_dates, detect_decimal_comma, op_convert_boolean, op_convert_numeric,
                                             op_drop_empty_columns, op_drop_rows_missing, op_fill_constant, op_flag_outliers, op_impute,
                                             op_normalize_case, op_parse_dates, op_remove_exact_duplicates, op_standardize_nulls,
                                             op_standardize_units, op_trim_whitespace, parse_dates, parse_number, run_actions)

TOK = {"", "n/a", "na", "null", "none", "nan", "#n/a"}
CX = ExecContext(null_tokens=TOK)


def df_of(**cols):
    return pd.DataFrame({k: pd.Series(v, dtype=object) for k, v in cols.items()})


# ---- number parsing -------------------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [("12", 12.0), ("-7", -7.0), ("+3", 3.0), (".5", 0.5), ("1e3", 1000.0), ("$1,234.50", 1234.5),
                                           ("1,234", 1234.0), ("(12.5)", -12.5), ("0", 0.0), ("  42  ", 42.0)])
def test_parse_number_valid(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("text", ["abc", "12 kg", "", "   ", "1.2.3", "--5", "inf", "nan", "12%", "1,23,456", None, float("nan")])
def test_parse_number_rejects_ambiguous_or_malformed(text):
    assert parse_number(text) is None


def test_decimal_comma_mode_and_ambiguity():
    assert parse_number("12,50", True) == 12.5
    assert parse_number("1.234,5", True) == 1234.5
    assert parse_number("1,000", True) is None          # one thousand or one point zero: never guess
    assert detect_decimal_comma(pd.Series(["1,5", "2,25", "3"]))
    assert not detect_decimal_comma(pd.Series(["1.5", "2,25"]))


# ---- null tokens --------------------------------------------------------------------------------------
def test_standardize_nulls_blank_and_placeholders_case_insensitive():
    df = df_of(a=["x", "", "N/A", " null ", "None", "real", np.nan])
    out, r = op_standardize_nulls(df, {}, CX)
    assert out["a"].isna().tolist() == [False, True, True, True, True, False, True]
    assert r.values_changed == 4          # blank, N/A, null, None (the NaN was already missing)
    assert df["a"].tolist()[1] == ""      # input untouched


def test_standardize_nulls_leaves_real_words_alone():
    out, r = op_standardize_nulls(df_of(a=["nothing", "nancy", "n/a "]), {}, CX)
    assert out["a"].tolist()[:2] == ["nothing", "nancy"] and r.values_changed == 1


# ---- whitespace and case -----------------------------------------------------------------------------------
def test_trim_whitespace_only_touches_edges():
    out, r = op_trim_whitespace(df_of(a=[" a b ", "c", "\td\n", np.nan]), {}, CX)
    assert out["a"].tolist()[:3] == ["a b", "c", "d"] and r.values_changed == 2


def test_trim_whitespace_skips_typed_columns():
    df = pd.DataFrame({"n": [1.0, 2.0], "d": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    out, r = op_trim_whitespace(df, {}, CX)
    assert r.values_changed == 0 and out["n"].dtype == "float64"


def test_normalize_case_configured_and_mapped():
    df = df_of(t=["sale", "SALE", "Sale", np.nan], c=["usd", "USD", "Eur", "x"])
    out, r = op_normalize_case(df, {"mappings": {"t": {"mode": "upper", "map": {}}, "c": {"mode": None, "map": {"usd": "USD"}}}}, CX)
    assert out["t"].tolist()[:3] == ["SALE"] * 3 and pd.isna(out["t"].iloc[3])
    assert out["c"].tolist() == ["USD", "USD", "Eur", "x"]


def test_normalize_case_missing_column_is_ignored():
    out, r = op_normalize_case(df_of(a=["x"]), {"mappings": {"zzz": {"mode": "upper"}}}, CX)
    assert r.values_changed == 0 and out["a"].tolist() == ["x"]


# ---- numbers ---------------------------------------------------------------------------------------------------
def test_convert_numeric_clean_column_becomes_integer_or_float():
    df = df_of(i=["1", "2", "3", np.nan], f=["1.5", "$2.00", "1,000.25", np.nan])
    out, r = op_convert_numeric(df, {"columns": ["i", "f"]}, CX)
    assert str(out["i"].dtype) == "Int64" and out["i"].tolist()[:3] == [1, 2, 3] and pd.isna(out["i"].iloc[3])
    assert out["f"].dtype == "float64" and out["f"].iloc[2] == 1000.25


def test_convert_numeric_unreadable_defaults_to_leaving_text_and_lists_rows():
    df = df_of(q=["1", "two", "3"])
    out, r = op_convert_numeric(df, {"columns": ["q"]}, CX)
    assert out["q"].tolist() == ["1", "two", "3"]            # nothing guessed
    assert r.skipped_rows == [3] and r.skipped_count == 1 and r.status == "skipped"


def test_convert_numeric_set_missing_only_when_chosen():
    df = df_of(q=["1", "two", "3"])
    out, r = op_convert_numeric(df, {"columns": ["q"], "on_unparseable": {"q": "set_missing"}}, CX)
    assert pd.isna(out["q"].iloc[1]) and out["q"].iloc[2] == 3 and r.skipped_rows == [3]


def test_convert_numeric_all_missing_and_already_numeric_untouched():
    df = pd.DataFrame({"a": pd.Series([np.nan, np.nan], dtype=object), "b": [1.0, 2.0]})
    out, r = op_convert_numeric(df, {"columns": ["a", "b"]}, CX)
    assert out["b"].tolist() == [1.0, 2.0]


# ---- dates -------------------------------------------------------------------------------------------------
def test_analyze_dates_detects_order_from_evidence():
    assert analyze_dates(pd.Series(["25/12/2026", "03/04/2026"])).order == "dmy"
    assert analyze_dates(pd.Series(["12/25/2026"])).order == "mdy"
    assert analyze_dates(pd.Series(["03/04/2026"])).order == "ambiguous"
    assert analyze_dates(pd.Series(["25/12/2026", "12/25/2026"])).order == "conflict"
    assert analyze_dates(pd.Series(["2026-01-02", "2026-03-04"])).order == "none"
    assert analyze_dates(pd.Series(["2026-01-02T10:00:00Z", "2026-01-02T10:00:00+05:30"])).tz_mix == "all"
    assert analyze_dates(pd.Series(["2026-01-02T10:00:00Z", "2026-01-02T10:00:00"])).tz_mix == "mixed"


def test_parse_dates_iso_dmy_names_and_invalid():
    s = pd.Series(["2026-09-01", "13/09/2026", "Sep 5, 2026", "2026-02-30", "12/25/26", np.nan], dtype=object)
    out, bad = parse_dates(s, "dmy")
    assert out.iloc[0] == pd.Timestamp("2026-09-01") and out.iloc[1] == pd.Timestamp("2026-09-13") and out.iloc[2] == pd.Timestamp("2026-09-05")
    assert bad.tolist() == [False, False, False, True, True, False]       # impossible date and 2-digit year are unreadable, not guessed


def test_parse_dates_order_changes_result():
    s = pd.Series(["03/04/2026"], dtype=object)
    assert parse_dates(s, "dmy")[0].iloc[0] == pd.Timestamp("2026-04-03")
    assert parse_dates(s, "mdy")[0].iloc[0] == pd.Timestamp("2026-03-04")


def test_op_parse_dates_ambiguous_is_left_alone_until_order_given():
    df = df_of(d=["03/04/2026", "05/06/2026"])
    out, r = op_parse_dates(df, {"columns": ["d"]}, CX)
    assert out["d"].tolist() == ["03/04/2026", "05/06/2026"] and r.status == "skipped"
    out2, r2 = op_parse_dates(df, {"columns": ["d"], "order": {"d": "dmy"}}, CX)
    assert pd.api.types.is_datetime64_any_dtype(out2["d"]) and out2["d"].iloc[0] == pd.Timestamp("2026-04-03")


def test_op_parse_dates_invalid_requires_choice():
    df = df_of(d=["2026-01-01", "2026-02-30"])
    out, _ = op_parse_dates(df, {"columns": ["d"]}, CX)
    assert not pd.api.types.is_datetime64_any_dtype(out["d"])
    out2, r2 = op_parse_dates(df, {"columns": ["d"], "on_unparseable": {"d": "set_missing"}}, CX)
    assert pd.isna(out2["d"].iloc[1]) and r2.skipped_rows == [3]


def test_op_parse_dates_mixed_timezone_column_left_as_text():
    df = df_of(d=["2026-01-01T00:00:00Z", "2026-01-02T00:00:00"])
    out, r = op_parse_dates(df, {"columns": ["d"]}, CX)
    assert not pd.api.types.is_datetime64_any_dtype(out["d"])


def test_op_parse_dates_time_zone_all_converted_to_utc():
    df = df_of(d=["2026-01-01T05:30:00+05:30", "2026-01-01T00:00:00Z"])
    out, _ = op_parse_dates(df, {"columns": ["d"]}, CX)
    assert out["d"].iloc[0] == out["d"].iloc[1] == pd.Timestamp("2026-01-01 00:00:00")


# ---- booleans ----------------------------------------------------------------------------------------------------
def test_convert_boolean_only_when_every_value_is_known():
    df = df_of(a=["Yes", "no", "TRUE", np.nan], b=["yes", "maybe", np.nan, np.nan])
    out, r = op_convert_boolean(df, {"columns": ["a", "b"]}, CX)
    assert out["a"].tolist()[:3] == [True, False, True] and pd.isna(out["a"].iloc[3])
    assert out["b"].dropna().tolist() == ["yes", "maybe"]      # one unknown word keeps the whole column as text


# ---- units ---------------------------------------------------------------------------------------------------------
UNIT = {"value_col": "t", "unit_col": "u", "target_unit": "C", "conversions": {"C": [1, 0], "F": [5 / 9, -160 / 9], "K": [1, -273.15]}}


def test_standardize_units_converts_known_and_reports_unknown():
    df = pd.DataFrame({"t": [20.0, 212.0, 300.0, 50.0, 10.0, 5.0], "u": pd.Series(["C", "f", "K", "XX", None, "°C"], dtype=object)})
    out, r = op_standardize_units(df, UNIT, CX)
    assert out["t"].round(3).tolist()[:3] == [20.0, 100.0, 26.85]
    assert out["t"].tolist()[3:5] == [50.0, 10.0]                 # unknown and missing unit: unchanged
    assert out["u"].iloc[1] == "C" and out["u"].iloc[3] == "XX"
    assert r.skipped_rows == [5, 6] and r.values_changed == 2


def test_standardize_units_requires_numeric_values():
    df = df_of(t=["20", "30"], u=["C", "F"])
    out, r = op_standardize_units(df, UNIT, CX)
    assert r.status == "skipped" and out["t"].tolist() == ["20", "30"]


def test_standardize_units_missing_columns_skipped():
    out, r = op_standardize_units(pd.DataFrame({"x": [1]}), UNIT, CX)
    assert r.status == "skipped"


# ---- fills, duplicates, removals -----------------------------------------------------------------------------------------
def test_fill_constant_numeric_and_text():
    df = pd.DataFrame({"d": [1.0, np.nan], "s": pd.Series(["a", np.nan], dtype=object)})
    out, r = op_fill_constant(df, {"fills": {"d": "0", "s": "none", "zzz": "1"}}, CX)
    assert out["d"].tolist() == [1.0, 0.0] and out["s"].tolist() == ["a", "none"] and r.values_changed == 2


def test_fill_constant_rejects_non_numeric_value_for_numeric_column():
    df = pd.DataFrame({"d": [1.0, np.nan]})
    out, r = op_fill_constant(df, {"fills": {"d": "zero"}}, CX)
    assert out["d"].isna().sum() == 1 and r.status == "no_change"


def test_duplicates_keyed_removes_only_identical_rows_with_key():
    df = df_of(k=["1", "1", "1", "2", np.nan, np.nan], v=["a", "a", "b", "c", "x", "x"])
    out, r = op_remove_exact_duplicates(df, {"key": ["k"], "policy": "keyed"}, CX)
    assert r.removed_rows == [3]                                   # row 3 repeats row 2; same key but different value (row 4) stays
    assert len(out) == 5 and r.rows_in == 6 and r.rows_out == 5
    assert out.index.tolist() == [0, 2, 3, 4, 5]                    # original positions preserved; null-key twins are not merged


def test_duplicates_policy_all_requires_explicit_choice_semantics():
    df = df_of(a=["x", "x", "y"])
    out, r = op_remove_exact_duplicates(df, {"policy": "all"}, CX)
    assert len(out) == 2 and r.removed_rows == [3]
    out2, r2 = op_remove_exact_duplicates(df, {"policy": "keyed", "key": []}, CX)
    assert len(out2) == 3 and r2.status == "skipped"


def test_impute_median_mode_and_unknown():
    df = pd.DataFrame({"n": [1.0, 3.0, np.nan], "c": pd.Series(["a", "a", np.nan], dtype=object)})
    out, r = op_impute(df, {"numeric_columns": ["n"], "categorical_columns": ["c"]}, CX)
    assert out["n"].tolist() == [1.0, 3.0, 2.0] and out["c"].tolist() == ["a", "a", "a"]
    out2, _ = op_impute(df, {"categorical_columns": ["c"], "categorical_strategy": "unknown"}, CX)
    assert out2["c"].iloc[2] == "Unknown"


def test_drop_empty_columns_only_fully_empty():
    df = pd.DataFrame({"a": [np.nan, np.nan], "b": [1, np.nan]})
    out, r = op_drop_empty_columns(df, {"columns": ["a", "b"]}, CX)
    assert list(out.columns) == ["b"] and r.columns_dropped == ["a"]


def test_drop_rows_missing_required_identifier():
    df = df_of(id=["1", np.nan, "3"], v=["a", "b", "c"])
    out, r = op_drop_rows_missing(df, {"columns": ["id"]}, CX)
    assert len(out) == 2 and r.removed_rows == [3]


def test_flag_outliers_adds_columns_and_changes_nothing_else():
    vals = list(range(20)) + [500]
    df = pd.DataFrame({"x": vals})
    out, r = op_flag_outliers(df, {"columns": ["x"], "k": 3.0}, CX)
    assert out["x"].tolist() == vals and out["x__outlier_flag"].sum() == 1 and r.columns_added == ["x__outlier_flag"]


def test_flag_outliers_ignores_constant_and_low_variety():
    out, r = op_flag_outliers(pd.DataFrame({"x": [1] * 4, "y": [1, 2, 1, 2]}), {"columns": ["x", "y"]}, CX)
    assert r.status == "no_change"


# ---- the runner ------------------------------------------------------------------------------------------------------------
def _act(i, op, params, kind=SAFE):
    return Action(f"A{i}", op, kind, op, "s", columns=[], params=params)


def test_runner_orders_steps_and_chains_row_counts():
    df = df_of(a=[" x", "x ", "N/A"])
    acts = [_act(2, "trim_whitespace", {}), _act(1, "standardize_nulls", {}), _act(3, "remove_exact_duplicates", {"policy": "all"})]
    out, log = run_actions(df, acts, CX)
    assert [e.operation for e in log] == ["standardize_nulls", "trim_whitespace", "remove_exact_duplicates"]
    assert len(out) == 2 and [e.rows_in for e in log] == [3, 3, 3] and log[-1].rows_out == 2


def test_runner_isolates_failures_and_logs_them():
    df = df_of(a=["x"])
    acts = [_act(1, "convert_numeric", {"columns": "not-a-list"}), _act(2, "no_such_op", {}), _act(3, "trim_whitespace", {})]
    out, log = run_actions(df, acts, CX)
    statuses = {e.operation: e.status for e in log}
    assert statuses["no_such_op"] == "failed" and statuses["trim_whitespace"] in ("applied", "no_change")
    assert out["a"].tolist() == ["x"]


def test_runner_records_not_approved_actions_without_running_them():
    df = df_of(a=[" x"])
    out, log = run_actions(df, [], CX, not_approved=[_act(9, "trim_whitespace", {})])
    assert out["a"].tolist() == [" x"] and log[0].status == "not_approved"


def test_runner_never_mutates_input():
    df = df_of(a=[" x", "N/A"])
    snap = df.copy(deep=True)
    run_actions(df, [_act(1, "standardize_nulls", {}), _act(2, "trim_whitespace", {})], CX)
    pd.testing.assert_frame_equal(df, snap)
