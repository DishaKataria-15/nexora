"""upload -> profile -> plan -> confirm -> transform -> validate -> export, checked against the generator's recorded ground truth."""
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from nexora.cleaning import DECISION, REVIEW, SAFE
from nexora.cleaning.validation import frame_fingerprint
from nexora.validation.ingestion import IngestError
from nexora.validation.profiles import ProfileError

from .conftest import AS_OF, SAMPLES, decide


def run(ws, fname, profile, goal="analytics", approvals=None, **choice_by_op):
    h = ws.ingest(fname, (SAMPLES / fname).read_bytes())
    plan = ws.make_plan(h, profile, goal, as_of=AS_OF)
    res, vid = ws.run_preparation(h, plan, approvals or {}, decide(plan, **choice_by_op), as_of=AS_OF)
    return h, plan, res, vid


# =============================== retail ======================================================================
def test_retail_plan_is_explainable_and_conservative(ws):
    h = ws.ingest("retail_orders_messy.csv", (SAMPLES / "retail_orders_messy.csv").read_bytes())
    plan = ws.make_plan(h, "retail", "analytics", as_of=AS_OF)
    for a in plan.actions:
        assert a.title and a.summary, a.action_id                       # every action has a plain-language explanation
        if a.kind != REVIEW:
            assert a.rationale and a.expected_impact, a.action_id        # ... and details to expand
    ops = {(a.op, a.kind) for a in plan.actions}
    assert ("remove_exact_duplicates", SAFE) in ops and ("parse_dates", DECISION) in ops and ("drop_rows_missing", DECISION) in ops
    assert all(a.default_choice and a.choice_params.get(a.default_choice, "x") is None for a in plan.by_kind(DECISION))   # default = leave unchanged


def test_retail_defaults_change_nothing_ambiguous(ws, truth):
    h, plan, res, _ = run(ws, "retail_orders_messy.csv", "retail")
    t, p = truth["retail"], res.prepared
    assert res.after["rows"] == t["rows"] - t["exact_duplicate_copies"]                     # only exact repeats removed
    assert not pd.api.types.is_datetime64_any_dtype(p["ship_date"])                          # ambiguous dates not guessed
    assert not pd.api.types.is_numeric_dtype(p["quantity"])                                  # 'two' not silently dropped or converted
    assert p["order_id"].isna().sum() == len(t["missing_order_id_rows"])                     # identifiers never invented, rows kept
    assert any("Day first" in c.label for a in plan.by_kind(DECISION) for c in a.choices)
    assert res.reconciliation["ok"] and res.source_unchanged


def test_retail_removes_only_the_appended_copies_not_near_duplicates(ws, truth):
    _, plan, res, _ = run(ws, "retail_orders_messy.csv", "retail")
    t = truth["retail"]
    n_base = t["rows"] - t["exact_duplicate_copies"] - t["near_duplicate_rows"]
    removed = [r for e in res.log for r in e.removed_rows]
    assert len(removed) == 5 and all(n_base + 2 <= r < n_base + 2 + 5 for r in removed)   # the 5 exact copies, nothing else
    assert (res.prepared.iloc[-3:]["quantity"].astype(str) == "9").all()                    # near-duplicates stay
    assert any("share a record key" in a.title for a in plan.actions if a.kind == REVIEW)    # ... and are listed for review


def test_retail_with_decisions_applied(ws, truth):
    h, plan, res, _ = run(ws, "retail_orders_messy.csv", "retail", convert_numeric="set_missing", parse_dates="dmy", drop_rows_missing="drop")
    t, p, raw = truth["retail"], res.prepared, h.raw
    assert res.after["rows"] == t["rows"] - 5 - len(t["missing_order_id_rows"])
    assert pd.api.types.is_datetime64_any_dtype(p["ship_date"]) and pd.api.types.is_datetime64_any_dtype(p["order_date"])
    assert p["order_id"].notna().all() and p["quantity"].isna().sum() == len(t["unreadable_quantity_rows"])
    for r in t["ambiguous_ship_date_rows"]:                                                # day-first, checked against an independent parse of the raw text
        i = r - 2
        if i in p.index:
            assert p.loc[i, "ship_date"] == pd.to_datetime(raw.loc[i, "ship_date"], format="%d/%m/%Y")
    assert res.reconciliation["ok"]


def test_retail_normalisations_match_independent_counts(ws):
    h, plan, res, _ = run(ws, "retail_orders_messy.csv", "retail")
    raw, p = h.raw, res.prepared
    assert set(p["order_type"].dropna()) <= {"SALE", "RETURN"} and set(p["currency"].dropna()) == {"USD"}
    assert not p["sku"].astype(str).str.endswith(" ").any()
    assert (p["discount_amount"] >= 0).all() and p["discount_amount"].notna().all()           # blank discount -> 0 per profile
    log = {e.operation: e for e in res.log if e.status != "not_approved"}
    assert log["trim_whitespace"].values_changed == int((raw["sku"] != raw["sku"].str.strip()).sum()) == 7
    placeholder = int(raw.apply(lambda s: s.astype(str).str.strip().str.lower().isin({"", "n/a", "null"})).to_numpy().sum())
    assert log["standardize_nulls"].values_changed == placeholder
    case_changes = int(((raw["order_type"] != raw["order_type"].str.upper()) | (raw["currency"] != raw["currency"].str.upper())).sum())
    assert log["normalize_case"].values_changed == case_changes == 18


def test_retail_negative_prices_and_total_mismatches_are_reported_not_fixed(ws, truth):
    _, _, res, _ = run(ws, "retail_orders_messy.csv", "retail")
    t = truth["retail"]
    assert (res.prepared["unit_price"] < 0).sum() == len(t["negative_price_rows"])           # values untouched
    flagged = {int(f["row_ref"]) for f in res.findings_after if f["rule_id"] == "RET-007" and f.get("row_ref")}
    assert set(t["total_mismatch_rows"]) <= flagged
    assert any("flagged" in u for u in res.unresolved)


def test_source_is_immutable_everywhere(ws):
    data = (SAMPLES / "retail_orders_messy.csv").read_bytes()
    h = ws.ingest("retail_orders_messy.csv", data)
    before = frame_fingerprint(h.raw)
    plan = ws.make_plan(h, "retail", "analytics", as_of=AS_OF)
    res, _ = ws.run_preparation(h, plan, {}, decide(plan, parse_dates="dmy", drop_rows_missing="drop"), as_of=AS_OF)
    assert frame_fingerprint(h.raw) == before and res.source_unchanged
    assert ws.repo.read_raw_bytes(h.dataset_id) == data                                       # stored original is byte-identical
    is_root = getattr(os, "geteuid", lambda: 1)() == 0
    assert not os.access(ws.repo.get_dataset(h.dataset_id)["raw_path"], os.W_OK) or is_root  # read-only on disk (root ignores modes)


def test_log_is_complete_and_consistent(ws):
    _, _, res, _ = run(ws, "retail_orders_messy.csv", "retail")
    ran = [e for e in res.log if e.status != "not_approved"]
    assert all(e.operation and e.reason is not None and e.parameters is not None and e.timestamp for e in res.log)
    assert ran[0].rows_in == res.before["rows"] and ran[-1].rows_out == res.after["rows"]
    assert all(a.rows_out == b.rows_in for a, b in zip(ran, ran[1:]))
    assert sum(e.rows_in - e.rows_out for e in ran) == res.before["rows"] - res.after["rows"]
    assert {e.action_id for e in res.log if e.status == "not_approved"} == set(res.approved["not_run"])   # what was not done is recorded too


def test_disabled_action_leaves_data_untouched_and_is_logged(ws):
    h = ws.ingest("retail_orders_messy.csv", (SAMPLES / "retail_orders_messy.csv").read_bytes())
    plan = ws.make_plan(h, "retail", "analytics", as_of=AS_OF)
    trim = next(a for a in plan.actions if a.op == "trim_whitespace")
    res, _ = ws.run_preparation(h, plan, {trim.action_id: False}, decide(plan), as_of=AS_OF)
    assert res.prepared["sku"].astype(str).str.endswith(" ").sum() == 7
    assert next(e for e in res.log if e.action_id == trim.action_id).status == "not_approved"


def test_preparation_is_reproducible(ws):
    _, _, a, _ = run(ws, "retail_orders_messy.csv", "retail")
    _, _, b, _ = run(ws, "retail_orders_messy.csv", "retail")
    pd.testing.assert_frame_equal(a.prepared, b.prepared)
    assert [(e.operation, e.values_changed, e.rows_out) for e in a.log] == [(e.operation, e.values_changed, e.rows_out) for e in b.log]


def test_exports_are_valid_and_consistent(ws):
    h, plan, res, vid = run(ws, "retail_orders_messy.csv", "retail")
    files = ws.preparation_exports(res, h, plan)
    assert set(files) == {"prepared_data.csv", "preparation_report.html", "preparation_report.json", "transformation_log.csv"}
    csv = pd.read_csv(io.BytesIO(files["prepared_data.csv"]))
    assert len(csv) == res.after["rows"] and list(csv.columns) == list(res.prepared.columns)
    rep = json.loads(files["preparation_report.json"])
    assert rep["after"]["rows"] == len(csv) and rep["reconciliation"]["ok"] and rep["source_unchanged"]
    log = pd.read_csv(io.BytesIO(files["transformation_log.csv"]))
    assert len(log) == len(res.log) and {"operation", "rows_in", "rows_out", "status"} <= set(log.columns)
    html = files["preparation_report.html"].decode()
    assert "Reconciliation" in html and "Before and after" in html
    reloaded, _ = ws.repo.load_version(vid)                                                    # persisted version round-trips
    assert len(reloaded) == len(res.prepared) and pd.api.types.is_datetime64_any_dtype(reloaded["order_date"])


def test_validation_improves_measurably(ws):
    _, _, res, _ = run(ws, "retail_orders_messy.csv", "retail")
    total = lambda rs: sum(r["rows_flagged"] for r in rs)  # noqa: E731
    assert total(res.rule_summary_after) < total(res.rule_summary_before)                     # fewer flagged rows overall
    b = {r["rule_id"]: r for r in res.rule_summary_before}
    a = {r["rule_id"]: r for r in res.rule_summary_after}
    assert b["RET-013"]["status"] == "failed" and a["RET-013"]["status"] == "passed"           # fully fixed: lowercase currency codes
    for rid in ("RET-002", "RET-003", "RET-004", "RET-012"):
        assert a[rid]["rows_flagged"] < b[rid]["rows_flagged"], rid
    assert a["RET-008"]["status"] == "passed"                                                   # the generator keeps ship >= order, so no new contradiction
    assert res.after["columns_with_format_issues"] < res.before["columns_with_format_issues"]
    assert res.after["exact_duplicate_rows"] == 0 < res.before["exact_duplicate_rows"]


def test_cleaning_can_reveal_problems_that_were_hidden_and_says_so(ws):
    # Unreadable dates hide a ship-before-order contradiction; once the dates are read it must be reported, not hidden.
    csv = ("order_id,order_date,customer_id,sku,quantity,unit_price,discount_amount,order_total,order_type,return_reason,currency,stock_on_hand,ship_date\n"
           "O1,25/09/2026,C1,S1,1,10.00,,10.00,SALE,,USD,5,2026-09-20\n"
           "O2,2026-09-01,C2,S1,1,10.00,,10.00,SALE,,USD,5,2026-09-03\n")
    h = ws.ingest("hidden.csv", csv.encode())
    plan = ws.make_plan(h, "retail", "analytics", as_of=AS_OF)
    res, _ = ws.run_preparation(h, plan, {}, decide(plan), as_of=AS_OF)
    after = {r["rule_id"]: r for r in res.rule_summary_after}
    assert after["RET-008"]["status"] == "failed" and after["RET-008"]["rows_flagged"] == 1
    assert any("Shipped on or after" in u for u in res.unresolved)


# =============================== finance ===================================================================
def test_finance_flow(ws, truth):
    h, plan, res, _ = run(ws, "finance_transactions_messy.csv", "finance")
    t, p, raw = truth["finance"], res.prepared, h.raw
    assert res.after["rows"] == t["rows"] - t["exact_duplicate_copies"] and res.reconciliation["ok"]
    assert pd.api.types.is_numeric_dtype(p["amount"]) and pd.api.types.is_datetime64_any_dtype(p["posting_date"])
    paren = raw.index[raw["amount"].str.startswith("(")]
    assert len(paren) == 5 and all(p.loc[i, "amount"] < 0 for i in paren if i in p.index)
    comma = raw.index[raw["amount"].str.contains(",")]
    assert len(comma) == 8 and all(p.loc[i, "amount"] >= 1000 for i in comma if i in p.index)
    assert p["transaction_id"].isna().sum() == len(t["missing_transaction_id_rows"])              # ids not invented
    assert set(p["currency"].dropna()) == {"USD"} and set(p["txn_type"].dropna()) <= {"DEBIT", "CREDIT", "REVERSAL"}
    assert not p["account_id"].astype(str).str.endswith(" ").any()
    mism = {int(f["row_ref"]) for f in res.findings_after if f["rule_id"] == "FIN-007" and f.get("row_ref")}
    assert set(t["balance_mismatch_rows"]) <= mism                                                # ledger errors surfaced, not "fixed"
    assert p["balance_after"].astype(float).isin([1.0, 2.0, 3.0]).sum() == 3


def test_finance_drop_missing_ids_only_when_chosen(ws, truth):
    _, _, res, _ = run(ws, "finance_transactions_messy.csv", "finance", drop_rows_missing="drop")
    assert res.prepared["transaction_id"].notna().all()
    assert res.after["rows"] == truth["finance"]["rows"] - 4 - 2


# =============================== manufacturing ==================================================================
def test_manufacturing_flow(ws, truth):
    h, plan, res, _ = run(ws, "manufacturing_sensors_messy.csv", "manufacturing")
    t, p, raw = truth["manufacturing"], res.prepared, h.raw
    assert res.after["rows"] == t["rows"] - t["exact_duplicate_copies"] and res.reconciliation["ok"]
    assert set(p["temperature_unit"].dropna()) == {"C"}
    f_rows = [r - 2 for r in t["fahrenheit_rows"]]
    for i in f_rows:                                                                                # converted values equal the formula applied to the raw text
        assert p.loc[i, "temperature"] == pytest.approx((float(raw.loc[i, "temperature"]) - 32) * 5 / 9, abs=1e-5)
    assert p.loc[f_rows, "temperature"].between(60, 85).all()                                       # and now sit alongside the Celsius readings
    assert p.loc[t["vibration_spike_row"] - 2, "vibration"] == 25.0                                 # outliers are kept, never removed
    assert p["temperature"].isna().sum() == len(t["missing_temperature_rows"])                      # missing readings are not invented
    b = next(r for r in res.rule_summary_before if r["rule_id"] == "MFG-011")
    a = next(r for r in res.rule_summary_after if r["rule_id"] == "MFG-011")
    assert b["status"] == "failed" and a["status"] != "failed"
    assert (p["machine_state"].dropna() == "RUNNING").all()


# =============================== goals and scope ======================================================================
def test_ml_goal_does_not_fill_values_globally(ws):
    h = ws.ingest("manufacturing_sensors_messy.csv", (SAMPLES / "manufacturing_sensors_messy.csv").read_bytes())
    ml = ws.make_plan(h, "manufacturing", "supervised", as_of=AS_OF)
    an = ws.make_plan(h, "manufacturing", "analytics", as_of=AS_OF)
    assert not any(a.op == "impute" for a in ml.actions) and any(a.op == "impute" for a in an.actions)
    assert any("training data only" in w for w in ml.warnings)


def test_optional_imputation_is_off_unless_chosen_and_never_touches_ids(ws):
    h = ws.ingest("manufacturing_sensors_messy.csv", (SAMPLES / "manufacturing_sensors_messy.csv").read_bytes())
    plan = ws.make_plan(h, "manufacturing", "analytics", as_of=AS_OF)
    imp = next(a for a in plan.actions if a.op == "impute")
    assert imp.default_enabled is False and "device_id" not in imp.columns
    res, _ = ws.run_preparation(h, plan, {imp.action_id: True}, decide(plan), as_of=AS_OF)
    assert res.prepared["temperature"].notna().all() and res.reconciliation["ok"]


def test_generic_profile_on_ml_sample(ws):
    h, plan, res, _ = run(ws, "customer_churn.csv", "generic", goal="supervised")
    p = res.prepared
    assert pd.api.types.is_datetime64_any_dtype(p["signup_date"]) and pd.api.types.is_numeric_dtype(p["monthly_charges"])
    assert res.after["rows"] == len(h.raw) and res.reconciliation["ok"]
    assert p["tenure_months"].isna().sum() == h.raw["tenure_months"].isin(["", "N/A"]).sum() > 0     # missing stays missing for ML goals


# =============================== malformed inputs =====================================================================
def test_empty_and_header_only_files(ws):
    with pytest.raises(IngestError):
        ws.ingest("empty.csv", b"")
    with pytest.raises(IngestError):
        ws.ingest("header_only.csv", b"a,b,c\n")


def test_unsupported_type_and_encoding(ws):
    with pytest.raises(IngestError):
        ws.ingest("data.exe", b"MZ....")
    with pytest.raises(IngestError):
        ws.ingest("bad.csv", bytes(range(256)) * 20)


def test_duplicate_column_names_are_renamed_with_warning(ws):
    h = ws.ingest("dups.csv", b"a,a,b\n1,2,3\n4,5,6\n")
    assert len(set(h.raw.columns)) == 3 and any("duplicate" in w.lower() or "rename" in w.lower() for w in h.ingest.warnings)


def test_mixed_types_and_malformed_dates_do_not_crash_and_are_not_guessed(ws):
    csv = "id,when,amount\n1,2026-01-05,10\n2,not a date,twelve\n3,2026-13-45,5\n4,2026-02-01,7\n"
    h = ws.ingest("mixed.csv", csv.encode())
    plan = ws.make_plan(h, "generic", "analytics", as_of=AS_OF)
    res, _ = ws.run_preparation(h, plan, {}, decide(plan), as_of=AS_OF)
    assert not pd.api.types.is_datetime64_any_dtype(res.prepared["when"]) and not pd.api.types.is_numeric_dtype(res.prepared["amount"])
    assert res.prepared["when"].tolist()[1] == "not a date"                                       # left as text, nothing destroyed
    assert res.reconciliation["ok"]


def test_excel_upload_end_to_end(ws):
    df = pd.DataFrame({"order_id": ["A1", "A2", "A2"], "amount": ["$1,000.50", "20", "20"], "when": ["2026-01-01", "2026-01-02", "2026-01-02"]})
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Sheet1", index=False)
        df.head(1).to_excel(xw, sheet_name="Other", index=False)
    h = ws.ingest("book.xlsx", buf.getvalue(), sheet="Sheet1")
    plan = ws.make_plan(h, "generic", "analytics", as_of=AS_OF)
    res, _ = ws.run_preparation(h, plan, {}, decide(plan, remove_exact_duplicates="remove"), as_of=AS_OF)
    assert res.prepared["amount"].iloc[0] == 1000.5 and len(res.prepared) == 2
    assert ws.open_dataset(h.dataset_id).raw.shape == (3, 3)


def test_large_file_completes_and_reconciles(ws):
    rng = np.random.RandomState(0)
    n = 40_000
    df = pd.DataFrame({"id": [f"R{i}" for i in range(n)], "amount": [f"{v:.2f}" for v in rng.gamma(2, 50, n)], "cat": rng.choice(["a", "A", "b"], n),
                       "d": (pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.randint(0, 300, n), unit="D")).strftime("%Y-%m-%d")})
    df.loc[::40, "amount"] = "N/A"
    h = ws.ingest("big.csv", df.to_csv(index=False).encode())
    plan = ws.make_plan(h, "generic", "analytics", as_of=AS_OF)
    res, _ = ws.run_preparation(h, plan, {}, decide(plan), as_of=AS_OF)
    assert res.reconciliation["ok"] and res.after["rows"] == n and res.prepared["amount"].isna().sum() == n // 40


CUSTOM = """
profile_id: mini
name: Mini
version: "1.0.0"
coverage: template
description: test
guardrail: test
settings: {datetime_formats: ["ISO8601"]}
fields:
  code: {type: identifier, required: true}
rules: []
changelog: [{version: "1.0.0", date: "2026-10-26", note: x}]
"""


def test_custom_profile_and_builtin_id_protection(ws):
    assert ws.add_custom_profile(CUSTOM).profile_id == "mini" and "mini" in ws.profiles
    with pytest.raises(ProfileError):
        ws.add_custom_profile(CUSTOM.replace("mini", "retail"))
    with pytest.raises(ProfileError):
        ws.add_custom_profile("::: not yaml :::\n- [")
    with pytest.raises(ProfileError):
        ws.add_custom_profile("- just\n- a list")


def test_referential_check_uses_reference_tables_when_supplied(ws):
    prof = """
profile_id: refdemo
name: Ref demo
version: "1.0.0"
coverage: template
description: test
guardrail: test
settings: {datetime_formats: ["ISO8601"]}
fields:
  dept: {type: category, required: true}
rules:
  - id: REF-001
    name: Department exists
    type: referential
    fields: [dept]
    params: {reference_table: departments, reference_column: code}
    severity: High
    explanation: Department code is not in the departments table.
    recommendation: Fix the code or add the department.
    rationale: Referential integrity.
changelog: [{version: "1.0.0", date: "2026-10-26", note: x}]
"""
    p = ws.add_custom_profile(prof)
    assert ws.required_reference_tables(p) == {"departments": "code"}
    h = ws.ingest("people.csv", b"name,dept\nA,D1\nB,D2\nC,D9\n")
    refs = {"departments": pd.DataFrame({"code": ["D1", "D2", "D3"]})}
    plan = ws.make_plan(h, "refdemo", "analytics", as_of=AS_OF, reference_tables=refs)
    res, _ = ws.run_preparation(h, plan, {}, decide(plan), as_of=AS_OF, reference_tables=refs)
    flagged = next(r for r in res.rule_summary_after if r["rule_id"] == "REF-001")
    assert flagged["status"] == "failed" and flagged["rows_flagged"] == 1                   # only D9 is unknown
    assert len(res.prepared) == 3                                                           # reported, never deleted
