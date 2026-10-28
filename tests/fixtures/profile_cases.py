"""Per-domain test data: a CLEAN dataset (must produce no hard findings) plus VIOLATIONS.

A violation is (rule_id, changes, row_index). ``changes`` is a dict of column -> value applied to the
clean row ``row_index`` (or a function receiving the clean rows and returning such a dict).
The test asserts that the rule flags that row. All data is synthetic.
"""
from __future__ import annotations

from typing import Any, Callable

Case = tuple  # (rule_id, changes | callable, row_index)


def rows(n: int, make: Callable[[int], dict[str, Any]]) -> list[dict[str, Any]]:
    return [make(i) for i in range(n)]


def v(rule: str, changes, row: int = 0) -> Case:
    return (rule, changes, row)


def copy_from(src: int, *cols: str, **extra):
    """Make row N look like clean row `src` in the given columns (creates a duplicate)."""
    return lambda clean: {**{c: clean[src][c] for c in cols}, **extra}


CASES: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------- healthcare
CASES["healthcare"] = {
    "clean": rows(3, lambda i: dict(
        patient_id=f"P{i+1}", visit_id=f"V{i+1}", admit_time="2026-03-01T08:00:00", discharge_time="2026-03-03T10:00:00",
        birth_date="1980-05-14", sex="F", measurement_type="glucose", measurement_value="5.4", measurement_unit="mmol/L",
        measured_at="2026-03-01T09:00:00", heart_rate="72")),
    "violations": [
        v("HC-001", {"patient_id": ""}), v("HC-003", {"admit_time": "not-a-date"}),
        v("HC-004", {"discharge_time": "2026-02-01T00:00:00"}), v("HC-005", {"measured_at": "2026-02-01T00:00:00"}),
        v("HC-006", {"birth_date": "2090-01-01"}), v("HC-007", {"sex": "Q"}), v("HC-008", {"measurement_value": "abc"}),
        v("HC-009", {"measurement_unit": ""}), v("HC-010", {"measurement_unit": "mg/dL"}), v("HC-011", {"heart_rate": "-3"}),
        v("HC-012", copy_from(1, "patient_id", "measurement_type", "measured_at")),
        v("HC-013", {"measurement_type": ""}), v("HC-014", {"measurement_value": ""}),
    ],
}
# ------------------------------------------------------------------- finance
_fin = [("T1", "-50.00", "DEBIT", "1000.00", "950.00"), ("T2", "200.00", "CREDIT", "950.00", "1150.00"),
        ("T3", "-20.00", "DEBIT", "1150.00", "1130.00")]
CASES["finance"] = {
    "clean": [dict(transaction_id=t, account_id="A1", posting_date=f"2026-09-0{i+1}", amount=a, currency="USD", txn_type=ty,
                   balance_before=b0, balance_after=b1, original_transaction_id="") for i, (t, a, ty, b0, b1) in enumerate(_fin)],
    "violations": [
        v("FIN-001", {"account_id": ""}), v("FIN-002", {"transaction_id": "T2"}), v("FIN-003", {"amount": "abc"}),
        v("FIN-004", {"posting_date": "01/09/2026"}), v("FIN-005", {"currency": "usd"}), v("FIN-006", {"txn_type": "XYZ"}),
        v("FIN-007", {"balance_after": "1.00"}),
        v("FIN-008", {"account_id": "A1", "posting_date": "2026-09-02", "amount": "200.00", "currency": "USD", "txn_type": "CREDIT"}),
        v("FIN-009", {"txn_type": "REFUND", "original_transaction_id": ""}),
        v("FIN-010", {"txn_type": "REFUND", "original_transaction_id": "T404"}),
        v("FIN-011", {"amount": "2000000.00"}), v("FIN-012", {"posting_date": "2099-01-01"}),
    ],
}
# ------------------------------------------------------------------ aviation
CASES["aviation"] = {
    "clean": rows(4, lambda i: dict(
        flight_id="FL1", aircraft_id="N1", timestamp=f"2026-06-01T10:00:0{i}", seq_no=str(i + 1), sensor_id="ALT",
        reading=str(10000 + 50 * i), unit="ft", flight_phase="climb", departure_time="2026-06-01T09:00:00",
        arrival_time="2026-06-01T12:00:00")),
    "violations": [
        v("AV-001", {"unit": ""}), v("AV-002", {"timestamp": "bad"}), v("AV-003", {"reading": "ERR"}),
        v("AV-004", copy_from(1, "timestamp")), v("AV-005", {"timestamp": "2026-06-01T10:01:00"}, 3),
        v("AV-006", {"timestamp": "2026-05-31T10:00:00"}, 1), v("AV-007", {"seq_no": "9"}, 3),
        v("AV-008", {"unit": "m"}), v("AV-009", {"flight_phase": "flying"}),
        v("AV-010", {"arrival_time": "2026-06-01T08:00:00"}),
    ],
}
# ------------------------------------------------------------- manufacturing
CASES["manufacturing"] = {
    "clean": rows(3, lambda i: dict(
        device_id="M1", timestamp=f"2026-08-03T06:0{i}:00", temperature="70.0", temperature_unit="C", vibration="2.0",
        machine_state="RUNNING", output_count="10", last_calibration_date="2026-07-01")),
    "violations": [
        v("MFG-001", {"device_id": ""}), v("MFG-002", {"timestamp": "bad"}), v("MFG-003", {"vibration": "ERR"}),
        v("MFG-004", copy_from(1, "timestamp")), v("MFG-005", {"timestamp": "2026-08-03T07:00:00"}, 2),
        v("MFG-006", {"timestamp": "2026-08-03T05:00:00"}, 1), v("MFG-007", {"machine_state": "BROKEN"}),
        v("MFG-008", {"last_calibration_date": ""}), v("MFG-009", {"last_calibration_date": "2027-01-01"}),
        v("MFG-010", {"machine_state": "STOPPED", "output_count": "4"}), v("MFG-011", {"temperature_unit": "K"}),
        v("MFG-012", {"temperature": "200"}),
    ],
}
# -------------------------------------------------------------------- retail
_ret = [("O1", "SKU-A", "1", "20.00", "", "20.00"), ("O2", "SKU-B", "2", "15.00", "", "30.00"),
        ("O3", "SKU-C", "3", "10.00", "5.00", "25.00")]
CASES["retail"] = {
    "clean": [dict(order_id=o, order_date="2026-09-01", customer_id=f"C{i+1}", sku=s, quantity=q, unit_price=p,
                   discount_amount=d, order_total=t, order_type="SALE", return_reason="", currency="USD",
                   stock_on_hand="50", ship_date="2026-09-03") for i, (o, s, q, p, d, t) in enumerate(_ret)],
    "violations": [
        v("RET-001", {"sku": ""}), v("RET-002", {"order_id": "O2", "sku": "SKU-B"}), v("RET-003", {"unit_price": "ten"}),
        v("RET-004", {"order_date": "bad"}), v("RET-005", {"unit_price": "-5", "order_total": "-5"}),
        v("RET-006", {"quantity": "0", "order_total": "0"}), v("RET-007", {"order_total": "99"}),
        v("RET-008", {"ship_date": "2026-08-01"}), v("RET-009", {"order_type": "BUY"}),
        v("RET-010", {"order_type": "RETURN", "quantity": "-1", "order_total": "-20"}),
        v("RET-011", {"stock_on_hand": "-1"}),
        v("RET-012", copy_from(1, "customer_id", "sku", "order_date", "quantity", "unit_price", order_id="O9")),
        v("RET-013", {"currency": "usd"}),
    ],
}
# ----------------------------------------------------------------- logistics
_lg = [("2026-09-01T08:00:00", "PICKED_UP", ""), ("2026-09-02T08:00:00", "IN_TRANSIT", ""),
       ("2026-09-03T09:00:00", "DELIVERED", "2026-09-03T09:00:00")]
CASES["logistics"] = {
    "clean": [dict(shipment_id="S1", event_time=t, event_type=e, scan_seq=str(i + 1), origin="Indore", destination="Delhi",
                   weight="10", weight_unit="kg", pickup_date="2026-09-01T08:00:00", promised_date="2026-09-05T00:00:00",
                   delivered_date=d) for i, (t, e, d) in enumerate(_lg)],
    "violations": [
        v("LOG-001", {"shipment_id": ""}), v("LOG-002", {"event_time": "bad"}), v("LOG-003", {"event_type": "LOST"}),
        v("LOG-004", {"scan_seq": "7"}, 2), v("LOG-005", {"event_time": "2026-08-30T08:00:00"}, 1),
        v("LOG-006", {"delivered_date": "2026-08-01T00:00:00"}, 2), v("LOG-007", {"delivered_date": ""}, 2),
        v("LOG-008", {"weight_unit": "stone"}), v("LOG-009", {"weight": "-1"}), v("LOG-010", {"weight": "heavy"}),
        v("LOG-011", {"destination": "Indore"}), v("LOG-012", {"event_time": "2026-09-20T09:00:00"}, 2),
    ],
}
# ----------------------------------------------------------------- insurance
CASES["insurance"] = {
    "clean": rows(3, lambda i: dict(
        policy_id=f"P{i+1}", claim_id=f"C{i+1}", customer_id=f"U{i+1}", policy_start="2026-01-01", policy_end="2026-12-31",
        loss_date=f"2026-06-0{i+1}", claim_date="2026-06-10", claim_amount=str(1000 + i), coverage_limit="5000",
        premium="400", claim_status="OPEN")),
    "violations": [
        v("INS-001", {"claim_id": ""}), v("INS-002", {"claim_id": "C2"}), v("INS-003", {"loss_date": "bad"}),
        v("INS-004", {"policy_end": "2025-01-01"}),
        v("INS-005", {"loss_date": "2025-12-01"}), v("INS-006", {"loss_date": "2027-03-01", "claim_date": "2027-03-02"}),
        v("INS-007", {"claim_date": "2026-05-01"}), v("INS-008", {"claim_amount": "9000"}),
        v("INS-009", {"claim_amount": "-5"}), v("INS-010", copy_from(1, "policy_id", "loss_date", "claim_amount")),
        v("INS-011", {"claim_status": "WEIRD"}), v("INS-013", {"premium": "-1"}),
    ],
}
# ----------------------------------------------------------------- education
CASES["education"] = {
    "clean": rows(3, lambda i: dict(
        student_id=f"S100{i+1}", course_id=f"CS10{i+1}", term="2026-FALL", enrollment_date="2026-08-15",
        term_start="2026-08-20", term_end="2026-12-20", grade="A", score="90", attendance_date="2026-09-01",
        attendance_status="PRESENT")),
    "violations": [
        v("EDU-001", {"term": ""}), v("EDU-002", copy_from(1, "student_id", "course_id", "term")),
        v("EDU-003", {"enrollment_date": "bad"}), v("EDU-004", {"term_end": "2026-01-01"}),
        v("EDU-005", {"enrollment_date": "2027-01-05"}), v("EDU-006", {"attendance_date": "2026-07-01"}),
        v("EDU-007", {"attendance_date": "2027-01-15"}), v("EDU-008", {"grade": "Z"}), v("EDU-009", {"score": "150"}),
        v("EDU-010", {"score": "absent"}), v("EDU-011", {"attendance_status": "MAYBE"}), v("EDU-012", {"student_id": "x"}),
    ],
}
# --------------------------------------------------------------- agriculture
CASES["agriculture"] = {
    "clean": rows(3, lambda i: dict(
        field_id="F1", crop="wheat", sensor_id="SN1", timestamp=f"2026-05-01T0{i}:00:00", soil_moisture="25",
        moisture_unit="%", temperature="20", yield_value="4500", yield_unit="kg_per_ha", planting_date="2026-03-01",
        harvest_date="2026-08-01")),
    "violations": [
        v("AGR-001", {"field_id": ""}), v("AGR-002", {"timestamp": "bad"}), v("AGR-003", {"temperature": "hot"}),
        v("AGR-004", copy_from(1, "timestamp")), v("AGR-005", {"timestamp": "2026-05-02T00:00:00"}, 2),
        v("AGR-006", {"harvest_date": "2026-01-01"}), v("AGR-007", {"yield_unit": "t_per_ha"}),
        v("AGR-008", {"soil_moisture": "150"}), v("AGR-009", {"yield_value": "-1"}),
        v("AGR-010", {"yield_value": "50000"}), v("AGR-011", {"crop": ""}),
    ],
}
# ------------------------------------------------------------------- energy
CASES["energy"] = {
    "clean": rows(4, lambda i: dict(
        meter_id="MT1", timestamp=f"2026-07-01T00:{15*i:02d}:00", cumulative_reading=str(100 + 10 * i),
        interval_consumption="10", unit="kWh")),
    "violations": [
        v("ENE-001", {"meter_id": ""}), v("ENE-002", {"timestamp": "bad"}), v("ENE-003", {"cumulative_reading": "x"}),
        v("ENE-004", copy_from(1, "timestamp")), v("ENE-005", {"timestamp": "2026-07-01T05:00:00"}, 3),
        v("ENE-006", {"timestamp": "2026-06-30T23:00:00"}, 1), v("ENE-007", {"cumulative_reading": "50"}, 2),
        v("ENE-008", {"unit": "BTU"}), v("ENE-009", {"interval_consumption": "-5"}),
    ],
}
# ------------------------------------------------------------------- telecom
CASES["telecom"] = {
    "clean": rows(3, lambda i: dict(
        event_id=f"E{i+1}", subscriber_id=f"S{i+1}", device_id=f"D{i+1}", event_time=f"2026-09-01T08:{5*i:02d}:00",
        event_type="CALL", duration_sec="60", cell_id="C1", kpi_seq=str(i + 1), latency_ms="20", packet_loss_pct="0.1",
        throughput_mbps="50")),
    "violations": [
        v("TEL-001", {"event_id": ""}), v("TEL-002", {"event_id": "E2"}),
        v("TEL-003", copy_from(1, "subscriber_id", "event_time", "event_type", "duration_sec")),
        v("TEL-004", {"event_time": "bad"}), v("TEL-005", {"latency_ms": "fast"}), v("TEL-006", {"packet_loss_pct": ""}),
        v("TEL-007", {"packet_loss_pct": "150"}), v("TEL-008", {"duration_sec": "-1"}), v("TEL-009", {"event_type": "PIGEON"}),
        v("TEL-010", {"event_time": "2026-09-01T12:00:00"}, 2), v("TEL-011", {"kpi_seq": "9"}, 2),
    ],
}
# ------------------------------------------------------------- cybersecurity
CASES["cybersecurity"] = {
    "clean": rows(3, lambda i: dict(
        event_id=f"E{i+1}", timestamp=f"2026-09-01T10:0{i}:00", source_host="H1", log_source="L1", user="alice",
        event_type="login_success", severity="info", source_ip=f"10.0.0.{i+5}", bytes_out="100")),
    "violations": [
        v("SEC-001", {"severity": ""}), v("SEC-002", {"timestamp": "bad"}), v("SEC-003", {"severity": "catastrophic"}),
        v("SEC-004", {"event_id": "E2"}),
        v("SEC-005", copy_from(1, "timestamp", "source_host", "event_type", "user", "source_ip")),
        v("SEC-006", {"source_ip": "999.1.1.1"}), v("SEC-007", {"timestamp": "2026-09-01T09:00:00"}, 1),
        v("SEC-008", {"timestamp": "2026-09-01T20:00:00"}, 2), v("SEC-009", {"user": ""}),
        v("SEC-010", {"timestamp": "2099-01-01T00:00:00"}),
    ],
}
# ---------------------------------------------------------------- government
CASES["government"] = {
    "clean": rows(3, lambda i: dict(
        record_id=f"R{i+1}", parent_record_id="" if i == 0 else "R1", applicant_id=f"A{i+1}", service_code="LIC",
        region_code="RG1", postal_code="452001", application_date="2026-09-01", decision_date="2026-09-10",
        status="APPROVED")),
    "violations": [
        v("GOV-001", {"status": ""}), v("GOV-002", {"record_id": "R2"}), v("GOV-003", {"application_date": "bad"}),
        v("GOV-004", {"decision_date": "2026-08-01"}), v("GOV-005", {"status": "MAYBE"}), v("GOV-006", {"service_code": "XXX"}),
        v("GOV-007", {"parent_record_id": "R404"}), v("GOV-008", {"decision_date": ""}), v("GOV-009", {"postal_code": "!"}),
        v("GOV-010", copy_from(1, "applicant_id", "service_code", "application_date")),
        v("GOV-011", {"application_date": "2099-01-01", "decision_date": ""}), v("GOV-012", {"region_code": ""}),
    ],
}
# -------------------------------------------------------------- construction
CASES["construction"] = {
    "clean": rows(3, lambda i: dict(
        project_id="PR1", site_id="S1", work_order_id=f"W{i+1}", parent_work_order_id="" if i == 0 else "W1",
        material="cement", quantity="100", planned_quantity="100", unit="bag", start_date="2026-05-01",
        end_date="2026-05-10", inspection_date="2026-05-11", inspection_result="PASS")),
    "violations": [
        v("CON-001", {"site_id": ""}), v("CON-002", {"work_order_id": "W2"}), v("CON-003", {"start_date": "bad"}),
        v("CON-004", {"end_date": "2026-04-01"}), v("CON-005", {"inspection_date": "2026-04-01"}),
        v("CON-006", {"quantity": "lots"}), v("CON-007", {"quantity": "-5"}), v("CON-008", {"unit": "parsec"}),
        v("CON-009", {"quantity": "200"}), v("CON-010", {"parent_work_order_id": "W404"}),
        v("CON-011", {"inspection_result": "MEH"}), v("CON-012", {"inspection_result": ""}),
    ],
}
# --------------------------------------------------------------- environment
CASES["environment"] = {
    "clean": rows(3, lambda i: dict(
        station_id="ST1", timestamp=f"2026-07-01T0{i}:00:00", temperature="25", temperature_unit="C", humidity_pct="60",
        rainfall_mm="0", calibration_date="2026-01-01")),
    "violations": [
        v("ENV-001", {"station_id": ""}), v("ENV-002", {"timestamp": "bad"}), v("ENV-003", {"rainfall_mm": "wet"}),
        v("ENV-004", copy_from(1, "timestamp")), v("ENV-005", {"timestamp": "2026-07-01T10:00:00"}, 2),
        v("ENV-006", {"timestamp": "2026-06-30T00:00:00"}, 1), v("ENV-007", {"temperature_unit": "X"}),
        v("ENV-008", {"humidity_pct": "120"}), v("ENV-009", {"rainfall_mm": "-1"}), v("ENV-010", {"temperature": "100"}),
        v("ENV-011", {"calibration_date": ""}), v("ENV-012", {"calibration_date": "2027-01-01"}),
    ],
}
# ------------------------------------------------------------------------ hr
CASES["hr"] = {
    "clean": rows(3, lambda i: dict(
        employee_id=f"E{i+1}", department_id="D1", employment_status="ACTIVE", hire_date="2020-01-01",
        termination_date="", effective_date="2026-01-01", pay_period_start="2026-09-01", pay_period_end="2026-09-30",
        pay_date="2026-10-01", pay_amount="3000")),
    "violations": [
        v("HR-001", {"department_id": ""}), v("HR-002", copy_from(1, "employee_id", "effective_date")),
        v("HR-003", {"hire_date": "bad"}), v("HR-004", {"termination_date": "2019-01-01"}),
        v("HR-005", {"pay_period_end": "2026-08-01"}), v("HR-006", {"pay_date": "2026-09-15"}),
        v("HR-007", {"effective_date": "2019-01-01"}), v("HR-008", {"employment_status": "GONE"}),
        v("HR-009", {"employment_status": "TERMINATED", "termination_date": ""}), v("HR-010", {"pay_amount": "lots"}),
        v("HR-011", {"pay_amount": "-100"}),
        v("HR-013", copy_from(1, "employee_id", "pay_period_start", "pay_period_end")),
    ],
}
# --------------------------------------------------------------- real_estate
CASES["real_estate"] = {
    "clean": rows(3, lambda i: dict(
        property_id=f"PR{i+1}", listing_id=f"L{i+1}", listing_date="2026-08-01", sale_date="", price="300000",
        currency="USD", area="1200", area_unit="sqft", property_type="HOUSE", postcode="452001",
        address=f"{i+1} Main St", bedrooms="3")),
    "violations": [
        v("RE-001", {"price": ""}), v("RE-002", {"listing_id": "L2"}),
        v("RE-003", copy_from(1, "address", "postcode", "property_type", "area")), v("RE-004", {"listing_date": "bad"}),
        v("RE-005", {"sale_date": "2026-07-01"}), v("RE-006", {"listing_date": "2099-01-01"}),
        v("RE-007", {"price": "cheap"}), v("RE-008", {"area": "-10"}), v("RE-009", {"area_unit": "sqm"}),
        v("RE-010", {"postcode": "!"}), v("RE-011", {"property_type": "CASTLE"}), v("RE-012", {"bedrooms": "50"}),
        v("RE-013", {"currency": "usd"}),
    ],
}
# ------------------------------------------------------------ transportation
CASES["transportation"] = {
    "clean": rows(4, lambda i: dict(
        trip_id="T1", vehicle_id="V1", timestamp=f"2026-09-01T08:00:{30*i%60:02d}".replace(":00:", f":0{(30*i)//60}:") ,
        latitude="22.7", longitude="75.8", speed_kmh="40", distance="1.0", distance_unit="km",
        trip_start="2026-09-01T08:00:00", trip_end="2026-09-01T09:00:00")),
    "violations": [
        v("TRN-001", {"vehicle_id": ""}), v("TRN-002", {"timestamp": "bad"}), v("TRN-003", {"speed_kmh": "fast"}),
        v("TRN-004", {"trip_end": "2026-09-01T07:00:00"}), v("TRN-005", copy_from(1, "timestamp")),
        v("TRN-006", {"timestamp": "2026-09-01T09:30:00"}, 3), v("TRN-007", {"timestamp": "2026-09-01T07:00:00"}, 1),
        v("TRN-008", {"latitude": "95"}), v("TRN-009", {"longitude": "200"}), v("TRN-010", {"distance_unit": "furlong"}),
        v("TRN-011", {"speed_kmh": "-3"}), v("TRN-012", {"speed_kmh": "400"}), v("TRN-013", {"latitude": ""}),
    ],
}
