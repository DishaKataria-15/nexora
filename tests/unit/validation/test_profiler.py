import pandas as pd

from nexora.validation.profiler import mapping_warnings, profile_dataframe, suggest_mapping
from tests.helpers import prof


def _df(**cols):
    return pd.DataFrame({k: [str(x) for x in v] for k, v in cols.items()})


def test_type_inference_and_counts():
    n = 30
    df = _df(id=[f"ID{i}" for i in range(n)], qty=[str(i) for i in range(n)], price=[f"{i}.5" for i in range(n)],
             when=[f"2026-01-{(i % 28) + 1:02d}" for i in range(n)], flag=["yes", "no"] * 15,
             color=["red", "blue", "green"] * 10, zip=["00123"] * n, empty=[""] * n)
    s = profile_dataframe(df)
    t = {c.name: c.inferred_type for c in s.columns}
    assert t == {"id": "text", "qty": "integer", "price": "float", "when": "datetime", "flag": "boolean",
                 "color": "category", "zip": "text", "empty": "empty"}
    assert any("leading zeros" in i for i in s.column("zip").issues)
    assert s.column("qty").stats["max"] == 29 and s.column("empty").missing_pct == 1.0
    assert s.candidate_dates == ["when"] and "qty" in s.candidate_numeric and "id" in s.candidate_ids


def test_missing_and_duplicate_rows():
    s = profile_dataframe(_df(a=["1", "1", "", "N/A", "2"], b=["x", "x", "y", "z", "w"]))
    assert s.column("a").missing_count == 2 and s.duplicate_rows == 2 and s.n_rows == 5


def test_format_and_unit_hints_are_reported_not_applied():
    df = _df(weight=["12 kg", "15 kg", "9 kg", "11 kg"], d=["2026-01-01", "01/02/2026", "2026-03-04", "05/06/2026"],
             name=["Kg", "kg", "KG", "kg"], pad=[" a", "b ", "c", "d"])
    s = profile_dataframe(df)
    assert any("units attached" in i for i in s.column("weight").issues)
    assert any("variant" in i or "spelling" in i for i in s.column("name").issues)
    assert any("spaces" in i for i in s.column("pad").issues)
    assert df["weight"][0] == "12 kg"        # profiling never modifies data


def test_mixed_date_formats_are_flagged():
    s = profile_dataframe(_df(d=["2026-01-01", "01/02/2026", "2026-03-04", "05/06/2026", "2026-05-05"]))
    c = s.column("d")
    assert c.inferred_type == "datetime" and c.issues


def test_suggest_mapping_exact_alias_fuzzy_none_and_no_double_use():
    p = prof([], fields={"patient_id": {"aliases": ["mrn"]}, "visit_id": {"aliases": ["encounter"]},
                         "heart_rate": {"aliases": ["pulse"]}, "sex": {}})
    got = {m.field_name: (m.column, m.confidence) for m in suggest_mapping(
        ["Patient ID", "ENCOUNTER", "heartrate", "unrelated"], p)}
    assert got["patient_id"] == ("Patient ID", "exact")
    assert got["visit_id"] == ("ENCOUNTER", "alias")
    assert got["heart_rate"] == ("heartrate", "exact") or got["heart_rate"][1] in ("fuzzy", "alias")
    assert got["sex"] == (None, "none")
    cols = [c for c, _ in got.values() if c]
    assert len(cols) == len(set(cols))


def test_fuzzy_match_is_labelled_for_user_confirmation():
    p = prof([], fields={"transaction_amount": {}})
    (m,) = suggest_mapping(["transaction_amnt"], p)
    assert m.confidence == "fuzzy" and m.column == "transaction_amnt"


def test_mapping_warnings_for_wrong_types_and_double_mapping():
    p = prof([], fields={"qty": {"type": "number"}, "when": {"type": "datetime"}, "other": {}})
    s = profile_dataframe(_df(name=["a", "b", "c"], num=["1", "2", "3"]))
    w = mapping_warnings({"qty": "name", "when": "num", "other": "num"}, s, p)
    assert any("expects a number" in x for x in w) and any("date/time" in x for x in w) and any("both" in x for x in w)
    assert mapping_warnings({"qty": "num"}, s, p) == []
