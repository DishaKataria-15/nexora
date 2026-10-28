"""All 18 domain profiles: loading, documentation, clean data and one targeted violation per rule."""
import pandas as pd
import pytest

from tests.fixtures.profile_cases import CASES
from tests.helpers import identity_mapping, rows_of, result_of, run

DOMAINS = ["healthcare", "finance", "aviation", "manufacturing", "logistics", "retail", "insurance", "education",
           "agriculture", "energy", "telecom", "cybersecurity", "government", "construction", "environment", "hr",
           "real_estate", "transportation"]
FLAGSHIP = {"healthcare", "finance", "aviation", "manufacturing", "retail"}


def test_all_domains_load(builtin_profiles):
    assert sorted(k for k in builtin_profiles if k != "generic") == sorted(DOMAINS)
    assert builtin_profiles["generic"].rules == []  # the generic profile has no domain rules by design
    assert set(CASES) == set(DOMAINS)


@pytest.mark.parametrize("domain", DOMAINS)
def test_profile_is_documented_and_not_a_placeholder(builtin_profiles, domain):
    p = builtin_profiles[domain]
    assert len(p.rules) >= 9, "each profile needs a meaningful rule set"
    assert p.guardrail and p.description and p.version
    assert (p.coverage == "tested_end_to_end") == (domain in FLAGSHIP)
    for r in p.rules:
        assert r.explanation and r.recommendation and r.rationale, r.rule_id
        assert r.rule_id.split("-")[0].isalpha()


def test_profiles_are_not_copies_of_each_other(builtin_profiles):
    sigs = {d: frozenset(r.name for r in p.rules) for d, p in builtin_profiles.items()}
    assert len(set(sigs.values())) == len(sigs)
    prefixes = {r.rule_id.rsplit("-", 1)[0] for p in builtin_profiles.values() for r in p.rules}
    assert len(prefixes) == 18


def _frame(rows):
    return pd.DataFrame(rows).astype(str)


@pytest.mark.parametrize("domain", DOMAINS)
def test_clean_data_has_no_hard_findings(builtin_profiles, domain):
    p = builtin_profiles[domain]
    df = _frame(CASES[domain]["clean"])
    out = run(df, p, identity_mapping(df, p))
    bad = [(f.rule_id, f.row_ref, f.observed_value) for f in out.findings
           if f.category not in ("Informational",) and f.rule_id != "SCHEMA-001"]
    assert bad == []
    assert not [r for r in out.rule_results if r.status == "error"]


def _violation_params():
    return [pytest.param(d, rule, ch, row, id=rule) for d in DOMAINS for rule, ch, row in CASES[d]["violations"]]


@pytest.mark.parametrize("domain,rule_id,changes,row", _violation_params())
def test_rule_flags_its_violation(builtin_profiles, domain, rule_id, changes, row):
    p = builtin_profiles[domain]
    clean = CASES[domain]["clean"]
    rows = [dict(r) for r in clean]
    rows[row].update(changes(clean) if callable(changes) else changes)
    df = _frame(rows)
    out = run(df, p, identity_mapping(df, p))
    res = result_of(out, rule_id)
    assert res.status == "failed", f"{rule_id} status {res.status}: {res.message}"
    assert row + 2 in rows_of(out, rule_id)


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_enabled_rule_has_a_test_case(builtin_profiles, domain):
    """Rules not exercised by a violation case must be exempt for a stated reason."""
    p = builtin_profiles[domain]
    covered = {rule for rule, _, _ in CASES[domain]["violations"]}
    exempt = {r.rule_id for r in p.rules if r.rule_type in ("statistical_anomaly", "freshness") or not r.enabled
              or r.params.get("reference_table")}
    missing = [r.rule_id for r in p.rules if r.rule_id not in covered | exempt]
    assert missing == []


def test_reference_table_rules_skip_without_file_and_run_with_it(builtin_profiles):
    p = builtin_profiles["insurance"]
    df = _frame(CASES["insurance"]["clean"])
    out = run(df, p, identity_mapping(df, p))
    assert result_of(out, "INS-012").status == "skipped"
    out = run(df, p, identity_mapping(df, p), reference_tables={"policies": pd.DataFrame({"policy_id": ["P1", "P2"]})})
    res = result_of(out, "INS-012")
    assert res.status == "failed" and rows_of(out, "INS-012") == [4]  # P3 is not in the register
    h = builtin_profiles["hr"]
    df = _frame(CASES["hr"]["clean"])
    out = run(df, h, identity_mapping(df, h), reference_tables={"departments": pd.DataFrame({"department_id": ["D1"]})})
    assert result_of(out, "HR-012").status == "passed"


@pytest.mark.parametrize("domain,rule_id", [("healthcare", "HC-002"), ("aviation", "AV-011"), ("aviation", "AV-013"),
                                            ("finance", "FIN-013")])
def test_rules_can_be_enabled_or_disabled_per_run(builtin_profiles, domain, rule_id):
    p = builtin_profiles[domain]
    df = _frame(CASES[domain]["clean"])
    default_on = p.rule(rule_id).enabled
    out = run(df, p, identity_mapping(df, p), rule_overrides={rule_id: {"enabled": not default_on}})
    status = result_of(out, rule_id).status
    assert (status == "disabled") if default_on else (status != "disabled")


def test_healthcare_unique_visit_rule_when_enabled(builtin_profiles):
    p = builtin_profiles["healthcare"]
    rows = [dict(r) for r in CASES["healthcare"]["clean"]]
    rows[1]["visit_id"] = rows[0]["visit_id"]
    df = _frame(rows)
    assert result_of(run(df, p, identity_mapping(df, p)), "HC-002").status == "disabled"
    out = run(df, p, identity_mapping(df, p), rule_overrides={"HC-002": {"enabled": True}})
    assert rows_of(out, "HC-002") == [2, 3]


def test_aviation_range_limits_are_off_until_configured(builtin_profiles):
    p = builtin_profiles["aviation"]
    rows = [dict(r) for r in CASES["aviation"]["clean"]]
    rows[0]["reading"] = "99999999"
    df = _frame(rows)
    assert result_of(run(df, p, identity_mapping(df, p)), "AV-011").status == "disabled"
    out = run(df, p, identity_mapping(df, p), rule_overrides={"AV-011": {"enabled": True}})
    assert rows_of(out, "AV-011") == [2]
    assert {f.category for f in out.findings if f.rule_id == "AV-011"} == {"Suspicious"}


def test_energy_rollover_is_information_not_error(builtin_profiles):
    p = builtin_profiles["energy"]
    rows = [dict(r) for r in CASES["energy"]["clean"]]
    rows[1]["cumulative_reading"] = "99990"
    rows[2]["cumulative_reading"] = "5"
    rows[3]["cumulative_reading"] = "15"
    df = _frame(rows)
    out = run(df, p, identity_mapping(df, p))
    ene7 = [f for f in out.findings if f.rule_id == "ENE-007"]
    assert [(f.row_ref, f.category, f.severity) for f in ene7] == [(4, "Informational", "Info")]
    assert "rollover" in ene7[0].explanation


def test_freshness_rules_use_the_fixed_as_of_time(builtin_profiles):
    p = builtin_profiles["cybersecurity"]
    df = _frame(CASES["cybersecurity"]["clean"])
    on = {"SEC-011": {"enabled": True}}
    stale = run(df, p, identity_mapping(df, p), rule_overrides=on, as_of="2026-09-03T00:00:00+00:00")
    assert result_of(stale, "SEC-011").status == "failed"
    fresh = run(df, p, identity_mapping(df, p), rule_overrides=on, as_of="2026-09-01T11:00:00+00:00")
    assert result_of(fresh, "SEC-011").status == "passed"
    assert fresh.summary["dimensions"]["freshness"] == 1.0 and stale.summary["dimensions"]["freshness"] == 0.0
