from nexora.validation.scoring import compute_quality_summary, live_counts
from tests.helpers import frame, prof, run
import pandas as pd


def test_dimensions_are_reported_separately_and_unmeasured_ones_are_none():
    p = prof([{"id": "R", "type": "required", "fields": ["a"]}, {"id": "N", "type": "numeric", "fields": ["b"]},
              {"id": "U", "type": "unique", "fields": ["id"]}])
    out = run(frame(id=["1", "1", "2", "3"], a=["x", "", "y", "z"], b=["1", "2", "x", "4"]), p)
    d = out.summary["dimensions"]
    assert d["completeness"] == 0.75 and d["validity"] == 0.75 and d["uniqueness"] == 0.5
    assert d["consistency"] is None and d["freshness"] is None
    assert out.summary["composite"] == round((0.75 + 0.75 + 0.5) / 3, 4)
    assert "weighted average" in out.summary["composite_explanation"]


def test_composite_uses_profile_weights_and_ignores_unmeasured():
    p = prof([{"id": "R", "type": "required", "fields": ["a"]}, {"id": "U", "type": "unique", "fields": ["id"]}],
             weights={"completeness": 3, "uniqueness": 1})
    out = run(frame(id=["1", "1"], a=["x", ""]), p)
    assert out.summary["composite"] == round((3 * 0.5 + 1 * 0.0) / 4, 4)


def test_suspicious_values_and_anomalies_do_not_reduce_quality_dimensions():
    p = prof([{"id": "G", "type": "range", "fields": ["a"], "params": {"max": 5}},
              {"id": "S", "type": "statistical_anomaly", "fields": ["a"], "params": {"min_samples": 5}}])
    out = run(frame(a=["1", "2", "1", "2", "1", "2", "100"]), p)
    assert out.summary["findings_total"] >= 2
    assert all(v is None for v in out.summary["dimensions"].values()) and out.summary["composite"] is None
    assert out.summary["findings_by_category"]["Statistical anomaly"] == 1


def test_live_counts_follow_review_status():
    df = pd.DataFrame({"severity": ["High", "Low", "Critical"], "status": ["Open", "Open", "Resolved"]})
    c = live_counts(df)
    assert c == {"total": 3, "unresolved": 2, "unresolved_high_severity": 1, "by_status": {"Open": 2, "Resolved": 1}}
    assert live_counts(pd.DataFrame())["total"] == 0
