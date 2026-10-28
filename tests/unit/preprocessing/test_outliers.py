import numpy as np
import pandas as pd
import pytest

from nexora.cleaning.transformations import ExecContext, op_treat_outliers
from nexora.preprocessing.outliers import (OutlierConfig, OutlierConfigError, OutlierTreatment, apply_treatment, column_bounds, detect,
                                           validate_config)

CX = ExecContext(null_tokens={""})


@pytest.fixture()
def df():
    base = list(np.linspace(10, 20, 40))
    return pd.DataFrame({"x": base + [100.0, -50.0], "y": list(range(42)), "cat": ["a"] * 42, "flag": [True] * 42})


# ---- bounds per method -----------------------------------------------------------------------------------------------------
def test_iqr_bounds_match_hand_calculation():
    v = np.arange(1, 101, dtype=float)
    lo, hi, _ = column_bounds(v, OutlierConfig("iqr", iqr_k=1.5))
    q1, q3 = np.percentile(v, [25, 75])
    assert (lo, hi) == pytest.approx((q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)))


def test_zscore_and_modified_zscore_bounds():
    v = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    lo, hi, _ = column_bounds(v, OutlierConfig("zscore", z_threshold=2))
    assert (lo, hi) == pytest.approx((v.mean() - 2 * v.std(ddof=1), v.mean() + 2 * v.std(ddof=1)))
    lo, hi, _ = column_bounds(v, OutlierConfig("modified_zscore", mz_threshold=3.5))
    mad = np.median(np.abs(v - np.median(v)))
    assert hi == pytest.approx(np.median(v) + 3.5 * mad / 0.6745)


@pytest.mark.parametrize("method,vals,note", [("iqr", [5.0] * 10, "zero"), ("zscore", [5.0] * 10, "zero"), ("modified_zscore", [1, 1, 1, 1, 1, 1, 9, 10.0], "zero"),
                                               ("iqr", [1.0, np.nan], "too few")])
def test_degenerate_columns_never_flag_and_explain_why(method, vals, note):
    lo, hi, msg = column_bounds(np.array(vals, dtype=float), OutlierConfig(method))
    assert lo == -np.inf and hi == np.inf and note in msg


def test_detect_reports_counts_and_ignores_categorical_and_boolean(df):
    r = detect(df, OutlierConfig("iqr", columns=["x", "cat", "flag", "y"]))
    assert r.columns == ["x", "y"] and r.per_column == {"x": 2, "y": 0} and r.n_flagged == 2 and r.pct_flagged == pytest.approx(4.76)
    assert set(r.row_mask[r.row_mask].index) == {40, 41}
    with pytest.raises(OutlierConfigError, match="numeric"):
        detect(df, OutlierConfig("iqr", columns=["cat"]))


def test_isolation_forest_is_multivariate_and_seeded():
    rng = np.random.RandomState(0)
    a = rng.normal(size=300)
    d = pd.DataFrame({"a": a, "b": a + rng.normal(0, 0.05, 300)})
    d.loc[0, ["a", "b"]] = [2.0, -2.0]          # each value is ordinary on its own; the combination is not
    cfg = OutlierConfig("isolation_forest", contamination=0.01, seed=1)
    r1, r2 = detect(d, cfg), detect(d, cfg)
    assert r1.multivariate and r1.row_mask[0] and r1.row_mask.equals(r2.row_mask) and r1.scores is not None
    assert not detect(d, OutlierConfig("zscore", z_threshold=3)).row_mask[0]
    with pytest.raises(OutlierConfigError, match="whole rows"):
        apply_treatment(d, r1, cfg, "clip")


def test_domain_bounds_use_profile_limits(df):
    r = detect(df, OutlierConfig("domain_bounds", bounds={"x": (0, 50)}))
    assert r.columns == ["x"] and r.per_column["x"] == 2
    with pytest.raises(OutlierConfigError, match="no numeric limits"):
        detect(df, OutlierConfig("domain_bounds"))


@pytest.mark.parametrize("cfg,msg", [(OutlierConfig("iqr", iqr_k=0.1), "IQR"), (OutlierConfig("zscore", z_threshold=0.5), "Z-score"),
                                     (OutlierConfig("isolation_forest", contamination=0.9), "Contamination"), (OutlierConfig("bogus"), "Unknown"),
                                     (OutlierConfig("iqr", winsor_limits=(0.5, 0.6)), "Winsorizing"), (OutlierConfig("isolation_forest", n_estimators=1), "trees")])
def test_parameter_ranges_are_validated(cfg, msg):
    with pytest.raises(OutlierConfigError, match=msg):
        validate_config(cfg)


# ---- treatments --------------------------------------------------------------------------------------------------------------
def test_treatments_change_only_flagged_values_and_never_the_input(df):
    cfg = OutlierConfig("iqr", columns=["x"])
    r = detect(df, cfg)
    snap = df.copy()
    clip, s1 = apply_treatment(df, r, cfg, "clip")
    lo, hi = r.bounds["x"]
    assert clip.loc[40, "x"] == pytest.approx(hi) and clip.loc[41, "x"] == pytest.approx(lo) and s1["values_modified"] == {"x": 2}
    assert clip.loc[:39, "x"].equals(df.loc[:39, "x"])
    med, s2 = apply_treatment(df, r, cfg, "median")
    assert med.loc[40, "x"] == pytest.approx(df.loc[:39, "x"].median())
    rem, s3 = apply_treatment(df, r, cfg, "remove")
    assert len(rem) == 40 and s3["rows_removed"] == [40, 41]
    flag, s4 = apply_treatment(df, r, cfg, "flag")
    assert flag["x__outlier_flag"].sum() == 2 and flag["x"].equals(df["x"])
    win, s5 = apply_treatment(df, r, OutlierConfig("iqr", columns=["x"], winsor_limits=(0.05, 0.95)), "winsorize")
    assert win["x"].max() == pytest.approx(df["x"].quantile(0.95))
    rep, s6 = apply_treatment(df, r, cfg, "report")
    assert rep.equals(df)
    pd.testing.assert_frame_equal(df, snap)


def test_log_refuses_negative_columns_and_large_removals_warn(df):
    cfg = OutlierConfig("iqr", columns=["x"])
    out, s = apply_treatment(df, detect(df, cfg), cfg, "log")
    assert out["x"].equals(df["x"]) and "negative" in s["warnings"][0]
    many = pd.DataFrame({"x": [1.0] * 20 + [100.0] * 3 + list(range(20))})
    r = detect(many, OutlierConfig("zscore", z_threshold=1.0))
    _, s = apply_treatment(many, r, OutlierConfig("zscore", z_threshold=1.0), "remove")
    assert s["warnings"] and "large share" in s["warnings"][0]


def test_cleaning_executor_logs_counts_and_removed_rows(df):
    out, res = op_treat_outliers(df, {"config": OutlierConfig("iqr", columns=["x"]).to_dict(), "treatment": "remove"}, CX)
    assert len(out) == 40 and res.removed_rows == [42, 43] and "2 row(s)" in res.notes and res.status == "applied"
    out2, res2 = op_treat_outliers(df, {"config": OutlierConfig("iqr", columns=["x"]).to_dict(), "treatment": "report"}, CX)
    assert out2.equals(df) and res2.status == "no_change"


# ---- pipeline transformer: learned on training rows only -------------------------------------------------------------------------
def test_pipeline_treatment_learns_bounds_from_training_rows_only():
    rng = np.random.RandomState(0)
    train = rng.normal(0, 1, (200, 2))
    test = np.array([[1e6, 0.0], [-1e6, 0.0]])
    t = OutlierTreatment("iqr", "clip").fit(train)
    expected = [column_bounds(train[:, j], OutlierConfig("iqr")) for j in range(2)]
    assert np.allclose(t.lower_, [e[0] for e in expected]) and np.allclose(t.upper_, [e[1] for e in expected])
    out = t.transform(test)
    assert out[0, 0] == pytest.approx(t.upper_[0]) and out[1, 0] == pytest.approx(t.lower_[0])   # applied with TRAINING bounds
    m = OutlierTreatment("zscore", "median").fit(train)
    assert m.transform(test)[0, 0] == pytest.approx(m.median_[0])


def test_pipeline_treatment_rejects_row_level_choices():
    with pytest.raises(OutlierConfigError, match="cannot be removed"):
        OutlierTreatment("iqr", "remove").fit(np.zeros((5, 1)))
    with pytest.raises(OutlierConfigError, match="IQR, Z-score"):
        OutlierTreatment("isolation_forest", "clip").fit(np.zeros((5, 1)))
