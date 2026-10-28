import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from nexora.preprocessing.features import MAX_GENERATED, FeatureConfig, FeatureConfigError, FeatureGenerator, plan_features, preview
from nexora.preprocessing.pipeline import DatetimeFeatures, PreprocessOptions, build_pipeline_steps, feature_names, select_features


@pytest.fixture()
def df():
    rng = np.random.RandomState(0)
    n = 120
    return pd.DataFrame({"a": rng.normal(10, 2, n), "b": rng.uniform(0, 5, n), "neg": rng.normal(0, 1, n), "c": rng.choice(["x", "y", "z"], n),
                         "t": [" ".join(["w"] * (i % 7 + 1)) for i in range(n)], "y": rng.randint(0, 2, n)})


def test_plan_lists_names_and_validates_types(df):
    cfg = FeatureConfig(log=["a"], square=["b"], ratios=[["a", "b"]], differences=[["a", "b"]], interactions=[["a", "b"]], bins=["a"],
                        quantile_rank=["b"], missing_indicators=["a"], frequency_encode=["c"], text_stats=["t"], poly_columns=["a", "b"], poly_degree=2)
    names = [g.name for g in plan_features(cfg, ["a", "b", "neg"], ["c"], ["t"], list(df.columns))]
    assert names == ["log(a)", "b^2", "poly(a*a)", "poly(a*b)", "poly(b*b)", "a*b", "a/b", "a-b", "bin(a)", "rank(b)", "missing(a)", "freq(c)", "len(t)", "words(t)"]
    with pytest.raises(FeatureConfigError, match="numeric"):
        plan_features(FeatureConfig(log=["c"]), ["a"], ["c"], [], list(df.columns))
    with pytest.raises(FeatureConfigError, match="two different"):
        plan_features(FeatureConfig(ratios=[["a", "a"]]), ["a"], [], [], ["a"])
    with pytest.raises(FeatureConfigError, match="degree"):
        plan_features(FeatureConfig(poly_columns=["a"], poly_degree=5), ["a"], [], [], ["a"])


def test_feature_explosion_is_capped():
    cols = [f"v{i}" for i in range(30)]
    pairs = [[a, b] for i, a in enumerate(cols) for b in cols[i + 1:]]
    with pytest.raises(FeatureConfigError, match="limit is"):
        plan_features(FeatureConfig(interactions=pairs), cols, [], [], cols)
    assert len(pairs) > MAX_GENERATED


def test_values_are_correct_and_safe(df):
    d = df.copy()
    d.loc[0, "b"] = 0.0
    cfg = FeatureConfig(log=["neg", "b"], ratios=[["a", "b"]], square=["a"], text_stats=["t"])
    out = FeatureGenerator(cfg.to_dict()).fit(d).transform(d)
    assert np.isnan(out.loc[0, "a/b"]) and out.loc[1, "a/b"] == pytest.approx(d.loc[1, "a"] / d.loc[1, "b"])   # division by zero -> missing
    assert out.loc[3, "a^2"] == pytest.approx(d.loc[3, "a"] ** 2)
    assert out.loc[5, "log(neg)"] == pytest.approx(np.sign(d.loc[5, "neg"]) * np.log1p(abs(d.loc[5, "neg"])))       # signed log learned from training
    assert out.loc[5, "log(b)"] == pytest.approx(np.log1p(d.loc[5, "b"]))
    assert out.loc[2, "words(t)"] == 3 and out.loc[2, "len(t)"] == len(d.loc[2, "t"])
    assert np.isfinite(out.select_dtypes("number").to_numpy(dtype=float)[~np.isnan(out.select_dtypes("number").to_numpy(dtype=float))]).all()


def test_learned_features_use_training_rows_only(df):
    train, test = df.iloc[:80], df.iloc[80:].copy()
    test["c"] = "never_seen"
    test["a"] = 1e6
    cfg = FeatureConfig(bins=["a"], frequency_encode=["c"], quantile_rank=["a"])
    g = FeatureGenerator(cfg.to_dict()).fit(train)
    edges = g.bin_edges_["a"]
    assert np.allclose(edges, np.unique(np.quantile(train["a"], np.linspace(0, 1, 6)))[1:-1])
    out = g.transform(test)
    assert (out["freq(c)"] == 0).all()                                                # unseen category -> 0, not refitted
    assert (out["rank(a)"] == 1.0).all() and (out["bin(a)"] == "bin4").all()         # placed against TRAINING distribution
    g2 = FeatureGenerator(cfg.to_dict()).fit(pd.concat([train, test]))
    e2 = g2.bin_edges_["a"]
    assert len(e2) != len(edges) or not np.allclose(e2, edges)                         # proof the fit depends on what it sees


def test_preview_flags_constant_duplicate_and_invalid(df):
    d = df.assign(zero=0.0, b2=df["b"])
    sample, q = preview(d, FeatureConfig(square=["zero", "b", "b2"], ratios=[["a", "zero"]]), ["a", "b", "zero", "b2"], [], [])
    issues = dict(zip(q["feature"], q["issue"]))
    assert "constant" in issues["zero^2"] and "identical to b^2" in issues["b2^2"] and "valid" in issues["a/zero"]
    assert list(sample.columns) == ["zero^2", "b^2", "b2^2", "a/zero"]


def test_generated_features_flow_through_the_model_pipeline(df):
    opts = PreprocessOptions(features={"square": ["a"], "bins": ["b"], "frequency_encode": ["c"]})
    fs = select_features(df, "y", options=opts)
    pipe = Pipeline(build_pipeline_steps(fs, opts, True) + [("m", LogisticRegression(max_iter=500))]).fit(df[fs.all], df["y"])
    names = feature_names(pipe)
    assert "a^2" in names and "freq(c)" in names and any(n.startswith("bin(b)_") for n in names)


def test_date_parts_and_cyclical_encoding():
    tr = pd.DataFrame({"d": pd.to_datetime(["2026-01-03 10:30", "2026-04-04 00:00", "2026-12-31 23:00"])})
    f = DatetimeFeatures(parts=["year", "quarter", "week", "weekend", "hour", "minute"], cyclical=True).fit(tr)
    out = f.transform(tr)
    names = list(f.get_feature_names_out())
    row = dict(zip(names, out[0]))
    assert row["d__quarter"] == 1 and row["d__weekend"] == 1 and row["d__hour"] == 10 and row["d__minute"] == 30 and row["d__week"] == 1
    assert row["d__month_sin"] == pytest.approx(np.sin(2 * np.pi / 12)) and "d__hour_sin" in names
    legacy = DatetimeFeatures().fit(tr)
    assert list(legacy.get_feature_names_out())[:5] == ["d__year", "d__month", "d__weekday", "d__dayofyear", "d__days_since_start"]
    with pytest.raises(ValueError, match="Unknown date part"):
        DatetimeFeatures(parts=["century"]).fit(tr).transform(tr)
    missing = DatetimeFeatures(parts=["month"]).fit(tr).transform(pd.DataFrame({"d": pd.to_datetime([None, "2026-02-01"])}))
    assert np.isnan(missing[0, 0]) and missing[1, 0] == 2
