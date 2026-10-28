import numpy as np
import pandas as pd
import pytest

from nexora.ml import imbalance as IMB
from nexora.ml.splitting import SplitConfig, SplitError, make_split, recommend_split
from nexora.preprocessing import leakage_checks as L


# ---- splitting ---------------------------------------------------------------------------------------------------------
def test_recommend_split_depends_on_size():
    small, text_s = recommend_split(300)
    big, text_b = recommend_split(5000)
    assert (small.val, small.test) == (0.0, 0.2) and (big.train, big.val, big.test) == (0.7, 0.15, 0.15)
    assert "cross-validation" in text_s and "validation" in text_b


@pytest.mark.parametrize("strategy", ["random", "stratified"])
def test_split_is_disjoint_complete_and_reproducible(clf_df, strategy):
    cfg = SplitConfig(strategy, 0.7, 0.15, 0.15, seed=7)
    a = make_split(len(clf_df), cfg, y=clf_df["y"])
    b = make_split(len(clf_df), cfg, y=clf_df["y"])
    allidx = np.concatenate([a.train, a.val, a.test])
    assert sorted(allidx) == list(range(len(clf_df))) and len(set(allidx)) == len(allidx)
    assert all(np.array_equal(getattr(a, k), getattr(b, k)) for k in ("train", "val", "test"))
    c = make_split(len(clf_df), SplitConfig(strategy, 0.7, 0.15, 0.15, seed=8), y=clf_df["y"])
    assert not np.array_equal(a.test, c.test)                                              # different seed, different split


def test_stratified_keeps_class_share(clf_df):
    r = make_split(len(clf_df), SplitConfig("stratified", 0.7, 0.15, 0.15, seed=1), y=clf_df["y"])
    base = (clf_df["y"] == "yes").mean()
    for part in (r.train, r.val, r.test):
        assert abs((clf_df["y"].iloc[part] == "yes").mean() - base) < 0.03


def test_stratified_falls_back_with_visible_warning_when_a_class_is_tiny():
    y = pd.Series(["a"] * 50 + ["b"] * 2)
    r = make_split(len(y), SplitConfig("stratified", 0.8, 0.0, 0.2, seed=1), y=y)
    assert r.effective_strategy == "random" and any("too few" in w for w in r.warnings)


def test_group_split_keeps_each_entity_on_one_side():
    groups = pd.Series(np.repeat(np.arange(60), 5))
    r = make_split(300, SplitConfig("group", 0.7, 0.15, 0.15, seed=3, group_col="g"), groups=groups)
    sets = [set(groups.iloc[i]) for i in (r.train, r.val, r.test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_chronological_split_orders_by_time_even_if_rows_are_shuffled():
    t = pd.Series(pd.date_range("2026-01-01", periods=100)).sample(frac=1, random_state=0).reset_index(drop=True)
    r = make_split(100, SplitConfig("chronological", 0.7, 0.15, 0.15, time_col="d"), times=t)
    assert t.iloc[r.train].max() < t.iloc[r.val].min() <= t.iloc[r.val].max() < t.iloc[r.test].min()


@pytest.mark.parametrize("cfg,msg", [
    (SplitConfig("random", 0.5, 0.2, 0.2), "add up"), (SplitConfig("random", 0.9, 0.1, 0.0), "test"),
    (SplitConfig("group", 0.8, 0, 0.2), "entity"), (SplitConfig("chronological", 0.8, 0, 0.2), "date column"),
    (SplitConfig("bogus"), "Unknown")])
def test_invalid_split_configs_raise_readable_errors(cfg, msg):
    with pytest.raises(SplitError, match=msg):
        make_split(100, cfg)


def test_split_rejects_tiny_data_and_bad_time_or_group_values():
    with pytest.raises(SplitError, match="too few"):
        make_split(5, SplitConfig())
    with pytest.raises(SplitError, match="invalid dates"):
        make_split(20, SplitConfig("chronological", 0.8, 0, 0.2, time_col="d"), times=pd.Series(["2026-01-01"] * 19 + ["bad"]))
    with pytest.raises(SplitError, match="at least 3"):
        make_split(20, SplitConfig("group", 0.8, 0, 0.2, group_col="g"), groups=pd.Series(["a"] * 10 + ["b"] * 10))


# ---- leakage -------------------------------------------------------------------------------------------------------------------
def test_flags_perfect_categorical_predictor_and_missingness_leak():
    rng = np.random.RandomState(0)
    y = rng.choice(["stay", "leave"], 400, p=[0.8, 0.2])
    df = pd.DataFrame({"noise": rng.normal(size=400), "copy": np.where(y == "leave", "L", "S"),
                       "reason": pd.Series(np.where(y == "leave", "price", None), dtype=object), "y": y})
    w = L.check_target_leakage(df, "y", "classification", ["noise", "copy", "reason"])
    flagged = {c for x in w for c in x.columns}
    assert flagged == {"copy", "reason"} and all(x.suggest_exclude for x in w)


def test_flags_numeric_near_copy_of_regression_target(reg_df):
    reg_df["leak"] = reg_df["t"] * 2 + 0.001
    w = L.check_target_leakage(reg_df, "t", "regression", ["a", "b", "g", "leak"])
    assert [c for x in w for c in x.columns] == ["leak"]


def test_honest_features_are_not_flagged(clf_df):
    assert L.check_target_leakage(clf_df, "y", "classification", ["x1", "x2", "cat"]) == []


def test_post_outcome_names_and_duplicate_and_split_warnings():
    assert L.check_post_outcome_names(["age", "refund_amount", "final_status"], "churn")[0].columns == ["refund_amount", "final_status"]
    df = pd.DataFrame({"a": [1, 1, 2], "y": [0, 0, 1]})
    assert "1 row" in L.check_duplicates(df, ["a"], "y")[0].message
    assert L.duplicates_across_splits(pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [2, 3]})) == 1
    kinds = {w.kind for w in L.check_split_strategy(df, "random", ["when"], ["customer_id"])}
    assert kinds == {"time_split", "group_split"} and L.check_split_strategy(df, "chronological", ["when"], []) == []


# ---- imbalance ---------------------------------------------------------------------------------------------------------------
def test_class_distribution_and_levels():
    info = IMB.class_distribution(pd.Series(["a"] * 90 + ["b"] * 10))
    assert info.table["rows"].tolist() == [90, 10] and info.table["share %"].tolist() == [90.0, 10.0]
    assert info.level == "moderate" and info.ratio == pytest.approx(10 / 90) and info.smallest_class == "b"
    assert IMB.class_distribution(pd.Series(["a"] * 5 + ["b"] * 5)).level == "balanced"
    assert "mislead" in IMB.metric_advice(info) and "reasonable" in IMB.metric_advice(IMB.class_distribution(pd.Series(["a", "b"] * 5)))


def test_options_reflect_data_compatibility():
    info = IMB.class_distribution(pd.Series(["a"] * 90 + ["b"] * 10))
    num = {o.key: o for o in IMB.strategy_options(info, all_numeric_features=True)}
    cat = {o.key: o for o in IMB.strategy_options(info, all_numeric_features=False)}
    assert num["smote"].available and not cat["smote"].available and "categories" in cat["smote"].why
    assert num["class_weight"].recommended and not num["none"].recommended
    tiny = {o.key: o for o in IMB.strategy_options(IMB.class_distribution(pd.Series(["a"] * 50 + ["b"] * 3)), True)}
    assert any("only 3" in w for w in tiny["random_over"].warnings + tiny["smote"].warnings)


def test_missing_dependency_is_reported_not_crashed(monkeypatch):
    monkeypatch.setattr(IMB, "IMBLEARN", False)
    info = IMB.class_distribution(pd.Series(["a"] * 90 + ["b"] * 10))
    opts = {o.key: o for o in IMB.strategy_options(info, True)}
    assert not opts["random_over"].available and not opts["smote"].available and "imbalanced-learn" in opts["smote"].why
    with pytest.raises(RuntimeError, match="imbalanced-learn"):
        IMB.build_sampler("smote", pd.Series(["a", "b"]), 0)
    assert IMB.build_sampler("class_weight", pd.Series(["a", "b"]), 0) is None


@pytest.mark.skipif(not IMB.IMBLEARN, reason="imbalanced-learn not installed")
def test_samplers_build_and_smote_neighbours_adapt_to_tiny_classes():
    y = pd.Series(["a"] * 50 + ["b"] * 3)
    assert IMB.build_sampler("smote", y, 0).k_neighbors == 2
    assert IMB.build_sampler("random_over", y, 0) is not None and IMB.build_sampler("random_under", y, 0) is not None
    with pytest.raises(ValueError):
        IMB.build_sampler("nonsense", y, 0)
