import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from nexora.ml import evaluation as E
from nexora.preprocessing.pipeline import (DatetimeFeatures, FrameCoercer, PreprocessOptions, build_pipeline_steps, feature_names, select_features)


# ---- classification metrics (hand-checkable) -------------------------------------------------------------------------------
def test_binary_metrics_match_hand_calculation():
    yt = ["n", "n", "n", "n", "p", "p", "p", "p"]
    yp = ["n", "n", "n", "p", "p", "p", "n", "n"]
    proba = np.array([[.9, .1], [.8, .2], [.7, .3], [.4, .6], [.2, .8], [.3, .7], [.6, .4], [.9, .1]])   # columns: n, p
    r = E.classification_metrics(yt, yp, proba, ["n", "p"], positive="p")
    m = r["metrics"]
    assert m["accuracy"] == pytest.approx(5 / 8) and m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(f1_score(yt, yp, pos_label="p")) and m["f1"] == pytest.approx(4 / 7)
    assert m["roc_auc"] == pytest.approx(roc_auc_score([0, 0, 0, 0, 1, 1, 1, 1], proba[:, 1]))
    assert r["confusion_matrix"] == [[3, 1], [2, 2]] and r["labels"] == ["n", "p"] and r["positive_class"] == "p"
    assert r["per_class"]["rows"].tolist() == [4, 4]


def test_roc_and_pr_unavailable_when_invalid_not_faked():
    r = E.classification_metrics(["a", "a", "a"], ["a", "a", "a"], np.array([[.9, .1]] * 3), ["a", "b"], "b")
    assert r["metrics"]["roc_auc"] is None and "one class" in r["unavailable"]["roc_auc"]
    r2 = E.classification_metrics(["a", "b"], ["a", "b"], None, ["a", "b"], "b")
    assert r2["metrics"]["roc_auc"] is None and "probabilities" in r2["unavailable"]["pr_auc"]


def test_multiclass_uses_macro_and_needs_all_classes_for_auc():
    classes = ["a", "b", "c"]
    yt, yp = ["a", "b", "c", "a"], ["a", "b", "b", "a"]
    proba = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .6, .3], [.7, .2, .1]])
    r = E.classification_metrics(yt, yp, proba, classes)
    assert r["averaging"].startswith("macro") and r["metrics"]["roc_auc"] is not None and len(r["per_class"]) == 3
    missing = E.classification_metrics(["a", "b"], ["a", "b"], proba[:2], classes)
    assert missing["metrics"]["roc_auc"] is None and "absent" in missing["unavailable"]["roc_auc"]


def test_regression_metrics_and_mathematical_validity():
    r = E.regression_metrics([10, 20, 30], [12, 18, 33])
    assert r["metrics"]["mae"] == pytest.approx((2 + 2 + 3) / 3) and r["metrics"]["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert r["metrics"]["mape"] == pytest.approx(np.mean([.2, .1, .1]) * 100)
    assert r["metrics"]["r2"] == pytest.approx(1 - 17 / 200)
    z = E.regression_metrics([0, 5, 10], [1, 5, 9])
    assert z["metrics"]["mape"] is None and "zero" in z["unavailable"]["mape"]            # MAPE is undefined with zeros: not reported
    c = E.regression_metrics([5, 5, 5], [5, 5, 5])
    assert c["metrics"]["r2"] is None and "identical" in c["unavailable"]["r2"]


def test_clustering_metrics_and_validity(blobs_df):
    from sklearn.cluster import KMeans
    X = blobs_df.values
    labels = KMeans(3, n_init=5, random_state=0).fit_predict(X)
    r = E.clustering_metrics(X, labels)
    assert r["n_clusters"] == 3 and r["metrics"]["silhouette"] > 0.6 and r["metrics"]["davies_bouldin"] < 0.6 and sum(r["sizes"].values()) == len(X)
    one = E.clustering_metrics(X, np.zeros(len(X), dtype=int))
    assert one["metrics"]["silhouette"] is None and "at least 2" in one["unavailable"]["silhouette"]
    noisy = labels.copy()
    noisy[:10] = -1
    assert E.clustering_metrics(X, noisy)["noise_rows"] == 10


def test_interpretation_is_built_from_numbers_and_flags_small_or_weak_results():
    res = E.classification_metrics(["a", "b"] * 5, ["a", "b"] * 5, None, ["a", "b"], "b")
    base = E.classification_metrics(["a", "b"] * 5, ["a"] * 10, None, ["a", "b"], "b")
    text = " ".join(E.interpret("classification", res, base, {"a": 90.0, "b": 10.0}))
    assert "Only 10 rows" in text and "beats that baseline" in text and "imbalanced" in text
    assert "does not clearly beat" in " ".join(E.interpret("classification", base, base, None))
    assert "not accuracy" in " ".join(E.interpret("clustering", {"metrics": {"silhouette": .3}, "n": 500}, None))


def test_metric_catalogue_is_consistent():
    for t in ("classification", "regression", "clustering"):
        for k in E.selection_metrics(t, 3):
            assert k in E.METRIC_HELP and E.metric_direction(k) in ("higher", "lower")
    assert E.default_primary("classification", "severe") == "f1" and E.default_primary("classification", "balanced") == "accuracy"


# ---- preprocessing: learned from training data only ---------------------------------------------------------------------------------
def _pipe(df, cols, model):
    fs = select_features(df, "y", include=cols)
    return Pipeline(build_pipeline_steps(fs, PreprocessOptions(scaling="standard"), True) + [("model", model)]), fs


def test_scaler_and_imputer_statistics_come_from_training_rows_only():
    from sklearn.linear_model import LinearRegression
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan], "y": [1, 2, 3, 4]})
    test = pd.DataFrame({"x": [1000.0, np.nan], "y": [5, 6]})
    pipe, _ = _pipe(train, ["x"], LinearRegression())
    pipe.fit(train[["x"]], train["y"])
    prep = pipe.named_steps["prep"].named_transformers_["num"]
    assert prep.named_steps["impute"].statistics_[0] == 2.0                        # median of training rows, unaffected by 1000
    assert prep.named_steps["scale"].mean_[0] == pytest.approx(2.0)
    out = pipe.named_steps["prep"].transform(pipe.named_steps["coerce"].transform(test[["x"]]))
    assert out[1, 0] == pytest.approx(0.0)                                          # missing test value filled with the TRAIN median
    assert out[0, 0] > 100                                                          # and scaled with the train mean/std, not re-fit


def test_unseen_category_and_missing_columns_are_handled_or_explained():
    from sklearn.linear_model import LogisticRegression
    train = pd.DataFrame({"c": ["a", "b", "a", "b", "a", "b"], "n": [1, 2, 3, 4, 5, 6.0], "y": [0, 1, 0, 1, 0, 1]})
    pipe, _ = _pipe(train, ["c", "n"], LogisticRegression())
    pipe.fit(train[["c", "n"]], train["y"])
    assert len(pipe.predict(pd.DataFrame({"c": ["zzz"], "n": [3.0]}))) == 1          # unknown category ignored, not an error
    with pytest.raises(ValueError, match="missing from the data"):
        pipe.predict(pd.DataFrame({"n": [3.0]}))


def test_pipeline_is_serialisable_and_reproduces_predictions(tmp_path, clf_df):
    from sklearn.ensemble import RandomForestClassifier
    pipe, fs = _pipe(clf_df, ["x1", "x2", "cat"], RandomForestClassifier(50, random_state=0))
    pipe.fit(clf_df[fs.all], clf_df["y"])
    joblib.dump(pipe, tmp_path / "p.joblib")
    again = joblib.load(tmp_path / "p.joblib")
    assert (again.predict(clf_df[fs.all]) == pipe.predict(clf_df[fs.all])).all()
    names = feature_names(pipe)
    assert "x1" in names and any(n.startswith("cat_") for n in names)


def test_datetime_features_learn_origin_from_training_and_handle_missing():
    tr = pd.DataFrame({"d": pd.to_datetime(["2026-01-01", "2026-01-03", None])})
    f = DatetimeFeatures().fit(tr)
    out = f.transform(pd.DataFrame({"d": pd.to_datetime(["2026-01-11", None])}))
    names = list(f.get_feature_names_out())
    assert out[0, names.index("d__days_since_start")] == 10.0 and np.isnan(out[1]).all()


def test_feature_selection_excludes_with_reasons_and_never_includes_target():
    df = pd.DataFrame({"id": [f"R{i}" for i in range(50)], "const": 1, "empty": np.nan, "num": np.arange(50) * 1.0, "cat": ["a", "b"] * 25,
                       "when": pd.date_range("2026-01-01", periods=50), "y": [0, 1] * 25})
    fs = select_features(df, "y", exclude=["cat"])
    assert fs.numeric == ["num"] and fs.datetime == ["when"] and fs.categorical == []
    assert "identifier" in fs.excluded["id"] and "Same value" in fs.excluded["const"] and fs.excluded["empty"] == "No values" and fs.excluded["cat"] == "Excluded by you"
    assert "y" not in fs.all and "y" not in fs.excluded
    with pytest.raises(ValueError, match="No usable feature"):
        build_pipeline_steps(select_features(df[["const", "y"]], "y"), PreprocessOptions(), False)


def test_coercer_handles_nullable_boolean_and_int_columns():
    df = pd.DataFrame({"b": pd.array([True, None, False], dtype="boolean"), "i": pd.array([1, None, 3], dtype="Int64")})
    out = FrameCoercer(["b", "i"]).transform(df)
    assert out["i"].dtype == "float64" and out["b"].tolist()[0] == "true" and pd.isna(out["b"].iloc[1])


def test_text_column_is_used_only_when_enabled_and_is_fitted_on_training_text():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(0)
    words_pos, words_neg = ["great", "love", "excellent"], ["awful", "broken", "refund"]
    texts = [" ".join(rng.choice(words_pos if i % 2 else words_neg, 12)) + " " + "filler " * 10 for i in range(120)]
    df = pd.DataFrame({"review": texts, "y": [i % 2 for i in range(120)]})
    off = select_features(df, "y")
    assert "review" in off.excluded and "text" in off.excluded["review"].lower() and off.text == []
    on = select_features(df, "y", options=PreprocessOptions(text_columns=["review"]))
    assert on.text == ["review"]
    pipe = Pipeline(build_pipeline_steps(on, PreprocessOptions(text_columns=["review"]), True) + [("model", LogisticRegression())])
    pipe.fit(df.iloc[:90][["review"]], df.iloc[:90]["y"])
    vocab = pipe.named_steps["prep"].named_transformers_["text0"].named_steps["tfidf"].vocabulary_
    assert "great" in vocab and "zzzunseen" not in vocab and pipe.score(df.iloc[90:][["review"]], df.iloc[90:]["y"]) > 0.9
