"""Explainability and feature selection inside experiments."""
import numpy as np
import pandas as pd
import pytest

from nexora.ml import explain as XP
from nexora.ml import model_registry as MR
from nexora.ml import training as T
from nexora.preprocessing.pipeline import PreprocessOptions

CV_SPLIT = {"strategy": "stratified", "train": 0.8, "val": 0.0, "test": 0.2, "seed": 5, "group_col": None, "time_col": None}


@pytest.fixture()
def signal_df():
    rng = np.random.RandomState(4)
    n = 400
    df = pd.DataFrame({"strong": rng.normal(size=n), "noise1": rng.normal(size=n), "noise2": rng.normal(size=n), "noise3": rng.normal(size=n)})
    df["y"] = np.where(df["strong"] + 0.2 * rng.normal(size=n) > 0, "pos", "neg")
    return df


def _run(df, model, **kw):
    return T.run_selection(df, T.ExperimentConfig(task="classification", target="y", models=[model], split=CV_SPLIT, cv_folds=3, seed=1, **kw))


@pytest.mark.parametrize("model,kind", [("random_forest", "impurity"), ("logistic_regression", "coefficient"), ("gradient_boosting", None)])
def test_native_importance_ranks_the_real_signal_first(signal_df, model, kind):
    exp = _run(signal_df, model)
    ni = exp.native_importance
    if kind is None:                                   # HistGradientBoosting has no native importance; nothing is invented
        assert ni is None
        return
    assert ni["kind"] == kind and ni["rows"][0]["feature"].endswith("strong")
    if kind == "coefficient":
        assert ni["rows"][0]["direction"] == "raises the prediction"


@pytest.mark.skipif(not XP.SHAP_AVAILABLE, reason="shap not installed")
@pytest.mark.parametrize("model", ["random_forest", "gradient_boosting", "logistic_regression"])
def test_shap_ranks_the_real_signal_first(signal_df, model):
    exp = _run(signal_df, model, shap=True)
    assert exp.shap["available"], exp.shap.get("reason")
    assert exp.shap["rows"][0]["feature"].endswith("strong") and exp.shap["n_rows"] <= XP.SHAP_MAX_ROWS


def test_shap_declines_unsupported_models_with_a_reason(signal_df, monkeypatch):
    exp = _run(signal_df, "knn", shap=True)
    assert exp.shap["available"] is False and "tree and linear" in exp.shap["reason"]
    monkeypatch.setattr(XP, "SHAP_AVAILABLE", False)
    assert "pip install shap" in XP.shap_importance(exp._state["pipeline"], signal_df.drop(columns="y"))["reason"]


def test_feature_selection_runs_inside_the_experiment_and_is_reported(signal_df):
    opts = PreprocessOptions(selection={"method": "kbest", "k": 2}).to_dict()
    exp = _run(signal_df, "logistic_regression", preprocess=opts)
    sel = exp.pipeline_report["selection"]
    kept = [r["feature"] for r in sel["features"] if r["kept"]] if "features" in sel else sel.get("kept")
    assert any(str(k).endswith("strong") for k in kept) and len(kept) == 2
    assert len(exp.native_importance["rows"]) == 2


def test_reduction_inside_the_experiment_reports_variance(signal_df):
    opts = PreprocessOptions(reduction={"method": "pca", "n_components": 2}).to_dict()
    exp = _run(signal_df, "logistic_regression", preprocess=opts)
    red = exp.pipeline_report["reduction"]
    assert red["outputs"] == 2 and red["inputs"] == 4 and 0 < red["explained_total"] <= 1
    assert exp.candidates[0].status == "ok"


def test_explanations_use_no_test_rows(signal_df):
    a = _run(signal_df, "random_forest", shap=XP.SHAP_AVAILABLE)
    test = a._state["split"].test
    bad = signal_df.copy()
    bad.loc[test, "noise1"] = 1e6          # labels are left alone so the stratified split itself is unchanged
    bad.loc[test, "strong"] = -bad.loc[test, "strong"]
    b = _run(bad, "random_forest", shap=XP.SHAP_AVAILABLE)
    assert a.native_importance == b.native_importance
    if XP.SHAP_AVAILABLE:
        assert [r["feature"] for r in a.shap["rows"]] == [r["feature"] for r in b.shap["rows"]]
        assert [r["mean_abs_shap"] for r in a.shap["rows"]] == pytest.approx([r["mean_abs_shap"] for r in b.shap["rows"]])


@pytest.mark.skipif(not MR.XGB_AVAILABLE, reason="xgboost not installed")
def test_native_importance_unwraps_the_xgboost_label_wrapper(signal_df):
    assert _run(signal_df, "xgboost").native_importance["rows"][0]["feature"].endswith("strong")
