"""Model registry: every model trains through the real pipeline, metadata is consistent, settings are validated."""
import joblib
import numpy as np
import pandas as pd
import pytest

from nexora.ml import model_registry as MR
from nexora.ml import training as T

CV_SPLIT = {"strategy": "stratified", "train": 0.8, "val": 0.0, "test": 0.2, "seed": 5, "group_col": None, "time_col": None}
RSPLIT = {"strategy": "random", "train": 0.7, "val": 0.15, "test": 0.15, "seed": 5, "group_col": None, "time_col": None}

CLF = [k for k, s in MR.REGISTRY.items() if s.task == "classification" and s.available]
REG = [k for k, s in MR.REGISTRY.items() if s.task == "regression" and s.available]
CLU = [k for k, s in MR.REGISTRY.items() if s.task == "clustering"]


def test_required_models_are_registered_with_complete_metadata():
    need = {"svc", "knn", "gaussian_nb", "multinomial_nb", "bernoulli_nb", "adaboost", "ridge", "lasso", "elastic_net", "svr", "knn_regressor",
            "adaboost_regressor", "gaussian_mixture", "mean_shift", "spectral", "optics", "xgboost", "lightgbm"}
    assert need <= set(MR.REGISTRY)
    for s in MR.REGISTRY.values():
        assert s.description and s.tradeoff and s.speed in ("fast", "medium", "slow")
        md = s.metadata()
        assert set(md) >= {"scaling", "probabilities", "predicts_new_rows", "native_missing_values", "row_limit", "available"}
        for p in s.params:                       # declared defaults respect their own ranges
            if p.kind in ("int", "float") and p.default is not None:
                assert (p.low is None or p.default >= p.low) and (p.high is None or p.default <= p.high), (s.key, p.name)
            if p.kind == "choice":
                assert p.default in p.choices, (s.key, p.name)
        for name, values in s.search_space.items():   # search spaces only name declared settings and stay small
            assert name in {p.name for p in s.params}, (s.key, name)
            assert MR.validate_params(s, {name: values[0]}) is not None
        assert int(np.prod([len(v) for v in s.search_space.values()] or [1])) <= 48
        est = s.factory(0, False, "cpu") if s.available else None
        for p in s.params:                       # every declared setting exists on the (possibly wrapped) estimator
            if est is not None:
                assert s.param_prefix + p.name in est.get_params(), (s.key, p.name)
        if s.task == "clustering" and s.k_param:
            assert s.k_param in s.factory(0, False, "cpu").get_params()


@pytest.mark.parametrize("key", CLF)
def test_every_classifier_trains_scores_and_saves(key, clf_df, tmp_path):
    exp = T.run_selection(clf_df, T.ExperimentConfig(task="classification", target="y", models=[key], split=CV_SPLIT, seed=3, cv_folds=3,
                                                     primary_metric="balanced_accuracy"))
    c = exp.candidates[0]
    assert c.status == "ok", c.error
    assert 0.3 < c.score <= 1
    T.finalize_experiment(exp)
    m = exp.test_result["metrics"]
    assert 0.5 < m["accuracy"] <= 1
    pred = exp._state["test_predictions"]
    assert set(pred["predicted"]) <= {"yes", "no"}                     # text labels come back even for XGBoost
    if MR.REGISTRY[key].proba:
        assert "probability_yes" in pred and pred["probability_yes"].between(0, 1).all()
    art = joblib.load(T.save_artifact(exp, str(tmp_path / "m.joblib")))
    assert list(art["pipeline"].predict(clf_df.drop(columns="y").head(3))) == list(exp._state["pipeline"].predict(clf_df.drop(columns="y").head(3)))


@pytest.mark.parametrize("key", REG)
def test_every_regressor_trains_and_beats_the_mean(key, reg_df):
    exp = T.run_selection(reg_df, T.ExperimentConfig(task="regression", target="t", models=[key], split=RSPLIT, seed=5, primary_metric="rmse"))
    c = exp.candidates[0]
    assert c.status == "ok", c.error
    assert c.score < exp.baseline_selection["metrics"]["rmse"], key     # all of these learn the clear signal in the fixture


@pytest.mark.parametrize("key", CLU)
def test_every_clustering_method_runs(key, blobs_df):
    exp = T.run_selection(blobs_df, T.ExperimentConfig(task="clustering", models=[key], cluster_k_min=2, cluster_k_max=4))
    ok = [c for c in exp.candidates if c.status == "ok"]
    assert ok and exp.test_result["n_clusters"] >= 2
    if MR.REGISTRY[key].k_param:
        assert sorted(c.params["k"] for c in exp.candidates) == [2, 3, 4]
    if key in ("kmeans", "gaussian_mixture", "agglomerative", "spectral", "birch"):
        assert exp.test_result["n_clusters"] == 3                        # three well-separated blobs


def test_validate_params_rejects_bad_values_with_readable_messages():
    s = MR.REGISTRY["svc"]
    assert MR.validate_params(s, {"C": "2", "kernel": "linear"}) == {"C": 2.0, "kernel": "linear"}
    with pytest.raises(MR.ModelConfigError, match="between"):
        MR.validate_params(s, {"C": -1})
    with pytest.raises(MR.ModelConfigError, match="one of"):
        MR.validate_params(s, {"kernel": "magic"})
    with pytest.raises(MR.ModelConfigError, match="no setting"):
        MR.validate_params(s, {"bogus": 1})
    with pytest.raises(MR.ModelConfigError, match="whole number"):
        MR.validate_params(MR.REGISTRY["knn"], {"n_neighbors": "many"})


def test_user_settings_reach_the_fitted_estimator(clf_df):
    exp = T.run_selection(clf_df, T.ExperimentConfig(task="classification", target="y", models=["knn", "xgboost"] if MR.XGB_AVAILABLE else ["knn"],
                                                     split=CV_SPLIT, seed=3, cv_folds=3,
                                                     model_params={"knn": {"n_neighbors": 11, "weights": "distance"}, "xgboost": {"max_depth": 2}}))
    fitted = exp._state["pipeline"].named_steps["model"]
    if exp.selected_key == "knn":
        assert fitted.n_neighbors == 11 and fitted.weights == "distance"
    assert next(c for c in exp.candidates if c.key == "knn").params == {"n_neighbors": 11, "weights": "distance"}
    with pytest.raises(T.ExperimentError, match="between"):
        T.run_selection(clf_df, T.ExperimentConfig(task="classification", target="y", models=["knn"], split=CV_SPLIT, model_params={"knn": {"n_neighbors": 0}}))


def test_row_limits_skip_a_model_with_a_reason_instead_of_hanging(clf_df, monkeypatch):
    monkeypatch.setattr(MR.REGISTRY["svc"], "max_rows", 100)
    exp = T.run_selection(clf_df, T.ExperimentConfig(task="classification", target="y", models=["svc", "logistic_regression"], split=CV_SPLIT, cv_folds=3))
    svc = next(c for c in exp.candidates if c.key == "svc")
    assert svc.status == "skipped" and "limited to 100 rows" in svc.error
    assert exp.selected_key == "logistic_regression"


def test_multinomial_nb_gets_non_negative_inputs_and_blocks_pca(clf_df):
    exp = T.run_selection(clf_df, T.ExperimentConfig(task="classification", target="y", models=["multinomial_nb"], split=CV_SPLIT, cv_folds=3))
    pipe = exp._state["pipeline"]
    Xt = pipe[:-1].transform(clf_df.drop(columns="y"))
    assert np.asarray(Xt).min() >= 0
    ok, reasons, _ = MR.check_compatibility(MR.REGISTRY["multinomial_nb"], 400, 5, {"reduction": {"method": "pca", "n_components": 2}})
    assert not ok and "non-negative" in reasons[0]


def test_unavailable_optional_package_is_reported_not_crashing(clf_df, monkeypatch):
    monkeypatch.setattr(MR, "XGB_AVAILABLE", False)
    assert "xgboost" not in [m.key for m in MR.models_for("classification")]
    assert "xgboost" in [m.key for m in MR.models_for("classification", include_unavailable=True)]
    exp = T.run_selection(clf_df, T.ExperimentConfig(task="classification", target="y", models=["xgboost", "decision_tree"], split=CV_SPLIT, cv_folds=3))
    x = next(c for c in exp.candidates if c.key == "xgboost")
    assert x.status == "skipped" and "pip install xgboost" in x.error


def test_label_encoded_wrapper_round_trips_text_labels():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(120, 3))
    y = np.where(X[:, 0] > 0, "high", "low")
    from sklearn.tree import DecisionTreeClassifier
    m = MR.LabelEncodedClassifier(DecisionTreeClassifier(random_state=0)).fit(X, y)
    assert list(m.classes_) == ["high", "low"] and set(m.predict(X)) == {"high", "low"} and m.predict_proba(X).shape == (120, 2)


def test_spectral_and_optics_are_marked_as_unable_to_predict_new_rows():
    assert not MR.REGISTRY["spectral"].predicts_new and not MR.REGISTRY["optics"].predicts_new and not MR.REGISTRY["dbscan"].predicts_new
    assert MR.REGISTRY["kmeans"].predicts_new and MR.REGISTRY["gaussian_mixture"].predicts_new


def test_missing_values_reach_native_models_only_through_imputation_and_still_work(clf_df):
    df = clf_df.copy()
    df.loc[::7, "x1"] = np.nan
    for key in ("gaussian_nb", "svc"):
        exp = T.run_selection(df, T.ExperimentConfig(task="classification", target="y", models=[key], split=CV_SPLIT, cv_folds=3))
        assert exp.candidates[0].status == "ok"
    assert pd.isna(df["x1"]).sum() > 0
