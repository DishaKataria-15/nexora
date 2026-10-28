import json
import time

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator

from nexora.ml import compute as CP
from nexora.ml import imbalance as IMB
from nexora.ml import model_registry as MR
from nexora.ml import training as T
from nexora.ml.splitting import SplitConfig, make_split
from nexora.ml.task_detection import recommend_targets, task_for_column

SPLIT = {"strategy": "random", "train": 0.7, "val": 0.15, "test": 0.15, "seed": 5, "group_col": None, "time_col": None}
CV_SPLIT = {"strategy": "stratified", "train": 0.8, "val": 0.0, "test": 0.2, "seed": 5, "group_col": None, "time_col": None}


def cfg_clf(**kw):
    base = dict(task="classification", target="y", models=["logistic_regression", "decision_tree"], split={**CV_SPLIT, "seed": 3}, seed=3, primary_metric="f1")
    base.update(kw)
    return T.ExperimentConfig(**base)


# ---- the three task workflows on small synthetic fixtures ----------------------------------------------------------------------------
def test_classification_workflow_selection_then_one_time_test(clf_df):
    exp = T.run_selection(clf_df, cfg_clf(models=["logistic_regression", "random_forest", "decision_tree", "gradient_boosting"]))
    assert exp.status == "selection_complete" and exp.test_result is None and not exp.test_evaluated   # test rows untouched so far
    assert exp.selection_basis.endswith("cross-validation on the training set") and len(exp.candidates) == 4
    assert all(c.status == "ok" and c.score is not None and c.score_std is not None for c in exp.candidates)
    best = max(exp.candidates, key=lambda c: c.score)
    assert exp.selected_key == best.key
    T.finalize_experiment(exp)
    assert exp.status == "finalized" and exp.test_evaluated
    m = exp.test_result["metrics"]
    assert 0.7 < m["accuracy"] <= 1 and m["roc_auc"] > 0.7 and exp.test_baseline["metrics"]["f1"] == 0.0     # majority baseline never predicts "yes"
    assert exp.interpretation and exp._state["test_predictions"].shape[0] == exp.split_info["sizes"]["test"]
    with pytest.raises(T.TestSetAlreadyUsed):
        T.finalize_experiment(exp)


def test_regression_workflow_uses_validation_set(reg_df):
    cfg = T.ExperimentConfig(task="regression", target="t", models=["linear_regression", "random_forest_regressor", "gradient_boosting_regressor"],
                             split=SPLIT, seed=5, primary_metric="rmse")
    exp = T.run_selection(reg_df, cfg)
    assert exp.selection_basis == "the validation set" and exp.split_info["sizes"]["val"] > 0
    assert exp.selected_key == "linear_regression"                                  # the data is linear: the simplest model should win on its merits
    T.finalize_experiment(exp)
    m = exp.test_result["metrics"]
    assert m["r2"] > 0.95 and m["rmse"] < 1.0
    assert exp.test_baseline["metrics"]["rmse"] > 5 * m["rmse"]


def test_clustering_workflow_finds_the_three_groups(blobs_df):
    exp = T.run_selection(blobs_df, T.ExperimentConfig(task="clustering", models=["kmeans", "agglomerative", "dbscan"]))
    assert exp.status == "finalized" and exp.test_result["n_clusters"] == 3
    km = {c.params.get("k"): c.score for c in exp.candidates if c.key.startswith("kmeans")}
    assert max(km, key=km.get) == 3
    assert len({c.key for c in exp.candidates}) == len(exp.candidates)              # every candidate is uniquely identifiable
    assert sum(r["rows"] for r in exp.cluster_profile) == len(blobs_df) and exp.projection
    with pytest.raises(T.ExperimentError, match="no held-out"):
        T.finalize_experiment(exp)


# ---- test-set discipline ------------------------------------------------------------------------------------------------------------------
def test_selection_never_reads_test_rows(reg_df):
    """Scramble the targets AND features of exactly the test rows: nothing chosen during selection may change."""
    cfg = T.ExperimentConfig(task="regression", target="t", models=["linear_regression", "random_forest_regressor"], split=SPLIT, seed=5, primary_metric="rmse")
    a = T.run_selection(reg_df, cfg)
    test_idx = make_split(len(reg_df), SplitConfig(**SPLIT)).test
    tampered = reg_df.copy()
    rng = np.random.RandomState(9)
    tampered.loc[test_idx, "t"] = rng.normal(1e6, 1e5, len(test_idx))
    tampered.loc[test_idx, "a"] = rng.normal(1e3, 1e2, len(test_idx))
    b = T.run_selection(tampered, cfg)
    # parallel tree ensembles may differ in the last floating-point digit between runs; tampering would change scores by orders of magnitude
    assert [c.key for c in a.candidates] == [c.key for c in b.candidates]
    assert [c.score for c in a.candidates] == pytest.approx([c.score for c in b.candidates], rel=1e-9)
    assert a.selected_key == b.selected_key
    assert [r["feature"] for r in a.importance] == [r["feature"] for r in b.importance]
    assert [r["importance"] for r in a.importance] == pytest.approx([r["importance"] for r in b.importance], rel=1e-6, abs=1e-12)
    T.finalize_experiment(a)
    T.finalize_experiment(b)
    assert a.test_result["metrics"]["rmse"] != b.test_result["metrics"]["rmse"]    # only the final, one-time evaluation looks at them


def test_finalize_requires_selection_first():
    with pytest.raises(T.ExperimentError, match="selection first"):
        T.finalize_experiment(T.Experiment(config=cfg_clf()))


def test_experiments_are_reproducible(clf_df):
    a = T.run_selection(clf_df, cfg_clf(models=["random_forest", "logistic_regression"]))
    b = T.run_selection(clf_df, cfg_clf(models=["random_forest", "logistic_regression"]))
    assert [(c.key, c.score) for c in a.candidates] == [(c.key, c.score) for c in b.candidates] and a.split_info == b.split_info
    assert a.dataset_fingerprint == b.dataset_fingerprint and a.environment["scikit-learn"]


# ---- resampling stays inside training data / folds -----------------------------------------------------------------------------------------------
@pytest.mark.skipif(not IMB.IMBLEARN, reason="imbalanced-learn not installed")
@pytest.mark.parametrize("protocol", ["cv", "val"])
def test_resampling_sees_only_training_rows(clf_df, monkeypatch, protocol):
    from imblearn.over_sampling import RandomOverSampler

    seen: list[int] = []

    class Spy(RandomOverSampler):
        def _fit_resample(self, X, y, **kw):
            seen.append(len(X))
            return super()._fit_resample(X, y, **kw)

    monkeypatch.setattr(IMB, "build_sampler", lambda strategy, y, seed: Spy(random_state=seed))
    split = CV_SPLIT if protocol == "cv" else {**SPLIT, "strategy": "stratified"}
    exp = T.run_selection(clf_df, cfg_clf(models=["logistic_regression"], imbalance="random_over", cv_folds=4, split=split))
    n_train, n_val = exp.split_info["sizes"]["train"], exp.split_info["sizes"]["val"]
    assert max(seen) <= n_train                                   # no fit ever saw validation or test rows
    if protocol == "cv":
        assert n_val == 0 and len(seen) == 5                      # 4 folds + the final refit on the training part
        assert all(abs(n - 0.75 * n_train) <= 2 for n in seen[:4]) and seen[4] == n_train      # each fold resampled only its OWN training part
    else:
        assert n_val > 0 and seen == [n_train]                    # one fit, on the training rows only; validation rows are scored as-is


@pytest.mark.skipif(not IMB.IMBLEARN, reason="imbalanced-learn not installed")
def test_real_resamplers_run_and_class_weight_is_applied(clf_df):
    for strat in ("random_over", "random_under", "class_weight"):
        exp = T.run_selection(clf_df, cfg_clf(models=["logistic_regression"], imbalance=strat))
        assert exp.candidates[0].status == "ok", strat
    exp = T.run_selection(clf_df.drop(columns=["cat"]), cfg_clf(models=["logistic_regression"], imbalance="smote"))
    assert exp.candidates[0].status == "ok"


# ---- validation, failure handling, dependencies, compute ---------------------------------------------------------------------------------------------
def test_missing_labels_need_explicit_confirmation_and_are_never_invented(clf_df):
    df = clf_df.copy()
    df["y"] = df["y"].astype(object)
    df.loc[:9, "y"] = None
    with pytest.raises(T.ExperimentError, match="never invents labels"):
        T.run_selection(df, cfg_clf())
    exp = T.run_selection(df, cfg_clf(drop_missing_target=True))
    assert sum(exp.split_info["sizes"].values()) == len(df) - 10 and any("left out" in w for w in exp.warnings)


@pytest.mark.parametrize("make,msg", [
    (lambda d: d.assign(y="one"), "only one class"), (lambda d: d.assign(y=np.arange(len(d))), "too many categories|looks like"),
    (lambda d: d.assign(y=["a"] * (len(d) - 1) + ["b"]), "only one row")])
def test_bad_classification_targets_are_explained(clf_df, make, msg):
    with pytest.raises(T.ExperimentError, match=msg):
        T.run_selection(make(clf_df), cfg_clf())


def test_bad_regression_targets_and_config_errors(reg_df):
    base = dict(task="regression", target="t", models=["linear_regression"], split=SPLIT, primary_metric="rmse")
    with pytest.raises(T.ExperimentError, match="not numeric"):
        T.run_selection(reg_df.assign(t="x"), T.ExperimentConfig(**base))
    with pytest.raises(T.ExperimentError, match="same value"):
        T.run_selection(reg_df.assign(t=1.0), T.ExperimentConfig(**base))
    with pytest.raises(T.ExperimentError, match="Choose the column"):
        T.run_selection(reg_df, T.ExperimentConfig(**{**base, "target": None}))
    with pytest.raises(T.ExperimentError, match="not available for regression"):
        T.run_selection(reg_df, T.ExperimentConfig(**{**base, "models": ["random_forest"]}))
    with pytest.raises(T.ExperimentError, match="at least one model"):
        T.run_selection(reg_df, T.ExperimentConfig(**{**base, "models": []}))
    with pytest.raises(T.ExperimentError, match="not a valid selection metric"):
        T.run_selection(reg_df, T.ExperimentConfig(**{**base, "primary_metric": "f1"}))
    with pytest.raises(T.ExperimentError, match="only applies to classification"):
        T.run_selection(reg_df, T.ExperimentConfig(**{**base, "imbalance": "smote"}))


def test_one_failing_model_does_not_stop_the_comparison(clf_df, monkeypatch):
    def boom(seed, cw, dev):
        raise RuntimeError("synthetic failure")
    monkeypatch.setitem(MR.REGISTRY, "decision_tree", MR.ModelSpec("decision_tree", "classification", "Decision Tree", boom, "d", "t", True, False))
    exp = T.run_selection(clf_df, cfg_clf(models=["decision_tree", "logistic_regression"]))
    st = {c.key: c for c in exp.candidates}
    assert st["decision_tree"].status == "failed" and "synthetic failure" in st["decision_tree"].error and st["logistic_regression"].status == "ok"
    assert exp.selected_key == "logistic_regression"


def test_all_models_failing_is_a_clear_error(clf_df, monkeypatch):
    def boom(seed, cw, dev):
        raise RuntimeError("nope")
    monkeypatch.setitem(MR.REGISTRY, "logistic_regression", MR.ModelSpec("logistic_regression", "classification", "LR", boom, "d", "t", True, True))
    with pytest.raises(T.ExperimentError, match="Every model failed"):
        T.run_selection(clf_df, cfg_clf(models=["logistic_regression"]))


def test_gpu_request_without_support_is_an_error_not_a_silent_fallback(clf_df, monkeypatch):
    monkeypatch.setattr(T, "detect_compute", lambda: CP.ComputeInfo(4, [], False))
    with pytest.raises(T.ExperimentError, match="Every model failed.*GPU"):
        T.run_selection(clf_df, cfg_clf(models=["logistic_regression"], compute="gpu"))


def test_compute_options_never_offer_unavailable_gpu():
    none = CP.ComputeInfo(8, [], False)
    gpu_no_backend = CP.ComputeInfo(8, ["Fake GPU"], False)
    ready = CP.ComputeInfo(8, ["Fake GPU"], True)
    assert CP.device_options(none, True) == ["auto", "cpu"] and CP.device_options(gpu_no_backend, True) == ["auto", "cpu"]
    assert CP.device_options(ready, False) == ["auto", "cpu"] and CP.device_options(ready, True) == ["auto", "cpu", "gpu"]
    assert CP.resolve_device("auto", False, ready)[0] == "cpu" and CP.resolve_device("cpu", True, ready)[0] == "cpu"
    assert CP.resolve_device("gpu", True, ready)[0] == "gpu"
    for info, cap in ((none, True), (gpu_no_backend, True), (ready, False)):
        with pytest.raises(CP.ComputeError):
            CP.resolve_device("gpu", cap, info)
    assert isinstance(CP.detect_compute().cpu_count, int)


def test_registry_covers_required_models_and_explains_them():
    assert {"logistic_regression", "decision_tree", "random_forest", "gradient_boosting", "linear_regression", "decision_tree_regressor",
            "random_forest_regressor", "gradient_boosting_regressor", "kmeans", "dbscan", "agglomerative"} <= set(MR.REGISTRY)
    assert all(s.description and s.tradeoff for s in MR.REGISTRY.values())


def test_recommendations_use_data_characteristics_and_state_tradeoffs():
    small = MR.recommend_models("classification", 200, 4, 2)
    large = MR.recommend_models("classification", 20000, 8, 3, imbalance_ratio=0.1)
    assert small[0].key == "logistic_regression" and "200" in small[0].reason
    assert large[0].key == "gradient_boosting" and "class weighting" in large[0].reason
    assert MR.recommend_models("classification", 5000, 5, 1, prefer="interpretable")[0].key == "logistic_regression"
    assert MR.recommend_models("regression", 100, 3, 0)[0].key == "linear_regression"
    assert MR.recommend_models("clustering", 500, 3, 0)[0].key == "kmeans"
    assert all(r.tradeoff and r.reason.endswith(".") for r in small + large)


def test_task_and_target_suggestions():
    df = pd.DataFrame({"id": [f"R{i}" for i in range(100)], "price": np.random.RandomState(0).uniform(1, 500, 100), "churned": ["y", "n"] * 50,
                       "city": (["a", "b", "c", "d"] * 25), "when": pd.date_range("2026-01-01", periods=100)})
    assert task_for_column(df["price"])[0] == "regression" and task_for_column(df["churned"])[0] == "classification"
    assert task_for_column(df["id"])[0] is None and task_for_column(df["when"])[0] is None
    assert recommend_targets(df, "classification")[0]["column"] == "churned" and recommend_targets(df, "regression")[0]["column"] == "price"


# ---- artifacts and background jobs ---------------------------------------------------------------------------------------------------------------
def test_artifact_roundtrip_predicts_like_the_session_model(clf_df, tmp_path):
    exp = T.run_selection(clf_df, cfg_clf(models=["random_forest"]))
    art = T.load_artifact(T.save_artifact(exp, str(tmp_path / "m.joblib")))
    X = clf_df[art["feature_columns"]]
    assert (art["pipeline"].predict(X) == exp._state["pipeline"].predict(X)).all()
    assert art["config"]["seed"] == 3 and art["environment"]["scikit-learn"] and art["classes"] == ["no", "yes"]


def test_artifact_without_model_is_explained(tmp_path):
    with pytest.raises(T.ExperimentError, match="no fitted model"):
        T.save_artifact(T.Experiment(config=cfg_clf()), str(tmp_path / "x.joblib"))
    assert "_state" not in T.Experiment(config=cfg_clf()).to_dict()


def test_experiment_to_dict_is_json_safe(clf_df):
    exp = T.run_selection(clf_df, cfg_clf())
    T.finalize_experiment(exp)
    s = json.dumps(exp.to_dict())
    assert '"pipeline":' not in s and "Pipeline(" not in s and json.loads(s)["test_evaluated"] is True     # no fitted objects, only the report


def test_training_job_reports_real_progress_and_result(clf_df):
    job = T.TrainingJob(lambda progress, token: T.run_selection(clf_df, cfg_clf(), cancel=token, progress=progress)).start()
    job.join(60)
    assert job.state == "done" and job.fraction == 1.0 and job.result.status == "selection_complete" and job.elapsed > 0


def test_training_job_can_be_cancelled_between_models(clf_df):
    class Slow(BaseEstimator):
        def fit(self, X, y):
            time.sleep(0.4)
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            return np.full(len(X), self.classes_[0])

    MR.REGISTRY["slow"] = MR.ModelSpec("slow", "classification", "Slow", lambda s, c, d: Slow(), "d", "t", True, False)
    try:
        cfg = cfg_clf(models=["slow", "logistic_regression", "decision_tree"], split={**SPLIT, "strategy": "stratified"})
        job = T.TrainingJob(lambda p, tok: T.run_selection(clf_df, cfg, cancel=tok, progress=p)).start()
        time.sleep(0.15)
        job.cancel()
        job.join(30)
        assert job.state == "cancelled" and job.result is None and "no results were saved" in job.message
    finally:
        MR.REGISTRY.pop("slow", None)


def test_training_job_surfaces_errors_as_state_not_exceptions(clf_df):
    job = T.TrainingJob(lambda p, t: T.run_selection(clf_df, cfg_clf(target="missing_col"), cancel=t)).start()
    job.join(30)
    assert job.state == "failed" and "Choose the column" in job.error
