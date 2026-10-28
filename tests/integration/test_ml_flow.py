"""Whole-product happy paths through the service layer, with genuine task-specific metrics."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from nexora.eda import charts as C
from nexora.eda import profiling as P
from nexora.ml import training as T
from nexora.ml.splitting import recommend_split
from nexora.reporting import zip_bytes
from nexora.validation.ingestion import IngestError

from .conftest import AS_OF, SAMPLES, decide


def prepared(ws, fname, profile="generic", goal="supervised"):
    h = ws.ingest(fname, (SAMPLES / fname).read_bytes())
    plan = ws.make_plan(h, profile, goal, as_of=AS_OF)
    res, vid = ws.run_preparation(h, plan, {}, decide(plan), as_of=AS_OF)
    df, note = ws.working_frame(vid, h)
    return h, plan, res, vid, df, note


def run_job(ws, df, cfg, vid):
    job = ws.run_experiment_job(df, cfg, vid)
    job.join(120)
    assert job.state == "done", job.error
    return job.result


def test_definition_of_done_happy_path_regression(ws):
    # 1-4. upload, profile, review/confirm plan, transform + validate
    h, plan, res, vid, df, note = prepared(ws, "house_prices.csv")
    assert res.reconciliation["ok"] and res.source_unchanged and "Prepared version" in note
    exports = ws.preparation_exports(res, h, plan)
    assert len(pd.read_csv(io.BytesIO(exports["prepared_data.csv"]))) == len(h.raw)          # 5. prepared data export
    # 6. explore: every recommended chart is computed from this data
    specs = C.recommend_charts(df, target="price_thousands_usd")
    assert specs and all(len(C.build_figure(df, s).data) for s in specs)
    assert P.dataset_overview(df)["rows"] == 900
    # 7. supervised task with a confirmed target, split, models, metric
    split, _ = recommend_split(len(df))
    cfg = T.ExperimentConfig(task="regression", target="price_thousands_usd", dataset_name=h.name, dataset_version=vid,
                             models=["linear_regression", "random_forest_regressor", "gradient_boosting_regressor"],
                             split={**split.__dict__, "strategy": "random", "seed": 11}, seed=11, primary_metric="rmse")
    exp = run_job(ws, df, cfg, vid)
    assert exp.split_info["sizes"] == {"train": 720, "val": 0, "test": 180}                # 900 rows < 1000: 80/20 with cross-validation
    assert exp.selection_basis.endswith("cross-validation on the training set")
    assert "identifier" in exp.feature_set["excluded"]["house_id"]
    eid = ws.save_experiment(exp, vid)
    T.finalize_experiment(exp)
    ws.save_experiment(exp, vid, eid)
    m = exp.test_result["metrics"]
    # 8. genuine metrics: independently recomputed from the saved predictions
    pred = exp._state["test_predictions"]
    assert m["rmse"] == pytest.approx(float(np.sqrt(((pred["predicted"] - pred["actual"]) ** 2).mean())))
    assert m["mae"] == pytest.approx(float((pred["predicted"] - pred["actual"]).abs().mean()))
    assert m["r2"] > 0.85 and m["rmse"] < exp.test_baseline["metrics"]["rmse"] / 2
    assert len(pred) == 180 and np.allclose(pred["actual"].values, df.loc[pred["row_in_dataset"].values, "price_thousands_usd"].values)   # rows trace back
    # 9. exports: report, config, predictions, model
    files = ws.experiment_exports(exp, h.name)
    assert {"experiment_report.html", "experiment_config.json", "test_predictions.csv", "model_pipeline.joblib"} <= set(files)
    cfgj = json.loads(files["experiment_config.json"])
    assert cfgj["config"]["seed"] == 11 and cfgj["split"]["sizes"]["test"] == 180 and cfgj["environment"]["scikit-learn"]
    rep = files["experiment_report.html"].decode()
    assert f"{m['rmse']:.4g}" in rep and "used once" in rep
    stored = ws.repo.get_experiment(eid)
    assert stored["status"] == "finalized" and stored["test_evaluated"] == 1 and stored["result"]["test_result"]["metrics"]["rmse"] == pytest.approx(m["rmse"])
    assert ws.repo.read_raw_bytes(h.dataset_id) == (SAMPLES / "house_prices.csv").read_bytes()   # the original is exactly what was uploaded


def test_classification_with_leakage_detected_excluded_and_score_dropping(ws):
    h, plan, res, vid, df, _ = prepared(ws, "customer_churn.csv")
    cfg = T.ExperimentConfig(task="classification", target="churned", dataset_name=h.name, dataset_version=vid,
                             models=["logistic_regression", "gradient_boosting"],
                             split={"strategy": "stratified", "train": .7, "val": .15, "test": .15, "seed": 4, "group_col": None, "time_col": None},
                             seed=4, primary_metric="f1", imbalance="class_weight")
    leaky = run_job(ws, df, cfg, vid)
    assert {c for w in leaky.leakage for c in w["columns"] if w["kind"] == "target_leakage"} == {"cancellation_reason"}
    assert any(w["kind"] == "post_outcome_name" and "cancellation_reason" in w["columns"] for w in leaky.leakage)
    cfg.features_exclude = ["cancellation_reason"]
    honest = run_job(ws, df, cfg, vid)
    assert not [w for w in honest.leakage if w["kind"] == "target_leakage"]
    T.finalize_experiment(leaky)
    T.finalize_experiment(honest)
    assert leaky.test_result["metrics"]["accuracy"] > honest.test_result["metrics"]["accuracy"] + 0.1          # leakage inflates scores
    hm = honest.test_result["metrics"]
    assert 0.6 < hm["accuracy"] < 0.95 and hm["roc_auc"] > 0.65 and hm["f1"] > honest.test_baseline["metrics"]["f1"]
    shares = {r["class"]: r["share %"] for r in honest.class_distribution}
    assert max(shares.values()) > 75 and any("imbalanced" in t for t in honest.interpretation)
    assert sum(sum(r) for r in honest.test_result["confusion_matrix"]) == honest.split_info["sizes"]["test"]


def test_repeated_testing_is_surfaced(ws):
    h, plan, res, vid, df, _ = prepared(ws, "house_prices.csv")
    cfg = T.ExperimentConfig(task="regression", target="price_thousands_usd", models=["linear_regression"], seed=2,
                             split={"strategy": "random", "train": .8, "val": 0.0, "test": .2, "seed": 2, "group_col": None, "time_col": None}, primary_metric="rmse")
    first = run_job(ws, df, cfg, vid)
    assert first.prior_test_uses == 0
    T.finalize_experiment(first)
    ws.save_experiment(first, vid)
    second = run_job(ws, df, cfg, vid)
    assert second.prior_test_uses == 1
    assert "1 earlier experiment" in ws.experiment_exports(second, "d")["experiment_report.html"].decode()


def test_chronological_prediction_and_random_split_warning(ws):
    h, plan, res, vid, df, _ = prepared(ws, "daily_demand.csv")
    base = dict(task="regression", target="units_sold", models=["linear_regression", "random_forest_regressor"], seed=1, primary_metric="rmse")
    rnd = run_job(ws, df, T.ExperimentConfig(**base, split={"strategy": "random", "train": .7, "val": .15, "test": .15, "seed": 1, "group_col": None, "time_col": None}), vid)
    assert any(w["kind"] == "time_split" for w in rnd.leakage)                                   # random split on dated data is flagged
    chrono = run_job(ws, df, T.ExperimentConfig(**base, split={"strategy": "chronological", "train": .7, "val": .15, "test": .15, "seed": 1,
                                                              "group_col": None, "time_col": "date"}), vid)
    assert not any(w["kind"] == "time_split" for w in chrono.leakage)
    sp, dates = chrono._state["split"], chrono._state["X"]["date"]
    assert dates.iloc[sp.train].max() < dates.iloc[sp.val].min() and dates.iloc[sp.val].max() < dates.iloc[sp.test].min()
    T.finalize_experiment(chrono)
    assert chrono.test_result["metrics"]["r2"] > 0.2 and chrono.test_result["metrics"]["rmse"] < chrono.test_baseline["metrics"]["rmse"]


def test_clustering_workflow_end_to_end(ws):
    h, plan, res, vid, df, _ = prepared(ws, "customer_segments.csv", goal="unsupervised")
    exp = run_job(ws, df, T.ExperimentConfig(task="clustering", models=["kmeans", "dbscan"], dataset_name=h.name, dataset_version=vid, seed=3), vid)
    assert "customer_id" in exp.feature_set["excluded"] and exp.status == "finalized"
    assert exp.test_result["n_clusters"] in (3, 4) and exp.test_result["metrics"]["silhouette"] > 0.4
    assert sum(r["rows"] for r in exp.cluster_profile) == len(df)
    files = ws.experiment_exports(exp, h.name)
    labels = pd.read_csv(io.BytesIO(files["cluster_assignments.csv"]))
    assert len(labels) == len(df) and "Clusters are groups" in " ".join(exp.interpretation)
    assert "experiment_report.html" in zipfile.ZipFile(io.BytesIO(zip_bytes(files))).namelist()


def test_unprepared_data_can_be_explored_and_labelled_as_such(ws):
    h = ws.ingest("retail_orders_messy.csv", (SAMPLES / "retail_orders_messy.csv").read_bytes())
    df, note = ws.working_frame(None, h)
    assert "Unprepared" in note and len(df) == len(h.raw)                                         # explore-only never removes rows
    assert pd.api.types.is_numeric_dtype(df["order_total"]) and not pd.api.types.is_numeric_dtype(h.raw["order_total"])
    assert C.recommend_charts(df)


def test_saved_dataset_reopens_with_its_version(ws):
    h, plan, res, vid, df, _ = prepared(ws, "house_prices.csv")
    again = ws.open_dataset(h.dataset_id)
    assert again.raw.shape == h.raw.shape and again.name == h.name
    df2, _ = ws.working_frame(vid, again)
    pd.testing.assert_frame_equal(df2.reset_index(drop=True), df.reset_index(drop=True), check_dtype=False)
    with pytest.raises(KeyError):
        ws.open_dataset("missing")


def test_upload_limits_are_enforced_with_clear_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXORA_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("NEXORA_MAX_ROWS", "5")
    from nexora.workflow import Workspace
    with pytest.raises(IngestError, match="rows"):
        Workspace().ingest("big.csv", ("a\n" + "\n".join(str(i) for i in range(20))).encode())
