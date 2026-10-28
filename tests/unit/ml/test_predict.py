"""Scoring new rows with a saved model."""
import joblib
import pytest

from nexora.ml import training as T
from nexora.ml.predict import PredictionError, predict_frame

CV_SPLIT = {"strategy": "stratified", "train": 0.8, "val": 0.0, "test": 0.2, "seed": 5, "group_col": None, "time_col": None}


def _saved(exp, tmp_path):
    return joblib.load(T.save_artifact(exp, str(tmp_path / "m.joblib")))


def test_saved_classifier_scores_new_rows_identically(clf_df, tmp_path):
    exp = T.run_selection(clf_df, T.ExperimentConfig(task="classification", target="y", models=["random_forest"], split=CV_SPLIT, cv_folds=3))
    out, notes = predict_frame(_saved(exp, tmp_path), clf_df.head(40))
    assert list(out["predicted_y"]) == list(exp._state["pipeline"].predict(clf_df.drop(columns="y").head(40)))
    assert {"probability_yes", "probability_no"} <= set(out.columns) and any("'y'" in n for n in notes)


def test_missing_columns_and_empty_input_are_refused(clf_df, tmp_path):
    exp = T.run_selection(clf_df, T.ExperimentConfig(task="classification", target="y", models=["decision_tree"], split=CV_SPLIT, cv_folds=3))
    p = _saved(exp, tmp_path)
    with pytest.raises(PredictionError, match="missing column"):
        predict_frame(p, clf_df.drop(columns=["x1"]))
    with pytest.raises(PredictionError, match="no rows"):
        predict_frame(p, clf_df.head(0))


def test_clustering_models_that_cannot_assign_new_rows_say_so(blobs_df, tmp_path):
    km = T.run_selection(blobs_df, T.ExperimentConfig(task="clustering", models=["kmeans"], cluster_k_min=3, cluster_k_max=3))
    out, _ = predict_frame(_saved(km, tmp_path), blobs_df.head(10))
    assert list(out["predicted_group"]) == list(km._state["labels"][:10])
    db = T.run_selection(blobs_df, T.ExperimentConfig(task="clustering", models=["dbscan"]))
    with pytest.raises(PredictionError, match="cannot assign new rows"):
        predict_frame(_saved(db, tmp_path), blobs_df.head(10))
