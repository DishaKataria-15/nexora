"""Cross-validation folds and bounded hyperparameter search."""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from nexora.ml import training as T
from nexora.ml.cv import CVConfig, CVError, make_folds
from nexora.ml.splitting import SplitConfig, make_split
from nexora.ml.tuning import TuningConfig, TuningError, candidate_params, tune

CV_SPLIT = {"strategy": "stratified", "train": 0.8, "val": 0.0, "test": 0.2, "seed": 5, "group_col": None, "time_col": None}
VAL_SPLIT = {"strategy": "random", "train": 0.7, "val": 0.15, "test": 0.15, "seed": 5, "group_col": None, "time_col": None}


# ---- folds -----------------------------------------------------------------------------------------------------------------------------
def test_time_series_folds_always_validate_on_later_rows_and_respect_the_gap():
    rng = np.random.RandomState(0)
    times = pd.Series(pd.date_range("2024-01-01", periods=120, freq="D")).sample(frac=1, random_state=1).reset_index(drop=True)  # shuffled order
    f = make_folds(120, CVConfig(method="timeseries", folds=4, gap=3), task="regression", split_strategy="chronological", times=times)
    assert f.method == "timeseries" and len(f.folds) == 4 and "gap of 3" in f.description
    for a, b in f.folds:
        assert times.iloc[a].max() < times.iloc[b].min()
        assert (times.iloc[b].min() - times.iloc[a].max()).days == 4          # 3 rows skipped between them
    assert len(f.folds[0][0]) < len(f.folds[-1][0])                          # forward chaining: training grows


def test_auto_picks_time_series_for_chronological_and_group_for_group_splits():
    times = pd.Series(pd.date_range("2024-01-01", periods=60, freq="D"))
    assert make_folds(60, CVConfig(), task="regression", split_strategy="chronological", times=times).method == "timeseries"
    g = pd.Series(np.repeat(np.arange(12), 5))
    f = make_folds(60, CVConfig(folds=4), task="regression", split_strategy="group", groups=g)
    assert f.method == "group"
    for a, b in f.folds:
        assert not set(g.iloc[a]) & set(g.iloc[b])
    w = make_folds(60, CVConfig(method="kfold"), task="regression", split_strategy="chronological", times=times).warnings
    assert any("validated partly on the past" in x for x in w)


def test_repeated_stratified_folds_count_and_class_shares():
    y = pd.Series(["a"] * 80 + ["b"] * 20)
    f = make_folds(100, CVConfig(method="repeated", folds=5, repeats=3), task="classification", split_strategy="stratified", y=y)
    assert len(f.folds) == 15 and "repeated 3 times" in f.description
    for _, b in f.folds:
        assert (y.iloc[b] == "b").sum() == 4
    assert len({tuple(b) for _, b in f.folds}) == 15                          # repeats reshuffle


def test_small_classes_fall_back_to_unstratified_with_a_warning():
    y = pd.Series(["a"] * 40 + ["b"] * 3)
    f = make_folds(43, CVConfig(folds=5), task="classification", split_strategy="random", y=y)
    assert f.method == "kfold" and any("not stratified" in w for w in f.warnings)


@pytest.mark.parametrize("cfg,msg", [(CVConfig(folds=1), "between 2"), (CVConfig(folds=30), "between 2"), (CVConfig(repeats=11), "repeats"),
                                     (CVConfig(method="magic"), "Unknown"), (CVConfig(method="timeseries", gap=-1), "negative"),
                                     (CVConfig(folds=20, repeats=10), "more than")])
def test_invalid_cv_settings_are_rejected(cfg, msg):
    with pytest.raises(CVError, match=msg):
        make_folds(100, cfg, task="regression", split_strategy="random", times=pd.Series(pd.date_range("2024", periods=100)))


def test_time_series_without_dates_is_a_readable_error():
    with pytest.raises(CVError, match="date column"):
        make_folds(50, CVConfig(method="timeseries"), task="regression", split_strategy="random")


# ---- the search engine on its own -----------------------------------------------------------------------------------------------------
@pytest.fixture()
def tiny():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    y = pd.Series(np.where(X["a"] + 0.3 * rng.normal(size=300) > 0, "y", "n"))
    folds = make_folds(300, CVConfig(folds=3), task="classification", split_strategy="random", y=y).folds
    make = lambda p: Pipeline([("model", DecisionTreeClassifier(random_state=0))]).set_params(**p)
    score = lambda pipe, Xs, ys: float((pipe.predict(Xs) == ys).mean())
    return X, y, folds, make, score


SPACE = {"model__max_depth": [1, 2, 4, 8], "model__min_samples_leaf": [1, 20]}


def test_grid_search_tries_every_combination_and_returns_the_best(tiny):
    X, y, folds, make, score = tiny
    r = tune(SPACE, TuningConfig(method="grid"), make_pipe=make, score_fn=score, X=X, y=y, folds=folds, higher_is_better=True)
    assert len(r.trials) == 8 == r.grid_size and not r.stopped
    assert r.best_score == max(t["score"] for t in r.trials)
    assert all(len(t["fold_scores"]) == 3 for t in r.trials)
    manual = np.mean([score(make(r.best_params).fit(X.iloc[a], y.iloc[a]), X.iloc[b], y.iloc[b]) for a, b in folds])
    assert r.best_score == pytest.approx(manual)                             # the reported score is reproducible


def test_grid_larger_than_the_limit_is_refused(tiny):
    with pytest.raises(TuningError, match="more than the limit"):
        candidate_params(SPACE, TuningConfig(method="grid", max_trials=5), 0)


def test_random_search_samples_without_repeats_and_lower_is_better_works(tiny):
    X, y, folds, make, score = tiny
    r = tune(SPACE, TuningConfig(method="random", n_iter=5), make_pipe=make, score_fn=lambda *a: 1 - score(*a), X=X, y=y, folds=folds,
             higher_is_better=False, seed=1)
    assert len(r.trials) == 5 and len({tuple(sorted(t["params"].items())) for t in r.trials}) == 5
    assert r.best_score == min(t["score"] for t in r.trials)


def test_halving_spends_more_rows_on_fewer_combinations(tiny):
    X, y, folds, make, score = tiny
    r = tune(SPACE, TuningConfig(method="halving", n_iter=8, factor=2), make_pipe=make, score_fn=score, X=X, y=y, folds=folds, higher_is_better=True)
    rounds = sorted({t["round"] for t in r.trials})
    assert len(rounds) >= 2
    first = [t for t in r.trials if t["round"] == rounds[0]]
    last = [t for t in r.trials if t["round"] == rounds[-1]]
    assert len(last) < len(first) and last[0]["rows_per_fold"] > first[0]["rows_per_fold"]
    assert r.best_params in [t["params"] for t in last]


def test_time_budget_stops_the_search_and_says_so(tiny):
    X, y, folds, make, score = tiny
    import time
    slow = lambda pipe, Xs, ys: (time.sleep(0.4), score(pipe, Xs, ys))[1]
    r = tune(SPACE, TuningConfig(method="grid", time_budget_s=1), make_pipe=make, score_fn=slow, X=X, y=y, folds=folds, higher_is_better=True)
    assert 1 <= len(r.trials) < 8 and "budget" in r.stopped and r.best_params


def test_a_failing_combination_is_recorded_not_fatal_and_cancel_propagates(tiny):
    X, y, folds, make, score = tiny
    r = tune({"model__max_depth": [2, -5]}, TuningConfig(method="grid"), make_pipe=make, score_fn=score, X=X, y=y, folds=folds, higher_is_better=True)
    assert r.best_params == {"model__max_depth": 2} and any(t["error"] for t in r.trials)
    tok = T.CancelToken()
    tok.cancel()
    with pytest.raises(T.TrainingCancelled):
        tune(SPACE, TuningConfig(method="grid"), make_pipe=make, score_fn=score, X=X, y=y, folds=folds, higher_is_better=True,
             check=tok.check, passthrough=(T.TrainingCancelled,))


# ---- inside the experiment ----------------------------------------------------------------------------------------------------------
def test_tuned_experiment_uses_the_best_settings_in_the_final_model(clf_df):
    cfg = T.ExperimentConfig(task="classification", target="y", models=["decision_tree", "logistic_regression"], split=CV_SPLIT, seed=3,
                             primary_metric="balanced_accuracy", cv={"method": "stratified", "folds": 3}, tuning={"method": "grid", "max_trials": 20})
    exp = T.run_selection(clf_df, cfg)
    dt = next(c for c in exp.candidates if c.key == "decision_tree")
    assert dt.tuning and len(dt.tuning["trials"]) == 12
    best = {k.split("__")[-1]: v for k, v in dt.tuning["best_params"].items()}
    assert dt.params == best and dt.score == pytest.approx(dt.tuning["best_score"])
    assert len(dt.fold_scores) == 3 and exp.cv_info["method"] == "stratified"
    if exp.selected_key == "decision_tree":
        m = exp._state["pipeline"].named_steps["model"]
        assert all(getattr(m, k) == v for k, v in best.items())
    assert any("optimistic" in w for w in exp.warnings)


def test_user_fixed_settings_are_not_searched(clf_df):
    cfg = T.ExperimentConfig(task="classification", target="y", models=["decision_tree"], split=CV_SPLIT, seed=3, cv={"folds": 3},
                             model_params={"decision_tree": {"max_depth": 3}}, tuning={"method": "grid"})
    dt = T.run_selection(clf_df, cfg).candidates[0]
    assert all("max_depth" not in k for k in dt.tuning["space"]) and dt.params["max_depth"] == 3 and len(dt.tuning["trials"]) == 3


def test_tuning_never_reads_test_or_validation_rows(reg_df):
    """Tamper with the validation AND test rows: the tuning trials (CV on training rows) must be identical."""
    cfg = T.ExperimentConfig(task="regression", target="t", models=["decision_tree_regressor"], split=VAL_SPLIT, seed=5, primary_metric="rmse",
                             cv={"folds": 3}, tuning={"method": "random", "n_iter": 4})
    a = T.run_selection(reg_df, cfg)
    sp = make_split(len(reg_df), SplitConfig(**VAL_SPLIT))
    bad = reg_df.copy()
    rows = np.concatenate([sp.val, sp.test])
    bad.loc[rows, "t"] = 1e6
    bad.loc[rows, "a"] = -1e3
    b = T.run_selection(bad, cfg)
    ta, tb = a.candidates[0].tuning, b.candidates[0].tuning
    assert [t["params"] for t in ta["trials"]] == [t["params"] for t in tb["trials"]]
    assert [t["score"] for t in ta["trials"]] == pytest.approx([t["score"] for t in tb["trials"]])
    assert ta["best_params"] == tb["best_params"]
    assert a.candidates[0].score != b.candidates[0].score                     # validation scores DO see the tampered validation rows


def test_chronological_experiment_uses_forward_folds(reg_df):
    df = reg_df.assign(day=pd.date_range("2023-01-01", periods=len(reg_df), freq="D"))
    split = {"strategy": "chronological", "train": 0.8, "val": 0.0, "test": 0.2, "seed": 1, "group_col": None, "time_col": "day"}
    exp = T.run_selection(df, T.ExperimentConfig(task="regression", target="t", models=["ridge"], split=split, cv={"folds": 4, "gap": 2},
                                                 features_exclude=["day"]))
    assert exp.cv_info["method"] == "timeseries" and "forward-chaining" in exp.selection_basis
    assert len(exp.candidates[0].fold_scores) == 4


def test_invalid_tuning_settings_stop_the_experiment(clf_df):
    with pytest.raises(T.ExperimentError, match="trial limit"):
        T.run_selection(clf_df, T.ExperimentConfig(task="classification", target="y", models=["decision_tree"], split=CV_SPLIT, tuning={"method": "grid", "max_trials": 0}))
