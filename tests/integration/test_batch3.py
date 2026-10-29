"""Batch 3, ML Studio: top-3 comparison by default, a suggested goal, a caveat beside too-good test scores, a fuller
review before training, clearer clustering wording and descriptive group names."""
import numpy as np
import pandas as pd

from nexora.ml.task_detection import suggest_goal
from nexora.ml.training import describe_groups
from nexora.ui.pages.ml_studio import score_caveat

from .test_ui_flow import app, click, nav, open_sample  # noqa: F401

SAMPLES = "sample_data/synthetic/"


def test_goal_is_suggested_from_the_data():
    churn = pd.read_csv(SAMPLES + "customer_churn.csv", dtype=str)
    g = suggest_goal(churn)
    assert g["task"] == "classification" and g["column"] == "churned"
    rng = np.random.RandomState(0)
    plain = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60), "c": list("xyz") * 20})
    assert suggest_goal(plain)["task"] == "clustering" and suggest_goal(plain)["column"] is None
    assert suggest_goal(plain[["a", "c"]])["task"] == "explore"
    house = pd.DataFrame({"area": rng.normal(1000, 100, 60), "price_usd": rng.normal(300, 50, 60)})
    assert suggest_goal(house)["task"] == "regression" and suggest_goal(house)["column"] == "price_usd"
    assert suggest_goal(plain.assign(annual_income_k=rng.normal(50, 9, 60)))["task"] == "clustering"   # income is usually an input


def test_goal_step_shows_and_preselects_the_suggestion(app):
    open_sample(app, "customer_churn.csv")
    nav(app, "ML Studio")
    assert any("Suggested" in m.value and "churned" in m.value for m in app.markdown)
    assert app.radio(key="ml_w_goal").value == "classification"


def test_caveat_for_near_perfect_or_leaky_scores():
    assert score_caveat("classification", {"accuracy": 0.81}, [], []) is None
    msg = score_caveat("classification", {"accuracy": 0.995}, [], [])
    assert msg and "close to perfect" in msg
    leak = [{"level": "warning", "columns": ["refund_date"], "message": "x"}]
    msg = score_caveat("regression", {"r2": 0.7}, leak, [])
    assert "refund_date" in msg and "only known after the outcome" in msg
    assert score_caveat("clustering", {"silhouette": 0.99}, leak, []) is None


def test_groups_get_descriptive_names():
    df = pd.DataFrame({"income": [10.0, 12, 11, 90, 95, 92], "age": [20.0, 22, 21, 60, 62, 61], "city": ["A", "A", "A", "B", "B", "A"]})
    names = describe_groups(df, [0, 0, 0, 1, 1, 1], ["income", "age"], ["city"])
    assert "Low income" in names[0] or "Low age" in names[0]
    assert "high" in names[1].lower()
    assert describe_groups(df, [0, 0, 0, -1, -1, -1], ["income"], [])[-1] == "Noise (no group)"
    flat = pd.DataFrame({"x": [1.0, 2, 1, 2]})
    assert describe_groups(flat, [0, 1, 0, 1], ["x"], [])[0] in ("Low x", "Typical rows")


def _to_train(app, fname, goal):
    open_sample(app, fname)
    nav(app, "ML Studio")
    app.radio(key="ml_w_goal").set_value(goal).run()
    if goal == "clustering":
        click(app, "Next: Features")
    else:
        click(app, "Next: Target")
        click(app, "Next: Split")
        click(app, "Next: Preparation")
    click(app, "Next: Models")
    click(app, "Next: Train")


def test_review_table_lists_everything_that_decides_the_result(app):
    _to_train(app, "customer_churn.csv", "classification")
    table = app.dataframe[0].value
    settings = set(table["Setting"])
    for s in ("Input columns", "Left out", "Split", "How models are compared", "Preparation inside the model", "Class you care most about", "Balancing"):
        assert s in settings, s
    vals = dict(zip(table["Setting"], table["Value"]))
    assert "cancellation_reason" in vals["Left out"] and "train" in vals["Split"] and "rows" in vals["Split"]


def test_clustering_results_use_grouping_words_and_named_groups(app):
    _to_train(app, "customer_segments.csv", "clustering")
    click(app, "Start training")
    app.session_state["job"].join(180)
    app.run()
    app.run()
    md = " ".join(m.value for m in app.markdown)
    assert "Grouping methods compared" in md and "Chosen grouping" in md and "not correct answers" in md
    prof = app.session_state["exp"].cluster_profile
    assert all(r.get("name") for r in prof)


def test_group_names_use_dates_and_never_repeat():
    df = pd.DataFrame({"d": pd.date_range("2026-01-01", periods=8, freq="D"), "x": [1.0] * 8})
    names = describe_groups(df, [0, 0, 0, 0, 1, 1, 1, 1], ["x"], [], dates=["d"])
    assert names == {0: "Earlier d", 1: "Later d"}
    same = pd.DataFrame({"x": [1.0, 1.0, 1.0, 1.0]})
    n2 = describe_groups(same, [0, 0, 1, 1], ["x"], [])
    assert len(set(n2.values())) == 2                       # identical names get the group number added
