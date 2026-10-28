"""Drives the real Streamlit app (widgets, buttons, navigation) with Streamlit's AppTest."""
import pytest
from streamlit.testing.v1 import AppTest

from nexora.ml.training import TestSetAlreadyUsed, finalize_experiment
from nexora.ui.state import PAGES

from .conftest import ROOT

APP = str(ROOT / "app.py")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXORA_DATA_DIR", str(tmp_path / "data"))
    from nexora.ui import state
    state.workspace.clear()            # each test gets its own workspace and data folder
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception
    return at


def nav(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def click(at, label, key=None):
    btn = at.button(key=key) if key else next((b for b in at.button if b.label.startswith(label)), None)
    assert btn is not None, (label, [b.label for b in at.button])
    btn.click().run()
    assert not at.exception, [e.value for e in at.exception]


def open_sample(at, fname):
    nav(at, "Datasets")
    click(at, "", key=f"smp_{fname}")


def test_every_page_renders_on_an_empty_workspace_with_a_useful_empty_state(app):
    for page in PAGES:
        nav(app, page)
        assert app.title and app.title[0].value
    nav(app, "Home")
    assert any("No dataset open" in m.value for m in app.markdown)
    for page in ("Prepare Data", "Visualize & Dashboard", "ML Studio"):
        nav(app, page)
        assert any("Open a dataset first" in m.value for m in app.markdown) and any(b.label == "Go to Datasets" for b in app.button)
    nav(app, "Reports")
    assert any("No experiments yet" in m.value for m in app.markdown)


def test_navigation_buttons_actually_navigate(app):
    nav(app, "Prepare Data")
    click(app, "Go to Datasets")
    assert app.session_state["nav"] == "Datasets"


def test_sample_data_is_labelled_as_sample(app):
    nav(app, "Datasets")
    text = " ".join(m.value for m in app.markdown)
    assert "SAMPLE DATA" in text and "invented" in text
    click(app, "", key="smp_customer_churn.csv")
    nav(app, "Home")
    assert any("SAMPLE DATA" in m.value for m in app.markdown)


def test_prepare_flow_with_decisions_matches_engine_results(app):
    open_sample(app, "retail_orders_messy.csv")
    nav(app, "Prepare Data")
    click(app, "Next: choose goal")
    assert app.session_state["profile_id"] == "retail" and app.session_state["goal"] == "analytics"
    click(app, "Build preparation plan")
    labels = [c.label for c in app.checkbox]
    assert any("Remove repeated copies" in lab for lab in labels) and any("Standardize dates" in lab for lab in labels)
    optional = [c for c in app.checkbox if "missing value(s) with typical values" in c.label or "Mark extreme" in c.label]
    assert optional and not any(c.value for c in optional)                          # optional steps default to OFF
    radios = [r for r in app.radio if r.label == "Choose"]
    assert len(radios) == 3 and radios[0].value == "leave" and radios[2].value == "keep"          # decisions default to "leave unchanged"
    radios[1].set_value("dmy")
    radios[2].set_value("drop")
    app.run()
    click(app, "Confirm and prepare data")
    res = app.session_state["prep_result"]
    assert app.session_state["prep_step"] == 3 and app.session_state["version_id"]
    assert res.after["rows"] == 228 - 5 - 3 and res.reconciliation["ok"] and res.source_unchanged
    assert any("reconciliation checks passed" in s.value for s in app.success)
    assert len(list(app.get("download_button"))) == 5
    click(app, "Visualize this data")
    assert app.session_state["nav"] == "Visualize & Dashboard"


def test_untick_a_safe_step_and_it_is_not_applied(app):
    open_sample(app, "retail_orders_messy.csv")
    nav(app, "Prepare Data")
    click(app, "Next: choose goal")
    click(app, "Build preparation plan")
    next(c for c in app.checkbox if "Remove extra spaces" in c.label).set_value(False)
    app.run()
    click(app, "Confirm and prepare data")
    assert app.session_state["prep_result"].prepared["sku"].astype(str).str.endswith(" ").sum() == 7


def test_explore_dashboard_add_filter_reorder_remove_export(app):
    open_sample(app, "customer_churn.csv")
    nav(app, "Visualize & Dashboard")
    assert any("Not prepared" in m.value for m in app.markdown)
    assert [t.label for t in app.tabs] == ["Overview", "Columns", "Relationships", "Charts", "Dashboard", "Dimensions"]
    for key in ("add_c1", "add_c2", "add_c3"):
        if any(b.key == key for b in app.button):
            click(app, "", key=key)
    dash = app.session_state["dashboard"]
    assert len(dash) >= 2
    first, second = dash[0]["spec_id"], dash[1]["spec_id"]
    app.multiselect(key="dash_cols").set_value(["contract"]).run()
    app.multiselect(key="df_contract").set_value(["Two year"]).run()
    assert not app.exception and any("Filters: contract in Two year" in m.value for m in app.markdown)
    click(app, "", key=f"dn_{first}")
    assert app.session_state["dashboard"][1]["spec_id"] == first and app.session_state["dashboard"][0]["spec_id"] == second
    click(app, "", key=f"rm_{first}")
    assert all(d["spec_id"] != first for d in app.session_state["dashboard"])
    assert any(d.label == "Download dashboard (HTML)" for d in app.get("download_button"))
    click(app, "Clear dashboard")
    assert app.session_state["dashboard"] == []


def test_explore_handles_identifier_columns_gracefully(app):
    open_sample(app, "customer_churn.csv")
    nav(app, "Visualize & Dashboard")
    app.selectbox(key="ex_col").set_value("customer_id").run()
    assert not app.exception and any("Identifiers are unique per row" in c.value for c in app.caption)


def _ml(app, task_label, fname):
    open_sample(app, fname)
    nav(app, "ML Studio")
    app.radio(key="ml_w_goal").set_value(task_label).run()


def test_ml_classification_path_with_leakage_and_single_test_evaluation(app):
    _ml(app, "Predict a category (classification)", "customer_churn.csv")
    assert any("not been prepared" in w.value for w in app.warning)
    click(app, "Next: Target")
    assert app.session_state["ml"]["target"] == "churned"
    assert any("cancellation_reason" in e.value for e in app.error)                  # leakage is called out
    click(app, "Exclude suspected leaky")
    assert app.session_state["ml"]["exclude"] == ["cancellation_reason"]
    click(app, "Next: Split")
    click(app, "Next: Preparation")
    assert any("recommended" in o for o in app.radio(key="ml_w_imb").options)
    click(app, "Next: Models")
    app.radio(key="ml_w_mode").set_value("Compare models").run()
    assert len(app.session_state["ml"]["models"]) == 4
    assert app.radio(key="ml_w_comp").options == ["Automatic (recommended)", "CPU"]              # no GPU offered when none is detected
    click(app, "Next: Train")
    click(app, "Start training")
    app.session_state["job"].join(120)
    app.run()
    assert not app.exception and app.session_state["ml"]["step"] == "Results"
    exp = app.session_state["exp"]
    assert exp.status == "selection_complete" and exp.test_result is None
    assert any("seen the test rows" in m.value for m in app.markdown)
    click(app, "Evaluate selected model on the test set")
    exp = app.session_state["exp"]
    assert exp.status == "finalized" and exp.test_result["metrics"]["roc_auc"] is not None
    assert not any(b.label.startswith("Evaluate selected") for b in app.button)                   # the one-time button is gone
    with pytest.raises(TestSetAlreadyUsed):
        finalize_experiment(exp)
    nav(app, "Reports")
    assert len(app.dataframe) >= 1


def test_ml_regression_path_shows_no_classification_controls(app):
    _ml(app, "Predict a number (regression)", "house_prices.csv")
    click(app, "Next: Target")
    assert app.session_state["ml"]["target"] == "price_thousands_usd"
    click(app, "Next: Split")
    click(app, "Next: Preparation")
    assert not any(r.key == "ml_w_imb" for r in app.radio) and not any("imbalance" in m.value.lower() for m in app.markdown)
    click(app, "Next: Models")
    assert app.session_state["ml"]["metric"] == "rmse"
    click(app, "Next: Train")
    click(app, "Start training")
    app.session_state["job"].join(120)
    app.run()
    assert app.session_state["exp"].config.task == "regression" and not app.exception


def test_ml_clustering_path_needs_no_target_and_no_split(app):
    _ml(app, "Discover groups (clustering)", "customer_segments.csv")
    click(app, "Next: Features")
    assert app.session_state["ml"]["step"] == "Features"
    assert any("No target is needed" in c.value for c in app.caption)
    click(app, "Next: Models")
    assert app.session_state["ml"]["metric"] == "silhouette"
    app.radio(key="ml_w_mode").set_value("Compare models").run()
    click(app, "Next: Train")
    click(app, "Start training")
    app.session_state["job"].join(120)
    app.run()
    exp = app.session_state["exp"]
    assert exp.status == "finalized" and not any(b.label.startswith("Evaluate selected") for b in app.button)
    assert any("not accuracy" in m.value for m in app.markdown)


def test_ml_explore_only_goal_sends_user_to_explore(app):
    _ml(app, "Explore and visualize only", "customer_churn.csv")
    assert not any(b.label.startswith("Next:") for b in app.button)
    click(app, "Open Visualize & Dashboard")
    assert app.session_state["nav"] == "Visualize & Dashboard"


def test_export_failure_is_shown_as_a_message_not_a_crash(app, monkeypatch):
    _ml(app, "Predict a number (regression)", "house_prices.csv")
    for lbl in ("Next: Target", "Next: Split", "Next: Preparation", "Next: Models", "Next: Train", "Start training"):
        click(app, lbl)
    app.session_state["job"].join(120)
    app.run()
    from nexora.workflow import Workspace

    def broken(self, exp, name):
        raise OSError("disk full")
    monkeypatch.setattr(Workspace, "experiment_exports", broken)
    app.run()
    assert not app.exception and any("Could not build the downloads" in e.value and "disk full" in e.value for e in app.error)


def test_settings_seed_and_delete_everything(app):
    open_sample(app, "house_prices.csv")
    nav(app, "Settings")
    app.number_input[0].set_value(123).run()
    assert app.session_state["seed"] == 123
    assert next(b for b in app.button if b.label == "Delete everything").disabled                 # needs the confirmation tick first
    app.checkbox(key="del_ok").set_value(True).run()
    click(app, "Delete everything")
    assert app.session_state["handle"] is None and any("deleted" in s.value for s in app.success)
    nav(app, "Datasets")
    assert any("Nothing saved yet" in m.value for m in app.markdown)
