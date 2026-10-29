"""Batch 4: mixed units and flagged summary in Anomalies, per-window backtest errors, clearer dashboard adds, the
Columns tab opening on a chartable column, rename/delete datasets, newest prepared version preselected, a simpler
column choice in Combine, comparing experiments, and saving anomaly/forecast runs in Reports."""
import numpy as np
import pandas as pd
import pytest

from nexora.ml.anomaly import mixed_unit_warnings
from nexora.persistence.repository import SqliteRepository
from nexora.ui.pages.forecasting import window_table
from nexora.ui.pages.reports import compare_table

from .test_ui_flow import app, click, nav, open_sample  # noqa: F401

SAMPLES = "sample_data/synthetic/"


def test_mixed_units_are_found_and_case_is_ignored():
    sensors = pd.read_csv(SAMPLES + "manufacturing_sensors_messy.csv", dtype=str, keep_default_na=False)
    [w] = mixed_unit_warnings(sensors)
    assert w["value_cols"] == ["temperature"] and set(w["units"]) == {"C", "F"}
    retail = pd.read_csv(SAMPLES + "retail_orders_messy.csv", dtype=str, keep_default_na=False)
    assert mixed_unit_warnings(retail) == []                      # 'USD' and 'usd' are one unit
    assert mixed_unit_warnings(sensors, exclude=["temperature"]) == []


def test_anomaly_page_warns_about_units_and_summarises_flags(app):
    open_sample(app, "manufacturing_sensors_messy.csv")
    nav(app, "Anomalies")
    assert any("more than one unit" in w.value for w in app.warning)
    click(app, "Leave 'temperature' out")
    assert "temperature" in app.session_state["an_excl"] and not any("more than one unit" in w.value for w in app.warning)
    click(app, "Find unusual rows")
    assert any(" rows flagged (" in m.value and m.value.startswith("#### ") for m in app.markdown)
    click(app, "Save to Reports")
    nav(app, "Reports")
    assert any("Anomalies" in str(df.value.get("Kind", pd.Series()).tolist()) for df in app.dataframe)


def test_forecast_window_table_and_saving(app):
    open_sample(app, "daily_demand.csv")
    nav(app, "Forecasting")
    click(app, "Backtest and forecast")
    tab = window_table(app.session_state["fc_res"])
    wcols = [c for c in tab.columns if c.startswith("Window")]
    assert len(wcols) == 3 and "Worst / best" in tab and len(tab) >= 3
    assert any("Error in each backtest window" in e.label for e in app.expander)
    click(app, "Save to Reports")
    assert app.button(key="fc_save").disabled


def test_dashboard_add_shows_a_lasting_confirmation(app):
    open_sample(app, "house_prices.csv")
    nav(app, "Visualize & Dashboard")
    assert app.selectbox(key="ex_col").value != "house_id"         # opens on a column that has a chart
    click(app, "Add to dashboard")
    assert len(app.session_state["dashboard"]) >= 1
    assert any("On the dashboard" in s.value for s in app.success)


def test_rename_and_delete_dataset(tmp_path):
    repo = SqliteRepository(tmp_path)
    did = repo.add_dataset("a", "a.csv", b"x\n1\n", "h", 1, 1, "upload")
    vid = repo.save_version(did, pd.DataFrame({"x": [1]}), {})
    repo.save_experiment(vid, {"config": {"task": "regression"}, "status": "finalized"})
    repo.save_experiment(None, {"config": {"task": "regression"}, "status": "finalized"})
    repo.rename_dataset(did, "  new   name ")
    assert repo.get_dataset(did)["name"] == "new name"
    with pytest.raises(ValueError):
        repo.rename_dataset(did, "   ")
    gone = repo.delete_dataset(did)
    assert gone == {"versions": 1, "experiments": 1}
    assert repo.get_dataset(did) is None and repo.list_versions(did) == [] and len(repo.list_experiments()) == 1
    assert not (tmp_path / "raw" / did).exists()


def test_runs_are_saved_listed_and_deleted(tmp_path):
    repo = SqliteRepository(tmp_path)
    rid = repo.save_run("forecast", "demand", None, {"headline": "h", "table": [{"a": 1}]}, {"forecast.csv": b"d,v\n"})
    [r] = repo.list_runs()
    assert r["kind"] == "forecast" and r["summary"]["headline"] == "h"
    assert repo.get_run(rid)["files"] == {"forecast.csv": b"d,v\n"}
    repo.delete_run(rid)
    assert repo.list_runs() == [] and repo.get_run(rid) is None


def test_datasets_page_rename_delete_and_prepared_default(app):
    open_sample(app, "retail_orders_messy.csv")
    nav(app, "Prepare Data")
    click(app, "Next: choose goal")
    click(app, "Build preparation plan")
    click(app, "Confirm and prepare data")
    nav(app, "Datasets")
    vsel = next(s for s in app.selectbox if s.label == "Version to work with")
    assert vsel.value.startswith("Prepared")                          # newest prepared version preselected
    did = app.session_state["handle"].dataset_id
    app.text_input(key=f"rn_{did}").set_value("My retail data")
    click(app, "Rename")
    assert app.session_state["handle"].name == "My retail data" and any("Renamed" in s.value for s in app.success)
    app.checkbox(key=f"del_ok_{did}").check().run()
    click(app, "Delete dataset")
    assert app.session_state["handle"] is None and any("Deleted" in s.value for s in app.success)


def test_compare_table_flags_settings_that_differ():
    base = {"config": {"task": "classification", "primary_metric": "f1", "seed": 42, "dataset_version": "v1",
                       "split": {"strategy": "stratified", "train": 0.7, "val": 0.15, "test": 0.15}},
            "candidates": [{"key": "rf|Random forest", "name": "Random forest", "score": 0.8}], "selected_key": "rf|Random forest",
            "test_evaluated": True, "test_result": {"metrics": {"f1": 0.78, "accuracy": 0.9}}}
    other = {**base, "config": {**base["config"], "seed": 7}, "test_evaluated": False}
    tab, notes = compare_table([("A", base), ("B", other)])
    assert list(tab["Experiment"]) == ["A", "B"] and tab.loc[0, "Test f1"] == 0.78 and pd.isna(tab.loc[1, "Test f1"])
    assert any("seed differs" in n for n in notes)
    _, same = compare_table([("A", base), ("C", base)])
    assert same == []
