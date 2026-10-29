"""UI coverage for the features added after the first release: every control here drives the real engine."""
import sqlite3

import pandas as pd
import pytest

from .test_ui_flow import _ml, app, click, nav, open_sample  # noqa: F401  (fixtures and helpers)


def _rerun_until_done(app, key="job"):
    app.session_state[key].join(180)
    app.run()
    assert not app.exception, [e.value for e in app.exception]


def test_prepare_outlier_decision_preview_and_override(app):
    open_sample(app, "manufacturing_sensors_messy.csv")
    nav(app, "Prepare Data")
    click(app, "Next: choose goal")
    click(app, "Build preparation plan")
    radios = [r for r in app.radio if r.label == "Outlier treatment"]
    assert len(radios) == 1 and radios[0].value == "report"                 # default: change nothing
    aid = radios[0].key.split("_", 2)[2]
    no = radios[0].key.split("_")[1]
    radios[0].set_value("clip").run()
    app.selectbox(key=f"ol_{no}_{aid}_method").set_value("zscore").run()
    click(app, "Preview effect")
    res = app.session_state[f"ol_{no}_{aid}_result"]
    assert "error" not in res and res["flagged"] >= 1 and res["summary"]["values_modified"]
    click(app, "Confirm and prepare data")
    pr = app.session_state["prep_result"]
    ov = pr.approved["overrides"]
    assert ov and next(iter(ov.values()))["config"]["method"] == "zscore"
    assert any(e.operation == "treat_outliers" and e.values_changed >= 1 and e.parameters["config"]["method"] == "zscore" for e in pr.log)


def test_ml_studio_preparation_tuning_and_results(app):
    _ml(app, "Predict a number (regression)", "house_prices.csv")
    click(app, "Next: Target")
    click(app, "Next: Split")
    click(app, "Next: Preparation")
    app.selectbox(key="ml_w_sel_m").set_value("kbest").run()
    app.number_input(key="ml_w_sel_k").set_value(5).run()
    app.radio(key="ml_w_ol_t").set_value("clip").run()
    click(app, "Next: Models")
    app.radio(key="ml_w_mode").set_value("Choose a model").run()
    app.selectbox(key="ml_w_pick").set_value("ridge").run()
    app.selectbox(key="ml_w_tu_m").set_value("grid").run()
    click(app, "Next: Train")
    click(app, "Start training")
    _rerun_until_done(app)
    exp = app.session_state["exp"]
    c = exp.candidates[0]
    assert c.status == "ok" and c.tuning and len(c.tuning["trials"]) == 5
    assert exp.pipeline_report["selection"]["after"] == 5 and exp.pipeline_report.get("outliers")
    md = " ".join(m.value for m in app.markdown)
    assert "#### Tuning" in md and "What the preparation steps learned" in md
    assert exp.config.tuning["method"] == "grid" and exp.config.preprocess["selection"]["k"] == 5


def test_ml_studio_model_settings_and_compatibility(app):
    _ml(app, "Predict a category (classification)", "customer_churn.csv")
    click(app, "Next: Target")
    click(app, "Next: Split")
    click(app, "Next: Preparation")
    click(app, "Next: Models")
    app.radio(key="ml_w_mode").set_value("Choose a model").run()
    app.selectbox(key="ml_w_pick").set_value("knn").run()
    app.number_input(key="ml_w_mp_knn_n_neighbors").set_value(15).run()
    assert app.session_state["ml"]["model_params"] == {"knn": {"n_neighbors": 15}}
    app.radio(key="ml_w_mode").set_value("Compare models").run()
    assert app.session_state["ml_w_cmp"] == ["gradient_boosting", "random_forest", "logistic_regression"]   # top 3 suggestions


def test_anomalies_page(app):
    open_sample(app, "customer_segments.csv")
    nav(app, "Anomalies")
    click(app, "Find unusual rows")
    res = app.session_state["an_res"]
    assert res.selected and len(res.top) == 50
    assert any(b.label == "Scores (CSV)" for b in app.get("download_button"))


def test_forecasting_page(app):
    open_sample(app, "daily_demand.csv")
    nav(app, "Forecasting")
    app.selectbox(key="fc_target").set_value("units_sold").run()
    click(app, "Backtest and forecast")
    res = app.session_state["fc_res"]
    assert res.selected and len(res.forecast) == 14 and {e["key"] for e in res.backtest} >= {"naive", "seasonal_naive"}
    assert not app.exception


def test_explore_dimensions_tab(app):
    open_sample(app, "customer_segments.csv")
    nav(app, "Visualize & Dashboard")
    click(app, "Draw")
    pr, _ = app.session_state["ex_dim_res"]
    assert pr.method == "pca" and len(pr.points) == 600


def test_datasets_database_tab_reads_sqlite_read_only(app, tmp_path):
    db = tmp_path / "shop.db"
    con = sqlite3.connect(db)
    pd.DataFrame({"id": range(30), "amount": [i * 2.5 for i in range(30)]}).to_sql("orders", con, index=False)
    con.close()
    nav(app, "Datasets")
    app.selectbox(key="db_kind").set_value("sqlite").run()
    app.text_input(key="db_path").input(str(db)).run()
    click(app, "", key="db_connect")
    assert app.session_state["db_tables"][0]["name"] == "orders"
    app.number_input(key="db_limit").set_value(10).run()
    click(app, "", key="db_add")
    h = app.session_state["handle"]
    assert h.source == "database" and len(h.raw) == 10


def test_datasets_web_api_tab_blocks_internal_addresses(app):
    nav(app, "Datasets")
    app.text_input(key="api_url").input("http://127.0.0.1:9/data").run()
    click(app, "", key="api_go")
    assert any("internal" in e.value for e in app.error)
    assert app.session_state["handle"] is None


def test_reports_use_a_saved_model(app):
    _ml(app, "Predict a number (regression)", "house_prices.csv")
    click(app, "Next: Target")
    click(app, "Next: Split")
    click(app, "Next: Preparation")
    click(app, "Next: Models")
    click(app, "Next: Train")
    click(app, "Start training")
    _rerun_until_done(app)
    nav(app, "Reports")
    app.radio(key="pr_src").set_value("The dataset that is open now").run()
    click(app, "", key="pr_go")
    out, notes = app.session_state["pr_out"]
    assert "predicted_price_thousands_usd" in out.columns and len(out) == 900
    assert any("price_thousands_usd" in n for n in notes)               # the target column is reported as ignored


def test_comparison_can_show_several_metrics_but_one_rule_selects(app):
    _ml(app, "Predict a category (classification)", "customer_churn.csv")
    click(app, "Next: Target")
    click(app, "Next: Split")
    click(app, "Next: Preparation")
    click(app, "Next: Models")
    app.radio(key="ml_w_mode").set_value("Compare models").run()
    app.multiselect(key="ml_w_extra").set_value(["recall", "precision", "roc_auc"]).run()
    click(app, "Next: Train")
    click(app, "Start training")
    _rerun_until_done(app)
    exp = app.session_state["exp"]
    assert exp.config.extra_metrics == ["recall", "precision", "roc_auc"]
    table = next(d for d in app.dataframe if "Model" in d.value.columns)
    assert {"Recall", "Precision", "ROC-AUC"} <= set(table.value.columns)
    best = max((c for c in exp.candidates if c.status == "ok"), key=lambda c: c.score)
    assert exp.selected_key == best.key                                     # selection still follows the primary metric only
    row = table.value[table.value["Model"] == best.name].iloc[0]
    assert row["Recall"] == pytest.approx(best.metrics["recall"])


def test_next_steps_after_opening_a_dataset_and_after_preparation(app):
    open_sample(app, "customer_churn.csv")
    labels = [b.label for b in app.button]
    for lab in ("Prepare this data", "Visualize this data", "Build a model", "Find unusual rows", "Forecast over time"):
        assert lab in labels
    click(app, "Build a model")
    assert app.session_state["nav"] == "ML Studio"
    nav(app, "Prepare Data")
    click(app, "Next: choose goal")
    click(app, "Build preparation plan")
    click(app, "Confirm and prepare data")
    assert "Prepare this data" not in [b.label for b in app.button]        # already done
    rec = next(b for b in app.button if b.label == "Build a model")
    assert rec.proto.type == "primary"                                     # the churn sample's goal is supervised ML
    click(app, "Forecast over time")
    assert app.session_state["nav"] == "Forecasting"


def test_every_working_page_can_open_a_dataset_itself_and_shows_a_summary(app):
    nav(app, "ML Studio")
    assert any("Open a dataset first" in m.value for m in app.markdown)
    app.selectbox(key="pk_smp_ml").set_value("House prices (regression demo)").run()
    click(app, "", key="pk_osmp_ml")
    assert app.session_state["handle"].name == "House prices (regression demo)" and app.session_state["nav"] == "ML Studio"
    cap = " ".join(c.value for c in app.caption)
    assert "900 rows x 10 columns" in cap and "% of cells missing" in cap
    for page in ("Prepare Data", "Visualize & Dashboard", "Anomalies", "Forecasting"):
        nav(app, page)
        assert any("**Dataset:** House prices" in m.value for m in app.markdown), page


def test_dashboard_palette_and_per_chart_colour(app):
    open_sample(app, "house_prices.csv")
    nav(app, "Visualize & Dashboard")
    app.selectbox(key="chart_palette").set_value("Colour-blind safe").run()
    click(app, "", key="add_c1")
    spec = app.session_state["dashboard"][0]
    from nexora.eda import charts as C
    from nexora.ui import state
    df = state.ws().autotype(app.session_state["handle"].raw)
    fig = C.build_figure(df, C.ChartSpec.from_dict(spec), "Colour-blind safe")
    assert fig.layout.colorway[0] == "#0072B2"
    app.color_picker[0].set_value("#ff00aa").run()
    assert app.session_state["dashboard"][0]["accent"] == "#ff00aa"
    fig = C.build_figure(df, C.ChartSpec.from_dict(app.session_state["dashboard"][0]), "Colour-blind safe")
    assert fig.layout.colorway[0] == "#ff00aa"
    click(app, "Use palette colour")
    assert app.session_state["dashboard"][0]["accent"] is None
    app.run()
    assert app.session_state["dashboard"][0]["accent"] is None             # the reset sticks (picker does not re-apply the old colour)
    assert app.color_picker[0].value.lower() == "#0072b2"
    assert not app.exception


def test_combine_two_datasets_from_the_datasets_page(app):
    import pandas as pd
    from nexora.ui import state
    w = state.ws()
    w.ingest_frame(pd.DataFrame({"user": ["u1", "u2", "u3", "u1"], "movie_id": ["1", "1", "2", "3"], "rating": ["4", "5", "3", "2"]}), "ratings", "upload")
    w.ingest_frame(pd.DataFrame({"movie_id": ["1", "2"], "title": ["Alpha", "Beta"]}), "movies", "upload")
    nav(app, "Datasets")
    app.multiselect(key="cb_sel").set_value(["ratings (4 rows)", "movies (2 rows)"]).run()
    assert app.multiselect(key="cb_l_1_movies (2 rows)").value == ["movie_id"]      # key suggested from names and values
    click(app, "", key="cb_prev")
    head, reps, n, _ = app.session_state["cb_preview"]
    assert n == 4 and reps[0].matched_rows == 3 and reps[0].unmatched_main == 1
    assert any("found no match" in w_.value for w_ in app.warning)
    click(app, "", key="cb_save")
    h = app.session_state["handle"]
    assert h.source == "combined" and list(h.raw.columns) == ["user", "movie_id", "rating", "title"] and len(h.raw) == 4
    app.radio(key="cb_mode_1_movies (2 rows)")                                      # widgets still there after saving


def test_help_assistant_in_the_sidebar(app):
    nav(app, "Datasets")
    app.sidebar.text_input(key="help_q").input("how do I join two files")             # typed, then submitted with the form
    next(b for b in app.sidebar.button if b.label == "Ask").click().run()
    hist = app.session_state["help_history"]
    assert hist[0]["found"] and hist[0]["id"] == "combine"
    assert any("Combine datasets (advanced)" in m.value for m in app.sidebar.markdown)
    app.sidebar.text_input(key="help_q").input("what is the weather")
    next(b for b in app.sidebar.button if b.label == "Ask").click().run()
    assert not app.session_state["help_history"][0]["found"]
    assert any("only answer questions about using Nexora" in m.value for m in app.sidebar.markdown)
    nav(app, "ML Studio")
    sug = next(b for b in app.sidebar.button if b.label == "What is leakage?")       # suggestions follow the page
    sug.click().run()
    assert app.session_state["help_history"][0]["id"] == "leakage"
    go = next((b for b in app.sidebar.button if b.label.startswith("Go to")), None)
    assert go is None                                                                  # already on ML Studio
    assert not app.exception


def test_leaky_columns_are_excluded_by_default_and_can_be_included_on_purpose(app):
    _ml(app, "Predict a category (classification)", "customer_churn.csv")
    click(app, "Next: Target")
    m = app.session_state["ml"]
    assert m["exclude"] == ["cancellation_reason"] and app.multiselect(key="ml_w_exclude").value == ["cancellation_reason"]
    click(app, "", key="keep_leak")
    m = app.session_state["ml"]
    assert m["exclude"] == [] and m["keep_leaky"]["churned"] == ["cancellation_reason"]
    assert any("may leak the answer" in e.value for e in app.error)                 # the warning stays while it is included
    app.run()
    assert app.session_state["ml"]["exclude"] == []                                  # not re-excluded on the next run
    app.multiselect(key="ml_w_exclude").set_value(["cancellation_reason", "region"]).run()
    assert app.session_state["ml"]["exclude"] == ["cancellation_reason", "region"]   # manual choices are kept


def test_switching_target_undoes_automatic_exclusions(app):
    _ml(app, "Predict a category (classification)", "customer_churn.csv")
    click(app, "Next: Target")
    assert "cancellation_reason" in app.session_state["ml"]["exclude"]
    other = [o for o in app.selectbox(key="ml_w_target").options if o != "churned"][0]
    app.selectbox(key="ml_w_target").set_value(other).run()
    assert "cancellation_reason" not in app.session_state["ml"]["exclude"] or other == "cancellation_reason"
