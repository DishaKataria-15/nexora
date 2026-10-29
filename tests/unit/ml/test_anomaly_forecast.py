"""Anomaly detection and forecasting workflows."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nexora.ml import anomaly as A
from nexora.ml import forecasting as F

SAMPLES = Path(__file__).resolve().parents[3] / "sample_data"


@pytest.fixture()
def planted():
    rng = np.random.RandomState(0)
    n = 500
    df = pd.DataFrame({"a": rng.normal(10, 1, n), "b": rng.normal(50, 5, n), "kind": rng.choice(["x", "y"], n)})
    idx = [7, 77, 177, 277, 377]
    df.loc[idx, "a"] = [25, -5, 30, 24, -8]
    df.loc[idx, "b"] = [120, 0, 130, 115, -10]
    df["is_fraud"] = "no"
    df.loc[idx, "is_fraud"] = "yes"
    return df, idx


@pytest.mark.parametrize("method", list(A.METHODS))
def test_every_method_ranks_planted_anomalies_at_the_top(planted, method):
    df, idx = planted
    r = A.run_anomaly(df.drop(columns="is_fraud"), A.AnomalyConfig(methods=[method], contamination=0.01))
    top = {t["row_in_dataset"] for t in r.top[:10]}
    assert set(idx) <= top, (method, top)
    asg = r.assignments()
    assert len(asg) == len(df) and asg.loc[idx, "flagged"].all()


def test_labels_only_evaluate_and_never_become_features(planted):
    df, idx = planted
    r = A.run_anomaly(df, A.AnomalyConfig(methods=["isolation_forest", "robust_z"], contamination=0.01, label_column="is_fraud"))
    assert "is_fraud" not in r.feature_set["numeric"] + r.feature_set["categorical"]
    ev = next(m for m in r.methods if m["key"] == "robust_z")["evaluation"]
    assert ev["known_anomalies"] == 5 and ev["recall"] == 1.0 and ev["pr_auc"] > 0.9
    assert any("only to evaluate" in w for w in r.warnings)
    assert r.selected in ("isolation_forest", "robust_z") and "agreement" in r.methods[0]
    # identical scores with and without the label column present: it had no influence on detection
    r2 = A.run_anomaly(df.drop(columns="is_fraud"), A.AnomalyConfig(methods=["isolation_forest"], contamination=0.01))
    assert np.allclose(r._scores["isolation_forest"], r2._scores["isolation_forest"])


def test_explanations_name_the_extreme_column(planted):
    df, idx = planted
    r = A.run_anomaly(df.drop(columns="is_fraud"), A.AnomalyConfig(methods=["robust_z"]))
    first = r.top[0]
    assert first["row_in_dataset"] in idx and ("a (" in first["why"] or "b (" in first["why"])


def test_row_caps_and_bad_settings(planted, monkeypatch):
    df, _ = planted
    monkeypatch.setitem(A.METHODS, "one_class_svm", (*A.METHODS["one_class_svm"][:2], 100))
    r = A.run_anomaly(df, A.AnomalyConfig(methods=["one_class_svm", "isolation_forest"]))
    assert r.methods[0]["status"] == "skipped" and "limited to 100" in r.methods[0]["error"]
    with pytest.raises(A.AnomalyError, match="between"):
        A.run_anomaly(df, A.AnomalyConfig(contamination=0.9))
    with pytest.raises(A.AnomalyError, match="Unknown"):
        A.run_anomaly(df, A.AnomalyConfig(methods=["magic"]))
    with pytest.raises(A.AnomalyError, match="20 rows"):
        A.run_anomaly(df.head(5), A.AnomalyConfig())


# ---- forecasting -----------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def weekly_daily():
    idx = pd.date_range("2024-01-01", periods=200, freq="D")
    rng = np.random.RandomState(1)
    y = 100 + 0.2 * np.arange(200) + 15 * (idx.dayofweek >= 5) + rng.normal(0, 2, 200)
    return pd.DataFrame({"day": idx.strftime("%Y-%m-%d"), "sales": y}).sample(frac=1, random_state=3)   # unordered rows


def test_backtest_never_uses_future_values(weekly_daily):
    cfg = F.ForecastConfig("day", "sales", horizon=7, folds=3, methods=["ridge_lags", "boosting_lags"])
    a = F.run_forecast(weekly_daily, cfg)
    df = weekly_daily.copy()
    d = pd.to_datetime(df["day"])
    last_cut = pd.Timestamp(a.fold_windows[0]["test_from"])
    df.loc[d >= last_cut + pd.Timedelta(days=7), "sales"] = 1e6           # corrupt everything after the first window
    b = F.run_forecast(df, cfg)
    ea = next(e for e in a.backtest if e["key"] == "ridge_lags")["folds"][0]
    eb = next(e for e in b.backtest if e["key"] == "ridge_lags")["folds"][0]
    assert ea == pytest.approx(eb)                                       # window 1 only ever saw data before its cut-off


def test_lag_models_beat_naive_baselines_on_a_seasonal_trend(weekly_daily):
    r = F.run_forecast(weekly_daily, F.ForecastConfig("day", "sales", horizon=14, folds=3, methods=["ridge_lags", "boosting_lags"]))
    assert r.frequency == "D" and r.n_points == 200
    bt = {e["key"]: e for e in r.backtest}
    assert {"naive", "seasonal_naive"} <= set(bt) and bt["naive"]["baseline"] and not bt["naive"]["requested"]
    assert r.selected in ("ridge_lags", "boosting_lags") and r.beats_baseline
    assert bt["ridge_lags"]["metrics"]["mae"] < bt["seasonal_naive"]["metrics"]["mae"]
    assert len(r.forecast) == 14 and all(f["lower_80"] <= f["forecast"] <= f["upper_80"] for f in r.forecast)
    assert r.forecast[0]["date"] == "2024-07-19"


def test_gaps_and_duplicate_dates_are_reported_and_handled(weekly_daily):
    df = weekly_daily.copy()
    df = df[~df["day"].isin(["2024-02-10", "2024-02-11"])]
    df = pd.concat([df, pd.DataFrame({"day": ["2024-03-01"], "sales": [5.0]})])
    r = F.run_forecast(df, F.ForecastConfig("day", "sales", horizon=7, folds=2, methods=["seasonal_naive"]))
    assert any("2 missing step(s)" in c for c in r.checks) and any("combined by sum" in c for c in r.checks)
    with pytest.raises(F.ForecastError, match="have no value"):
        F.run_forecast(df, F.ForecastConfig("day", "sales", fill="none", methods=["naive"]))


def test_when_the_baseline_wins_it_is_the_forecast():
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    rng = np.random.RandomState(0)
    walk = np.cumsum(rng.normal(0, 1, 120)) + 50                          # a random walk: repeating the last value is optimal
    r = F.run_forecast(pd.DataFrame({"d": idx, "v": walk}), F.ForecastConfig("d", "v", horizon=5, folds=4, methods=["moving_average"]))
    bt = {e["key"]: e for e in r.backtest}
    best = min(bt.values(), key=lambda e: e["metrics"]["mae"])
    assert r.selected == best["key"]                                      # the forecast always comes from the most accurate method
    if best["baseline"]:
        assert not r.beats_baseline and r.best_model == "moving_average" and any("is used for the forecast" in w for w in r.warnings)
    future = F._forecast(r.selected, pd.Series(walk, index=idx), pd.DataFrame(index=idx), pd.DataFrame(index=range(5)), 5, 7, [1, 2, 3, 7], 42)
    assert [f["forecast"] for f in r.forecast] == pytest.approx(list(future))


def test_monthly_series_and_holt_winters():
    idx = pd.date_range("2018-01-01", periods=72, freq="MS")
    y = 200 + 2 * np.arange(72) + 30 * np.sin(2 * np.pi * idx.month / 12)
    df = pd.DataFrame({"month": idx, "units": y})
    methods = ["ridge_lags"] + (["holt_winters"] if F.STATSMODELS_AVAILABLE else [])
    r = F.run_forecast(df, F.ForecastConfig("month", "units", horizon=6, folds=2, methods=methods))
    assert r.frequency.startswith("MS") and r.forecast[0]["date"] == "2024-01-01"
    if F.STATSMODELS_AVAILABLE:
        hw = next(e for e in r.backtest if e["key"] == "holt_winters")
        assert hw["status"] == "ok" and hw["metrics"]["mape"] < 5


def test_known_in_advance_columns_are_used(weekly_daily):
    df = weekly_daily.copy()
    d = pd.to_datetime(df["day"])
    df["promo"] = np.random.RandomState(7).binomial(1, 0.15, len(df))   # random, so calendar features cannot stand in for it
    df["sales"] = df["sales"] + 40 * df["promo"]
    base = F.run_forecast(df, F.ForecastConfig("day", "sales", horizon=14, folds=3, methods=["ridge_lags"]))
    ex = F.run_forecast(df, F.ForecastConfig("day", "sales", horizon=14, folds=3, methods=["ridge_lags"], exog=["promo"]))
    mae = lambda r: next(e for e in r.backtest if e["key"] == "ridge_lags")["metrics"]["mae"]
    assert mae(ex) < mae(base)
    assert any("known in advance" in w for w in ex.warnings)


def test_sample_daily_demand_runs_end_to_end():
    df = pd.read_csv(SAMPLES / "synthetic" / "daily_demand.csv")
    r = F.run_forecast(df, F.ForecastConfig("date", "units_sold", horizon=14, folds=3, methods=["ridge_lags", "boosting_lags"]))
    assert r.n_points >= 400 and r.selected and len(r.forecast) == 14


@pytest.mark.parametrize("kw,msg", [({"horizon": 0}, "horizon"), ({"folds": 50}, "backtest windows"), ({"methods": []}, "at least one"),
                                    ({"methods": ["magic"]}, "Unknown"), ({"horizon": 300}, "too few")])
def test_forecast_settings_are_validated(weekly_daily, kw, msg):
    with pytest.raises(F.ForecastError, match=msg):
        F.run_forecast(weekly_daily, F.ForecastConfig("day", "sales", **kw))


def test_irregular_dates_are_refused():
    t = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.cumsum(np.random.RandomState(0).randint(1, 40, 50)), unit="D")
    with pytest.raises(F.ForecastError, match="not evenly spaced"):
        F.run_forecast(pd.DataFrame({"d": t, "v": np.arange(50.0)}), F.ForecastConfig("d", "v", methods=["naive"]))
