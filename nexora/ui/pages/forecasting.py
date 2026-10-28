"""Forecasting page: backtest simple and machine-learning forecasts honestly, then forecast the future."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...eda import charts as C
from ...eda import profiling as P
from ...ml import forecasting as F
from ...reporting import zip_bytes
from .. import components, theme


def render() -> None:
    st.title("Forecasting")
    wf = components.dataset_bar("forecasting", empty_body="Forecasting needs a dataset with a date column and a number to forecast.")
    if wf is None:
        return
    df, note = wf
    st.markdown("Forecast a number over time. Every method is **backtested**: it is trained only on data before a cut-off and scored on what "
                "happened next, several times. Simple baselines are always included so you can see whether a model adds anything.")
    st.caption(note)
    dts = P.datetime_columns(df)
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    if not dts or not nums:
        theme.empty_state("No date column or no number", "Forecasting needs one date column and one numeric column. Prepare the data first so dates are recognised.")
        return
    S = st.session_state
    c1, c2 = st.columns(2)
    date_col = c1.selectbox("Date column", dts, key="fc_date")
    tcols = [c for c in nums if c != date_col]
    target = c2.selectbox("Value to forecast", tcols, index=len(tcols) - 1, key="fc_target", help="Usually the last numeric column is the outcome; change it if not.")
    c1, c2, c3 = st.columns(3)
    horizon = int(c1.number_input("Steps to forecast", 1, F.MAX_HORIZON, 14, key="fc_h"))
    folds = int(c2.number_input("Backtest windows", 1, F.MAX_FOLDS, 3, key="fc_folds"))
    season = int(c3.number_input("Season length in steps (0 = automatic)", 0, 400, 0, key="fc_season", help="7 for daily data with a weekly pattern, 12 for monthly."))
    methods = st.multiselect("Methods", [k for k in F.METHODS if k not in F.BASELINES and (k != "holt_winters" or F.STATSMODELS_AVAILABLE)],
                             default=["ridge_lags", "boosting_lags"], format_func=F.METHODS.get, key="fc_methods")
    if not F.STATSMODELS_AVAILABLE:
        st.caption("Holt-Winters is optional (pip install statsmodels).")
    with st.expander("More settings"):
        exog = st.multiselect("Columns known in advance (for example a planned promotion)", [c for c in nums if c not in (target,)], key="fc_exog",
                              help="Only use columns whose future values you will know. Using something you only learn afterwards makes the backtest unrealistically good.")
        agg = st.selectbox("If several rows share a date, combine them by", ["sum", "mean", "last"], key="fc_agg")
        fill = st.selectbox("Fill missing dates by", ["interpolate", "zero", "ffill", "none"], key="fc_fill",
                            format_func={"interpolate": "Interpolating", "zero": "Zero", "ffill": "Carrying the last value forward", "none": "Do not fill (stop instead)"}.get)
    if st.button("Backtest and forecast", type="primary", key="fc_go", disabled=not methods):
        cfg = F.ForecastConfig(date_col, target, horizon=horizon, season=season or None, methods=methods, folds=folds, exog=exog, aggregate=agg, fill=fill,
                               seed=int(S.get("seed") or 42))
        try:
            with st.spinner("Backtesting"):
                S["fc_res"] = F.run_forecast(df, cfg)
        except F.ForecastError as e:
            st.error(str(e))
    res = S.get("fc_res")
    if res is None:
        return
    st.markdown(f"**{res.n_points:,} time steps**, frequency `{res.frequency}`.")
    for c in res.checks:
        st.info(c)
    for w_ in res.warnings:
        st.warning(w_)
    rows = [{"Method": e["name"] + (" (baseline)" if e["baseline"] else ""), "Status": e["status"], "MAE": (e.get("metrics") or {}).get("mae"),
             "RMSE": (e.get("metrics") or {}).get("rmse"), "MAPE %": (e.get("metrics") or {}).get("mape"), "sMAPE %": (e.get("metrics") or {}).get("smape"),
             "Seconds": e["seconds"], "Note": e.get("error", "")} for e in res.backtest]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={k: st.column_config.NumberColumn(format="%.4g") for k in ("MAE", "RMSE", "MAPE %", "sMAPE %")})
    st.caption("Backtest windows: " + "; ".join(f"trained until {w['train_until']}, tested {w['test_from']} to {w['test_until']}" for w in res.fold_windows))
    st.markdown(f"**Selected:** {F.METHODS[res.selected]} (lowest average MAE in the backtest)." + ("" if res.beats_baseline else " It does not beat the baselines."))
    hist = pd.DataFrame(res.history)
    fc = pd.DataFrame(res.forecast)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["value"], mode="lines", name="history", line={"color": C.NAVY}))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["upper_80"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["lower_80"], mode="lines", line={"width": 0}, fill="tonexty", fillcolor="rgba(15,118,110,0.18)", name="rough 80% range"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc["forecast"], mode="lines+markers", name="forecast", line={"color": C.TEAL}))
    bt = pd.DataFrame(res.backtest_predictions)
    if not bt.empty:
        part = bt[bt["method"] == res.selected]
        fig.add_trace(go.Scatter(x=part["date"], y=part["predicted"], mode="markers", name="backtest predictions", marker={"color": C.MUTED, "size": 5}))
    st.plotly_chart(C._layout(fig, f"{target}: history and forecast", 420), width="stretch")
    st.caption("The range comes from the backtest errors at each step ahead. It is a rough guide, not a guarantee.")
    st.dataframe(fc, hide_index=True, width="stretch")
    files = {"forecast.csv": fc.to_csv(index=False).encode(), "backtest_predictions.csv": bt.to_csv(index=False).encode(),
             "forecast_summary.json": json.dumps(res.to_dict(), indent=2, default=str).encode()}
    c1, c2, _ = st.columns([1.4, 1.4, 4])
    c1.download_button("Forecast (CSV)", files["forecast.csv"], file_name="forecast.csv", mime="text/csv", key="fc_dl")
    c2.download_button("Everything (ZIP)", zip_bytes(files), file_name="nexora_forecast.zip", mime="application/zip", key="fc_zip")
