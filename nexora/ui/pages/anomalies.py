"""Anomaly detection page: score rows for how unusual they are, look at the most unusual ones, export the scores."""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...eda import charts as C
from ...ml import anomaly as A
from ...reporting import zip_bytes
from .. import components, state, theme


def _leave_out(cols: list[str]) -> None:
    cur = list(st.session_state.get("an_excl") or [])
    st.session_state["an_excl"] = cur + [c for c in cols if c not in cur]


def render() -> None:
    st.title("Anomalies")
    wf = components.dataset_bar("anomalies", empty_body="Anomaly detection works on a dataset you have opened.")
    if wf is None:
        return
    df, note = wf
    st.markdown("Find rows that look unusual compared with the rest. This is unsupervised: no labels are needed, and a flagged row is a "
                "**candidate for review**, not proof of an error or of fraud.")
    S = st.session_state
    methods = st.multiselect("Methods", list(A.METHODS), default=["isolation_forest", "robust_z"], format_func=lambda k: A.METHODS[k][0], key="an_methods")
    for k in methods:
        st.caption(f"**{A.METHODS[k][0]}**: {A.METHODS[k][1]}" + (f" Limited to {A.METHODS[k][2]:,} rows." if A.METHODS[k][2] else ""))
    c1, c2 = st.columns(2)
    auto = c1.checkbox("Let each method decide how many rows to flag", value=True, key="an_auto")
    share = None if auto else c2.slider("Expected share of unusual rows (%)", 0.1, 20.0, 1.0, 0.1, key="an_share")
    excl = st.multiselect("Leave these columns out", list(df.columns), key="an_excl", help="For example identifiers, or the outcome you want to check the detector against.")
    for i, uw in enumerate(A.mixed_unit_warnings(df, excl)):
        st.warning(uw["message"])
        st.button("Leave " + ", ".join(f"'{c}'" for c in uw["value_cols"]) + " out", key=f"an_unit_{i}", on_click=_leave_out, args=(uw["value_cols"],))
    label = st.selectbox("Column with known anomalies, used only to evaluate (optional)", ["(none)"] + list(df.columns), key="an_label")
    pos = None
    if label != "(none)":
        vals = sorted(df[label].dropna().astype(str).unique().tolist())[:50]
        pos = st.selectbox("Which value marks a known anomaly?", vals, index=len(vals) - 1 if vals else 0, key="an_pos")
    if st.button("Find unusual rows", type="primary", key="an_go"):
        cfg = A.AnomalyConfig(methods=methods, contamination="auto" if auto else share / 100, features_exclude=excl,
                              label_column=None if label == "(none)" else label, label_positive=pos, seed=int(S.get("seed") or 42))
        try:
            with st.spinner("Scoring rows"):
                S["an_res"] = A.run_anomaly(df, cfg)
        except A.AnomalyError as e:
            st.error(str(e))
    res = S.get("an_res")
    if res is None:
        return
    for w_ in res.warnings:
        st.info(w_)
    best = next(m for m in res.methods if m["key"] == res.selected)
    n_rows = len(res._scores[res.selected])
    st.markdown(f"#### {best['flagged']:,} of {n_rows:,} rows flagged ({best['flagged_pct']:g}%) by {best['name']}")
    st.caption("Flagged rows are candidates to review, not confirmed errors. " + (f"Rule used: {best['note']}." if best.get("note") else ""))
    for uw in res.unit_warnings:
        st.warning(uw["message"])
    rows = []
    for m in res.methods:
        r = {"Method": m["name"], "Status": m["status"], "Flagged": m.get("flagged"), "Flagged %": m.get("flagged_pct"), "Seconds": m.get("seconds"),
             "Note": m.get("error") or m.get("note", "")}
        ev = m.get("evaluation") or {}
        for k in ("precision", "recall", "pr_auc"):
            if k in ev:
                r[k] = round(ev[k], 3)
        rows.append(r)
    st.dataframe(pd.DataFrame(rows).dropna(axis=1, how="all"), hide_index=True, width="stretch")
    ok = [m for m in res.methods if m["status"] == "ok"]
    if len(ok) > 1 and "agreement" in ok[0]:
        st.caption("Agreement between methods (rank correlation of scores, 1 = identical ordering): "
                   + "; ".join(f"{m['name']} vs {A.METHODS[k][0]}: {v}" for m in ok[:1] for k, v in m["agreement"].items()))
    keys = [m["key"] for m in ok]
    sel = st.selectbox("Show results for", keys, index=keys.index(res.selected), format_func=lambda k: A.METHODS[k][0], key="an_sel")
    asg = res.assignments(sel)
    fig = go.Figure(go.Histogram(x=asg["anomaly_score"], marker_color=C.TEAL, nbinsx=60))
    fig.update_xaxes(title="Anomaly score (higher = more unusual)")
    st.plotly_chart(C._layout(fig, "Distribution of scores", 280), width="stretch", config=C.PLOTLY_CONFIG)
    st.markdown("#### Most unusual rows" + (" (selected method)" if sel == res.selected else ""))
    if sel == res.selected:
        st.dataframe(pd.DataFrame(res.top), hide_index=True, width="stretch")
    else:
        st.dataframe(theme.display_frame(df.reset_index(drop=True).assign(anomaly_score=asg["anomaly_score"])
                                         .sort_values("anomaly_score", ascending=False).head(50)), width="stretch")
    st.caption("Row numbers count from 0 in the working data. 'why' lists the values furthest from typical for that column (robust spreads from the median).")
    out = df.reset_index(drop=True).assign(anomaly_score=asg["anomaly_score"], flagged=asg["flagged"])
    files = {"anomaly_scores.csv": out.to_csv(index=False).encode(), "anomaly_summary.json": json.dumps(res.to_dict(), indent=2, default=str).encode()}
    c1, c2, c3, _ = st.columns([1.4, 1.4, 1.4, 2.6])
    c1.download_button("Scores (CSV)", files["anomaly_scores.csv"], file_name="anomaly_scores.csv", mime="text/csv", key="an_dl")
    c2.download_button("Everything (ZIP)", zip_bytes(files), file_name="nexora_anomalies.zip", mime="application/zip", key="an_zip")
    saved = S.get("an_saved") == id(res)
    if c3.button("Saved to Reports" if saved else "Save to Reports", key="an_save", disabled=saved,
                 help="Keeps this run (summary and files) under Reports > Anomaly and forecast runs."):
        h = state.handle()
        summary = {"headline": f"{best['flagged']:,} of {n_rows:,} rows flagged ({best['flagged_pct']:g}%) by {best['name']}",
                   "table": [{k: v for k, v in r.items() if k != "Note"} for r in rows],
                   "notes": [w_ for w_ in res.warnings] + [u["message"] for u in res.unit_warnings], "config": res.config.to_dict()}
        state.ws().repo.save_run("anomaly", h.name if h else "", S.get("version_id"), summary, files)
        S["an_saved"] = id(res)
        st.rerun()
