from __future__ import annotations

import streamlit as st

from .. import state, theme


FEATURES = [("Prepare", "Finds placeholders, mixed date and number formats, duplicates, mixed units and outliers, proposes fixes, and applies "
                        "only what you approve. The original file is never changed."),
            ("Visualize", "Summaries, distributions, correlations and a dashboard you can filter, recolour and download."),
            ("Model", "Predict a number or a category, or find groups. Fair model comparison, leakage warnings and a test set used once."),
            ("Anomalies and forecasts", "Score rows for how unusual they are, with reasons, and forecast over time with honest backtests.")]

# sample file -> (button label, page to open, what it shows)
QUICK = [("customer_churn.csv", "Predict churn", "ML Studio", "Classification with a leakage warning built in."),
         ("retail_orders_messy.csv", "Clean messy orders", "Prepare Data", "Mixed dates, placeholders, currency symbols, repeats."),
         ("daily_demand.csv", "Forecast demand", "Forecasting", "420 days of sales with a weekly pattern."),
         ("customer_segments.csv", "Find customer groups", "ML Studio", "Clustering with named groups.")]


def _intro() -> None:
    st.markdown("Nexora takes a spreadsheet from **raw file to trustworthy results**: clean it, chart it, model it and export everything, "
                "with every step explained in plain words and nothing changed without your say.")
    cols = st.columns(len(FEATURES))
    for col, (t, d) in zip(cols, FEATURES):
        with col.container(border=True):
            st.markdown(f"**{t}**")
            st.caption(d)
    st.markdown("#### Try a sample")
    st.caption("Synthetic datasets made for the demo, each opening on the page that shows it best.")
    cols = st.columns(len(QUICK))
    for col, (fname, label, page, about) in zip(cols, QUICK):
        with col.container(border=True):
            st.caption(about)
            if st.button(label, key=f"home_try_{fname}", width="stretch"):
                from .. import components
                components._open_sample(fname)
                state.goto(page)          # ML Studio then suggests the right goal (classification for churn, groups for segments)
                st.rerun()
    st.markdown("#### Or bring your own")


def render() -> None:
    st.title("Nexora")
    st.markdown("<div class='nx-tag'>Connect data, models and decisions.</div>", unsafe_allow_html=True)
    h = state.handle()
    if h is None:
        _intro()
        if st.button("Upload your own data", type="primary"):
            state.goto("Datasets")
            st.rerun()
        recent = state.ws().repo.list_datasets()[:5]
        if recent:
            st.subheader("Recent datasets")
            st.dataframe([{"Name": r["name"], "Rows": r["rows"], "Columns": r["cols"], "Added": theme.local_time(r["created_at"]),
                           "Source": "sample" if r["source"] == "sample" else "upload"} for r in recent], hide_index=True, width="stretch")
            st.caption("Open one from Datasets > My datasets.")
        return

    S = st.session_state
    prep = S.get("prep_result")
    exp = S.get("exp")
    st.subheader(f"Workspace: {h.name}")
    if h.source == "sample":
        st.markdown(theme.sample_badge(), unsafe_allow_html=True)
    rows = [("1. Dataset", theme.chip("Loaded", "ok"), f"{theme.fmt_int(len(h.raw))} rows, {h.raw.shape[1]} columns"),
            ("2. Prepare", theme.chip("Done", "ok") if (prep or state.has_prepared()) else theme.chip("Not started", "off"),
             (f"{theme.fmt_int(prep.after['rows'])} rows prepared; checks {'passed' if prep.reconciliation['ok'] else 'NEED ATTENTION'}" if prep
              else ("A prepared version is open" if state.has_prepared() else "Clean and validate before analysis (optional for exploring)"))),
            ("3. Visualize & Dashboard", theme.chip("In progress", "info") if S.get("dashboard") else theme.chip("Available", "off"),
             f"{len(S.get('dashboard', []))} chart(s) on the dashboard" if S.get("dashboard") else "Charts, correlations and dashboards"),
            ("4. Model", theme.chip("Done", "ok") if exp else theme.chip("Available", "off"),
             (f"Last experiment: {exp.config.task}, status {exp.status.replace('_', ' ')}" if exp else "Predict a number, predict a category, or discover groups"))]
    body = "".join(f"<tr><td style='padding:6px 14px 6px 0'><b>{theme.esc(a)}</b></td><td style='padding:6px 14px 6px 0'>{b}</td><td class='nx-muted'>{theme.esc(c)}</td></tr>" for a, b, c in rows)
    st.markdown(f"<div class='nx-card'><table style='border-collapse:collapse'>{body}</table></div>", unsafe_allow_html=True)
    prepared = bool(prep) or state.has_prepared()
    nxt = ("Prepare Data", "Prepare this dataset") if not prepared else (("Visualize & Dashboard", "Explore your data") if not S.get("dashboard") else ("ML Studio", "Open ML Studio"))
    c1, c2, _ = st.columns([1.2, 1.2, 4])
    if c1.button(nxt[1], type="primary"):
        state.goto(nxt[0])
        st.rerun()
    if c2.button("Open Datasets"):
        state.goto("Datasets")
        st.rerun()
