"""Nexora: connect data, models and decisions.  Launch with:  streamlit run app.py"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from nexora.config import configure_logging, get_settings  # noqa: E402
from nexora.ui import help_panel, state, theme  # noqa: E402
from nexora.ui.pages import anomalies, datasets, explore, forecasting, home, ml_studio, prepare, reports, settings  # noqa: E402

st.set_page_config(page_title="Nexora", page_icon="N", layout="wide", initial_sidebar_state="expanded")
configure_logging(get_settings().log_level)
theme.inject()
state.init()

S = st.session_state
if S.get("_goto"):
    S["nav"] = S["_goto"]
    S["_goto"] = None

with st.sidebar:
    st.markdown("### Nexora")
    st.markdown("<div class='nx-side-tag'>Connect data, models and decisions.</div>", unsafe_allow_html=True)
    st.radio("Navigate", state.PAGES, key="nav", label_visibility="collapsed")
    h = state.handle()
    st.divider()
    if h:
        st.markdown(f"**Dataset**<br>{theme.esc(h.name)}<br><span class='nx-side-tag'>{len(h.raw):,} rows, {h.raw.shape[1]} columns"
                    f"{'; prepared' if state.has_prepared() else '; not prepared'}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='nx-side-tag'>No dataset open</span>", unsafe_allow_html=True)
    st.divider()
    help_panel.render(S["nav"])

if get_settings().public_mode:
    st.info("**Public demo.** Your uploads, prepared data and models are visible only in this browser session and are deleted from the server "
            "within 24 hours. Please do not upload personal or confidential data. To keep your work, download it, or run Nexora on your own computer.",
            icon=None)

PAGE_FUNCS = {"Home": home.render, "Datasets": datasets.render, "Prepare Data": prepare.render, "Visualize & Dashboard": explore.render,
              "ML Studio": ml_studio.render, "Anomalies": anomalies.render, "Forecasting": forecasting.render, "Reports": reports.render,
              "Settings": settings.render}
try:
    PAGE_FUNCS[S["nav"]]()
except Exception as exc:  # last-resort guard: show a readable message rather than a stack trace
    if type(exc).__name__ in ("RerunException", "StopException"):
        raise
    logging.getLogger("nexora").exception("Unhandled error on page %s", S["nav"])
    st.error("Something went wrong on this page. Your data is safe (the original file is never modified). "
             "Try going back a step; if it keeps happening, check the terminal for details.")
    if st.checkbox("Show technical details"):
        st.exception(exc)
