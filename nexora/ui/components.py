"""Shared page furniture: the dataset bar (what is loaded, a short summary, a switcher) and the "what next?" chooser."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..eda import profiling as P
from ..workflow import SAMPLES
from . import state, theme

# page -> (button label, one-line description)
DESTINATIONS = {
    "Prepare Data": ("Prepare this data", "Clean and validate: placeholders, formats, duplicates, units, outliers."),
    "Visualize & Dashboard": ("Visualize this data", "Summaries, charts, correlations and a dashboard you can export."),
    "ML Studio": ("Build a model", "Predict a number or a category, or discover groups."),
    "Anomalies": ("Find unusual rows", "Score every row for how unusual it is, with the reason."),
    "Forecasting": ("Forecast over time", "Backtest and forecast a number along a date column."),
}


def summary(df: pd.DataFrame) -> str:
    n_num, n_cat, n_dt = len(P.numeric_columns(df)), len(P.categorical_columns(df)), len(P.datetime_columns(df))
    miss = 100 * float(df.isna().to_numpy().mean()) if df.size else 0.0
    return (f"{len(df):,} rows x {df.shape[1]} columns · {n_num} numeric, {n_cat} categorical, {n_dt} date · "
            f"{miss:.1f}% of cells missing")


def _close_popover() -> None:
    # Streamlit has no "close" call for popovers; giving it a new key on the next run renders it closed
    st.session_state["_pop_n"] = st.session_state.get("_pop_n", 0) + 1


def _open_saved(dataset_id: str, version_id: str | None) -> None:
    _close_popover()
    state.set_dataset(state.ws().open_dataset(dataset_id))
    st.session_state["version_id"] = version_id
    state.clear_frames()


def _open_sample(fname: str) -> None:
    meta = next(s for s in SAMPLES if s["file"] == fname)
    _close_popover()
    state.set_dataset(state.ws().load_sample(fname))
    st.session_state["profile_id"] = meta["domain"]
    st.session_state["goal"] = meta["goal"]


def picker(key: str) -> None:
    """Choose a saved dataset (raw or a prepared version) or a sample, without leaving the page."""
    w = state.ws()
    rows = w.repo.list_datasets()
    src = st.radio("Source", ["Saved datasets", "Sample data"] if rows else ["Sample data"], horizontal=True, key=f"pk_src_{key}")
    if src == "Saved datasets":
        ids = {f"{r['name']} ({r['rows']:,} rows, {theme.local_time(r['created_at'])})": r["id"] for r in rows}
        pick = st.selectbox("Dataset", list(ids), key=f"pk_ds_{key}")
        vers = w.repo.list_versions(ids[pick])
        vopts = {"Raw data (as uploaded)": None} | {f"Prepared {theme.local_time(v['created_at'])}: {v['rows']:,} rows": v["id"] for v in vers}
        vk = list(vopts)
        vpick = st.selectbox("Version", vk, index=1 if vers else 0, key=f"pk_v_{key}",      # versions are newest first
                             help="Prepared versions come from Prepare Data. The newest one is preselected; the raw file is always kept.")
        if st.button("Open", type="primary", key=f"pk_open_{key}"):
            _open_saved(ids[pick], vopts[vpick])
            st.rerun()
    else:
        labels = {s["label"]: s["file"] for s in SAMPLES}
        lab = st.selectbox("Sample", list(labels), key=f"pk_smp_{key}")
        st.caption(next(s["about"] for s in SAMPLES if s["label"] == lab))
        if st.button("Open sample", type="primary", key=f"pk_osmp_{key}"):
            _open_sample(labels[lab])
            st.rerun()
    st.caption("To upload a new file or connect a database or API, use the Datasets page.")


def dataset_bar(key: str, *, raw: bool = False, empty_body: str = "") -> tuple[pd.DataFrame, str] | None:
    """Top-of-page bar. Returns (frame, note) for the page to use, or None after showing an empty state with a picker."""
    h = state.handle()
    if h is None:
        theme.empty_state("Open a dataset first", empty_body or "Choose a dataset below, or upload one on the Datasets page.")
        c1, _ = st.columns([1.2, 4])
        if c1.button("Go to Datasets", type="primary", key=f"goto_ds_{key}"):
            state.goto("Datasets")
            st.rerun()
        with st.container(border=True):
            st.markdown("**Or choose one here**")
            picker(key)
        return None
    if raw:
        df, note = h.raw, "Original file as uploaded (preparation never changes it)"
    else:
        wf = state.working_frame()
        if wf is None:
            return None
        df, note = wf
    prepared = state.has_prepared() and not raw
    with st.container(border=True):
        c1, c2 = st.columns([5, 1.3])
        badge = theme.chip("Prepared data", "ok") if prepared else theme.chip("Not prepared" if not raw else "Raw file", "warn" if not raw else "info")
        c1.markdown(f"**Dataset:** {theme.esc(h.name)} &nbsp; {badge}" + (f" &nbsp; {theme.sample_badge()}" if h.source == "sample" else ""),
                    unsafe_allow_html=True)
        c1.caption(summary(state.typed_raw() if raw else df) + (" (types detected automatically)" if raw else "") + " · " + note
                   + ("" if prepared or raw else ". For trustworthy results, run Prepare Data first."))
        with c2.popover("Change dataset", width="stretch", key=f"chg_pop_{key}_{st.session_state.get('_pop_n', 0)}"):
            st.caption("Switching resets unfinished work on this page. Saved experiments and prepared versions stay in Reports.")
            picker(f"bar_{key}")
        if not prepared and not raw and key not in ("prepare",):
            if c2.button("Prepare it first", key=f"bar_prep_{key}", width="stretch", help="Open Prepare Data for this dataset."):
                state.goto("Prepare Data")
                st.rerun()
    return df, note


def next_steps(key: str, *, exclude: tuple[str, ...] = (), recommended: str | None = None, df: pd.DataFrame | None = None) -> None:
    """A row of cards: where to go next with the current dataset."""
    st.markdown("#### What would you like to do next?")
    pages = [p for p in DESTINATIONS if p not in exclude]
    cols = st.columns(len(pages))
    for col, page in zip(cols, pages):
        label, desc = DESTINATIONS[page]
        why = ""
        if df is not None and page == "Forecasting" and not P.datetime_columns(df):
            why = "Needs a date column (prepare the data so dates are recognised)."
        if df is not None and page == "Anomalies" and len(df) < 20:
            why = "Needs at least 20 rows."
        with col.container(border=True):
            st.markdown(f"**{page}**" + ("  \n*recommended*" if page == recommended else ""))
            st.caption(why or desc)
            if st.button(label, key=f"ns_{key}_{page}", type="primary" if page == recommended else "secondary", disabled=bool(why),
                         width="stretch"):
                state.goto(page)
                st.rerun()
