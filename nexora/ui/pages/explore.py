from __future__ import annotations

import zlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...config import get_settings

from ...eda import charts as C
from ...eda import profiling as P
from ...reporting import eda_summary_html
from .. import components, state, theme


def _show(df: pd.DataFrame, spec: C.ChartSpec, key: str) -> bool:
    try:
        fig = C.build_figure(df, spec, st.session_state.get("chart_palette"))
    except C.ChartError as e:
        st.info(f"This chart cannot be drawn: {e}")
        return False
    st.plotly_chart(fig, width="stretch", config=C.PLOTLY_CONFIG, key=key)
    return True


def _on_dashboard(spec: C.ChartSpec) -> bool:
    return any(C.ChartSpec.from_dict(d).key() == spec.key() for d in st.session_state["dashboard"])


def _add_spec(spec: C.ChartSpec) -> None:
    dash = st.session_state["dashboard"]
    if not _on_dashboard(spec):
        spec.spec_id = f"d{len(dash) + 1}_{zlib.crc32(spec.key().encode()) % 100000}"
        dash.append(spec.to_dict())
        st.toast(f"Added to the dashboard ({len(dash)} chart(s) there now).")


def _add(spec: C.ChartSpec, key: str) -> None:
    """'Add to dashboard', which turns into a lasting 'On the dashboard' note so it is clear the click worked."""
    if _on_dashboard(spec):
        n = len(st.session_state["dashboard"])
        st.success(f"On the dashboard. Open the **Dashboard** tab to see all {n} chart(s).", icon=":material/check_circle:")
    else:
        st.button("Add to dashboard", key=key, help="Adds this chart to the Dashboard tab.", on_click=_add_spec, args=(spec,))


def render() -> None:
    st.title("Visualize & Dashboard")
    wf = components.dataset_bar("explore", empty_body="Exploration needs a dataset. Samples are available in Datasets.")
    if wf is None:
        return
    df, note = wf
    c1, _ = st.columns([1.4, 3])
    _palette_picker(c1)
    if len(df) == 0:
        theme.empty_state("This dataset has no rows", "There is nothing to chart.")
        return
    tabs = st.tabs(["Overview", "Columns", "Relationships", "Charts", "Dashboard", "Dimensions"])
    with tabs[0]:
        _overview(df)
    with tabs[1]:
        _columns(df)
    with tabs[2]:
        _relationships(df)
    with tabs[3]:
        _charts(df)
    with tabs[4]:
        _dashboard(df)
    with tabs[5]:
        _dimensions(df)


def _dimensions(df: pd.DataFrame) -> None:
    """A low-dimensional picture of many columns at once (PCA, SVD, t-SNE or UMAP)."""
    from ...preprocessing import reduction as R
    st.markdown("See many columns at once as points in two dimensions. Similar rows land close together. This is a picture for exploration only.")
    cols = [c for c in df.columns if P.column_role(df[c], c) in ("numeric", "categorical", "boolean")]
    if len(cols) < 2:
        st.info("At least two numeric or categorical columns are needed.")
        return
    picked = st.multiselect("Columns", cols, default=cols[: min(len(cols), 12)], key="ex_dim_cols")
    methods = [m for m in R.VIEW_METHODS if m != "umap" or R.UMAP_AVAILABLE]
    c1, c2 = st.columns(2)
    meth = c1.selectbox("Method", methods, format_func=lambda k: R.VIEW_METHODS[k], key="ex_dim_m")
    color = c2.selectbox("Colour points by", ["(none)"] + list(df.columns), key="ex_dim_c")
    if not R.UMAP_AVAILABLE and not get_settings().public_mode:
        st.caption("UMAP is optional (pip install umap-learn).")
    if st.button("Draw", type="primary", key="ex_dim_go"):
        try:
            pr = R.project(df, picked, meth, 2, seed=int(st.session_state.get("seed") or 42))
            st.session_state["ex_dim_res"] = (pr, color)
        except R.ReductionError as e:
            st.error(str(e))
    res = st.session_state.get("ex_dim_res")
    if res:
        pr, color = res
        pts = pr.points.copy()
        fig = go.Figure()
        if color != "(none)" and color in df.columns:
            vals = df[color].loc[pts["row"]].astype(str).values
            for i, g in enumerate(pd.unique(vals)[:12]):
                mask = vals == g
                fig.add_trace(go.Scattergl(x=pts["dim1"][mask], y=pts["dim2"][mask], mode="markers", name=str(g),
                                           marker={"size": 6, "opacity": 0.7, "color": C.COLORWAY[i % len(C.COLORWAY)]}))
        else:
            fig.add_trace(go.Scattergl(x=pts["dim1"], y=pts["dim2"], mode="markers", marker={"size": 6, "opacity": 0.6, "color": C.TEAL}))
        st.plotly_chart(C._layout(fig, R.VIEW_METHODS[pr.method], 420), width="stretch", config=C.PLOTLY_CONFIG)
        if pr.explained_variance:
            st.caption("Share of variation shown: " + ", ".join(f"dimension {i + 1}: {100 * v:.1f}%" for i, v in enumerate(pr.explained_variance)))
        if pr.note:
            st.caption(pr.note)
        if pr.sampled:
            st.caption(f"A random sample of {len(pts):,} rows is shown.")
        if pr.loadings is not None:
            with st.expander("Which columns drive each dimension"):
                st.dataframe(pr.loadings, width="stretch")


def _overview(df: pd.DataFrame) -> None:
    ov = P.dataset_overview(df)
    roles = ", ".join(f"{v} {k}" for k, v in sorted(ov["role_counts"].items()))
    st.markdown(f"**{ov['rows']:,} rows and {ov['columns']} columns** ({roles}). {ov['missing_pct']}% of cells are missing; {ov['duplicate_rows']:,} row(s) repeat exactly.")
    obs = P.observations(df)
    st.markdown("#### What stands out")
    if obs:
        for o in obs:
            st.markdown(f"- {o.text}")
    else:
        st.caption("The automatic checks found nothing notable (no heavy missingness, skew, outliers or constant columns).")
    summ = P.column_summary(df)
    st.markdown("#### Columns")
    st.dataframe(summ, hide_index=True, width="stretch")
    miss = summ[summ["missing"] > 0].sort_values("missing %", ascending=False)
    if not miss.empty:
        st.markdown("#### Missing values")
        fig = go.Figure(go.Bar(x=miss["missing %"], y=miss["column"], orientation="h", marker_color=C.TEAL, text=[f"{v}%" for v in miss["missing %"]], textposition="outside"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="% of rows missing", range=[0, max(100, float(miss["missing %"].max()) * 1.1)])
        st.plotly_chart(C._layout(fig, "Missing values by column", height=max(260, 30 * len(miss) + 120)), width="stretch", config=C.PLOTLY_CONFIG)
    osum = P.outlier_summary(df)
    if not osum.empty and (osum["outliers"] > 0).any():
        st.markdown("#### Unusual values")
        st.dataframe(osum[osum["outliers"] > 0], hide_index=True, width="stretch")
        st.caption("Values beyond 1.5 times the interquartile range. They are review signals: unusual values can be real and important.")
    corr = P.top_correlations(P.correlation_matrix(df), 6)
    html = eda_summary_html(state.handle().name, ov, summ, [o.text for o in obs], osum, corr)
    st.download_button("Download data summary (HTML)", html, file_name="data_summary.html", mime="text/html")


def _columns(df: pd.DataFrame) -> None:
    cols = list(df.columns)
    chartable = [c for c in cols if P.column_role(df[c], c) in ("numeric", "categorical", "boolean", "datetime")]
    if st.session_state.get("ex_col") not in cols:
        st.session_state["ex_col"] = chartable[0] if chartable else cols[0]      # open on a column that has a chart, not an ID
    col = st.selectbox("Column", cols, key="ex_col")
    s = df[col]
    role = P.column_role(s, col)
    st.markdown(f"**{col}**: {role}, {s.dtype}; {int(s.isna().sum()):,} missing; {int(s.nunique()):,} distinct values.")
    if role == "numeric":
        d = s.describe()
        st.dataframe(pd.DataFrame({"Statistic": d.index, "Value": d.values.astype(float)}), hide_index=True, width="content")
        _show(df, C.ChartSpec("histogram", x=col), "ex_h")
        _show(df, C.ChartSpec("box", y=col), "ex_b")
        _add(C.ChartSpec("histogram", x=col, title=f"Distribution of {col}"), "add_col_h")
    elif role in ("categorical", "boolean"):
        _show(df, C.ChartSpec("bar", x=col), "ex_bar")
        _add(C.ChartSpec("bar", x=col), "add_col_b")
        if s.nunique() > C.TOP_N:
            st.caption(f"{s.nunique()} distinct values: the chart shows the {C.TOP_N} most frequent and groups the rest as 'Other'.")
    elif role == "datetime":
        st.caption(f"From {str(s.min())[:19]} to {str(s.max())[:19]}.")
        _show(df, C.ChartSpec("line", x=col), "ex_line")
        _add(C.ChartSpec("line", x=col), "add_col_l")
    else:
        vc = s.dropna().astype(str).value_counts().head(15)
        st.dataframe(pd.DataFrame({"Value": vc.index, "Rows": vc.values}), hide_index=True, width="content")
        st.caption({"identifier": "Identifiers are unique per row, so no chart is drawn.", "text": "Free text is summarised by its most frequent values.",
                    "constant": "This column has the same value in every row.", "empty": "This column has no values."}.get(role, ""))


def _relationships(df: pd.DataFrame) -> None:
    nums = P.numeric_columns(df)
    if len(nums) < 2:
        theme.empty_state("Not enough numeric columns", "Relationships need at least two numeric columns that vary.")
    else:
        method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True,
                          help="Pearson measures straight-line association; Spearman measures whether values rise together, even if not in a straight line.")
        corr = P.correlation_matrix(df, method)
        spec = C.ChartSpec("heatmap", columns=nums[:25], agg=method, title=f"{method.title()} correlation")
        if _show(df, spec, "rel_heat"):
            _add(spec, "add_heat")
        st.markdown("#### Strongest pairs")
        st.dataframe(P.top_correlations(corr, 8).round(3), hide_index=True, width="content")
        st.caption("Correlation shows that two columns move together. It does not show that one causes the other.")
    st.markdown("#### Build a comparison")
    cats = P.categorical_columns(df, 30)
    mode = st.radio("Compare", ["Two numeric columns", "A number across categories"], horizontal=True)
    if mode == "Two numeric columns" and len(nums) >= 2:
        x = st.selectbox("X axis", nums, key="rel_x")
        y = st.selectbox("Y axis", [n for n in nums if n != x], key="rel_y")
        color = st.selectbox("Colour by (optional)", ["(none)"] + cats, key="rel_c")
        spec = C.ChartSpec("scatter", x=x, y=y, color=None if color == "(none)" else color)
        if _show(df, spec, "rel_sc"):
            _add(spec, "add_sc")
    elif mode == "A number across categories" and nums and cats:
        c = st.selectbox("Category", cats, key="rel_cat")
        n = st.selectbox("Number", nums, key="rel_num")
        spec = C.ChartSpec("box", x=c, y=n)
        if _show(df, spec, "rel_box"):
            _add(spec, "add_box")
    else:
        st.caption("This comparison needs suitable columns.")


def _charts(df: pd.DataFrame) -> None:
    st.markdown("#### Recommended for this data")
    st.caption("Chosen from column types only. Each chart is computed from your data.")
    specs = C.recommend_charts(df)
    if not specs:
        theme.empty_state("No charts to recommend", "The columns are all empty, constant or identifiers.")
    for i in range(0, len(specs), 2):
        cols = st.columns(2)
        for col, s in zip(cols, specs[i:i + 2]):
            with col:
                st.caption(s.reason)
                if _show(df, s, f"rc_{s.spec_id}"):
                    _add(s, f"add_{s.spec_id}")
    st.markdown("#### Build your own")
    kind = st.selectbox("Chart type", ["histogram", "bar", "box", "scatter", "line"], key="cb_kind")
    nums, cats, dts = P.numeric_columns(df), P.categorical_columns(df), P.datetime_columns(df)
    spec = None
    if kind == "histogram" and nums:
        spec = C.ChartSpec("histogram", x=st.selectbox("Column", nums, key="cb_x"))
    elif kind == "bar" and (cats or nums):
        x = st.selectbox("Category column", cats or list(df.columns), key="cb_bx")
        agg = st.selectbox("Show", ["count", "mean", "sum", "median"], key="cb_agg")
        y = st.selectbox("Of", nums, key="cb_by") if agg != "count" and nums else None
        spec = C.ChartSpec("bar", x=x, y=y, agg=agg if y else "count")
    elif kind == "box" and nums:
        y = st.selectbox("Number", nums, key="cb_boy")
        x = st.selectbox("Group by (optional)", ["(none)"] + cats, key="cb_box")
        spec = C.ChartSpec("box", y=y, x=None if x == "(none)" else x)
    elif kind == "scatter" and len(nums) >= 2:
        x = st.selectbox("X", nums, key="cb_sx")
        y = st.selectbox("Y", [n for n in nums if n != x], key="cb_sy")
        c = st.selectbox("Colour by (optional)", ["(none)"] + cats, key="cb_sc")
        spec = C.ChartSpec("scatter", x=x, y=y, color=None if c == "(none)" else c)
    elif kind == "line" and dts:
        x = st.selectbox("Date column", dts, key="cb_lx")
        y = st.selectbox("Value (optional)", ["(row count)"] + nums, key="cb_ly")
        agg = st.selectbox("Combine with", ["mean", "sum", "median"], key="cb_lagg")
        spec = C.ChartSpec("line", x=x, y=None if y == "(row count)" else y, agg=agg)
    if spec is None:
        st.caption("This chart type needs columns the data does not have (for example a date column for a time series).")
    else:
        spec.title = C.default_title(spec)
        if _show(df, spec, "cb_fig"):
            _add(spec, "cb_add")


def _filter_widgets(df: pd.DataFrame) -> list[dict]:
    cols = st.multiselect("Filter the dashboard by", [c for c in df.columns if P.column_role(df[c], c) in ("categorical", "boolean", "numeric", "datetime")], key="dash_cols",
                          help="Charts on this tab are recomputed on the filtered rows.")
    filters = []
    for c in cols:
        s = df[c]
        role = P.column_role(s, c)
        if role in ("categorical", "boolean"):
            opts = sorted(s.dropna().astype(str).unique().tolist())[:200]
            sel = st.multiselect(c, opts, default=opts, key=f"df_{c}")
            if len(sel) != len(opts):
                filters.append({"column": c, "type": "category", "values": sel})
        elif role == "numeric":
            lo, hi = float(s.min()), float(s.max())
            if lo < hi:
                a, b = st.slider(c, lo, hi, (lo, hi), key=f"df_{c}")
                if (a, b) != (lo, hi):
                    filters.append({"column": c, "type": "range", "min": a, "max": b})
        elif role == "datetime":
            lo, hi = s.min().date(), s.max().date()
            if lo < hi:
                rng = st.date_input(c, (lo, hi), min_value=lo, max_value=hi, key=f"df_{c}")
                if isinstance(rng, tuple) and len(rng) == 2 and tuple(rng) != (lo, hi):
                    filters.append({"column": c, "type": "date", "start": str(rng[0]), "end": str(rng[1])})
    return filters


def _palette_picker(where=None) -> None:
    """One palette for every chart on this page and in the exported dashboard."""
    names = list(C.PALETTES)
    cur = st.session_state.get("chart_palette") or C.DEFAULT_PALETTE
    (where or st).selectbox("Colour palette", names, index=names.index(cur) if cur in names else 0, key="chart_palette",
                            help="Applies to every chart on this page and to the downloaded dashboard. 'Colour-blind safe' uses the Okabe-Ito colours.")


def _dashboard(df: pd.DataFrame) -> None:
    dash = st.session_state["dashboard"]
    if not dash:
        theme.empty_state("Your dashboard is empty", "Use 'Add to dashboard' under any chart in Columns, Relationships or Charts.")
        return
    pal = C.palette(st.session_state.get("chart_palette"))
    st.caption(f"Colours: {st.session_state.get('chart_palette') or C.DEFAULT_PALETTE} (change it at the top of the page). Each chart's main colour can be changed below it.")
    st.markdown(" ".join(f"<span style='display:inline-block;width:18px;height:18px;border-radius:3px;background:{c};margin:0 2px 8px 0'></span>"
                         for c in pal["colors"]), unsafe_allow_html=True)
    filters = _filter_widgets(df)
    fdf = C.apply_filters(df, filters)
    st.markdown(f"**{len(fdf):,} of {len(df):,} rows** shown. Filters: {C.describe_filters(filters)}.")
    items = []
    for i, d in enumerate(list(dash)):
        spec = C.ChartSpec.from_dict(d)
        with st.container(border=True):
            ok = _show(fdf, spec, f"dash_{spec.spec_id}")
            c1, c2, c3, c4, c5 = st.columns([1, 1, 1.2, 1.6, 3.2])
            cur = d.get("accent") or pal["primary"]
            new = c4.color_picker("Chart colour", cur, key=f"acc_{spec.spec_id}_{st.session_state.get('chart_palette')}_{d.get('accent_v', 0)}", label_visibility="collapsed",
                                  help="Main colour for this chart. Charts with several groups use the palette for the other groups.")
            if new.lower() != cur.lower():
                dash[i]["accent"] = new
                st.rerun()
            if d.get("accent") and c5.button("Use palette colour", key=f"accrst_{spec.spec_id}"):
                dash[i]["accent"] = None
                dash[i]["accent_v"] = d.get("accent_v", 0) + 1      # fresh colour picker, so it shows the palette colour again
                st.rerun()
            if c1.button("Up", key=f"up_{spec.spec_id}", disabled=i == 0):
                dash[i - 1], dash[i] = dash[i], dash[i - 1]
                st.rerun()
            if c2.button("Down", key=f"dn_{spec.spec_id}", disabled=i == len(dash) - 1):
                dash[i + 1], dash[i] = dash[i], dash[i + 1]
                st.rerun()
            if c3.button("Remove", key=f"rm_{spec.spec_id}"):
                dash.pop(i)
                st.rerun()
        if ok:
            items.append((spec.title or C.default_title(spec), C.build_figure(fdf, spec, st.session_state.get("chart_palette")), spec.reason or ""))
    html = C.dashboard_html(f"Dashboard: {state.handle().name}", items, f"{len(fdf):,} of {len(df):,} rows", C.describe_filters(filters))
    c1, c2, _ = st.columns([1.6, 1.2, 4])
    c1.download_button("Download dashboard (HTML)", html, file_name="dashboard.html", mime="text/html")
    if c2.button("Clear dashboard"):
        st.session_state["dashboard"] = []
        st.rerun()
    st.caption("Exports are interactive HTML files that open in any browser, without Nexora. PNG export is not included because it needs an extra image-rendering package.")
