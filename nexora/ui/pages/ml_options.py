"""Advanced ML Studio controls: in-pipeline preparation, model settings, validation and tuning, and extra results.

Everything set here ends up in the ExperimentConfig, is validated by the engine and is saved with the experiment.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...eda import charts as C
from ...ml import cv as CV
from ...ml import explain as XP
from ...ml import tuning as TU
from ...ml.model_registry import INSTALL_HINT, REGISTRY, check_compatibility, models_for
from ...preprocessing import features as FE
from ...preprocessing import outliers as OL
from ...preprocessing import selection as SE
from ...preprocessing.pipeline import PreprocessOptions, select_features


# ---------------------------------------------------------------------------------------------- preparation
def prep_advanced(df: pd.DataFrame, m: dict, task: str) -> list[str]:
    """Render the optional in-pipeline steps. Returns blocking problems (empty when everything is valid)."""
    p = m["pre"]
    problems: list[str] = []
    fs = select_features(df, m.get("target") if task != "clustering" else None, m["exclude"], None, PreprocessOptions(**p))

    with st.expander("Dates: calendar features", expanded=False):
        if not fs.datetime:
            st.caption("No date columns are used as features.")
        else:
            cur = p.get("date_parts") or list(FE.DEFAULT_DATE_PARTS)
            p["date_parts"] = st.multiselect("Calendar parts to extract", FE.DATE_PARTS, default=[x for x in cur if x in FE.DATE_PARTS], key="ml_w_dparts")
            p["cyclical"] = st.checkbox("Add cyclical (sine / cosine) encodings for month, weekday, day of year and hour", value=bool(p.get("cyclical")),
                                        key="ml_w_cyc", help="Lets models see that December is next to January.")
            if not p["date_parts"]:
                problems.append("Choose at least one calendar part, or switch date features off.")

    with st.expander("Unusual values (outliers) inside the model", expanded=False):
        st.caption("Bounds are learned from the training rows only and applied unchanged to validation, test and new data. Rows are never removed here.")
        o = dict(p.get("outliers") or {})
        opts = ["none", "clip", "winsorize", "median"]
        lab = {"none": "Leave values as they are", **{k: OL.TREATMENTS[k] for k in opts[1:]}}
        t = st.radio("Treatment", opts, index=opts.index(o.get("treatment", "none")) if o.get("treatment", "none") in opts else 0,
                     format_func=lab.get, key="ml_w_ol_t", horizontal=True)
        if t == "none":
            p["outliers"] = None
        else:
            meths = ["iqr", "zscore", "modified_zscore"]
            meth = st.selectbox("Detection", meths, index=meths.index(o.get("method", "iqr")) if o.get("method") in meths else 0,
                                format_func=OL.METHODS.get, key="ml_w_ol_m")
            st.caption(OL.METHOD_NOTES[meth])
            new = {"method": meth, "treatment": t}
            if meth == "iqr":
                new["iqr_k"] = st.slider("IQR multiplier", 0.5, 6.0, float(o.get("iqr_k", 1.5)), 0.25, key="ml_w_ol_k")
            elif meth == "zscore":
                new["z_threshold"] = st.slider("Z-score threshold", 1.5, 8.0, float(o.get("z_threshold", 3.0)), 0.25, key="ml_w_ol_z")
            else:
                new["mz_threshold"] = st.slider("Modified Z-score threshold", 2.0, 10.0, float(o.get("mz_threshold", 3.5)), 0.25, key="ml_w_ol_mz")
            if t == "winsorize":
                lo, hi = st.slider("Winsorize percentiles", 0, 100, tuple(int(100 * v) for v in o.get("winsor_limits", (0.01, 0.99))), key="ml_w_ol_w")
                new["winsor_limits"] = [lo / 100, hi / 100]
            p["outliers"] = new

    with st.expander("Feature engineering", expanded=False):
        st.caption(f"New columns are computed inside the pipeline (learned on training rows). At most {FE.MAX_GENERATED} generated columns.")
        f = FE.FeatureConfig.from_dict(p.get("features"))
        nums, cats, txt = fs.numeric, fs.categorical, fs.text
        if nums:
            f.log = st.multiselect("Log transform", nums, default=[c for c in f.log if c in nums], key="ml_w_fe_log", help="Signed log for columns with negatives.")
            f.square = st.multiselect("Square", nums, default=[c for c in f.square if c in nums], key="ml_w_fe_sq")
            f.poly_columns = st.multiselect("Polynomial terms (up to 6 columns)", nums, default=[c for c in f.poly_columns if c in nums], key="ml_w_fe_poly",
                                            max_selections=6)
            if f.poly_columns:
                f.poly_degree = int(st.select_slider("Polynomial degree", [2, 3], value=f.poly_degree if f.poly_degree in (2, 3) else 2, key="ml_w_fe_deg"))
            pairs = [f"{a} | {b}" for i, a in enumerate(nums) for b in nums[i + 1:]][:300]
            for attr, label in (("interactions", "Products of pairs (a x b)"), ("ratios", "Ratios (a / b)"), ("differences", "Differences (a - b)")):
                cur = [f"{a} | {b}" for a, b in getattr(f, attr) if f"{a} | {b}" in pairs]
                picked = st.multiselect(label, pairs, default=cur, key=f"ml_w_fe_{attr}")
                setattr(f, attr, [x.split(" | ") for x in picked])
            f.bins = st.multiselect("Bin into equal-frequency groups", nums, default=[c for c in f.bins if c in nums], key="ml_w_fe_bins")
            if f.bins:
                f.n_bins = int(st.slider("Number of bins", 2, 20, int(f.n_bins), key="ml_w_fe_nb"))
            f.quantile_rank = st.multiselect("Percentile rank", nums, default=[c for c in f.quantile_rank if c in nums], key="ml_w_fe_rank")
        f.missing_indicators = st.multiselect("Missing-value indicators", nums + cats, default=[c for c in f.missing_indicators if c in nums + cats], key="ml_w_fe_miss")
        if cats:
            f.frequency_encode = st.multiselect("Frequency encoding", cats, default=[c for c in f.frequency_encode if c in cats], key="ml_w_fe_freq")
        if txt:
            f.text_stats = st.multiselect("Text length and word count", txt, default=[c for c in f.text_stats if c in txt], key="ml_w_fe_txt")
        if f.is_empty():
            p["features"] = None
        else:
            try:
                planned = FE.plan_features(f, nums, cats, txt, list(df.columns))
                p["features"] = f.to_dict()
                st.markdown(f"**{len(planned)} column(s) will be generated.**")
                if st.button("Preview generated columns", key="ml_w_fe_prev"):
                    sample, quality = FE.preview(df.head(2000), f, nums, cats, txt)
                    st.session_state["ml_w_fe_preview"] = (sample, quality)
                pv = st.session_state.get("ml_w_fe_preview")
                if pv:
                    st.dataframe(pv[0], hide_index=True, width="stretch")
                    st.dataframe(pv[1], hide_index=True, width="stretch")
                    st.caption("Preview is fitted on up to 2,000 rows for display only; training refits on the training rows.")
            except FE.FeatureConfigError as e:
                problems.append(f"Feature engineering: {e}")
                st.error(str(e))

    with st.expander("Dimensionality reduction", expanded=False):
        r = dict(p.get("reduction") or {})
        meths = ["none", "pca", "svd", "agglomeration"]
        names = {"none": "None", "pca": "PCA (principal components)", "svd": "Truncated SVD", "agglomeration": "Feature agglomeration (merge similar features)"}
        meth = st.selectbox("Method", meths, index=meths.index(r.get("method", "none")) if r.get("method", "none") in meths else 0,
                            format_func=names.get, key="ml_w_red_m")
        if meth == "none":
            p["reduction"] = None
        else:
            nc = st.number_input("Number of components", 1, 500, int(r.get("n_components", 5)) if float(r.get("n_components", 5)) >= 1 else 5, key="ml_w_red_n")
            p["reduction"] = {"method": meth, "n_components": int(nc)}
            st.caption("Fitted on training rows only. Features are standardised first. Components are harder to interpret than original columns.")

    with st.expander("Feature selection", expanded=False):
        sl = dict(p.get("selection") or {})
        meths = list(SE.METHODS) if task != "clustering" else sorted(SE.UNSUPERVISED_OK)
        meth = st.selectbox("Method", meths, index=meths.index(sl.get("method", "none")) if sl.get("method", "none") in meths else 0,
                            format_func=SE.METHODS.get, key="ml_w_sel_m")
        if meth == "none":
            p["selection"] = None
        else:
            st.caption(SE.METHOD_NOTES.get(meth, "") + " Runs inside each training fold, so held-out targets are never used.")
            new = {"method": meth}
            if meth in ("kbest", "mutual_info", "rfe", "model"):
                new["k"] = int(st.number_input("Features to keep", 1, 1000, int(sl.get("k", 20)), key="ml_w_sel_k"))
            if meth == "percentile":
                new["percentile"] = int(st.slider("Percentage to keep", 1, 100, int(sl.get("percentile", 50)), key="ml_w_sel_p"))
            if meth == "variance":
                new["threshold"] = float(st.number_input("Minimum variance", 0.0, 100.0, float(sl.get("threshold", 0.0)), format="%g", key="ml_w_sel_t"))
            if meth == "l1":
                new["strength"] = float(st.number_input("Regularisation C (smaller keeps fewer)", 0.001, 100.0, float(sl.get("strength", 1.0)), format="%g", key="ml_w_sel_c"))
            p["selection"] = new
    return problems


# ---------------------------------------------------------------------------------------------- models
DEFAULT_COMPARE = {"classification": ["logistic_regression", "decision_tree", "random_forest", "gradient_boosting"],
                   "regression": ["linear_regression", "decision_tree_regressor", "random_forest_regressor", "gradient_boosting_regressor"],
                   "clustering": ["kmeans", "dbscan", "agglomerative"]}


def unavailable_note(task: str) -> None:
    missing = [s for s in models_for(task, include_unavailable=True) if not s.available]
    if missing:
        st.caption("Optional models not installed: " + "; ".join(f"{s.name} ({INSTALL_HINT.get(s.optional_package or '', '')})" for s in missing))


def compatibility(m: dict, n_rows: int, n_features: int) -> list[str]:
    """Show blocking reasons and warnings for the chosen models. Returns the keys that cannot run."""
    blocked = []
    for k in m["models"]:
        ok, reasons, warns = check_compatibility(REGISTRY[k], n_rows, n_features, m["pre"])
        for r in reasons:
            st.error(f"{REGISTRY[k].name}: {r}")
        for w in warns:
            st.warning(w)
        if not ok:
            blocked.append(k)
    return blocked


def _param_widget(key: str, p, current):
    wk = f"ml_w_mp_{key}_{p.name}"
    label = p.name.replace("_", " ")
    if p.kind == "choice":
        opts = list(p.choices)
        idx = opts.index(current) if current in opts else (opts.index(p.default) if p.default in opts else 0)
        return st.selectbox(label, opts, index=idx, format_func=lambda v: "none" if v is None else str(v), key=wk, help=p.help or None)
    if p.kind == "bool":
        return st.checkbox(label, value=bool(current if current is not None else p.default), key=wk, help=p.help or None)
    if p.default is None:                 # an optional limit such as max_depth
        on = st.checkbox(f"Limit {label}", value=current is not None, key=wk + "_on")
        if not on:
            return None
    lo, hi = p.low, p.high
    val = current if current is not None else (p.default if p.default is not None else (lo if lo is not None else 1))
    if p.kind == "int":
        return int(st.number_input(label, int(lo) if lo is not None else None, int(hi) if hi is not None else None, int(val), key=wk, help=p.help or None))
    return float(st.number_input(label, float(lo) if lo is not None else None, float(hi) if hi is not None else None, float(val), format="%g", key=wk,
                                 help=p.help or None))


def model_settings(m: dict) -> None:
    """Per-model settings; only values that differ from the defaults are sent to the engine."""
    if m.get("model_params") is None:
        m["model_params"] = {}
    mp = m["model_params"]
    keys = [k for k in m["models"] if REGISTRY[k].params]
    for k in list(mp):
        if k not in m["models"]:
            mp.pop(k)
    if not keys:
        return
    with st.expander("Model settings (optional)"):
        st.caption("Defaults are sensible starting points. Settings that you tune automatically below are searched instead of fixed.")
        for k in keys:
            spec = REGISTRY[k]
            st.markdown(f"**{spec.name}**")
            cols = st.columns(min(3, len(spec.params)))
            chosen = {}
            for i, p in enumerate(spec.params):
                with cols[i % len(cols)]:
                    v = _param_widget(k, p, (mp.get(k) or {}).get(p.name, p.default))
                if v != p.default:
                    chosen[p.name] = v
            if chosen:
                mp[k] = chosen
            else:
                mp.pop(k, None)


def validation_tuning(m: dict, task: str, has_val: bool) -> None:
    if task == "clustering":
        return
    if not m.get("cv"):
        m["cv"] = {"method": "auto", "folds": m.get("folds", 5), "repeats": 1, "gap": 0}
    if not m.get("tuning"):
        m["tuning"] = TU.TuningConfig().to_dict()
    cvd, tud = m["cv"], m["tuning"]
    with st.expander("Validation and tuning", expanded=False):
        st.markdown("**Cross-validation**" + ("" if not has_val else " (used for tuning; models are still compared on the validation set)"))
        meths = list(CV.METHODS)
        cvd["method"] = st.selectbox("Fold strategy", meths, index=meths.index(cvd.get("method", "auto")), format_func=CV.METHODS.get, key="ml_w_cv_m")
        c1, c2, c3 = st.columns(3)
        cvd["folds"] = int(c1.number_input("Folds", 2, CV.MAX_FOLDS, int(cvd.get("folds", m.get("folds", 5))), key="ml_w_cv_f"))
        m["folds"] = cvd["folds"]
        cvd["repeats"] = int(c2.number_input("Repeats", 1, CV.MAX_REPEATS, int(cvd.get("repeats", 1)), key="ml_w_cv_r",
                                             help="Several shuffles give a steadier estimate (not for time-series folds)."))
        cvd["gap"] = int(c3.number_input("Time-series gap (rows)", 0, 1000, int(cvd.get("gap", 0)), key="ml_w_cv_g",
                                         help="Rows left out between training and validation in time-series folds, to avoid look-ahead."))
        if m.get("strategy") == "chronological" and cvd["method"] not in ("auto", "timeseries"):
            st.warning("The split is by time; shuffled folds would validate models on the past. Prefer automatic or time-series folds.")
        st.markdown("**Hyperparameter tuning**")
        tmeths = list(TU.METHODS)
        tud["method"] = st.selectbox("Search", tmeths, index=tmeths.index(tud.get("method", "none")), format_func=TU.METHODS.get, key="ml_w_tu_m")
        if tud["method"] != "none":
            c1, c2, c3 = st.columns(3)
            if tud["method"] in ("random", "halving"):
                tud["n_iter"] = int(c1.number_input("Combinations to try", 2, TU.MAX_TRIALS, int(tud.get("n_iter", 10)), key="ml_w_tu_n"))
            tud["max_trials"] = int(c2.number_input("Trial limit per model", 1, TU.MAX_TRIALS, int(tud.get("max_trials", 30)), key="ml_w_tu_max"))
            tud["time_budget_s"] = float(c3.number_input("Time budget per model (seconds)", 5, TU.MAX_BUDGET_S, int(tud.get("time_budget_s", 300)), key="ml_w_tu_t"))
            st.caption("Each combination is scored with the folds above on the training rows only; preprocessing is re-learned inside every fold. "
                       "Search spaces are small, fixed lists per model; settings you fixed above are not searched.")
            for k in m["models"]:
                spec = REGISTRY[k]
                space = {a: v for a, v in spec.search_space.items() if a not in (m.get("model_params") or {}).get(k, {})}
                if space:
                    size = 1
                    for v in space.values():
                        size *= len(v)
                    st.caption(f"{spec.name}: " + "; ".join(f"{a} in {v}" for a, v in space.items()) + f" ({size} combinations)")
                    if tud["method"] == "grid" and size > tud["max_trials"]:
                        st.warning(f"{spec.name}: the grid has {size} combinations, above the trial limit; use random search or halving, or raise the limit.")
                else:
                    st.caption(f"{spec.name}: nothing to tune.")
        m["shap"] = st.checkbox("Also compute SHAP explanations for the selected model" + ("" if XP.SHAP_AVAILABLE else " (needs: pip install shap)"),
                                value=bool(m.get("shap")) and XP.SHAP_AVAILABLE, disabled=not XP.SHAP_AVAILABLE, key="ml_w_shap")


# ---------------------------------------------------------------------------------------------- results
def results_extras(exp) -> None:
    if exp.cv_info:
        st.caption(f"Cross-validation: {exp.cv_info.get('description', '')}.")
    tuned = [c for c in exp.candidates if c.tuning]
    if tuned:
        st.markdown("#### Tuning")
        for c in tuned:
            t = c.tuning
            with st.expander(f"{c.name}: {len(t['trials'])} combination(s) tried in {t['seconds']}s" + (f" ({t['stopped']})" if t.get("stopped") else "")):
                rows = [{**{k.split('__')[-1]: v for k, v in tr["params"].items()}, "score": tr["score"], "spread": tr["std"], "round": tr["round"],
                         "rows per fold": tr["rows_per_fold"], "seconds": tr["seconds"], "error": tr["error"]} for tr in t["trials"]]
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.markdown("**Chosen settings:** " + ", ".join(f"`{k.split('__')[-1]}` = {v}" for k, v in t["best_params"].items()))
    folds = [c for c in exp.candidates if c.fold_scores]
    if folds:
        fig = go.Figure()
        for c in folds:
            fig.add_trace(go.Box(y=c.fold_scores, name=c.name, boxpoints="all", marker_color=C.TEAL if c.key == exp.selected_key else C.MUTED))
        st.plotly_chart(C._layout(fig, "Score in each cross-validation fold", 300), width="stretch")
        st.caption("A wide spread means the score depends on which rows land in which fold; treat small differences between models with caution.")
    rep = exp.pipeline_report or {}
    if rep.get("reduction") or rep.get("selection") or rep.get("outliers"):
        st.markdown("#### What the preparation steps learned (training rows only)")
        if rep.get("outliers"):
            o = rep["outliers"]
            st.markdown(f"**Outliers:** {OL.TREATMENTS.get(o['treatment'], o['treatment'])} using {OL.METHODS.get(o['method'], o['method'])}.")
            st.dataframe(pd.DataFrame([{"column": k, "lower": v[0], "upper": v[1]} for k, v in o["bounds"].items()]), hide_index=True, width="content")
        if rep.get("reduction"):
            r = rep["reduction"]
            st.markdown(f"**Reduction:** {r['method'].upper()} turned {r['inputs']} prepared features into {r['outputs']}"
                        + (f", keeping {100 * r['explained_total']:.1f}% of the variance." if r.get("explained_total") is not None else "."))
        if rep.get("selection"):
            s = rep["selection"]
            kept = [x["feature"] for x in s.get("features", []) if x.get("kept")] if "features" in s else s.get("kept", [])
            st.markdown(f"**Selection:** kept {len(kept)} feature(s).")
            if "features" in s:
                st.dataframe(pd.DataFrame(s["features"]), hide_index=True, width="stretch")
    ni = exp.native_importance
    if ni:
        st.markdown("#### Model-specific importance")
        d = pd.DataFrame(ni["rows"]).head(15)
        fig = go.Figure(go.Bar(x=d["importance"], y=d["feature"], orientation="h", marker_color=C.TEAL))
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(C._layout(fig, "Impurity importance" if ni["kind"] == "impurity" else "Coefficient size", max(260, 26 * len(d) + 110)), width="stretch")
        st.caption(ni["note"])
    if exp.shap:
        st.markdown("#### SHAP explanation")
        if exp.shap.get("available"):
            d = pd.DataFrame(exp.shap["rows"]).head(15)
            fig = go.Figure(go.Bar(x=d["mean_abs_shap"], y=d["feature"], orientation="h", marker_color=C.TEAL))
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(C._layout(fig, "Mean |SHAP value| per prepared feature", max(260, 26 * len(d) + 110)), width="stretch")
            st.caption(exp.shap["note"] + " SHAP shows how the model uses features, not what causes the outcome.")
        else:
            st.info(exp.shap.get("reason", "SHAP is not available."))
