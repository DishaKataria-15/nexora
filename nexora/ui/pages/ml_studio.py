from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ...eda import charts as C
from ...eda import profiling as P
from ...ml import evaluation as E
from ...ml import imbalance as IMB
from ...ml.compute import detect_compute, device_options
from ...ml.model_registry import REGISTRY, models_for, recommend_models
from ...ml.splitting import STRATEGIES, SplitConfig, SplitError, make_split, recommend_split
from ...ml.task_detection import recommend_targets, suggest_goal, task_for_column
from ...ml.training import ExperimentConfig, ExperimentError, TestSetAlreadyUsed, finalize_experiment
from ...preprocessing import leakage_checks as L
from ...preprocessing.pipeline import PreprocessOptions, select_features
from ...reporting import zip_bytes
from .. import components, state, theme
from . import ml_options as MO

SUP = ["Goal", "Target and features", "Split", "Preparation", "Models", "Train", "Results"]
CLU = ["Goal", "Features", "Models", "Train", "Results"]
TASKS = {"regression": "Predict a number (regression)", "classification": "Predict a category (classification)",
         "clustering": "Discover groups (clustering)", "explore": "Explore and visualize only"}
TASK_HELP = {"regression": "For values such as a price or quantity. The result is a number with a typical error size.",
             "classification": "For labels such as yes/no or a type. The result is a class with a confidence.",
             "clustering": "Finds groups of similar rows without any answer key. Groups are patterns, not ground-truth labels.",
             "explore": "Charts and summaries only; no model is trained."}


def _ml() -> dict:
    S = st.session_state
    if S["ml"] is None:
        S["ml"] = {"task": None, "target": None, "exclude": [], "strategy": None, "shares": "recommended", "custom_test": 20, "custom_val": 0, "seed": S["seed"],
                   "group_col": None, "time_col": None, "imbalance": None, "pre": PreprocessOptions().to_dict(), "mode": "Compare models", "models": [],
                   "metric": None, "compute": "auto", "folds": 5, "positive": None, "drop_missing": False, "kmin": 2, "kmax": 8, "unit": "", "step": "Goal",
                   "last_mode": None, "model_params": {}, "cv": None, "tuning": None, "shap": False, "extra_metrics": []}
    return S["ml"]


def _set_step(name: str) -> None:
    st.session_state["ml"]["step"] = name


def _nav(m: dict, back: str | None, nxt: str | None, can_next: bool = True, why: str = "") -> None:
    c1, c2, c3 = st.columns([1.5, 1, 4])
    if nxt:
        c1.button("Next: " + nxt, type="primary", disabled=not can_next, on_click=_set_step, args=(nxt,), key=f"nx_{m['step']}")
    if back:
        c2.button("Back", on_click=_set_step, args=(back,), key=f"bk_{m['step']}")
    if not can_next and why:
        c3.caption(why)


def render() -> None:
    st.title("ML Studio")
    wf = components.dataset_bar("ml", empty_body="ML Studio trains models on a dataset you have opened.")
    if wf is None:
        return
    df, note = wf
    m = _ml()
    steps = CLU if m["task"] == "clustering" else SUP
    if m["step"] not in steps:
        m["step"] = "Goal"
    theme.stepper(steps, steps.index(m["step"]), key="ml")
    page = {"Goal": _goal, "Target and features": _target, "Features": _target, "Split": _split, "Preparation": _prep, "Models": _models, "Train": _train, "Results": _results}
    page[m["step"]](df, note, m)


def _exclude_widget(label: str, pool: list[str], m: dict, help_text: str | None = None) -> None:
    """Multiselect bound to session state (no 'default' argument, which would conflict with programmatic updates)."""
    st.session_state["ml_w_exclude"] = [c for c in st.session_state.get("ml_w_exclude", m["exclude"]) if c in pool]
    m["exclude"] = st.multiselect(label, pool, key="ml_w_exclude", help=help_text)


# ---------------------------------------------------------------------------------------------
def _goal(df, note, m) -> None:
    st.subheader("What do you want to achieve?")
    keys = list(TASKS)
    sug = suggest_goal(df)
    st.markdown(f"{theme.chip('Suggested', 'info')} &nbsp; **{TASKS[sug['task']]}**. {theme.esc(sug['reason'])}", unsafe_allow_html=True)
    if m["task"] is None and sug["column"]:
        m["target"] = sug["column"]
    cur = m["task"] if m["task"] in keys else (sug["task"] if sug["task"] != "explore" else "classification")
    m["task"] = st.radio("Goal", keys, index=keys.index(cur), format_func=lambda k: TASKS[k], captions=[TASK_HELP[k] for k in keys], key="ml_w_goal")
    if m["task"] == "explore":
        st.info("No model is needed for this. Charts, correlations and dashboards are in Visualize & Dashboard.")
        if st.button("Open Visualize & Dashboard", type="primary"):
            state.goto("Visualize & Dashboard")
            st.rerun()
        return
    m["models"] = [k for k in m["models"] if k in REGISTRY and REGISTRY[k].task == m["task"]]
    _nav(m, None, "Features" if m["task"] == "clustering" else "Target and features")


def _target(df, note, m) -> None:
    task = m["task"]
    opts = PreprocessOptions(**m["pre"])
    if task == "clustering":
        st.subheader("Which columns should define the groups?")
        st.caption("No target is needed. Clusters are groups of similar rows, not ground-truth labels.")
        base = select_features(df, None, [], None, opts)
        _exclude_widget("Exclude columns", [c for c in df.columns if c not in base.excluded], m,
                        "Leave out anything that should not influence the grouping (for example a record number).")
        fs = select_features(df, None, m["exclude"], None, opts)
        _feature_summary(fs)
        _nav(m, "Goal", "Models", bool(fs.all), "Select at least one usable column.")
        return
    st.subheader("What should the model predict?")
    recs = recommend_targets(df, task)
    valid = [c for c in df.columns if task_for_column(df[c], c)[0] == task]
    if not valid:
        theme.empty_state("No suitable target", f"No column in this data can be predicted as {'a number' if task == 'regression' else 'a category'}. Try the other goal.")
        _nav(m, "Goal", None)
        return
    ordered = [r["column"] for r in recs] + [c for c in valid if c not in [r["column"] for r in recs]]
    if recs:
        st.markdown("Suggested targets (a suggestion only, you decide):")
        st.dataframe(pd.DataFrame(recs)[["column", "reason", "missing %"]], hide_index=True, width="stretch")
    if m["target"] not in ordered:
        m["target"] = ordered[0]
    m["target"] = st.selectbox("Target column", ordered, index=ordered.index(m["target"]), key="ml_w_target")
    y = df[m["target"]]
    n_miss = int(y.isna().sum())
    ok = True
    if n_miss:
        st.warning(f"{n_miss:,} row(s) have no value in '{m['target']}'. Nexora never invents labels.")
        m["drop_missing"] = st.checkbox(f"Leave these {n_miss:,} row(s) out of training and testing", value=m["drop_missing"], key="ml_w_dropmiss")
        ok = m["drop_missing"]
    else:
        m["drop_missing"] = False
    if task == "classification":
        info = IMB.class_distribution(y.astype(str).where(y.notna()))
        st.markdown("**Class balance in the target**")
        c1, c2 = st.columns([1, 1.4])
        c1.dataframe(info.table, hide_index=True, width="stretch")
        fig = go.Figure(go.Bar(x=info.table["class"], y=info.table["rows"], marker_color=C.TEAL, text=[f"{s}%" for s in info.table["share %"]], textposition="outside"))
        c2.plotly_chart(C._layout(fig, "Rows per class", 260), width="stretch", config=C.PLOTLY_CONFIG)
        if info.n_classes > 30 or info.smallest_count < 2:
            ok = False
            st.error("Each class needs at least 2 rows, and this looks like too many classes. Merge rare classes or choose another target.")
    else:
        if y.nunique() > 1:
            st.plotly_chart(C.build_figure(df, C.ChartSpec("histogram", x=m["target"])), width="stretch", config=C.PLOTLY_CONFIG)
        m["unit"] = st.text_input("Unit of the target (optional, shown next to errors)", m["unit"], key="ml_w_unit", placeholder="for example: USD thousands")
    st.markdown("**Features**")
    work = df[df[m["target"]].notna()]
    _auto_exclude_leaks(work, m, task, opts)
    fs = select_features(df, m["target"], m["exclude"], None, opts)
    leaks = L.check_target_leakage(work, m["target"], task, fs.all) + L.check_post_outcome_names(fs.all, m["target"])
    for w_ in leaks:
        (st.error if w_.level == "warning" else st.warning)(w_.message)
    sus = sorted({c for w_ in leaks if w_.suggest_exclude for c in w_.columns} - set(m["exclude"]))
    if sus:
        def _excl(cols=tuple(sus)):
            cur = list(st.session_state["ml"]["exclude"])
            st.session_state["ml"]["exclude"] = cur + [c for c in cols if c not in cur]
            st.session_state["ml_w_exclude"] = list(st.session_state["ml"]["exclude"])
        st.button(f"Exclude suspected leaky column(s): {', '.join(sus)}", on_click=_excl, key="excl_leak")
    base = select_features(df, m["target"], [], None, opts)
    _exclude_widget("Also exclude these columns", [c for c in df.columns if c != m["target"] and c not in base.excluded], m)
    fs = select_features(df, m["target"], m["exclude"], None, opts)
    _feature_summary(fs)
    if not fs.all:
        ok = False
    _nav(m, "Goal", "Split", ok, "Resolve the items above to continue.")


def _auto_exclude_leaks(work: pd.DataFrame, m: dict, task: str, opts: PreprocessOptions) -> None:
    """Strong leakage (a column that alone reveals the target) is excluded by default; the user can include it on purpose."""
    target = m["target"]
    auto = m.setdefault("auto_excluded", {})          # target -> columns excluded automatically
    keep = m.setdefault("keep_leaky", {})             # target -> columns the user chose to include anyway
    if "ml_w_exclude" in st.session_state:            # the multiselect is the source of truth for what the user changed
        m["exclude"] = list(st.session_state["ml_w_exclude"])
        for c in [c for c in auto.get(target, []) if c not in m["exclude"]]:      # removed by hand = include on purpose
            keep[target] = sorted(set(keep.get(target, [])) | {c})
            auto[target] = [x for x in auto[target] if x != c]
    for other, cols in list(auto.items()):            # a different target: undo the old automatic exclusions
        if other != target:
            m["exclude"] = [c for c in m["exclude"] if c not in cols]
            auto.pop(other)
    all_fs = select_features(work, target, [c for c in m["exclude"] if c not in auto.get(target, [])], None, opts)
    strong = {c for w_ in L.check_target_leakage(work, target, task, all_fs.all) if w_.level == "warning" and w_.suggest_exclude
              for c in w_.columns}
    new = sorted(strong - set(keep.get(target, [])) - set(m["exclude"]))
    if new:
        m["exclude"] = list(m["exclude"]) + new
        auto[target] = sorted(set(auto.get(target, [])) | set(new))
    st.session_state["ml_w_exclude"] = [c for c in m["exclude"]]
    shown = [c for c in auto.get(target, []) if c in m["exclude"]]
    if shown:
        st.error(f"Excluded automatically because each one reveals '{target}' almost by itself (leakage): **{', '.join(shown)}**. "
                 "A model using it looks excellent in testing but fails on new data, because the value is only known after the outcome.")

        def _keep(cols=tuple(shown), t=target):
            mm = st.session_state["ml"]
            mm["keep_leaky"][t] = sorted(set(mm["keep_leaky"].get(t, [])) | set(cols))
            mm["exclude"] = [c for c in mm["exclude"] if c not in cols]
            mm["auto_excluded"][t] = [c for c in mm["auto_excluded"].get(t, []) if c not in cols]
            st.session_state["ml_w_exclude"] = list(mm["exclude"])
        st.button("Include anyway (only if it is truly known before the prediction)", on_click=_keep, key="keep_leak")


def _feature_summary(fs) -> None:
    st.markdown(f"**{len(fs.all)} feature(s) will be used:** {len(fs.numeric)} numeric, {len(fs.categorical)} categorical, {len(fs.datetime)} date, {len(fs.text)} text.")
    if fs.excluded:
        with st.expander(f"{len(fs.excluded)} column(s) not used"):
            st.dataframe(pd.DataFrame([{"Column": k, "Reason": v} for k, v in fs.excluded.items()]), hide_index=True, width="stretch")


def _shares(m: dict, n: int) -> tuple[float, float, float]:
    rec, _ = recommend_split(n)
    return {"recommended": (rec.train, rec.val, rec.test), "70/15/15": (0.7, 0.15, 0.15), "80/20": (0.8, 0.0, 0.2),
            "custom": (1 - (m["custom_test"] + m["custom_val"]) / 100, m["custom_val"] / 100, m["custom_test"] / 100)}[m["shares"]]


def _split(df, note, m) -> None:
    task = m["task"]
    st.subheader("How should the data be split?")
    work = df[df[m["target"]].notna()].reset_index(drop=True)
    n = len(work)
    rec, why = recommend_split(n)
    st.markdown(why)
    st.caption("The model learns from the training rows. Validation rows (or cross-validation) are used to compare models. Test rows are kept untouched until you choose the final model, and are used once.")
    dts = P.datetime_columns(work)
    groups = L.group_candidates(work)
    avail = (["stratified"] if task == "classification" else []) + ["random"] + (["group"] if groups else []) + (["chronological"] if dts else [])
    if m["strategy"] not in avail:
        m["strategy"] = avail[0]
    captions = {"stratified": "Keeps each class's share the same in every part.", "random": "Shuffles rows; fine when rows are independent and not time-ordered.",
                "group": "Keeps all rows of one entity (for example one customer) together.", "chronological": "Trains on earlier dates and tests on later ones."}
    m["strategy"] = st.radio("Split strategy", avail, index=avail.index(m["strategy"]), format_func=lambda k: STRATEGIES[k], captions=[captions[k] for k in avail])
    m["group_col"] = st.selectbox("Entity column", groups, index=groups.index(m["group_col"]) if m["group_col"] in groups else 0, key="ml_w_group") if m["strategy"] == "group" else None
    m["time_col"] = st.selectbox("Date column", dts, index=dts.index(m["time_col"]) if m["time_col"] in dts else 0, key="ml_w_time") if m["strategy"] == "chronological" else None
    labels = {"recommended": f"Recommended ({round(rec.train * 100)}/{round(rec.val * 100)}/{round(rec.test * 100)})", "70/15/15": "70% train / 15% validation / 15% test",
              "80/20": "80% train / 20% test (cross-validation inside training)", "custom": "Custom"}
    keys = list(labels)
    m["shares"] = st.radio("Shares", keys, index=keys.index(m["shares"]), format_func=lambda k: labels[k])
    if m["shares"] == "custom":
        c1, c2 = st.columns(2)
        m["custom_test"] = c1.slider("Test %", 5, 40, m["custom_test"], key="ml_w_ct")
        m["custom_val"] = c2.slider("Validation % (0 = use cross-validation)", 0, 30, m["custom_val"], key="ml_w_cv")
    tr, va, te = _shares(m, n)
    m["seed"] = int(st.number_input("Random seed", 0, 10**6, int(m["seed"]), key="ml_w_seed", help="Fixing the seed makes the split reproducible; it is saved with the experiment."))
    cfg = SplitConfig(m["strategy"], round(tr, 4), round(va, 4), round(te, 4), m["seed"], m["group_col"], m["time_col"])
    ok = True
    try:
        res = make_split(n, cfg, y=work[m["target"]].astype(str) if task == "classification" else None, groups=work[m["group_col"]] if m["group_col"] else None,
                         times=work[m["time_col"]] if m["time_col"] else None)
        st.dataframe(pd.DataFrame([{"Part": k.title(), "Rows": v, "Share %": round(100 * v / n, 1)} for k, v in res.sizes().items() if v or k != "val"]), hide_index=True, width="content")
        for w_ in res.warnings:
            st.warning(w_)
        if m["strategy"] == "chronological":
            tt = pd.to_datetime(work[m["time_col"]])
            st.caption(f"Training ends {str(tt.iloc[res.train].max())[:10]}; test period begins {str(tt.iloc[res.test].min())[:10]}.")
        if res.val.size == 0:
            m["folds"] = st.slider("Cross-validation folds", 2, 10, m["folds"], key="ml_w_folds", help="Models are compared by training and checking on rotating parts of the training rows.")
            if m.get("cv"):
                m["cv"]["folds"] = m["folds"]
                st.session_state.pop("ml_w_cv_f", None)
    except SplitError as e:
        ok = False
        st.error(str(e))
    for w_ in L.check_split_strategy(work, m["strategy"], dts, groups):
        st.warning(w_.message)
    _nav(m, "Target and features", "Preparation", ok, "Fix the split settings above.")


def _prep(df, note, m) -> None:
    task = m["task"]
    st.subheader("Preparation inside the model")
    st.caption("Everything below is learned from the training rows only, then applied to validation and test rows. Nothing from held-out data leaks in.")
    p = m["pre"]
    c1, c2 = st.columns(2)
    p["numeric_impute"] = c1.selectbox("Fill missing numbers with", ["median", "mean"], index=["median", "mean"].index(p["numeric_impute"]), key="ml_w_imp",
                                      help="The median is less affected by extreme values. Missing categories become their own '(missing)' category, so a blank stays visible to the model.")
    scal = ["auto", "standard", "minmax", "robust", "none"]
    p["scaling"] = c2.selectbox("Scaling", scal, index=scal.index(p["scaling"]), key="ml_w_scale",
                                help="Auto scales only for models that need it (linear and distance-based). Tree models ignore scale.")
    p["encoding"] = c1.selectbox("Categories", ["onehot", "ordinal"], index=["onehot", "ordinal"].index(p["encoding"]), key="ml_w_enc",
                                 format_func=lambda k: {"onehot": "One column per category (recommended)", "ordinal": "Numbered categories"}[k])
    p["skew_transform"] = c2.checkbox("Reduce skew in numeric features (Yeo-Johnson)", value=p["skew_transform"], key="ml_w_skew")
    p["use_datetime"] = st.checkbox("Use dates as calendar features (year, month, weekday, day of year, days since start)", value=p["use_datetime"], key="ml_w_dt")
    txt = [c for c in df.columns if P.column_role(df[c], c) == "text" and c != m["target"]]
    if txt:
        p["text_columns"] = st.multiselect("Use free-text columns (word weights)", txt, default=[c for c in p["text_columns"] if c in txt], key="ml_w_txt")
    fs = select_features(df, m["target"], m["exclude"], None, PreprocessOptions(**p))
    _feature_summary(fs)
    if task == "classification":
        st.markdown("#### Class imbalance")
        work = df[df[m["target"]].notna()]
        info = IMB.class_distribution(work[m["target"]].astype(str))
        st.dataframe(info.table, hide_index=True, width="content")
        st.markdown(IMB.metric_advice(info))
        all_num = not fs.categorical and not fs.datetime and not fs.text
        opts = IMB.strategy_options(info, all_num)
        avail = [o for o in opts if o.available]
        if m["imbalance"] not in [o.key for o in avail]:
            m["imbalance"] = next((o.key for o in avail if o.recommended), "none")
        lab = {o.key: o.label + ("  (recommended)" if o.recommended else "") for o in avail}
        m["imbalance"] = st.radio("How to handle it", [o.key for o in avail], index=[o.key for o in avail].index(m["imbalance"]), format_func=lambda k: lab[k], key="ml_w_imb")
        chosen = next(o for o in avail if o.key == m["imbalance"])
        st.caption(chosen.why)
        for w_ in chosen.warnings:
            st.warning(w_)
        st.caption("Resampling is applied to training data only, and inside each fold during cross-validation. Validation and test rows are never resampled.")
        na = [o for o in opts if not o.available]
        if na:
            with st.expander("Not available for this data"):
                for o in na:
                    st.markdown(f"- **{o.label}**: {o.why}")
    else:
        m["imbalance"] = "none"
    st.markdown("#### Optional steps inside the model")
    problems = MO.prep_advanced(df, m, task)
    _nav(m, "Split", "Models", not problems, " ".join(problems))


def _models(df, note, m) -> None:
    task = m["task"]
    st.subheader("Choose how to pick a model")
    if task != "clustering":
        work = df[df[m["target"]].notna()]
        fs = select_features(df, m["target"], m["exclude"], None, PreprocessOptions(**m["pre"]))
    else:
        work, fs = df, select_features(df, None, m["exclude"], None, PreprocessOptions(**m["pre"]))
    yinfo = IMB.class_distribution(work[m["target"]].astype(str)) if task == "classification" else None
    n_classes = yinfo.n_classes if yinfo else 2
    prefer = "balanced"
    if task != "clustering":
        prefer = st.selectbox("What matters most", ["balanced", "interpretable", "speed"], key="ml_w_prefer",
                              format_func={"balanced": "Best accuracy with sensible effort", "interpretable": "Easy to explain", "speed": "Fast to train"}.get)
    recs = recommend_models(task, len(work), len(fs.numeric) + len(fs.datetime), len(fs.categorical), imbalance_ratio=yinfo.ratio if yinfo else None, prefer=prefer)
    st.markdown("**Suggested starting points** (a guess from your data's size and types; only training shows what works)")
    for r in recs[:3]:
        st.markdown(f"{r.rank}. **{REGISTRY[r.key].name}**: {r.reason} *Trade-off:* {r.tradeoff}")
    modes = ["Recommended model", "Choose a model", "Compare models"]
    m["mode"] = st.radio("Path", modes, index=modes.index(m["mode"]), horizontal=True, key="ml_w_mode",
                         help="Recommended: train the top suggestion. Choose: train one model you pick. Compare: train several and select by score under the same protocol.")
    avail = [s.key for s in models_for(task)]
    name = {k: REGISTRY[k].name for k in avail}
    if m["mode"] == "Recommended model":
        m["models"] = [recs[0].key]
        st.info(f"Will train: **{name[recs[0].key]}**. It is a sensible default, not a claim that it is the best model.")
    elif m["mode"] == "Choose a model":
        cur = m["models"][0] if m["models"] and m["models"][0] in avail else recs[0].key
        pick = st.selectbox("Model", avail, index=avail.index(cur), format_func=name.get, key="ml_w_pick")
        m["models"] = [pick]
        st.caption(REGISTRY[pick].description + " " + REGISTRY[pick].tradeoff)
    else:
        if m.get("last_mode") != "Compare models" or "ml_w_cmp" not in st.session_state:
            # entering compare mode: start with the top 3 suggestions for this data; any other model can be added
            top3 = [r.key for r in recs if r.key in avail][:3]
            st.session_state["ml_w_cmp"] = top3 or [k for k in MO.DEFAULT_COMPARE.get(task, []) if k in avail] or list(avail)
        m["models"] = st.multiselect("Models to compare", avail, format_func=name.get, key="ml_w_cmp")
        st.caption("Starts with the top 3 suggestions above; add or remove any model. All models get the same data split, the same preparation and the same metric, "
                   "so the comparison is fair. The test set is not used here.")
    m["last_mode"] = m["mode"]
    MO.unavailable_note(task)
    blocked = MO.compatibility(m, len(work), len(fs.all))
    MO.model_settings(m)
    metrics = E.selection_metrics(task, n_classes)
    default_metric = E.default_primary(task, yinfo.level if yinfo else "balanced")
    if m["metric"] not in metrics:
        m["metric"] = default_metric
    m["metric"] = st.selectbox("Primary metric (used to choose between models)", list(metrics), index=list(metrics).index(m["metric"]), format_func=metrics.get, key="ml_w_metric",
                               help="Models are ranked by this metric on validation data or cross-validation.")
    others = [k for k in metrics if k != m["metric"]]
    st.session_state["ml_w_extra"] = [k for k in st.session_state.get("ml_w_extra", m.get("extra_metrics") or []) if k in others]
    m["extra_metrics"] = st.multiselect("Also compare models on (up to 4)", others, format_func=metrics.get, max_selections=4, key="ml_w_extra",
                                        help="Shown as extra columns in the comparison so you can weigh trade-offs. The primary metric still decides which model is selected, "
                                             "so the choice is made by one rule fixed in advance.")
    if task == "clustering":
        st.caption("Clustering metrics are internal measures of group shape. They are not accuracy, because there is no answer key.")
        c1, c2 = st.columns(2)
        m["kmin"] = int(c1.number_input("Smallest number of groups to try", 2, 20, m["kmin"], key="ml_w_kmin"))
        m["kmax"] = int(c2.number_input("Largest number of groups to try", m["kmin"], 30, max(m["kmax"], m["kmin"]), key="ml_w_kmax"))
    elif task == "classification" and n_classes == 2:
        classes = sorted(work[m["target"]].astype(str).unique())
        if m["positive"] not in classes:
            m["positive"] = yinfo.smallest_class
        m["positive"] = st.selectbox("Class you care most about (for precision, recall, F1)", classes, index=classes.index(m["positive"]), key="ml_w_pos")
    with st.expander("What do these metrics mean?"):
        for k in metrics:
            st.markdown(f"**{metrics[k]}** ({E.metric_direction(k)} is better): {E.METRIC_HELP[k]}")
    info = detect_compute()
    opts = device_options(info, any(REGISTRY[k].gpu_capable for k in m["models"]))
    if m["compute"] not in opts:
        m["compute"] = "auto"
    m["compute"] = st.radio("Compute", opts, index=opts.index(m["compute"]), horizontal=True, key="ml_w_comp",
                            format_func={"auto": "Automatic (recommended)", "cpu": "CPU", "gpu": "GPU"}.get)
    st.caption("CPU is fast enough for most tabular data, and a GPU is not always faster. " + info.note)
    if task != "clustering":
        tr, va, te = _shares(m, len(work))
        MO.validation_tuning(m, task, va > 0)
    runnable = [k for k in m["models"] if k not in blocked]
    _nav(m, "Features" if task == "clustering" else "Preparation", "Train", bool(runnable), "Choose at least one model that can run on this data.")


def _config(m: dict, df: pd.DataFrame) -> ExperimentConfig:
    h = state.handle()
    n = int(df[m["target"]].notna().sum()) if m["target"] and m["task"] != "clustering" else len(df)
    tr, va, te = _shares(m, n)
    split = SplitConfig(m["strategy"] or "random", round(tr, 4), round(va, 4), round(te, 4), m["seed"], m["group_col"], m["time_col"])
    return ExperimentConfig(task=m["task"], target=m["target"] if m["task"] != "clustering" else None, name=f"{m['task']} on {h.name}", dataset_name=h.name,
                            dataset_version=st.session_state.get("version_id") or "raw", features_exclude=list(m["exclude"]), split=asdict(split),
                            imbalance=m["imbalance"] or "none", models=list(m["models"]), primary_metric=m["metric"] or "", compute=m["compute"], seed=int(m["seed"]),
                            preprocess=dict(m["pre"]), cv_folds=int(m["folds"]), positive_label=m["positive"], drop_missing_target=bool(m["drop_missing"]),
                            cluster_k_min=int(m["kmin"]), cluster_k_max=int(m["kmax"]), model_params=dict(m.get("model_params") or {}),
                            cv=dict(m["cv"]) if m.get("cv") else None, tuning=dict(m["tuning"]) if m.get("tuning") else {"method": "none"},
                            shap=bool(m.get("shap")), extra_metrics=list(m.get("extra_metrics") or []))


def _review_rows(cfg: ExperimentConfig, m: dict, df: pd.DataFrame, S) -> list[tuple[str, str]]:
    """Everything that decides the result, in plain words, so the run can be checked before it starts."""
    clu = cfg.task == "clustering"
    opts = PreprocessOptions(**cfg.preprocess)
    fs = select_features(df, cfg.target, cfg.features_exclude, None, opts)
    mnames = E.selection_metrics(cfg.task, 2)
    work = df if clu or not cfg.target else (df[df[cfg.target].notna()] if cfg.drop_missing_target else df)
    n = len(work)
    rows = [("Goal", TASKS[cfg.task]),
            ("Dataset", f"{cfg.dataset_name} ({'prepared version' if S.get('version_id') else 'unprepared'}), {n:,} rows used")]
    if not clu:
        rows.append(("Target", cfg.target or "(none)"))
        if cfg.task == "classification" and cfg.positive_label:
            rows.append(("Class you care most about", str(cfg.positive_label)))
    kinds = ", ".join(f"{len(v)} {k}" for k, v in (("numeric", fs.numeric), ("categorical", fs.categorical), ("date", fs.datetime), ("text", fs.text)) if v)
    rows.append(("Input columns", f"{len(fs.all)} ({kinds or 'none'})"))
    excl = list(cfg.features_exclude) + [c for c in fs.excluded if c not in cfg.features_exclude and c != cfg.target]
    rows.append(("Left out", ", ".join(excl[:8]) + (f" and {len(excl) - 8} more" if len(excl) > 8 else "") if excl else "nothing"))
    rows.append(("Grouping methods" if clu else "Models", ", ".join(REGISTRY[k].name for k in cfg.models)))
    rows.append(("Primary metric", mnames.get(cfg.primary_metric, cfg.primary_metric) + (" (decides the choice)" if len(cfg.models) > 1 else "")))
    if cfg.extra_metrics:
        rows.append(("Also shown", ", ".join(mnames.get(k, k) for k in cfg.extra_metrics)))
    if clu:
        rows.append(("Number of groups tried", f"{cfg.cluster_k_min} to {cfg.cluster_k_max}"))
        rows.append(("Held-out data", "none (clustering has no answer key to test against)"))
    else:
        tr, va, te = cfg.split["train"], cfg.split["val"], cfg.split["test"]
        rows.append(("Split", f"{cfg.split['strategy']}: {tr:.0%} train, {va:.0%} validation, {te:.0%} test "
                              f"(about {round(n * tr):,} / {round(n * va):,} / {round(n * te):,} rows), seed {cfg.seed}"))
        cv = cfg.cv or {}
        rows.append(("How models are compared", "on the validation rows" if va > 0 else f"{cv.get('folds', cfg.cv_folds)}-fold cross-validation on the training rows"))
        if cfg.task == "classification":
            rows.append(("Balancing", IMB.STRATEGIES.get(cfg.imbalance, cfg.imbalance)))
    prep = [f"missing numbers: {opts.numeric_impute}", "missing categories: own '(missing)' category", f"scaling: {opts.scaling}",
            f"categories: {'one-hot' if opts.encoding == 'onehot' else 'ordinal'} (up to {opts.max_categories})"]
    if opts.skew_transform:
        prep.append("skew correction")
    rows.append(("Preparation inside the model", "; ".join(prep) + ". Learned from training rows only."))
    rows.append(("Extra steps", ", ".join(k for k in ("outliers", "features", "reduction", "selection") if cfg.preprocess.get(k)) or "none"))
    rows.append(("Tuning", cfg.tuning.get("method", "none")))
    rows.append(("Compute", cfg.compute))
    if not clu:
        rows.append(("Explanations (SHAP)", "yes" if cfg.shap else "no"))
    return rows


def _train(df, note, m) -> None:
    S = st.session_state
    job = S.get("job")
    if job is not None and job.state == "running":
        st.subheader("Training")
        _poll()
        return
    if job is not None and job.state == "done":
        exp = job.result
        S["exp"], S["job"] = exp, None
        S["exp_id"] = state.ws().save_experiment(exp, S.get("version_id"))
        m["step"] = "Results"
        st.rerun()
    if job is not None and job.state in ("failed", "cancelled"):
        (st.error if job.state == "failed" else st.warning)(job.error if job.state == "failed" else job.message)
        if job.state == "failed":
            st.caption("Nothing was saved. Adjust the settings and try again.")
        if st.button("Dismiss"):
            S["job"] = None
            st.rerun()
        return
    st.subheader("Review and train")
    cfg = _config(m, df)
    summary = _review_rows(cfg, m, df, S)
    st.dataframe(pd.DataFrame(summary, columns=["Setting", "Value"]), hide_index=True, width="stretch")
    st.caption("Training runs in the background. You can cancel between models; a single model's fit cannot be interrupted once it has started.")
    c1, c2, _ = st.columns([1.3, 1, 4])
    c2.button("Back", on_click=_set_step, args=("Models",), key="bk_train")
    if c1.button("Start training", type="primary"):
        S["job"] = state.ws().run_experiment_job(df, cfg, S.get("version_id"))
        st.rerun()


@st.fragment(run_every=1.0)
def _poll() -> None:
    job = st.session_state.get("job")
    if job is None:
        return
    if job.state != "running":
        st.rerun()
        return
    st.progress(min(max(job.fraction, 0.0), 1.0), text=f"{job.message}  ({job.elapsed:.0f}s elapsed)")
    if st.button("Cancel training", key="cancel_job"):
        job.cancel()


# ---------------------------------------------------------------------------------------------
def _results(df, note, m) -> None:
    S = st.session_state
    exp = S.get("exp")
    if exp is None:
        theme.empty_state("No results yet", "Train a model to see results here.")
        _nav(m, "Train", None)
        return
    task = exp.config.task
    clu = task == "clustering"
    st.subheader("Results")
    status = "Done" if clu else exp.status.replace('_', ' ').title()
    st.markdown(f"{theme.chip(status, 'ok' if exp.status == 'finalized' else 'info')} &nbsp; {'Grouping methods' if clu else 'Models'} were compared on **{theme.esc(exp.selection_basis)}**. "
                f"Seed {exp.config.seed}. Took {exp.seconds}s.", unsafe_allow_html=True)
    for w_ in exp.warnings:
        st.warning(w_)
    for lk in exp.leakage:
        (st.error if lk["level"] == "warning" else st.info)(lk["message"])
    metric = exp.config.primary_metric
    mlabel = E.selection_metrics(task, 2).get(metric, metric)
    st.markdown("#### " + ("Grouping methods compared" if clu else "Model comparison"))
    rows = [{"Method" if clu else "Model": c.name, "Status": ("Selected" if c.key == exp.selected_key else c.status.title()), mlabel: c.score, "Spread (folds)": c.score_std, "Groups": c.n_clusters,
             "Noise %": c.noise_pct, **{E.selection_metrics(task, 2).get(k, k): (c.metrics or {}).get(k) for k in (exp.config.extra_metrics or []) if k != metric},
             "Seconds": c.seconds, "Note": c.error or c.note} for c in exp.candidates]
    st.dataframe(pd.DataFrame(rows).dropna(axis=1, how="all"), hide_index=True, width="stretch")
    okc = [c for c in exp.candidates if c.status == "ok"]
    if len(okc) > 1:
        fig = go.Figure(go.Bar(x=[c.name for c in okc], y=[c.score for c in okc], marker_color=[C.TEAL if c.key == exp.selected_key else C.MUTED for c in okc]))
        fig.update_yaxes(title=f"{mlabel} ({E.metric_direction(metric)} is better)")
        st.plotly_chart(C._layout(fig, f"{mlabel} by {'method' if clu else 'model'}", 300), width="stretch", config=C.PLOTLY_CONFIG)
    sel = exp.selected
    if sel:
        spec = REGISTRY.get(sel.key.split("|")[0])
        if clu:
            st.markdown(f"**Chosen grouping:** {sel.name} with {sel.n_clusters} group(s), which had the best {mlabel} of the methods tried. "
                        "Groups are patterns the method found in these columns, not correct answers; a different method or number of groups can be just as valid.")
        else:
            st.markdown(f"**Selected:** {sel.name}: the best {mlabel} under this protocol. This describes this data and split; it is not a general ranking of the algorithms."
                        + (f" {spec.tradeoff}" if spec else ""))
    if exp.baseline_selection:
        bm = exp.baseline_selection["metrics"].get(metric)
        if bm is not None:
            st.caption(f"Baseline on the same validation data (always predicts the majority class / the average): {mlabel} = {bm:.3f}.")

    if task != "clustering" and not exp.test_evaluated:
        st.markdown("#### Final check on the test set")
        st.markdown("The selected model has **not** seen the test rows. Evaluating now gives an honest estimate of performance on new data. "
                    "It can be done **once** per experiment; changing your mind afterwards and re-testing would make the score optimistic.")
        if exp.prior_test_uses:
            st.warning(f"{exp.prior_test_uses} earlier experiment(s) on this data version already looked at this same test split. Scores get more optimistic with each repeat.")
        if not exp._state:
            st.info("This experiment was reloaded without its fitted model, so it cannot be tested. Train it again.")
        elif st.button("Evaluate selected model on the test set (one time)", type="primary"):
            try:
                finalize_experiment(exp)
                state.ws().save_experiment(exp, S.get("version_id"), S.get("exp_id"))
            except (TestSetAlreadyUsed, ExperimentError) as e:
                st.error(str(e))
            st.rerun()
    elif exp.test_result:
        _final(exp, task, m)
    MO.results_extras(exp)
    if exp.importance:
        st.markdown("#### What the model relies on")
        imp = pd.DataFrame(exp.importance).head(12)
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["feature"], orientation="h", marker_color=C.TEAL))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Drop in score when the column is shuffled")
        st.plotly_chart(C._layout(fig, "Permutation importance (computed without the test set)", max(260, 28 * len(imp) + 110)), width="stretch", config=C.PLOTLY_CONFIG)
        st.caption("Importance describes what this model used, not what causes the outcome.")
    _exports(exp)
    c1, c2, _ = st.columns([1.4, 1.4, 4])
    if c1.button("Start a new experiment"):
        S.update({"exp": None, "exp_id": None})
        m["step"] = "Goal"
        st.rerun()
    if c2.button("Change settings"):
        S.update({"exp": None, "exp_id": None})
        m["step"] = "Models"
        st.rerun()


def _final(exp, task, m) -> None:
    res, base = exp.test_result, exp.test_baseline
    st.markdown("#### " + ("Final results on the held-out test set" if task != "clustering" else "How clearly separated are the groups?"))
    unit = f" ({m['unit']})" if m.get("unit") and task == "regression" else ""
    rows = []
    for k, v in res["metrics"].items():
        b = (base or {}).get("metrics", {}).get(k) if base else None
        label = {"mae": "MAE", "rmse": "RMSE", "r2": "R²", "mape": "MAPE (%)", "median_ae": "Median abs. error"}.get(k, k.replace("_", " ").title())
        rows.append({"Metric": label + (unit if k in ("mae", "rmse", "median_ae") else ""), "Value": v, "Baseline": b, "What it means": E.METRIC_HELP.get(k, "")})
    table = pd.DataFrame(rows)
    if "Baseline" in table and table["Baseline"].isna().all():
        table = table.drop(columns="Baseline")                     # clustering has no baseline to compare with
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Value": st.column_config.NumberColumn(format="%.3f"), "Baseline": st.column_config.NumberColumn(format="%.3f")})
    caveat = score_caveat(task, res["metrics"], exp.leakage, (exp.config.features_exclude or []))
    if caveat:
        st.warning(caveat)
    if res.get("unavailable"):
        with st.expander("Metrics that could not be computed"):
            for k, v in res["unavailable"].items():
                st.markdown(f"- **{k}**: {v}")
    if task == "classification":
        lab = res["labels"]
        fig = go.Figure(go.Heatmap(z=res["confusion_matrix"], x=lab, y=lab, text=res["confusion_matrix"], texttemplate="%{text}", colorscale=[[0, "#f8fafc"], [1, C.TEAL]], showscale=False))
        fig.update_xaxes(title="Predicted", type="category")
        fig.update_yaxes(title="Actual", autorange="reversed", type="category")
        c1, c2 = st.columns([1, 1.2])
        c1.plotly_chart(C._layout(fig, "Confusion matrix", 340), width="stretch", config=C.PLOTLY_CONFIG)
        c2.markdown("**By class**")
        c2.dataframe(res["per_class"], hide_index=True, width="stretch", column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ("precision", "recall", "f1")})
        c2.caption(f"Precision, recall and F1 use the {res['averaging']}.")
    elif task == "regression" and exp._state.get("test_predictions") is not None:
        tp = exp._state["test_predictions"]
        fig = go.Figure(go.Scattergl(x=tp["actual"], y=tp["predicted"], mode="markers", marker={"color": C.TEAL, "opacity": 0.6, "size": 6}, name="test rows"))
        lo, hi = float(min(tp["actual"].min(), tp["predicted"].min())), float(max(tp["actual"].max(), tp["predicted"].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line={"color": C.NAVY, "dash": "dash"}, name="perfect"))
        fig.update_xaxes(title="Actual")
        fig.update_yaxes(title="Predicted")
        c1, c2 = st.columns(2)
        c1.plotly_chart(C._layout(fig, "Predicted vs actual (test rows)", 340), width="stretch", config=C.PLOTLY_CONFIG)
        h = go.Figure(go.Histogram(x=tp["error"], marker_color=C.TEAL))
        h.update_xaxes(title="Prediction error (predicted - actual)")
        c2.plotly_chart(C._layout(h, "Errors", 340), width="stretch", config=C.PLOTLY_CONFIG)
    elif task == "clustering":
        proj = pd.DataFrame(exp.projection)
        gname = {str(r["cluster"]): r.get("name") for r in exp.cluster_profile}
        if not proj.empty:
            fig = go.Figure()
            for i, (g, part) in enumerate(proj.groupby("cluster")):
                fig.add_trace(go.Scattergl(x=part["x"], y=part["y"], mode="markers", name=(f"Group {g}: {gname[g]}" if gname.get(g) else f"Group {g}") if g != "noise" else "Noise",
                                           marker={"size": 6, "opacity": 0.7, "color": C.COLORWAY[i % len(C.COLORWAY)] if g != "noise" else "#94a3b8"}))
            fig.update_xaxes(title="Component 1")
            fig.update_yaxes(title="Component 2")
            st.plotly_chart(C._layout(fig, "Groups in two dimensions (PCA view of the prepared features)", 380), width="stretch", config=C.PLOTLY_CONFIG)
            st.caption("A flat picture of many columns: groups that overlap here may still be separate in the full data.")
        st.markdown("**Group profiles** (averages in original units). Names describe what sets each group apart from all rows; they are a summary, not a label from the data.")
        st.dataframe(pd.DataFrame(exp.cluster_profile).astype({"cluster": str}), hide_index=True, width="stretch")
    st.markdown("**Interpretation and caveats**")
    for t in exp.interpretation:
        st.markdown(f"- {t}")


def score_caveat(task: str, metrics: dict, leakage: list[dict], excluded: list[str]) -> str | None:
    """A warning to show beside the test score when it may be too good to be true, or None."""
    if task == "clustering":
        return None
    high = {"accuracy": 0.98, "balanced_accuracy": 0.98, "f1": 0.98, "f1_macro": 0.98, "f1_weighted": 0.98, "roc_auc": 0.99, "r2": 0.98}
    near_perfect = [k for k, lim in high.items() if isinstance(metrics.get(k), (int, float)) and metrics[k] >= lim]
    warned = [lk for lk in leakage or [] if lk.get("level") == "warning"]
    if not near_perfect and not warned:
        return None
    parts = []
    if near_perfect:
        parts.append("This score is close to perfect, which is rare on real data.")
    if warned:
        cols = sorted({c for lk in warned for c in lk.get("columns", [])})
        parts.append("Some input columns looked like they could reveal the answer" + (f" ({', '.join(cols[:5])})" if cols else "") + ".")
    elif excluded:
        parts.append(f"Columns left out: {', '.join(excluded[:5])}.")
    parts.append("Check that no input column is only known after the outcome (for example a cancellation date when predicting churn); "
                 "if one is, the score will not hold on new data.")
    return " ".join(parts)


def _exports(exp) -> None:
    st.markdown("#### Downloads")
    name = state.handle().name if state.handle() else exp.config.dataset_name
    try:
        files = state.ws().experiment_exports(exp, name)
    except Exception as e:
        st.error(f"Could not build the downloads: {e}")
        return
    mime = {"html": "text/html", "json": "application/json", "csv": "text/csv", "joblib": "application/octet-stream"}
    cols = st.columns(min(len(files) + 1, 6))
    for col, (fn, data) in zip(cols, files.items()):
        col.download_button(fn, data, file_name=fn, mime=mime.get(fn.rsplit(".", 1)[-1], "application/octet-stream"), key=f"dlx_{fn}")
    cols[min(len(files), len(cols) - 1)].download_button("Everything (ZIP)", zip_bytes(files), file_name="nexora_experiment.zip", mime="application/zip", key="dlx_zip")
    if "model_pipeline.joblib" in files:
        st.caption("The .joblib file is a pickle: only load model files you created yourself. It contains the full fitted pipeline (preparation and model).")
