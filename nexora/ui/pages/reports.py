from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from ...reporting import experiment_report_html
from .. import state, theme


def _headline(res: dict) -> str:
    cfg = res.get("config", {})
    m = (res.get("test_result") or {}).get("metrics") or {}
    k = cfg.get("primary_metric")
    if k in m and m[k] is not None:
        return f"{k}: {m[k]:.4g} ({'test set' if cfg.get('task') != 'clustering' else 'all rows'})"
    sel = next((c for c in res.get("candidates", []) if c["key"] == res.get("selected_key")), None)
    return f"{k}: {sel['score']:.4g} (selection only; test set not used)" if sel and sel.get("score") is not None else ""


def compare_table(results: list[tuple[str, dict]]) -> tuple[pd.DataFrame, list[str]]:
    """One row per experiment (same dataset and task), with the settings that decide whether scores are comparable.

    results: [(label, result_dict)]. Returns (table, notes) where notes say which settings differ between the rows.
    """
    rows, test_keys = [], []
    for _, r in results:
        for k in ((r.get("test_result") or {}).get("metrics") or {}):
            if k not in test_keys:
                test_keys.append(k)
    for label, r in results:
        cfg = r.get("config", {})
        sel = next((c for c in r.get("candidates", []) if c["key"] == r.get("selected_key")), {}) or {}
        sp = cfg.get("split", {})
        row = {"Experiment": label, "Selected model": sel.get("name", ""), "Primary metric": cfg.get("primary_metric"),
               "Selection score": sel.get("score"), "Models tried": len(r.get("candidates", [])),
               "Data version": "prepared" if cfg.get("dataset_version") not in (None, "raw") else "raw",
               "Split": f"{sp.get('strategy', '')} {sp.get('train', 0):.0%}/{sp.get('val', 0):.0%}/{sp.get('test', 0):.0%}" if sp else "none",
               "Seed": cfg.get("seed"), "Left out": ", ".join(cfg.get("features_exclude") or []) or "nothing"}
        tm = (r.get("test_result") or {}).get("metrics") or {}
        for k in test_keys:
            row[f"Test {k}"] = tm.get(k) if r.get("test_evaluated") or cfg.get("task") == "clustering" else None
        rows.append(row)
    tab = pd.DataFrame(rows)
    notes = []
    for col, what in (("Primary metric", "primary metric"), ("Data version", "data version (raw or prepared)"), ("Split", "split"),
                      ("Seed", "seed"), ("Left out", "set of left-out columns")):
        if col in tab and tab[col].astype(str).nunique() > 1:
            notes.append(f"The {what} differs between these experiments, so their scores are not directly comparable.")
    return tab, notes


def _compare(full: dict, exps: list[dict]) -> None:
    st.markdown("#### Compare experiments")
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in exps:
        cfg = full[e["id"]]["result"].get("config", {})
        groups.setdefault((cfg.get("dataset_name", "?"), cfg.get("task", "?")), []).append(e)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    if not multi:
        st.caption("Train two or more experiments on the same dataset and goal to compare them here.")
        return
    glabels = {f"{ds}: {task} ({len(v)} experiments)": k for k, v in multi.items() for ds, task in [k]}
    g = multi[glabels[st.selectbox("Dataset and goal", list(glabels), key="rep_cmp_g",
                                   help="Only experiments on the same dataset with the same goal are compared.")]]
    labels = {f"{e['name']} ({theme.local_time(e['created_at'])}, {e['id'][:4]})": e["id"] for e in g}
    pick = st.multiselect("Experiments", list(labels), default=list(labels)[:4], max_selections=6, key="rep_cmp_pick")
    if len(pick) < 2:
        st.caption("Pick at least two.")
        return
    tab, notes = compare_table([(p, full[labels[p]]["result"]) for p in pick])
    nums = [c for c in tab.columns if c == "Selection score" or c.startswith("Test ")]
    st.dataframe(tab.dropna(axis=1, how="all"), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.4f") for c in nums})
    for n in notes:
        st.warning(n)
    if not notes:
        st.caption("Same metric, data version, split, seed and inputs: the scores are directly comparable.")
    st.caption("Test columns are empty for experiments whose test set has not been used yet.")


def _runs(w) -> None:
    runs = w.repo.list_runs()
    if not runs:
        theme.empty_state("No saved runs yet", "Use 'Save to Reports' on the Anomalies or Forecasting page to keep a run here.")
        return
    kind = {"anomaly": "Anomalies", "forecast": "Forecast"}
    st.dataframe(pd.DataFrame([{"Kind": kind.get(r["kind"], r["kind"]), "Dataset": r["dataset_name"], "Result": r["summary"].get("headline", ""),
                                "Data version": "prepared" if r.get("version_id") else "raw", "Saved": theme.local_time(r["created_at"])} for r in runs]),
                 hide_index=True, width="stretch")
    labels = {f"{kind.get(r['kind'], r['kind'])}: {r['dataset_name']} ({theme.local_time(r['created_at'])}, {r['id'][:4]})": r["id"] for r in runs}
    rid = labels[st.selectbox("Open a run", list(labels), key="rep_run")]
    run = w.repo.get_run(rid)
    if not run:
        return
    sm = run["summary"]
    st.markdown(f"**{theme.esc(sm.get('headline', ''))}**", unsafe_allow_html=True)
    if sm.get("table"):
        st.dataframe(pd.DataFrame(sm["table"]), hide_index=True, width="stretch")
    for t in sm.get("notes", []):
        st.caption(t)
    cols = st.columns(min(len(run["files"]) + 1, 5))
    mime = {"csv": "text/csv", "json": "application/json"}
    for col, (fn, data) in zip(cols, run["files"].items()):
        col.download_button(fn, data, file_name=fn, mime=mime.get(fn.rsplit(".", 1)[-1], "application/octet-stream"), key=f"run_dl_{rid}_{fn}")
    if st.button("Delete this run", key=f"run_del_{rid}"):
        w.repo.delete_run(rid)
        st.rerun()


def render() -> None:
    st.title("Reports and experiments")
    w = state.ws()
    tab_e, tab_r, tab_p, tab_u = st.tabs(["Experiments", "Anomaly and forecast runs", "Preparation reports", "Use a saved model"])
    with tab_u:
        _use_model(w)
    with tab_r:
        _runs(w)
    with tab_e:
        exps = w.repo.list_experiments()
        if not exps:
            theme.empty_state("No experiments yet", "Experiments you train in ML Studio are saved here with their settings, seed and results.")
        else:
            full = {e["id"]: w.repo.get_experiment(e["id"]) for e in exps}
            rows = []
            for e in exps:
                r = full[e["id"]]["result"]
                sel = next((c["name"] for c in r.get("candidates", []) if c["key"] == r.get("selected_key")), "")
                rows.append({"Name": e["name"], "Task": e["task"], "Status": e["status"].replace("_", " "), "Selected model": sel, "Headline": _headline(r),
                             "Test set used": "yes" if e["test_evaluated"] else ("n/a" if e["task"] == "clustering" else "no"), "Seed": e["seed"],
                             "Created": theme.local_time(e["created_at"])})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            names = {f"{e['name']} ({theme.local_time(e['created_at'])}, {e['id'][:4]})": e["id"] for e in exps}
            _compare(full, exps)
            one = st.selectbox("Open an experiment", list(names), key="rep_one")
            rec = full[names[one]]
            r = rec["result"]
            ds = r.get("config", {}).get("dataset_name", "")
            c1, c2, c3 = st.columns(3)
            c1.download_button("Report (HTML)", experiment_report_html(r, ds), file_name="experiment_report.html", mime="text/html", key="rep_html")
            c2.download_button("Settings (JSON)", json.dumps({"config": r["config"], "split": r.get("split_info"), "environment": r.get("environment")}, indent=2),
                               file_name="experiment_config.json", mime="application/json", key="rep_cfg")
            if rec.get("artifact_path") and Path(rec["artifact_path"]).exists():
                c3.download_button("Model (joblib)", Path(rec["artifact_path"]).read_bytes(), file_name="model_pipeline.joblib", key="rep_art")
            with st.expander("Reproducibility details"):
                st.json({"seed": r["config"]["seed"], "split": r.get("split_info"), "dataset_version": r["config"].get("dataset_version"),
                         "dataset_fingerprint": r.get("dataset_fingerprint"), "environment": r.get("environment"), "preprocessing": r["config"].get("preprocess"),
                         "balancing": r["config"].get("imbalance")})
            cand = pd.DataFrame(r.get("candidates", []))
            if not cand.empty:
                view = cand[[c for c in ("name", "status", "score", "score_std", "seconds", "error") if c in cand.columns]].copy()
                for c in ("score", "score_std", "seconds"):
                    if c in view:
                        view[c] = pd.to_numeric(view[c], errors="coerce")      # None becomes an empty cell, not the word "None"
                view = view.rename(columns={"name": "Model", "status": "Status", "score": "Score", "score_std": "Spread (folds)", "seconds": "Seconds",
                                            "error": "Note"})
                if "Note" in view:
                    view["Note"] = view["Note"].fillna("")
                st.dataframe(view, hide_index=True, width="stretch", column_config={"Score": st.column_config.NumberColumn(format="%.4f"),
                                                                                   "Spread (folds)": st.column_config.NumberColumn(format="%.4f"),
                                                                                   "Seconds": st.column_config.NumberColumn(format="%.2f")})
            for t in r.get("interpretation", []):
                st.markdown(f"- {t}")
    with tab_p:
        vers = w.repo.list_versions()
        if not vers:
            theme.empty_state("No preparation runs yet", "Prepared datasets and their reports are listed here after you run Prepare Data.")
        else:
            dsn = {d["id"]: d["name"] for d in w.repo.list_datasets()}
            st.dataframe(pd.DataFrame([{"Dataset": dsn.get(v["dataset_id"], v["dataset_id"]), "Rows": v["rows"], "Columns": v["cols"], "Profile": v["meta"].get("profile_id"),
                                        "Goal": v["meta"].get("goal"), "Checks passed": "yes" if v["meta"].get("reconciliation_ok") else "NO",
                                        "Created": theme.local_time(v["created_at"])} for v in vers]), hide_index=True, width="stretch")
            lab = {f"{dsn.get(v['dataset_id'], '?')} {theme.local_time(v['created_at'])} ({v['id'][:4]})": v for v in vers}
            v = lab[st.selectbox("Open a preparation report", list(lab))]
            rep = v["meta"].get("report", {})
            if rep:
                b, a = rep.get("before", {}), rep.get("after", {})
                st.dataframe(pd.DataFrame([{"Measure": k.replace("_", " "), "Before": b.get(k), "After": a.get(k)}
                                           for k in ("rows", "columns", "missing_cells", "exact_duplicate_rows", "columns_with_format_issues")]), hide_index=True, width="stretch")
                st.dataframe(pd.DataFrame([{"Step": e["step"], "Action": e["title"], "Status": e["status"], "Rows in": e["rows_in"], "Rows out": e["rows_out"],
                                            "Values changed": e["values_changed"]} for e in rep.get("log", [])]), hide_index=True, width="stretch")
                st.download_button("Report data (JSON)", json.dumps(rep, indent=2, default=str), file_name="preparation_report.json", mime="application/json")
            csv = w.repo.version_csv_path(v["id"])
            if csv and Path(csv).exists():
                st.download_button("Prepared data (CSV)", Path(csv).read_bytes(), file_name="prepared_data.csv", mime="text/csv")


def _use_model(w) -> None:
    from ...ml.predict import PredictionError
    from ...validation.ingestion import IngestError
    exps = [e for e in w.repo.list_experiments() if (w.repo.get_experiment(e["id"]) or {}).get("artifact_path")]
    if not exps:
        theme.empty_state("No saved models", "Models are saved automatically when you train in ML Studio.")
        return
    st.markdown("Score new rows with a model you trained here. The saved pipeline repeats every preparation step exactly as it was learned "
                "from the training rows. The new data needs the same input columns.")
    names = {f"{e['name']} ({e['task']}, {theme.local_time(e['created_at'])}, {e['id'][:4]})": e["id"] for e in exps}
    pick = st.selectbox("Model", list(names), key="pr_model")
    src = st.radio("Rows to score", ["Upload a file", "The dataset that is open now"], horizontal=True, key="pr_src")
    df = None
    if src == "Upload a file":
        f = st.file_uploader("CSV, TSV or Excel file", type=["csv", "tsv", "xlsx"], key="pr_file")
        if f is not None:
            try:
                df = w.read_upload(f.name, f.getvalue()).dataframe
                df = w.autotype(df)
            except IngestError as e:
                st.error(str(e))
    else:
        wf = state.working_frame()
        if wf is None:
            st.info("No dataset is open.")
        else:
            df = wf[0]
    if df is not None and st.button("Score rows", type="primary", key="pr_go"):
        try:
            out, notes = w.predict_saved(names[pick], df)
            st.session_state["pr_out"] = (out, notes)
        except PredictionError as e:
            st.error(str(e))
    res = st.session_state.get("pr_out")
    if res:
        out, notes = res
        for n in notes:
            st.caption(n)
        st.dataframe(theme.display_frame(out.head(200)), hide_index=True, width="stretch")
        st.download_button("Predictions (CSV)", out.to_csv(index=False).encode(), file_name="predictions.csv", mime="text/csv", key="pr_dl")
        st.caption("Predictions are only as good as the test results of this model, and only for data like the data it was trained on.")
