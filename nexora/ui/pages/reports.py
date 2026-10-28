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


def render() -> None:
    st.title("Reports and experiments")
    w = state.ws()
    tab_e, tab_p, tab_u = st.tabs(["Experiments", "Preparation reports", "Use a saved model"])
    with tab_u:
        _use_model(w)
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
                             "Created": e["created_at"][:16].replace("T", " ")})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            names = {f"{e['name']} ({e['created_at'][:16].replace('T', ' ')}, {e['id'][:4]})": e["id"] for e in exps}
            pick = st.multiselect("Compare experiments", list(names), max_selections=4, help="Select two or more to compare their candidate models side by side.")
            if len(pick) >= 2:
                comp = []
                for p in pick:
                    r = full[names[p]]["result"]
                    for c in r.get("candidates", []):
                        comp.append({"Experiment": p, "Model": c["name"], "Primary metric": r["config"].get("primary_metric"), "Selection score": c.get("score"), "Status": c["status"]})
                st.dataframe(pd.DataFrame(comp), hide_index=True, width="stretch")
                st.caption("Scores from different experiments are comparable only if they used the same data, split and metric.")
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
                st.dataframe(cand[[c for c in ("name", "status", "score", "score_std", "seconds", "error") if c in cand.columns]], hide_index=True, width="stretch")
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
                                        "Created": v["created_at"][:16].replace("T", " ")} for v in vers]), hide_index=True, width="stretch")
            lab = {f"{dsn.get(v['dataset_id'], '?')} {v['created_at'][:16].replace('T', ' ')} ({v['id'][:4]})": v for v in vers}
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
    names = {f"{e['name']} ({e['task']}, {e['created_at'][:16].replace('T', ' ')}, {e['id'][:4]})": e["id"] for e in exps}
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
        st.dataframe(out.head(200), hide_index=True, width="stretch")
        st.download_button("Predictions (CSV)", out.to_csv(index=False).encode(), file_name="predictions.csv", mime="text/csv", key="pr_dl")
        st.caption("Predictions are only as good as the test results of this model, and only for data like the data it was trained on.")
