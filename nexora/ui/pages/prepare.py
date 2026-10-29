from __future__ import annotations

import pandas as pd
import streamlit as st

from ...cleaning import DECISION, GOALS, OPTIONAL, REVIEW, SAFE, Action, default_mapping
from ...preprocessing.outliers import METHOD_NOTES, OutlierConfig, OutlierConfigError, validate_config
from ...preprocessing.outliers import METHODS as OUTLIER_METHODS
from ...preprocessing.outliers import apply_treatment as apply_outlier_treatment
from ...preprocessing.outliers import detect as detect_outliers
from ...reporting import zip_bytes
from ...validation.profiler import profile_dataframe
from .. import components, state, theme

STEPS = ["Profile", "Goal and domain", "Review plan", "Result"]


def _extra_copies(raw: pd.DataFrame) -> int:
    """Rows identical to an earlier row (the first occurrence is not counted), the same rule the plan uses."""
    return int(raw.astype(str).apply(lambda s: s.str.strip()).duplicated(keep="first").sum()) if len(raw) else 0


def _go(step: int) -> None:
    st.session_state["prep_step"] = step


def render() -> None:
    st.title("Prepare Data")
    if components.dataset_bar("prepare", raw=True, empty_body="Preparation works on a dataset you have uploaded or opened from the samples.") is None:
        return
    h = state.handle()
    step = st.session_state["prep_step"]
    if step == 3 and st.session_state.get("prep_result") is None:
        step = st.session_state["prep_step"] = 0
    if step == 2 and st.session_state.get("plan") is None:
        step = st.session_state["prep_step"] = 1
    theme.stepper(STEPS, step, key="prep")
    {0: _profile, 1: _goal, 2: _plan, 3: _result}[step](h)


# ----------------------------------------------------------------------------------------------
def _profile(h) -> None:
    st.subheader("What is in this dataset")
    w = state.ws()
    prof = w.profiles["generic"]
    sp = profile_dataframe(h.raw, list(prof.null_tokens))
    miss = sum(c.missing_count for c in sp.columns)
    cells = max(h.raw.shape[0] * h.raw.shape[1], 1)
    st.markdown(f"**{len(h.raw):,} rows** and **{h.raw.shape[1]} columns**. About **{100 * miss / cells:.1f}%** of cells are empty or placeholders, "
                f"and **{_extra_copies(h.raw):,}** row(s) repeat an earlier row exactly.")
    rows = [{"Column": c.name, "Looks like": c.inferred_type, "Missing %": round(100 * c.missing_pct, 1), "Distinct values": c.unique_count,
             "Example": (str(c.samples[0]) if c.samples else ""), "Note": "; ".join(c.issues[:2])} for c in sp.columns]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if sp.candidate_ids:
        st.caption("Likely identifiers (never filled in or invented): " + ", ".join(sp.candidate_ids))
    st.button("Next: choose goal and domain", type="primary", on_click=_go, args=(1,))


def _goal(h) -> None:
    w = state.ws()
    S = st.session_state
    st.subheader("What do you want to do with this data?")
    goal_keys = list(GOALS)
    S["goal"] = st.radio("Intended use", goal_keys, index=goal_keys.index(S["goal"]) if S["goal"] in goal_keys else 0, format_func=lambda k: GOALS[k],
                         help="This changes what Nexora offers. For machine learning, missing values are handled later inside the model pipeline, using training data only.")
    profs = w.profiles
    first = ["generic", "retail", "finance", "manufacturing"]
    order = first + sorted(k for k in profs if k not in first)
    S["profile_id"] = st.selectbox("Domain profile", order, index=order.index(S["profile_id"]) if S["profile_id"] in order else 0,
                                   format_func=lambda k: profs[k].name)
    p = profs[S["profile_id"]]
    st.caption(w.profile_depth(p) + ". " + (p.description or "").strip())
    with st.expander("What this profile checks"):
        st.markdown(f"Version {p.version}. {len(p.rules)} rule(s) and {len(p.fields)} expected field(s).")
        if p.guardrail:
            st.info(p.guardrail.strip())
        if p.rules:
            st.dataframe([{"Check": r.name, "Severity": r.severity, "Why": r.rationale} for r in p.rules[:60]], hide_index=True, width="stretch")
    cols = list(h.raw.columns)
    suggested = default_mapping(cols, p)
    mapping: dict[str, str | None] = {}
    with st.expander(f"Review column mapping ({sum(1 for v in suggested.values() if v)} of {len(p.fields)} fields matched automatically)"):
        if not p.fields:
            st.caption("This profile has no expected fields, so no mapping is needed.")
        for f, spec in p.fields.items():
            opts = ["(not mapped)"] + cols
            cur = suggested.get(f)
            sel = st.selectbox(f"{f}" + (" (required)" if spec.get("required") else ""), opts, index=opts.index(cur) if cur in opts else 0,
                               key=f"mp_{p.profile_id}_{f}", help=spec.get("description", "") or None)
            mapping[f] = None if sel == "(not mapped)" else sel
    need = w.required_reference_tables(p)
    if need:
        with st.expander(f"Reference tables for cross-checks ({len(need)} needed by this profile)"):
            st.caption("Some checks confirm that values exist in another table (for example every department code is a real department). "
                       "Upload each table as CSV or Excel; without it those checks are skipped.")
            refs = S["reference_tables"] = S.get("reference_tables") or {}
            for tname, col in need.items():
                up_ref = st.file_uploader(f"Table '{tname}' (must contain column '{col}')", type=["csv", "tsv", "xlsx"], key=f"ref_{p.profile_id}_{tname}")
                if up_ref is not None:
                    try:
                        tbl = w.read_upload(up_ref.name, up_ref.getvalue()).dataframe
                        if col and col not in tbl.columns:
                            refs.pop(tname, None)
                            st.error(f"That file has no column '{col}'.")
                        else:
                            refs[tname] = tbl
                            st.success(f"'{tname}' loaded: {len(tbl):,} rows.")
                    except Exception as e:
                        st.error(str(e))
    with st.expander("Advanced: use your own domain profile (YAML)"):
        up = st.file_uploader("Profile file", type=["yaml", "yml"], key="profile_upload")
        if up is not None and st.button("Add profile", key="add_prof"):
            try:
                np_ = w.add_custom_profile(up.getvalue().decode("utf-8"))
                S["profile_id"] = np_.profile_id
                st.success(f"Added '{np_.name}'.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not use that profile: {e}")
    c1, c2, _ = st.columns([1.6, 1, 4])
    c2.button("Back", on_click=_go, args=(0,), key="b1")
    if c1.button("Build preparation plan", type="primary"):
        with st.spinner("Analysing your data and testing each step on a copy..."):
            plan = w.make_plan(h, S["profile_id"], S["goal"], mapping, reference_tables=dict(S.get("reference_tables") or {}))
        S["plan"], S["mapping"] = plan, mapping
        S["plan_no"] = S.get("plan_no", 0) + 1
        S["prep_step"] = 2
        st.rerun()


def _details(a: Action) -> None:
    with st.expander("View details"):
        if a.columns:
            st.markdown("**Affected columns:** " + ", ".join(f"`{c}`" for c in a.columns[:12]) + (" ..." if len(a.columns) > 12 else ""))
        st.markdown(f"**Expected impact:** {a.expected_impact or 'None'}")
        if a.rationale:
            st.markdown(f"**Why:** {a.rationale}")
        if a.alternatives:
            st.markdown("**Alternatives:**\n" + "\n".join(f"- {x}" for x in a.alternatives))
        if a.examples:
            st.markdown("**Examples (row numbers match your file, header = row 1):**")
            st.dataframe(pd.DataFrame(a.examples).rename(columns={"row": "Row", "before": "Before", "after": "After"}), hide_index=True, width="stretch")
        if a.rule_ref:
            st.caption(f"Technical reference: {a.rule_ref}")


def _outlier_cfg(a: Action, no: int) -> OutlierConfig:
    """The detection settings currently chosen in the widgets for this action (defaults from the plan)."""
    S = st.session_state
    base = OutlierConfig.from_dict(a.params.get("config") or {})
    k = f"ol_{no}_{a.action_id}"
    base.method = S.get(f"{k}_method", base.method)
    base.columns = list(S.get(f"{k}_cols", base.columns))
    base.iqr_k = float(S.get(f"{k}_iqr", base.iqr_k))
    base.z_threshold = float(S.get(f"{k}_z", base.z_threshold))
    base.mz_threshold = float(S.get(f"{k}_mz", base.mz_threshold))
    cont = S.get(f"{k}_cont", 0.0)
    base.contamination = "auto" if not cont else float(cont) / 100
    lo, hi = S.get(f"{k}_win", (1, 99))
    base.winsor_limits = (lo / 100, hi / 100)
    base.bounds = {c: tuple(v) for c, v in (a.params.get("domain_bounds") or {}).items()}
    return base


def _outlier_decision(plan, a: Action, no: int) -> None:
    """Outlier treatment: choose a treatment, adjust detection, preview the effect on the scratch copy before confirming."""
    S = st.session_state
    st.markdown(f"**{a.title}**")
    keys = [c.key for c in a.choices]
    labels = {c.key: c.label for c in a.choices}
    treat = st.radio("Outlier treatment", keys, index=keys.index(a.default_choice) if a.default_choice in keys else 0,
                     format_func=lambda x, lab=labels: lab[x], key=f"ch_{no}_{a.action_id}", horizontal=True)
    st.caption(a.summary)
    k = f"ol_{no}_{a.action_id}"
    cfg0 = OutlierConfig.from_dict(a.params.get("config") or {})
    with st.expander("Detection settings and preview"):
        methods = list(OUTLIER_METHODS)
        if not a.params.get("domain_bounds"):
            methods.remove("domain_bounds")
        m = st.selectbox("Detection method", methods, index=methods.index(cfg0.method) if cfg0.method in methods else 0,
                         format_func=lambda x: OUTLIER_METHODS[x], key=f"{k}_method")
        st.caption(METHOD_NOTES[m])
        st.multiselect("Columns to check", cfg0.columns, default=cfg0.columns, key=f"{k}_cols")
        if m == "iqr":
            st.slider("IQR multiplier (k)", 0.5, 6.0, float(cfg0.iqr_k), 0.25, key=f"{k}_iqr", help="1.5 is the classic rule; 3 flags only far-out values.")
        elif m == "zscore":
            st.slider("Z-score threshold", 1.5, 8.0, float(cfg0.z_threshold), 0.25, key=f"{k}_z")
        elif m == "modified_zscore":
            st.slider("Modified Z-score threshold", 2.0, 10.0, float(cfg0.mz_threshold), 0.25, key=f"{k}_mz")
        elif m == "isolation_forest":
            st.number_input("Expected share of unusual rows (%; 0 = automatic)", 0.0, 50.0, 0.0, 0.5, key=f"{k}_cont")
        elif m == "domain_bounds":
            st.caption("Limits: " + ", ".join(f"{c}: {lo} to {hi}" for c, (lo, hi) in a.params["domain_bounds"].items()))
        if treat == "winsorize":
            st.slider("Winsorize limits (percentiles)", 0, 100, (1, 99), key=f"{k}_win")
        if a.params.get("policy") == "report_only" and treat != "report":
            st.warning("This domain profile recommends reporting unusual values only; changing them can hide real events.")
        if st.button("Preview effect", key=f"{k}_preview"):
            cfg = _outlier_cfg(a, no)
            try:
                if plan.preview is None:
                    raise OutlierConfigError("The preview copy is not available; rebuild the plan.")
                rep = detect_outliers(plan.preview, cfg)
                _, summ = apply_outlier_treatment(plan.preview, rep, cfg, treat)
                S[f"{k}_result"] = {"table": rep.summary_table(), "flagged": rep.n_flagged, "pct": rep.pct_flagged, "summary": summ}
            except OutlierConfigError as e:
                S[f"{k}_result"] = {"error": str(e)}
        res = S.get(f"{k}_result")
        if res:
            if "error" in res:
                st.error(res["error"])
            else:
                st.markdown(f"**{res['flagged']:,} row(s) flagged ({res['pct']}%).**")
                st.dataframe(res["table"], hide_index=True, width="stretch")
                sm = res["summary"]
                eff = []
                if sm.get("rows_removed"):
                    eff.append(f"{len(sm['rows_removed']):,} row(s) would be removed")
                if sm.get("values_modified"):
                    eff.append("values changed: " + ", ".join(f"{c} ({n})" for c, n in sm["values_modified"].items()))
                if sm.get("columns_added"):
                    eff.append("new column(s): " + ", ".join(sm["columns_added"]))
                st.caption("With the selected treatment: " + ("; ".join(eff) if eff else "no change to the data") + ".")
                for w in sm.get("warnings", []):
                    st.warning(w)
    _details(a)


def _plan(h) -> None:
    S = st.session_state
    plan = S["plan"]
    no = S.get("plan_no", 0)
    sm = plan.summary
    st.subheader("Review the plan")
    st.markdown(f"Nexora tested these steps on a **copy** of your data. **{sm['n_safe']}** safe step(s) will run once you confirm; "
                f"**{sm['n_decisions']}** need or allow your input; **{sm['n_review']}** item(s) are shown for review only. "
                f"Domain checks currently failing: **{sm['issues_found_before']}**; expected to remain after preparation: **{sm['issues_expected_after']}**.")
    for msg in plan.warnings:
        st.warning(msg)
    safe, dec, rev = plan.by_kind(SAFE), plan.by_kind(DECISION) + plan.by_kind(OPTIONAL), plan.by_kind(REVIEW)

    st.markdown("#### Runs automatically after you confirm")
    if not safe:
        st.caption("No automatic fixes are needed.")
    for a in safe:
        st.checkbox(f"**{a.title}**", value=True, key=f"ap_{no}_{a.action_id}", help="Untick to skip this step.")
        st.caption(f"{a.summary} {a.expected_impact}")
        _details(a)

    if dec:
        st.markdown("#### Needs your decision (nothing changes unless you choose)")
        for a in dec:
            if a.kind == OPTIONAL:
                st.checkbox(f"**{a.title}**", value=False, key=f"ap_{no}_{a.action_id}")
            elif a.op == "treat_outliers":
                _outlier_decision(plan, a, no)
                continue
            else:
                st.markdown(f"**{a.title}**")
                keys = [c.key for c in a.choices]
                labels = {c.key: c.label for c in a.choices}
                st.radio("Choose", keys, index=keys.index(a.default_choice) if a.default_choice in keys else 0, format_func=lambda k, lab=labels: lab[k],
                         key=f"ch_{no}_{a.action_id}", label_visibility="collapsed")
            st.caption(f"{a.summary} {a.expected_impact if a.kind == OPTIONAL else ''}")
            _details(a)

    if rev:
        st.markdown("#### For your review (Nexora will not change these)")
        for a in rev:
            with st.expander(a.title):
                st.markdown(a.summary)
                if a.expected_impact:
                    st.caption(a.expected_impact)
                if a.rationale:
                    st.markdown(f"**Suggested next step:** {a.rationale}")
                if a.rule_ref:
                    st.caption(f"Technical reference: {a.rule_ref}")

    st.divider()
    c1, c2, _ = st.columns([1.6, 1, 4])
    c2.button("Back", on_click=_go, args=(1,), key="b2")
    if c1.button("Confirm and prepare data", type="primary", help="Runs every ticked step once, then re-checks the result."):
        approvals = {a.action_id: bool(S.get(f"ap_{no}_{a.action_id}", a.kind == SAFE)) for a in safe + [x for x in dec if x.kind == OPTIONAL]}
        choices = {a.action_id: S.get(f"ch_{no}_{a.action_id}", a.default_choice) for a in dec if a.kind == DECISION}
        overrides = {}
        for a in dec:
            if a.op == "treat_outliers" and choices.get(a.action_id) not in (None, "report"):
                cfg = _outlier_cfg(a, no)
                try:
                    validate_config(cfg, choices[a.action_id])
                except OutlierConfigError as e:
                    st.error(f"Outlier treatment: {e}")
                    return
                overrides[a.action_id] = {"config": cfg.to_dict()}
        bar = st.progress(0.0, text="Starting")
        res, vid = state.ws().run_preparation(h, plan, approvals, choices, overrides=overrides, reference_tables=dict(S.get("reference_tables") or {}),
                                              progress=lambda f, m: bar.progress(min(max(f, 0.0), 1.0), text=m))
        S["prep_result"], S["version_id"] = res, vid
        state.clear_frames()
        S["prep_step"] = 3
        st.rerun()


def _result(h) -> None:
    S = st.session_state
    res, plan = S["prep_result"], S["plan"]
    rec = res.reconciliation
    st.subheader("Result")
    if rec["ok"] and res.source_unchanged:
        st.success(f"Prepared data saved: {res.after['rows']:,} rows, {res.after['columns']} columns. All {len(rec['checks'])} reconciliation checks passed and your original file is unchanged.")
    else:
        st.error("A reconciliation check failed. Do not rely on this result until you review the checks below.")
    b, a = res.before, res.after
    failing = lambda rs: sum(1 for r in rs if r["status"] == "failed")  # noqa: E731
    t = pd.DataFrame([{"Measure": m, "Before": x, "After": y} for m, x, y in [
        ("Rows", b["rows"], a["rows"]), ("Columns", b["columns"], a["columns"]), ("Missing cells", b["missing_cells"], a["missing_cells"]),
        ("Repeated identical rows", b["exact_duplicate_rows"], a["exact_duplicate_rows"]),
        ("Columns with format problems", b["columns_with_format_issues"], a["columns_with_format_issues"]),
        ("Domain checks failing", failing(res.rule_summary_before), failing(res.rule_summary_after))]])
    st.markdown("#### Before and after")
    st.dataframe(t, hide_index=True, width="stretch")
    with st.expander("Reconciliation checks"):
        st.dataframe(pd.DataFrame([{"Check": c["check"], "Result": "Pass" if c["ok"] else "FAIL", "Detail": c["detail"]} for c in rec["checks"]]), hide_index=True, width="stretch")
    st.markdown("#### What was done")
    labels = {"applied": "Applied", "no_change": "Nothing to change", "skipped": "Skipped", "failed": "Failed", "not_approved": "Not applied"}
    log = pd.DataFrame([{"Step": e.step, "Action": e.title, "Status": labels.get(e.status, e.status), "Rows in": e.rows_in, "Rows out": e.rows_out,
                         "Values changed": e.values_changed, "Skipped": e.skipped_count, "Notes": e.notes} for e in res.log])
    st.dataframe(log, hide_index=True, width="stretch")
    for e in [e for e in res.log if e.removed_rows]:
        with st.expander(f"Rows removed by '{e.title}' ({len(e.removed_rows):,})"):
            st.write(", ".join(map(str, e.removed_rows[:300])) + (" ..." if len(e.removed_rows) > 300 else ""))
            st.caption("Row numbers refer to your original file (header = row 1). Nothing was deleted from the original.")
    st.markdown("#### Still needs attention")
    if res.unresolved:
        for u in res.unresolved:
            st.markdown(f"- {u}")
    else:
        st.markdown("Nothing is flagged by the domain checks.")
    flagged = [r for r in res.rule_summary_after if r["status"] == "failed"]
    if flagged:
        with st.expander("Rows still flagged by domain checks"):
            st.dataframe(pd.DataFrame([{"Check": r["name"], "Severity": r["severity"], "Rows": r["rows_flagged"]} for r in flagged]), hide_index=True, width="stretch")
            fa = pd.DataFrame(res.findings_after)
            if not fa.empty:
                cols = [c for c in ("rule_id", "row_ref", "columns", "explanation") if c in fa.columns]
                st.dataframe(fa[cols].head(200).astype(str), hide_index=True, width="stretch")
    st.markdown("#### Prepared data")
    st.dataframe(theme.display_frame(res.prepared.head(50)), width="stretch")
    files = state.ws().preparation_exports(res, h, plan)
    st.markdown("#### Downloads")
    cols = st.columns(5)
    spec = [("Prepared data (CSV)", "prepared_data.csv", "text/csv"), ("Report (HTML)", "preparation_report.html", "text/html"),
            ("Step log (CSV)", "transformation_log.csv", "text/csv"), ("Report data (JSON)", "preparation_report.json", "application/json")]
    for col, (lab, fn, mime) in zip(cols, spec):
        col.download_button(lab, files[fn], file_name=fn, mime=mime, key=f"dl_{fn}")
    cols[4].download_button("Everything (ZIP)", zip_bytes(files), file_name="nexora_preparation.zip", mime="application/zip", key="dl_zip")
    st.divider()
    goal = plan.goal
    components.next_steps("prep_done", exclude=("Prepare Data",), df=res.prepared,
                          recommended={"supervised": "ML Studio", "unsupervised": "ML Studio"}.get(goal, "Visualize & Dashboard"))
    st.caption("These pages now use the prepared version. You can switch versions from the dataset bar at the top of any page.")
    if st.button("Start over with this dataset", key="prep_restart"):
        S.update({"prep_step": 0, "plan": None, "prep_result": None})
        st.rerun()
