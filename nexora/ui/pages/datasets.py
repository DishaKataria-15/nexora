from __future__ import annotations

import pandas as pd
import streamlit as st

from ...validation.ingestion import IngestError, list_excel_sheets
from ...workflow import SAMPLES
from .. import components, state, theme


def _preview(df) -> None:
    st.dataframe(df.head(50), hide_index=True, width="stretch")
    st.caption(f"Showing the first {min(50, len(df))} of {len(df):,} rows. Values are shown exactly as stored in the file.")


def render() -> None:
    st.title("Datasets")
    w = state.ws()
    h = state.handle()
    if h:
        st.markdown(f"**Current dataset:** {theme.esc(h.name)} &nbsp; {theme.chip(f'{len(h.raw):,} rows x {h.raw.shape[1]} columns', 'info')}"
                    + (f" &nbsp; {theme.sample_badge()}" if h.source == "sample" else "")
                    + (f" &nbsp; {theme.chip('prepared version selected', 'ok')}" if state.has_prepared() else ""), unsafe_allow_html=True)
        wf = state.working_frame()
        typed = wf[0] if wf else h.raw              # types detected (or the prepared version), so the counts mean something
        st.caption(components.summary(typed) + ("" if state.has_prepared() else " (types detected automatically)"))
        with st.expander("Preview current dataset"):
            _preview(h.raw)
        components.next_steps("datasets", recommended="Prepare Data" if not state.has_prepared() else None, df=typed)
        st.divider()
        st.markdown("#### Open or add another dataset")

    tab_up, tab_sample, tab_db, tab_api, tab_mine, tab_comb = st.tabs(["Upload", "Sample data", "Database", "Web API", "My datasets",
                                                                          "Combine datasets (advanced)"])
    with tab_comb:
        _combine(w)
    with tab_db:
        _database(w)
    with tab_api:
        _web_api(w)
    with tab_up:
        st.markdown(f"Upload a **CSV, TSV or Excel (.xlsx)** file. Limits: {w.settings.max_upload_mb} MB, {w.settings.max_rows:,} rows, {w.settings.max_cols} columns. "
                    "Your original file is stored read-only and is never modified.")
        f = st.file_uploader("Choose a file", type=["csv", "tsv", "xlsx"], key="uploader")
        if f is not None:
            data = f.getvalue()
            sheet: str | int = 0
            if f.name.lower().endswith(".xlsx"):
                sheets = list_excel_sheets(data)
                if len(sheets) > 1:
                    sheet = st.selectbox("Worksheet", sheets, help="Only one worksheet is read at a time.")
            res = None
            try:
                res = w.read_upload(f.name, data, sheet)
            except IngestError as e:
                st.error(str(e))
            if res is not None:
                for msg in res.warnings:
                    st.warning(msg)
                enc = f", encoding {res.encoding}" if res.encoding else ""
                st.markdown(f"**{theme.esc(res.filename)}**: {res.n_rows:,} rows, {res.n_cols} columns{enc}. Fingerprint `{res.sha256[:12]}`.")
                _preview(res.dataframe)
                if st.button("Use this dataset", type="primary", key="use_upload"):
                    state.set_dataset(w.ingest(f.name, data, sheet=sheet))
                    state.goto("Prepare Data")
                    st.rerun()
        else:
            st.caption("No file chosen yet.")

    with tab_sample:
        st.markdown(f"{theme.sample_badge()} &nbsp; These files are **invented** for demonstration. Counts and results you see from them are computed live from the file, "
                    "but they describe no real business.", unsafe_allow_html=True)
        for s in SAMPLES:
            c1, c2 = st.columns([5, 1.3])
            c1.markdown(f"**{s['label']}**<br><span class='nx-muted'>{theme.esc(s['about'])}</span>", unsafe_allow_html=True)
            if c2.button("Open sample", key=f"smp_{s['file']}"):
                state.set_dataset(w.load_sample(s["file"]))
                st.session_state["profile_id"] = s["domain"]
                st.session_state["goal"] = s["goal"]
                st.rerun()

    with tab_mine:
        rows = w.repo.list_datasets()
        if not rows:
            theme.empty_state("Nothing saved yet", "Datasets you upload or open from the samples are listed here.")
        else:
            st.dataframe([{"Name": r["name"], "Rows": r["rows"], "Columns": r["cols"], "File": r["filename"], "Added": r["created_at"][:16].replace("T", " "),
                           "Kind": "sample" if r["source"] == "sample" else "upload"} for r in rows], hide_index=True, width="stretch")
            ids = {f"{r['name']} ({r['created_at'][:16].replace('T', ' ')}, {r['id'][:4]})": r["id"] for r in rows}
            pick = st.selectbox("Open a saved dataset", list(ids))
            vers = w.repo.list_versions(ids[pick])
            vopts = {"Raw data (no preparation)": None} | {f"Prepared {v['created_at'][:16].replace('T', ' ')}: {v['rows']:,} rows ({v['id'][:4]})": v["id"] for v in vers}
            vpick = st.selectbox("Version to work with", list(vopts), help="Prepared versions were saved by Prepare Data; the raw file is always kept.")
            if st.button("Open", key="open_saved"):
                state.set_dataset(w.open_dataset(ids[pick]))
                st.session_state["version_id"] = vopts[vpick]
                state.clear_frames()
                st.rerun()


# ---------------------------------------------------------------------------------------------- connectors
def _database(w) -> None:
    from ...connectors import database as D
    if w.settings.public_mode:
        st.info("Database connections are switched off in the public demo, because the server would connect on behalf of every visitor. "
                "Run Nexora on your own computer to use them, or export your table as CSV and upload it.")
        return
    st.markdown("Read a table or the result of a **SELECT** query. Access is **read-only**: Nexora never writes to your database. "
                "For the strongest protection, connect with a database user that only has read permission.")
    kinds = list(D.DIALECTS)
    dialect = st.selectbox("Database type", kinds, format_func=lambda k: D.DIALECTS[k]["label"], key="db_kind")
    ok, why = D.driver_status(dialect)
    if not ok:
        st.warning(why)
    if dialect == "sqlite":
        path = st.text_input("Path to the SQLite file on this computer", key="db_path", placeholder="for example C:/data/shop.db")
        cfg = D.DBConfig("sqlite", database=path, timeout_s=w.settings.db_timeout_s)
    else:
        c1, c2 = st.columns([3, 1])
        host = c1.text_input("Host", key="db_host")
        port = c2.number_input("Port", 1, 65535, int(D.DIALECTS[dialect]["port"]), key="db_port")
        c1, c2 = st.columns(2)
        dbname = c1.text_input("Database name", key="db_name")
        user = c2.text_input("User name", key="db_user")
        src = st.radio("Password", ["Type it now (kept in memory for this session only)", "Read it from an environment variable"], key="db_pwsrc", horizontal=True)
        pw, env = "", ""
        if src.startswith("Type"):
            pw = st.text_input("Password", type="password", key="db_pw")
        else:
            env = st.text_input("Environment variable name", key="db_env", placeholder="for example SHOP_DB_PASSWORD")
        schema = st.text_input("Schema (optional)", key="db_schema") or None
        cfg = D.DBConfig(dialect, database=dbname, host=host, port=int(port), username=user, password=pw, password_env=env, schema=schema,
                         timeout_s=w.settings.db_timeout_s)
        st.caption("Passwords are never written to disk, logs or saved datasets; error messages are redacted.")
    if st.button("Connect", key="db_connect", disabled=not ok):
        try:
            old = st.session_state.pop("db_conn", None)
            if old is not None:
                old.close()
            conn = w.connect_database(cfg)
            st.session_state["db_conn"] = conn
            st.session_state["db_tables"] = conn.list_tables()
        except D.ConnectorError as e:
            st.error(str(e))
    conn = st.session_state.get("db_conn")
    if conn is None:
        return
    st.success(f"Connected (read-only): {conn.cfg.describe()}")
    tables = st.session_state.get("db_tables") or []
    mode = st.radio("Read", ["A table or view", "A SELECT query"], key="db_mode", horizontal=True)
    limit = int(st.number_input("Row limit", 1, D.MAX_ROW_LIMIT, min(w.settings.db_row_limit, D.MAX_ROW_LIMIT), key="db_limit"))
    try:
        if mode == "A table or view":
            if not tables:
                st.info("No tables or views were found.")
                return
            names = [t["name"] for t in tables]
            t = st.selectbox("Table", names, format_func=lambda n: n + ("  (view)" if next(x for x in tables if x["name"] == n)["kind"] == "view" else ""), key="db_table")
            if st.button("Preview 200 rows", key="db_prev"):
                st.session_state["db_preview"] = conn.read_table(t, limit=200)
            full = st.checkbox("Read every row (ignores the row limit)", key="db_full")
            confirm = False
            if full:
                n = conn.count_rows(t)
                st.warning(f"'{t}' has {n:,} rows. Reading all of them can be slow and expensive for the database.")
                confirm = st.checkbox(f"Yes, read all {n:,} rows", key="db_full_ok")
            if st.button("Add as dataset", type="primary", key="db_add", disabled=full and not confirm):
                h, res = w.ingest_database(conn, table=t, limit=limit, full=full, confirm_full=confirm, name=t)
                st.session_state["db_done"] = res.note()
                state.set_dataset(h)
                st.rerun()
        else:
            sql = st.text_area("SELECT query", key="db_sql", placeholder="SELECT region, SUM(amount) AS total FROM orders WHERE year = :year GROUP BY region",
                               help="One SELECT statement. Use :name for values and fill them in below; values are sent separately, never pasted into the SQL.")
            import re
            names = sorted(set(re.findall(r"(?<![:\w]):([A-Za-z_]\w*)", D._strip_literals(sql or ""))))
            params = {n: st.text_input(f"Value for :{n}", key=f"db_p_{n}") for n in names}
            c1, c2 = st.columns(2)
            if c1.button("Preview 200 rows", key="db_qprev"):
                st.session_state["db_preview"] = conn.run_query(sql, params, limit=200)
            if c2.button("Add as dataset", type="primary", key="db_qadd"):
                h, res = w.ingest_database(conn, sql=sql, params=params, limit=limit, name="query result")
                st.session_state["db_done"] = res.note()
                state.set_dataset(h)
                st.rerun()
    except (D.ConnectorError, ValueError) as e:
        st.error(str(e))
    pv = st.session_state.get("db_preview")
    if pv is not None:
        st.caption(pv.note())
        st.dataframe(pv.frame.head(200), hide_index=True, width="stretch")
    if st.session_state.get("db_done"):
        st.success("Added: " + st.session_state["db_done"])


def _web_api(w) -> None:
    from ...connectors import web_api as W
    st.markdown("Fetch **JSON or CSV** from a web address. Only http(s) addresses on the public internet are allowed; internal and private addresses "
                "are blocked" + (" (an administrator has allowed private addresses on this installation)." if w.settings.allow_private_api_hosts else "."))
    url = st.text_input("URL", key="api_url", placeholder="https://api.example.com/v1/items")
    c1, c2 = st.columns(2)
    auth = c1.selectbox("Authentication", list(W.AUTH), format_func=W.AUTH.get, key="api_auth")
    secret, user, key_name = "", "", "X-API-Key"
    if auth == "bearer":
        secret = c2.text_input("Token", type="password", key="api_secret")
    elif auth in ("api_key_header", "api_key_query"):
        key_name = c2.text_input("Header or parameter name", "X-API-Key" if auth == "api_key_header" else "api_key", key="api_keyname")
        secret = st.text_input("API key", type="password", key="api_secret2")
    elif auth == "basic":
        user = c2.text_input("User name", key="api_user")
        secret = st.text_input("Password", type="password", key="api_pw")
    if auth != "none":
        st.caption("Credentials are sent only to this host over https, kept in memory for this session, and never logged or saved.")
    c1, c2 = st.columns(2)
    path = c1.text_input("Path to the records in the JSON (optional)", key="api_path", placeholder="for example data.items")
    fmt = c2.selectbox("Response format", ["auto", "json", "csv"], key="api_fmt")
    pag = st.selectbox("Pagination", list(W.PAGINATION), format_func=W.PAGINATION.get, key="api_pag")
    cfg = W.APIConfig(url=url.strip(), auth=auth, secret=secret, username=user, key_name=key_name, records_path=path.strip(), response_format=fmt, pagination=pag)
    if pag != "none":
        c1, c2, c3 = st.columns(3)
        cfg.max_pages = int(c1.number_input("Page limit", 1, W.MAX_PAGES, 10, key="api_maxp"))
        if pag in ("page", "offset"):
            cfg.page_size = int(c2.number_input("Page size", 1, 1000, 100, key="api_psize"))
            if pag == "page":
                cfg.page_param = c3.text_input("Page parameter", "page", key="api_pp")
                cfg.size_param = st.text_input("Page-size parameter", "per_page", key="api_sp")
            else:
                cfg.offset_param = c3.text_input("Offset parameter", "offset", key="api_op")
                cfg.limit_param = st.text_input("Limit parameter", "limit", key="api_lp")
        elif pag == "cursor":
            cfg.cursor_path = c2.text_input("Where the next cursor is in the JSON", "next_cursor", key="api_cp")
            cfg.cursor_param = c3.text_input("Cursor parameter", "cursor", key="api_cparam")
        elif pag == "link":
            cfg.next_path = c2.text_input("JSON field with the next URL (empty = Link header)", "", key="api_np")
    cfg.max_rows = int(st.number_input("Row limit", 1, 1_000_000, 100_000, key="api_maxrows"))
    name = st.text_input("Dataset name", "api data", key="api_name")
    if st.button("Fetch and add as dataset", type="primary", key="api_go", disabled=not url.strip()):
        try:
            h, res = w.ingest_api(cfg, name=name)
            st.session_state["api_done"] = (res.note(), res.notes)
            state.set_dataset(h)
            st.rerun()
        except (W.ConnectorError, ValueError) as e:
            st.error(str(e))
    done = st.session_state.get("api_done")
    if done:
        st.success("Added: " + done[0])
        for n in done[1]:
            st.caption(n)


# ---------------------------------------------------------------------------------------------- combine
def _combine(w) -> None:
    from ...combine import AGGS, JOIN_TYPES, MODES, CombineError, Step, combine, suggest_keys
    S = st.session_state
    st.markdown("Combine **two or more saved datasets** into one table, for example ratings with movie titles. Upload or open each file first; "
                "they then appear here. The originals are not changed and the result is saved as a new dataset.")
    rows = w.repo.list_datasets()
    if len(rows) < 2:
        theme.empty_state("At least two datasets are needed", "Upload each file (or open samples) first; they are listed under My datasets.")
        return
    labels: dict[str, str] = {}
    for r in rows:
        lab = f"{r['name']} ({r['rows']:,} rows)"
        if lab in labels:
            lab = f"{r['name']} ({r['rows']:,} rows, {r['id'][:4]})"
        labels[lab] = r["id"]
    chosen = st.multiselect("Datasets to combine (2 to 5)", list(labels), max_selections=5, key="cb_sel")
    if len(chosen) < 2:
        st.caption("Pick at least two.")
        return
    frames = {lab: _raw(w, labels[lab]) for lab in chosen}
    base = st.selectbox("Main table (every step adds to this one)", chosen, key="cb_base",
                        help="Usually the table with one row per thing you want to analyse, for example one row per rating.")
    others = [c for c in chosen if c != base]
    steps: list[Step] = []
    known_cols = list(frames[base].columns)
    for i, o in enumerate(others, start=1):
        with st.container(border=True):
            st.markdown(f"**Step {i}: {theme.esc(o)}**", unsafe_allow_html=True)
            mode = st.radio("How", list(MODES), format_func=MODES.get, key=f"cb_mode_{i}_{o}")
            if mode == "stack":
                steps.append(Step("stack", o))
                st.caption("Columns are matched by name; columns that only one table has are kept and left empty for the other rows.")
                continue
            sug = suggest_keys(frames[base], frames[o])
            if sug:
                st.caption("Possible keys: " + "; ".join(f"`{a}` = `{b}` ({100 * c:.0f}% of values found)" for a, b, c in sug[:3]))
            c1, c2 = st.columns(2)
            dl = [sug[0][0]] if sug and sug[0][0] in known_cols else []
            dr = [sug[0][1]] if sug and sug[0][1] in frames[o].columns else []
            left_on = c1.multiselect("Key column(s) in the main table", known_cols, default=dl, key=f"cb_l_{i}_{o}")
            right_on = c2.multiselect(f"Matching column(s) in {o}", list(frames[o].columns), default=dr, key=f"cb_r_{i}_{o}",
                                      help="Pick them in the same order as on the left.")
            c1, c2 = st.columns([2, 1])
            how = c1.selectbox("Rows to keep", list(JOIN_TYPES), format_func=JOIN_TYPES.get, key=f"cb_how_{i}_{o}")
            ic = c2.checkbox("Ignore upper/lower case in keys", key=f"cb_ic_{i}_{o}")
            if mode == "join":
                cand = [c for c in frames[o].columns if c not in right_on]
                cols = st.multiselect("Columns to bring in", cand, default=cand, key=f"cb_cols_{i}_{o}")
                steps.append(Step("join", o, left_on, right_on, how, ic, columns=cols))
                known_cols += cols
            else:
                cand = [c for c in frames[o].columns if c not in right_on]
                c1, c2 = st.columns(2)
                scols = c1.multiselect("Columns to summarise", cand, default=cand[:1], key=f"cb_scol_{i}_{o}")
                funcs = c2.multiselect("Statistics", AGGS, default=["count", "mean"], key=f"cb_sf_{i}_{o}",
                                       help="mean, sum, min, max and median need numbers; count and nunique work for any column.")
                steps.append(Step("summary", o, left_on, right_on, how, ic, aggs={c: list(funcs) for c in scols}))
                known_cols += [f"{c}_{f}" for c in scols for f in funcs]
    name = st.text_input("Name for the combined dataset", f"{base.split(' (')[0]} combined", key="cb_name")
    c1, c2, _ = st.columns([1.2, 1.6, 3])
    if c1.button("Preview", key="cb_prev"):
        try:
            out, reps = combine(frames, base, steps, w.settings.max_rows)
            S["cb_preview"] = (out.head(50), reps, len(out), out.shape[1])
        except CombineError as e:
            S["cb_preview"] = None
            st.error(str(e))
    if c2.button("Combine and save", type="primary", key="cb_save"):
        try:
            h, reps, recipe = w.combine_datasets({lab: labels[lab] for lab in chosen}, base, steps, name.strip() or "combined data")
            S["cb_done"] = (h.name, len(h.raw), h.raw.shape[1], recipe)
            S["cb_preview"] = None
            state.set_dataset(h)
            st.rerun()
        except (CombineError, ValueError) as e:
            st.error(str(e))
    pv = S.get("cb_preview")
    if pv:
        head, reps, n, ncol = pv
        st.markdown(f"**Result: {n:,} rows x {ncol} columns.**")
        st.dataframe(pd.DataFrame([{"Step": r.step, "With": r.other, "How": r.mode, "Rows before": r.rows_before, "Rows after": r.rows_after,
                                    "Matched rows": r.matched_rows, "Unmatched main rows": r.unmatched_main,
                                    "Unused keys in other": r.unmatched_other_keys, "Repeated keys in other": r.duplicate_other_keys,
                                    "Columns added": len(r.columns_added)} for r in reps]).dropna(axis=1, how="all"), hide_index=True, width="stretch")
        for r in reps:
            for wmsg in r.warnings:
                st.warning(f"Step {r.step}: {wmsg}")
            if r.examples_unmatched:
                st.caption(f"Step {r.step}: examples of keys with no match: " + ", ".join(f"`{x}`" for x in r.examples_unmatched))
        st.dataframe(head, hide_index=True, width="stretch")
    done = S.get("cb_done")
    if done:
        st.success(f"Saved '{done[0]}' ({done[1]:,} rows x {done[2]} columns). It is now the open dataset; choose what to do next at the top of this page.")
        import json
        st.download_button("How it was combined (JSON)", json.dumps(done[3], indent=2), file_name="combine_recipe.json", mime="application/json", key="cb_recipe")


@st.cache_data(show_spinner="Loading dataset...", max_entries=8)
def _raw(_w, dataset_id: str):
    return _w.open_dataset(dataset_id).raw
