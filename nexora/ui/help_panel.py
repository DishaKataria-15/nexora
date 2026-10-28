"""Sidebar help assistant: type a question or click a suggested one; answers come from the built-in help (offline)."""
from __future__ import annotations

import streamlit as st

from ..help import assistant as A
from . import state

MAX_HISTORY = 6


def _show(entry: dict, page: str, key: str) -> None:
    st.markdown(entry["a"])
    target = entry.get("goto")
    if target and target != page:
        if st.button(f"Go to {target}", key=f"help_go_{key}_{entry['id']}", width="stretch"):
            state.goto(target)
            st.rerun()


def _ask(q: str, page: str) -> None:
    ans = A.ask(q, page)
    hist = st.session_state.setdefault("help_history", [])
    hist.insert(0, {"q": q, "found": ans.found, "id": ans.entry["id"] if ans.found else None, "related": [r["id"] for r in ans.related]})
    del hist[MAX_HISTORY:]


def render(page: str) -> None:
    with st.expander("Help: ask about Nexora", expanded=bool(st.session_state.get("help_history"))):
        with st.form("help_form", clear_on_submit=True, border=False):
            q = st.text_input("Your question", placeholder="e.g. combine two files", key="help_q", label_visibility="collapsed")
            sent = st.form_submit_button("Ask", width="stretch")
        if sent and q.strip():
            _ask(q.strip(), page)
        hist = st.session_state.get("help_history") or []
        if hist:
            last = hist[0]
            st.markdown(f"**You asked:** {last['q']}")
            if last["found"]:
                _show(A.entry(last["id"]), page, "last")
            else:
                st.markdown("I don't have an answer for that yet. I can only answer questions about using Nexora. Closest topics:")
            rel = [A.entry(i) for i in last["related"]] if last["found"] else A.ask(last["q"], page).related
            if rel:
                if last["found"]:
                    st.caption("Related")
                for r in rel:
                    if st.button(r["q"][0], key=f"help_rel_{r['id']}", width="stretch"):
                        _ask(r["q"][0], page)
                        st.rerun()
            if st.button("Clear", key="help_clear"):
                st.session_state["help_history"] = []
                st.rerun()
            st.divider()
        st.caption(f"Common questions on {page}:")
        for e in A.suggestions(page):
            if st.button(e["q"][0], key=f"help_sug_{e['id']}", width="stretch"):
                _ask(e["q"][0], page)
                st.rerun()
        st.caption("Works offline: your questions are not sent anywhere.")
