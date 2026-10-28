"""Visual system: restrained navy/slate/white with a single muted teal accent. Status always carries a text label."""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
:root{--nx-navy:#14213d;--nx-teal:#0f766e;--nx-slate:#475569;--nx-line:#dbe2ea;--nx-bg:#fbfaf8}
.block-container{padding-top:2.2rem;padding-bottom:3rem;max-width:1240px}
h1{font-size:1.65rem!important;letter-spacing:-.01em}h2{font-size:1.2rem!important}h3{font-size:1.02rem!important}
.nx-tag{color:var(--nx-slate);font-size:.95rem;margin:-.4rem 0 1.0rem}
.nx-stepper{display:flex;flex-wrap:wrap;gap:6px;margin:.2rem 0 1.1rem;padding:0;list-style:none}
.nx-step{flex:1 1 110px;padding:7px 10px 6px;border-bottom:3px solid #cbd5e1;font-size:.84rem;color:var(--nx-slate)}
.nx-step.done{border-color:var(--nx-teal);color:var(--nx-teal)}
.nx-step.current{border-color:var(--nx-navy);color:var(--nx-navy);font-weight:600}
.nx-chip{display:inline-block;padding:0 8px;border-radius:10px;font-size:.78rem;border:1px solid;white-space:nowrap}
.nx-chip.ok{color:#0f766e;border-color:#0f766e;background:#ecfdf5}.nx-chip.warn{color:#92400e;border-color:#b45309;background:#fffbeb}
.nx-chip.bad{color:#9f1239;border-color:#9f1239;background:#fff1f2}.nx-chip.info{color:#1e3a5f;border-color:#1e3a5f;background:#eff6ff}
.nx-chip.off{color:#475569;border-color:#94a3b8;background:#f1f5f9}
.nx-card{border:1px solid var(--nx-line);border-radius:6px;padding:12px 16px;background:#fff;margin-bottom:10px}
.nx-muted{color:var(--nx-slate);font-size:.88rem}
.nx-empty{border:1px dashed #94a3b8;border-radius:6px;padding:26px 18px;text-align:center;background:#fff;margin:10px 0}
.nx-sample{font-size:.8rem;border:1px solid #b45309;color:#92400e;background:#fffbeb;border-radius:4px;padding:0 6px}
[data-testid="stSidebar"] .nx-side-tag{color:#aab6c8;font-size:.82rem}
:focus-visible{outline:2px solid #0f766e!important;outline-offset:2px}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s: object) -> str:
    return html.escape(str(s))


def chip(text: str, kind: str = "info") -> str:
    return f"<span class='nx-chip {kind}'>{esc(text)}</span>"


def stepper(steps: list[str], current: int) -> None:
    items = []
    for i, s in enumerate(steps):
        cls = "done" if i < current else ("current" if i == current else "")
        mark = "&#10003; " if i < current else f"{i + 1}. "
        aria = " aria-current='step'" if i == current else ""
        items.append(f"<li class='nx-step {cls}'{aria}>{mark}{esc(s)}</li>")
    st.markdown(f"<ol class='nx-stepper' aria-label='Progress'>{''.join(items)}</ol>", unsafe_allow_html=True)


def empty_state(title: str, body: str) -> None:
    st.markdown(f"<div class='nx-empty'><b>{esc(title)}</b><br><span class='nx-muted'>{esc(body)}</span></div>", unsafe_allow_html=True)


def sample_badge() -> str:
    return "<span class='nx-sample'>SAMPLE DATA (synthetic)</span>"


def fmt_int(n: int | float) -> str:
    return f"{int(n):,}"
