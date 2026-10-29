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
.nx-footer{margin-top:3rem;padding-top:.8rem;border-top:1px solid var(--nx-line);color:var(--nx-slate);font-size:.82rem;text-align:center}
.nx-footer a{color:var(--nx-teal);text-decoration:none}.nx-footer a:hover{text-decoration:underline}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s: object) -> str:
    return html.escape(str(s))


def chip(text: str, kind: str = "info") -> str:
    return f"<span class='nx-chip {kind}'>{esc(text)}</span>"


_SCROLL_JS = """<script>
// {n}: Streamlit keeps the scroll position on reruns; after moving to another step, start at the top of the page
const d = window.parent.document;
for (const sel of ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', 'section.main']) {{
  const el = d.querySelector(sel); if (el) el.scrollTo({{top: 0, behavior: 'instant'}});
}}
window.parent.scrollTo(0, 0);
</script>"""


def footer() -> None:
    st.markdown("<div class='nx-footer'>Nexora · built by Disha Kataria · "
                "<a href='https://github.com/DishaKataria-15' target='_blank' rel='noopener'>GitHub</a> · "
                "<a href='https://www.linkedin.com/in/dishakataria15' target='_blank' rel='noopener'>LinkedIn</a></div>", unsafe_allow_html=True)


def scroll_to_top_on_change(key: str, value) -> None:
    """Scroll the page to the top when `value` changes between reruns (not on the first visit)."""
    sk = f"_scroll_{key}"
    prev = st.session_state.get(sk)
    st.session_state[sk] = value
    if prev is not None and prev != value:
        import streamlit.components.v1 as components
        n = st.session_state.get("_scroll_n", 0) + 1
        st.session_state["_scroll_n"] = n
        components.html(_SCROLL_JS.format(n=n), height=0)


def stepper(steps: list[str], current: int, key: str | None = None) -> None:
    if key:
        scroll_to_top_on_change(key, steps[current])
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


def local_time(iso: str | None) -> str:
    """'YYYY-MM-DD HH:MM' in the viewer's time zone (read from the browser), or UTC when it is unknown."""
    if not iso:
        return ""
    from datetime import datetime, timezone
    try:
        t = datetime.fromisoformat(str(iso))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        tz = None
        try:
            name = st.context.timezone
            if name:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(name)
        except Exception:
            tz = None
        return t.astimezone(tz or timezone.utc).strftime("%Y-%m-%d %H:%M") + ("" if tz else " UTC")
    except (ValueError, TypeError):
        return str(iso)[:16].replace("T", " ")


def display_frame(df):
    """A copy for on-screen tables: date-only columns as YYYY-MM-DD (no 00:00:00), the same way they are exported."""
    import pandas as pd
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            nn = s.dropna()
            if len(nn) == 0 or bool((nn.dt.normalize() == nn).all()):
                out[c] = s.map(lambda v: v.strftime("%Y-%m-%d") if pd.notna(v) else None).astype(object)
    return out


def fmt_int(n: int | float) -> str:
    return f"{int(n):,}"
