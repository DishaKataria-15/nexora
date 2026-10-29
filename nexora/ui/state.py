"""Session state helpers shared by all pages."""
from __future__ import annotations

import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ..config import get_settings
from ..workflow import DatasetHandle, Workspace

PAGES = ["Home", "Datasets", "Prepare Data", "Visualize & Dashboard", "ML Studio", "Anomalies", "Forecasting", "Reports", "Settings"]


SESSION_TTL_HOURS = 24


@st.cache_resource(show_spinner=False)
def workspace(folder: str) -> Workspace:
    """One Workspace per data folder (the argument is part of the cache key, so it must not start with an underscore)."""
    s = get_settings()
    if s.public_mode:
        _cleanup_sessions(s.data_dir / "sessions")
        return Workspace(settings=replace(s, data_dir=Path(folder)))
    return Workspace()


def _cleanup_sessions(root: Path) -> None:
    """Public mode: remove visitors' workspaces that have not been touched for a day."""
    import shutil
    import stat
    if not root.exists():
        return
    cutoff = time.time() - SESSION_TTL_HOURS * 3600
    for d in root.iterdir():
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                for p in d.rglob("*"):
                    p.chmod(p.stat().st_mode | stat.S_IWRITE)     # raw sources are stored read-only
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def data_dir() -> str:
    """The workspace folder for this visitor: shared locally, one per browser session when hosted publicly."""
    s = get_settings()
    if not s.public_mode:
        return str(s.data_dir)
    if "_sid" not in st.session_state:
        st.session_state["_sid"] = uuid.uuid4().hex
    d = s.data_dir / "sessions" / st.session_state["_sid"]
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def ws() -> Workspace:
    return workspace(data_dir())


DEFAULTS: dict[str, Any] = {
    "nav": "Home", "handle": None, "version_id": None, "prep_step": 0, "profile_id": "generic", "goal": "analytics", "mapping": None,
    "plan": None, "prep_result": None, "dashboard": [], "ml": None, "job": None, "exp": None, "exp_id": None,
    "seed": None, "_goto": None, "reference_tables": None,
}


def init() -> None:
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v if not isinstance(v, (list, dict)) else type(v)(v)
    if st.session_state["seed"] is None:
        st.session_state["seed"] = get_settings().default_seed


def goto(page: str) -> None:
    """Request navigation; applied at the top of the next run (before the nav widget is created)."""
    st.session_state["_goto"] = page


def handle() -> DatasetHandle | None:
    return st.session_state.get("handle")


def set_dataset(h: DatasetHandle) -> None:
    for k in ("version_id", "plan", "prep_result", "mapping", "exp", "exp_id", "job", "ml", "reference_tables"):
        st.session_state[k] = None
    st.session_state.update({"handle": h, "prep_step": 0, "dashboard": []})
    for k in ("an_res", "fc_res", "pr_out", "ex_dim_res"):
        st.session_state.pop(k, None)
    for k in [k for k in st.session_state if str(k).startswith(("ap_", "ch_", "ml_w_", "ref_", "df_", "mp_", "ol_"))]:
        del st.session_state[k]
    _frame.clear()


@st.cache_data(show_spinner="Loading data...", max_entries=4)
def _frame(dataset_id: str, version_id: str | None, folder: str) -> tuple[pd.DataFrame, str]:
    w = ws()
    h = w.open_dataset(dataset_id) if not version_id else None
    return w.working_frame(version_id, h)


def working_frame() -> tuple[pd.DataFrame, str] | None:
    h = handle()
    if h is None:
        return None
    return _frame(h.dataset_id, st.session_state.get("version_id"), data_dir())


def typed_raw() -> pd.DataFrame | None:
    """The original file with column types detected (nothing cleaned), for summaries of an unprepared dataset."""
    h = handle()
    return None if h is None else _frame(h.dataset_id, None, data_dir())[0]


def clear_frames() -> None:
    _frame.clear()


def has_prepared() -> bool:
    return bool(st.session_state.get("version_id"))
