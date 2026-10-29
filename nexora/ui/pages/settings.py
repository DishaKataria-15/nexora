from __future__ import annotations

import streamlit as st

from ...config import get_settings
from ...ml.compute import detect_compute
from ...ml.training import environment
from .. import state


def render() -> None:
    st.title("Settings")
    if st.session_state.pop("_deleted", False):
        st.success("All local data was deleted.")
    s = get_settings()
    w = state.ws()
    st.subheader("Defaults")
    st.session_state["seed"] = int(st.number_input("Default random seed", 0, 10**6, int(st.session_state["seed"]),
                                                   help="Used as the starting seed for new experiments, so results can be repeated."))
    st.caption("Applies to new experiments. Existing experiments keep the seed they were trained with.")
    st.subheader("Limits")
    if s.public_mode:
        # the demo does not show server paths, hardware or package versions
        st.table({"Setting": ["Maximum upload size", "Maximum rows", "Maximum columns"],
                  "Value": [f"{s.max_upload_mb} MB", f"{s.max_rows:,}", str(s.max_cols)]})
        st.caption("These are the limits of the online demo. Running Nexora on your own computer lets you raise them.")
    else:
        st.table({"Setting": ["Maximum upload size", "Maximum rows", "Maximum columns", "Data folder"],
                  "Value": [f"{s.max_upload_mb} MB", f"{s.max_rows:,}", str(s.max_cols), str(s.data_dir)]})
        st.caption("Change limits with the environment variables NEXORA_MAX_UPLOAD_MB, NEXORA_MAX_ROWS, NEXORA_MAX_COLS and NEXORA_DATA_DIR before starting Nexora.")
        st.subheader("Compute")
        info = detect_compute()
        st.markdown(f"**CPU cores:** {info.cpu_count}. **GPU:** {', '.join(info.gpu_devices) if info.gpu_devices else 'none detected'}.")
        st.caption(info.note + " A GPU option is offered only if a compatible device and a GPU-capable model are both available.")
        st.subheader("Software versions")
        env = environment()
        st.table({"Package": list(env), "Version": list(env.values())})
    st.subheader("Your data in this session" if s.public_mode else "Local data")
    if s.public_mode:
        st.markdown(f"{len(w.repo.list_datasets())} dataset(s) and {len(w.repo.list_experiments())} experiment(s) are stored on the demo server for "
                    "**this browser session only**. Other visitors cannot see them, and they are deleted within 24 hours or when the server restarts. "
                    "Download anything you want to keep.")
    else:
        st.markdown(f"{len(w.repo.list_datasets())} dataset(s) and {len(w.repo.list_experiments())} experiment(s) are stored on this computer only. "
                    "Nexora sends nothing anywhere; only the Database and Web API tabs contact the sources you enter.")
    with st.expander("Delete all workspace data" if s.public_mode else "Delete all local workspace data"):
        st.warning("This permanently removes every stored dataset, prepared version, experiment, saved run and model "
                   + ("from this browser session." if s.public_mode else "from this computer. Your original files elsewhere are not touched."))
        ok = st.checkbox("I understand this cannot be undone", key="del_ok")
        if st.button("Delete everything", disabled=not ok):
            w.repo.clear_all()
            for k in ("handle", "version_id", "plan", "prep_result", "exp", "exp_id", "job", "ml", "mapping", "reference_tables"):
                st.session_state[k] = None
            st.session_state.update({"dashboard": [], "prep_step": 0, "_deleted": True})
            st.session_state.pop("del_ok", None)
            state.clear_frames()
            st.rerun()
