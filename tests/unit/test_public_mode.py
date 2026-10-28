"""Public hosting: one private workspace per browser session, database tab off, old sessions cleaned up."""
import os
import time

import pytest
from streamlit.testing.v1 import AppTest

from nexora.config import get_settings

from ..integration.conftest import ROOT


def test_public_mode_switch(monkeypatch):
    monkeypatch.setenv("NEXORA_PUBLIC", "1")
    assert get_settings().public_mode
    monkeypatch.setenv("NEXORA_PUBLIC", "0")
    assert not get_settings().public_mode
    monkeypatch.delenv("NEXORA_PUBLIC")
    assert get_settings().public_mode is False          # this checkout is not under /mount/src


@pytest.fixture()
def public(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXORA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NEXORA_PUBLIC", "1")
    from nexora.ui import state
    state.workspace.clear()
    return tmp_path / "data"


def _visitor():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    return at


def test_each_visitor_gets_a_private_workspace(public):
    a, b = _visitor(), _visitor()
    a.sidebar.radio[0].set_value("Datasets").run()
    a.button(key="smp_customer_churn.csv").click().run()
    assert a.session_state["handle"] is not None
    b.sidebar.radio[0].set_value("Datasets").run()
    from nexora.ui import state
    dirs = sorted(p.name for p in (public / "sessions").iterdir())
    assert len(dirs) == 2 and a.session_state["_sid"] != b.session_state["_sid"]
    assert any("Public demo" in i.value for i in b.info)
    # visitor b sees no datasets from visitor a
    assert not any("Customer churn" in str(d.value) for d in b.dataframe)
    b.tabs  # noqa: B018
    assert any("switched off in the public demo" in i.value for i in b.info)


def test_stale_sessions_are_removed(public):
    old = public / "sessions" / "old"
    (old / "raw").mkdir(parents=True)
    f = old / "raw" / "x.csv"
    f.write_text("a\n1\n")
    os.chmod(f, 0o444)
    past = time.time() - 3 * 24 * 3600
    for p in (f, old / "raw", old):
        os.utime(p, (past, past))
    _visitor()
    assert not old.exists()
