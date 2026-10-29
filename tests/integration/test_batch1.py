"""Batch 1 fixes: readable errors that stay visible, honest privacy text on the hosted demo."""
import pytest
from streamlit.testing.v1 import AppTest

from .conftest import ROOT
from .test_ui_flow import app, nav  # noqa: F401


def test_error_details_stay_visible_without_a_rerun(app, monkeypatch):
    from nexora.ui.pages import reports
    monkeypatch.setattr(reports, "render", lambda: (_ for _ in ()).throw(RuntimeError("boom in reports")))
    app.sidebar.radio[0].set_value("Reports").run()
    assert any("RuntimeError: boom in reports" in e.value for e in app.error)
    assert any("boom in reports" in c.value for c in app.code)                       # details are on the page already
    assert not any(c.label == "Show technical details" for c in app.checkbox)


@pytest.mark.parametrize("public,needle", [("1", "this browser session only"), ("0", "stored on this computer only")])
def test_settings_privacy_text_matches_where_it_runs(tmp_path, monkeypatch, public, needle):
    monkeypatch.setenv("NEXORA_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("NEXORA_PUBLIC", public)
    from nexora.ui import state
    state.workspace.clear()
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Settings").run()
    assert any(needle in m.value for m in at.markdown)
