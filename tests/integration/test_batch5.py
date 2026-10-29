"""Batch 5: demo polish. Server details and install hints hidden on the public demo, the 'not prepared' message said
once, scrolling to the top on a new step, the Home introduction with sample shortcuts, and the footer."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from .test_ui_flow import app, click, nav  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def public_app(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXORA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NEXORA_PUBLIC", "1")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    return at


def test_demo_settings_hide_server_details(public_app):
    nav(public_app, "Settings")
    heads = [h.value for h in public_app.subheader]
    assert "Software versions" not in heads and "Compute" not in heads and "Your data in this session" in heads
    assert not any("Data folder" in str(t.value) for t in public_app.table)


def test_local_settings_still_show_them(app):
    nav(app, "Settings")
    assert "Software versions" in [h.value for h in app.subheader]


def test_home_introduction_sample_shortcut_and_footer(app):
    md = " ".join(m.value for m in app.markdown)
    assert "raw file to trustworthy results" in md and "Try a sample" in md
    assert "github.com/DishaKataria-15" in md and "linkedin.com/in/dishakataria15" in md
    click(app, "Predict churn")
    assert app.session_state["nav"] == "ML Studio" and app.session_state["handle"].name.startswith("Customer churn")
    assert app.radio(key="ml_w_goal").value == "classification"


def test_step_change_scrolls_to_top_only_when_the_step_changes(app):
    click(app, "Predict churn")
    n0 = app.session_state.get("_scroll_n", 0)
    app.run()
    assert app.session_state.get("_scroll_n", 0) == n0                  # a rerun on the same step does not scroll
    click(app, "Next: Target")
    assert app.session_state.get("_scroll_n", 0) == n0 + 1


def test_demo_installs_the_optional_models():
    req = (ROOT / "requirements.txt").read_text()
    for pkg in ("xgboost-cpu", "lightgbm", "statsmodels"):
        assert pkg in req
    assert (ROOT / "packages.txt").read_text().strip() == "libgomp1"
    assert (ROOT / "docs" / "demo.gif").stat().st_size < 3_000_000
