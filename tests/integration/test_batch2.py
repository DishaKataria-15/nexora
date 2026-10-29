"""Batch 2 fixes: consistent counts and summaries, date display, web API format and labels, constant detection,
outlier fences, popover closing, local timestamps, number formatting."""
import numpy as np
import pandas as pd
import pytest

from nexora.connectors.web_api import _guess_format
from nexora.eda.profiling import column_role
from nexora.preprocessing.outliers import OutlierConfig, detect
from nexora.ui import theme

from .test_ui_flow import app, click, nav, open_sample  # noqa: F401


def test_prepare_header_summarises_detected_types_and_duplicates_match_the_plan(app):
    open_sample(app, "retail_orders_messy.csv")
    nav(app, "Prepare Data")
    cap = " ".join(c.value for c in app.caption)
    assert "4 numeric, 6 categorical, 1 date" in cap and "12.7% of cells missing" in cap
    assert any("**5** row(s) repeat an earlier row exactly" in m.value for m in app.markdown)
    click(app, "Next: choose goal")
    click(app, "Build preparation plan")
    dup = next(a for a in app.session_state["plan"].actions if a.op == "remove_exact_duplicates")
    assert dup.affected_rows == 5                                                    # same number in Profile and plan


def test_sidebar_shows_the_prepared_row_count(app):
    open_sample(app, "retail_orders_messy.csv")
    nav(app, "Prepare Data")
    click(app, "Next: choose goal")
    click(app, "Build preparation plan")
    click(app, "Confirm and prepare data")
    assert any("223 rows" in m.value and "prepared" in m.value for m in app.sidebar.markdown)


def test_date_only_values_are_displayed_without_midnight():
    df = pd.DataFrame({"d": pd.to_datetime(["2026-01-05", "2026-02-01", None]), "t": pd.to_datetime(["2026-01-05 10:30", None, None])})
    out = theme.display_frame(df)
    assert out["d"].tolist()[:2] == ["2026-01-05", "2026-02-01"] and pd.isna(out["d"].iloc[2])
    assert pd.api.types.is_datetime64_any_dtype(out["t"])                            # real times are kept


@pytest.mark.parametrize("url,ctype,body,fmt", [("https://x.org/data.csv", "text/plain", b"a,b\n1,2\n", "csv"),
                                                ("https://x.org/data.csv", "application/octet-stream", b"a,b\n1,2", "csv"),
                                                ("https://x.org/feed", "text/plain", b'[{"a": 1}]', "json"),
                                                ("https://x.org/feed", "", b"a\tb\n1\t2", "tsv"),
                                                ("https://x.org/x.json", "text/plain", b'{"a": 1}', "json"),
                                                ("https://x.org/x", "application/json; charset=utf-8", b"{}", "json")])
def test_web_format_detection(url, ctype, body, fmt):
    assert _guess_format(url, ctype, body) == fmt


def test_one_value_plus_blanks_is_not_constant():
    assert column_role(pd.Series(["damaged"] * 3 + [None] * 20), "return_reason") == "categorical"
    assert column_role(pd.Series(["x"] * 10), "c") == "constant"


def test_negative_fence_on_non_negative_data_means_no_lower_limit():
    s = pd.DataFrame({"price": [1, 2, 3, 4, 5, 6, 7, 8, 500.0]})
    rep = detect(s, OutlierConfig(method="iqr", iqr_k=3))
    lo, hi = rep.bounds["price"]
    assert lo == -np.inf and np.isfinite(hi) and rep.per_column["price"] == 1
    row = rep.summary_table().iloc[0]
    assert row["lower bound"] == "none" and "no lower limit" in row["note"]
    neg = detect(pd.DataFrame({"t": [-5, -1, 0, 1, 2, 3, 4, 40.0]}), OutlierConfig(method="iqr", iqr_k=3))
    assert np.isfinite(neg.bounds["t"][0])                                         # columns that can be negative keep their fence


def test_change_dataset_popover_is_reset_after_opening(app):
    open_sample(app, "house_prices.csv")
    nav(app, "ML Studio")
    before = app.session_state.get("_pop_n", 0)
    app.radio(key="pk_src_bar_ml").set_value("Sample data").run()
    app.selectbox(key="pk_smp_bar_ml").set_value("Customer churn (classification demo)").run()
    click(app, "", key="pk_osmp_bar_ml")
    assert app.session_state["handle"].name.startswith("Customer churn") and app.session_state["_pop_n"] == before + 1


def test_local_time_formatting(monkeypatch):
    class Ctx:
        timezone = "Asia/Kolkata"
    monkeypatch.setattr(theme.st, "context", Ctx())
    assert theme.local_time("2026-10-29T06:00:00+00:00") == "2026-10-29 11:30"
    Ctx.timezone = None
    assert theme.local_time("2026-10-29T06:00:00+00:00") == "2026-10-29 06:00 UTC"
    assert theme.local_time("") == ""


def test_api_imports_are_labelled_and_named_from_the_url(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXORA_DATA_DIR", str(tmp_path / "d"))
    from nexora.connectors import web_api as W
    from nexora.workflow import Workspace

    class FakeResult:
        frame = pd.DataFrame({"a": [1, 2]})
        source = "https://example.org/files/sales_2025.csv"
    monkeypatch.setattr(W, "fetch_api", lambda cfg, allow_private=False: FakeResult())
    h, _ = Workspace().ingest_api(W.APIConfig("https://example.org/files/sales_2025.csv"))
    assert h.name == "sales_2025" and h.source == "web_api"
    from nexora.ui.pages.datasets import SOURCE_LABELS
    assert SOURCE_LABELS["web_api"] == "web API" and SOURCE_LABELS["combined"] == "combined"
