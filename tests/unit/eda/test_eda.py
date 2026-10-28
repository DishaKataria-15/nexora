import numpy as np
import pandas as pd
import pytest

from nexora.eda import charts as C
from nexora.eda import profiling as P


@pytest.fixture()
def mixed():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "id": [f"R{i}" for i in range(n)], "num": rng.normal(50, 10, n), "skewed": rng.exponential(2, n) ** 3, "const": 7, "empty": np.nan,
        "cat": rng.choice(["a", "b", "c"], n), "hicard": [f"v{i % 90}" for i in range(n)], "flag": rng.choice(["yes", "no"], n),
        "when": pd.date_range("2026-01-01", periods=n), "num2": rng.normal(0, 1, n)})
    df["num3"] = 0.8 * df["num"] + rng.normal(0, 4, n)
    return df


def test_column_roles(mixed):
    r = P.roles(mixed)
    assert r["id"] == "identifier" and r["num"] == "numeric" and r["const"] == "constant" and r["empty"] == "empty" and r["cat"] == "categorical"
    assert r["when"] == "datetime" and r["flag"] == "boolean"


def test_overview_and_summary_are_computed_from_the_data(mixed):
    ov = P.dataset_overview(mixed)
    assert ov["rows"] == 200 and ov["columns"] == 11 and ov["missing_cells"] == 200 and ov["duplicate_rows"] == 0
    summ = P.column_summary(mixed).set_index("column")
    assert summ.loc["num", "mean"] == pytest.approx(mixed["num"].mean()) and summ.loc["empty", "missing %"] == 100.0
    assert summ.loc["when", "earliest"].startswith("2026-01-01")


def test_observations_cite_real_numbers(mixed):
    texts = " ".join(o.text for o in P.observations(mixed))
    assert "'const' has the same value" in texts and "skewed" in texts and "'hicard' has 90 distinct" in texts
    assert P.observations(pd.DataFrame())[0].severity == "warning"
    assert "Only 5 rows" in P.observations(pd.DataFrame({"a": [1, 2, 3, 4, 5]}))[0].text


def test_correlation_matches_pandas_and_never_claims_cause():
    rng = np.random.RandomState(1)
    x = rng.normal(size=100)
    df = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 0.1, 100), "z": rng.normal(size=100)})
    c = P.correlation_matrix(df)
    assert c.loc["x", "y"] == pytest.approx(df["x"].corr(df["y"]))
    top = P.top_correlations(c, 1).iloc[0]
    assert {top["column A"], top["column B"]} == {"x", "y"}
    obs = [o.text for o in P.observations(df) if o.kind == "correlation"]
    assert obs and "not cause" in obs[0]
    assert P.correlation_matrix(df[["x"]]).empty and P.top_correlations(pd.DataFrame()).empty


def test_outlier_summary_flags_but_does_not_remove():
    df = pd.DataFrame({"v": list(range(30)) + [1000]})
    o = P.outlier_summary(df)
    assert o.loc[0, "outliers"] == 1 and len(df) == 31 and P.outlier_summary(pd.DataFrame({"c": [1] * 20})).empty


# ---- charts on awkward data -----------------------------------------------------------------------------------------------------------
def test_recommendations_skip_unusable_columns_and_each_has_a_reason(mixed):
    specs = C.recommend_charts(mixed)
    used = {c for s in specs for c in (s.x, s.y, s.color, *s.columns) if c}
    assert not ({"id", "const", "empty"} & used) and all(s.reason and s.title for s in specs)
    assert {"histogram", "bar", "line", "heatmap", "scatter"} <= {s.kind for s in specs}
    assert C.recommend_charts(pd.DataFrame()) == [] and C.recommend_charts(mixed[["id", "const", "empty"]]) == []
    ts = C.recommend_charts(mixed.assign(target=lambda d: d["cat"]), target="target")
    assert ts[0].kind == "bar" and ts[0].x == "target"


def test_every_recommended_chart_actually_builds(mixed):
    for s in C.recommend_charts(mixed):
        assert len(C.build_figure(mixed, s).data) >= 1, s


@pytest.mark.parametrize("spec,msg", [
    (C.ChartSpec("histogram", x="const"), "same value"), (C.ChartSpec("histogram", x="empty"), "no values"), (C.ChartSpec("histogram", x="cat"), "not numeric"),
    (C.ChartSpec("histogram", x="nope"), "not in the data"), (C.ChartSpec("scatter", x="num", y="const"), "same value"),
    (C.ChartSpec("bar", x="empty"), "no values"), (C.ChartSpec("heatmap", columns=["num"]), "At least two"), (C.ChartSpec("pie", x="cat"), "Unknown chart")])
def test_unplottable_charts_raise_readable_errors_not_crashes(mixed, spec, msg):
    with pytest.raises(C.ChartError, match=msg):
        C.build_figure(mixed, spec)


def test_empty_frame_and_filtered_to_nothing(mixed):
    with pytest.raises(C.ChartError, match="no rows"):
        C.build_figure(mixed.iloc[0:0], C.ChartSpec("histogram", x="num"))
    assert len(C.apply_filters(mixed, [{"column": "num", "type": "range", "min": 1e9, "max": 2e9}])) == 0


def test_high_cardinality_bar_groups_the_tail_and_reports_it(mixed):
    fig = C.build_figure(mixed, C.ChartSpec("bar", x="hicard"))
    labels = list(fig.data[0].x)
    assert len(labels) == C.TOP_N + 1 and labels[-1].startswith("Other (") and "75 values" in labels[-1]
    assert sum(fig.data[0].y) == mixed["hicard"].notna().sum()                      # every row is accounted for


def test_mixed_type_column_is_used_numerically_only_if_nearly_all_numeric():
    C.build_figure(pd.DataFrame({"m": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] * 3 + ["x"]}), C.ChartSpec("histogram", x="m"))
    with pytest.raises(C.ChartError, match="not numeric"):
        C.build_figure(pd.DataFrame({"m": ["1", "a", "b", "c"] * 5}), C.ChartSpec("histogram", x="m"))


def test_scatter_samples_large_data_and_says_so():
    big = pd.DataFrame({"a": np.arange(20000.0), "b": np.arange(20000.0) ** 0.5})
    fig = C.build_figure(big, C.ChartSpec("scatter", x="a", y="b"))
    assert len(fig.data[0].x) == C.MAX_SCATTER_POINTS and "random sample" in fig.layout.title.text


def test_time_series_aggregation_values_are_correct():
    d = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=10), "v": np.arange(10.0)})
    fig = C.build_figure(d, C.ChartSpec("line", x="t", y="v", agg="sum"))
    assert list(fig.data[0].y) == list(np.arange(10.0)) and "daily" in fig.layout.yaxis.title.text
    with pytest.raises(C.ChartError, match="single time period|Fewer than two"):
        C.build_figure(d.iloc[:1], C.ChartSpec("line", x="t", y="v"))


def test_heatmap_honours_correlation_method():
    d = pd.DataFrame({"a": np.arange(30.0), "b": np.arange(30.0) ** 3, "c": np.sin(np.arange(30.0))})
    p = C.build_figure(d, C.ChartSpec("heatmap", columns=["a", "b", "c"], agg="pearson")).data[0].z
    s = C.build_figure(d, C.ChartSpec("heatmap", columns=["a", "b", "c"], agg="spearman")).data[0].z
    assert s[0][1] == pytest.approx(1.0) and p[0][1] < 0.99


def test_filters_category_range_date_and_describe(mixed):
    f = [{"column": "cat", "type": "category", "values": ["a"]}, {"column": "num", "type": "range", "min": 40, "max": 60},
         {"column": "when", "type": "date", "start": "2026-02-01", "end": "2026-03-01"}]
    out = C.apply_filters(mixed, f)
    assert set(out["cat"]) <= {"a"} and out["num"].between(40, 60).all() and out["when"].between("2026-02-01", "2026-03-01 23:59:59").all() and 0 < len(out) < len(mixed)
    assert "cat in a" in C.describe_filters(f) and C.describe_filters([]) == "none"
    assert len(C.apply_filters(mixed, [{"column": "ghost", "type": "category", "values": []}])) == len(mixed)


def test_dashboard_export_is_self_contained_escaped_and_complete(mixed):
    figs = [("Chart <b>1</b>", C.build_figure(mixed, C.ChartSpec("histogram", x="num")), "note & more"), ("Two", C.build_figure(mixed, C.ChartSpec("bar", x="cat")), "")]
    html = C.dashboard_html("Dash <script>", figs, "200 of 200 rows", "none")
    assert html.count("plotly-graph-div") == 2 and "&lt;script&gt;" in html and "&lt;b&gt;1&lt;/b&gt;" in html
    assert 'src="http' not in html and "correlation does not imply causation" in html.lower()
    spec = C.ChartSpec("scatter", x="num", y="num2", color="cat", title="t", reason="r")
    assert C.ChartSpec.from_dict(spec.to_dict()).key() == spec.key()
