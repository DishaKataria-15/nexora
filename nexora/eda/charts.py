"""Chart recommendation and construction. Every figure is computed from the supplied data."""
from __future__ import annotations

import html
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import profiling as P

NAVY, TEAL, SLATE, AMBER, MUTED = "#14213d", "#0f766e", "#64748b", "#b45309", "#cbd5e1"
COLORWAY = [TEAL, "#1e3a5f", SLATE, AMBER, "#7c3aed", "#0369a1", "#9f1239", "#4d7c0f"]
# Colour palettes for charts and dashboards: a main colour for single-series charts, a sequence for groups,
# and the negative end of the correlation scale. "Colour-blind safe" is the Okabe-Ito palette.
PALETTES: dict[str, dict] = {
    "Nexora (teal)": {"primary": TEAL, "colors": COLORWAY, "negative": "#9f1239"},
    "Ocean blues": {"primary": "#1d4ed8", "colors": ["#1d4ed8", "#0ea5e9", "#0f766e", "#6366f1", "#0369a1", "#14b8a6", "#1e3a8a", "#38bdf8"], "negative": "#b45309"},
    "Sunset": {"primary": "#c2410c", "colors": ["#c2410c", "#be123c", "#d97706", "#7c2d12", "#db2777", "#a16207", "#9f1239", "#ea580c"], "negative": "#1e3a8a"},
    "Forest": {"primary": "#15803d", "colors": ["#15803d", "#65a30d", "#a16207", "#166534", "#4d7c0f", "#854d0e", "#14532d", "#84cc16"], "negative": "#7c2d12"},
    "Colour-blind safe": {"primary": "#0072B2", "colors": ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"], "negative": "#D55E00"},
    "Greyscale (print)": {"primary": "#334155", "colors": ["#0f172a", "#475569", "#94a3b8", "#1e293b", "#64748b", "#cbd5e1", "#334155", "#e2e8f0"], "negative": "#e2e8f0"},
}
DEFAULT_PALETTE = "Nexora (teal)"
MAX_SCATTER_POINTS = 5000
TOP_N = 15


class ChartError(ValueError):
    """The chart cannot be drawn from this data. The message is safe to show to the user."""


@dataclass
class ChartSpec:
    kind: str                     # histogram | box | bar | scatter | line | heatmap
    x: str | None = None
    y: str | None = None
    color: str | None = None
    agg: str = "count"            # count | mean | sum | median (bar and line)
    title: str = ""
    reason: str = ""
    columns: list[str] = field(default_factory=list)   # heatmap columns
    spec_id: str = ""
    accent: str | None = None     # optional main colour for this chart (hex); otherwise the palette's main colour

    def key(self) -> str:
        return "|".join(str(v) for v in (self.kind, self.x, self.y, self.color, self.agg, ",".join(self.columns)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ChartSpec":
        return ChartSpec(**{k: v for k, v in d.items() if k in ChartSpec.__dataclass_fields__})


def palette(name: str | None) -> dict:
    return PALETTES.get(name or DEFAULT_PALETTE, PALETTES[DEFAULT_PALETTE])


def _layout(fig: go.Figure, title: str, height: int = 380, colorway: list[str] | None = None) -> go.Figure:
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": NAVY}, "x": 0.0, "xanchor": "left"},
        font={"family": "Segoe UI, Inter, Arial, sans-serif", "size": 12, "color": NAVY},
        paper_bgcolor="white", plot_bgcolor="white", colorway=colorway or COLORWAY, height=height,
        margin={"l": 55, "r": 20, "t": 55, "b": 50}, legend={"orientation": "h", "y": -0.2},
    )
    fig.update_xaxes(showgrid=False, linecolor=MUTED, ticks="outside", tickcolor=MUTED, automargin=True)
    fig.update_yaxes(gridcolor="#e8edf3", zeroline=False, linecolor=MUTED, automargin=True)
    return fig


def _need(df: pd.DataFrame, *cols: str | None) -> None:
    for c in cols:
        if c and c not in df.columns:
            raise ChartError(f"Column '{c}' is not in the data.")


def _numeric(df: pd.DataFrame, c: str) -> pd.Series:
    s = df[c]
    if not P._is_numeric(s):
        conv = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() and conv.notna().sum() / max(s.notna().sum(), 1) >= 0.9:
            s = conv
        else:
            raise ChartError(f"'{c}' is not numeric, so it cannot be used on a numeric axis.")
    s = s.dropna().astype(float)
    s = s[np.isfinite(s)]
    if s.empty:
        raise ChartError(f"'{c}' has no values to plot.")
    return s


def _top_categories(s: pd.Series, n: int = TOP_N) -> pd.Series:
    """Keep the n most frequent values; group the rest as 'Other (k values)'."""
    s = s.astype(str)
    vc = s.value_counts()
    if len(vc) <= n:
        return s
    keep = set(vc.index[:n])
    other = len(vc) - n
    return s.where(s.isin(keep), f"Other ({other} values)")


def build_figure(df: pd.DataFrame, spec: ChartSpec, palette_name: str | None = None) -> go.Figure:
    """Draw ``spec`` from ``df``. Colours come from the named palette; ``spec.accent`` overrides the main colour."""
    pal = palette(palette_name)
    main = spec.accent or pal["primary"]
    seq = ([main] + [c for c in pal["colors"] if c.lower() != main.lower()]) if spec.accent else list(pal["colors"])
    neg = pal["negative"]

    def lay(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
        return _layout(fig, title, height, seq)
    k = spec.kind
    _need(df, spec.x, spec.y, spec.color, *spec.columns)
    if len(df) == 0:
        raise ChartError("There are no rows to plot (check your filters).")
    title = spec.title or default_title(spec)
    if k == "histogram":
        s = _numeric(df, spec.x)
        if s.nunique() == 1:
            raise ChartError(f"'{spec.x}' has the same value in every row ({s.iloc[0]:g}).")
        nb = int(min(60, max(8, len(np.histogram_bin_edges(s, bins="auto")) - 1)))
        fig = go.Figure(go.Histogram(x=s, nbinsx=nb, marker={"color": main, "line": {"color": "white", "width": 0.5}}, hovertemplate="%{x}: %{y} rows<extra></extra>"))
        fig.add_vline(x=float(s.median()), line={"color": NAVY, "dash": "dash", "width": 1}, annotation_text=f"median {s.median():.4g}", annotation_position="top")
        fig.update_xaxes(title=spec.x)
        fig.update_yaxes(title="rows")
        return lay(fig, title)
    if k == "box":
        if spec.x:
            cat = _top_categories(df[spec.x].dropna(), 12)
            y = pd.to_numeric(df.loc[cat.index, spec.y], errors="coerce")
            d = pd.DataFrame({"g": cat, "y": y}).dropna()
            if d.empty:
                raise ChartError("No rows have both values.")
            fig = go.Figure()
            for i, (g, part) in enumerate(d.groupby("g", sort=False)):
                fig.add_trace(go.Box(y=part["y"], name=str(g), marker={"color": seq[i % len(seq)]}, boxmean=False))
            fig.update_layout(showlegend=False)
            fig.update_xaxes(title=spec.x)
        else:
            s = _numeric(df, spec.y)
            if s.nunique() == 1:
                raise ChartError(f"'{spec.y}' has the same value in every row.")
            fig = go.Figure(go.Box(y=s, name=spec.y, marker={"color": main}, boxpoints="outliers"))
            fig.update_layout(showlegend=False)
        fig.update_yaxes(title=spec.y)
        return lay(fig, title)
    if k == "bar":
        if df[spec.x].notna().sum() == 0:
            raise ChartError(f"'{spec.x}' has no values to plot.")
        cat = _top_categories(df[spec.x].dropna())
        if spec.agg == "count" or not spec.y:
            agg = cat.value_counts().sort_values(ascending=False)
            ylab = "rows"
            vals = agg.values
            share = 100 * vals / max(vals.sum(), 1)
            text = [f"{v:,} ({s:.0f}%)" for v, s in zip(vals, share)]
        else:
            num = pd.to_numeric(df.loc[cat.index, spec.y], errors="coerce")
            g = pd.DataFrame({"g": cat, "y": num}).dropna().groupby("g")["y"].agg(spec.agg).sort_values(ascending=False)
            if g.empty:
                raise ChartError("No rows have both values.")
            agg, vals, ylab = g, g.values, f"{spec.agg} of {spec.y}"
            text = [f"{v:,.4g}" for v in vals]
        other = [i for i in agg.index if str(i).startswith("Other (")]
        if other:   # the grouped tail always goes last, however many rows it holds
            order = [i for i in agg.index if i not in other] + other
            pos = [list(agg.index).index(i) for i in order]
            agg, vals, text = agg.iloc[pos], np.asarray(vals)[pos], [text[j] for j in pos]
        colors = [MUTED if i in other else main for i in agg.index]
        fig = go.Figure(go.Bar(x=[str(i) for i in agg.index], y=vals, marker={"color": colors}, text=text, textposition="outside", cliponaxis=False))
        fig.update_xaxes(title=spec.x, type="category")
        fig.update_yaxes(title=ylab)
        return lay(fig, title)
    if k == "scatter":
        x, y = pd.to_numeric(df[spec.x], errors="coerce"), pd.to_numeric(df[spec.y], errors="coerce")
        d = pd.DataFrame({"x": x, "y": y})
        if spec.color:
            d["c"] = _top_categories(df[spec.color].astype(object).where(df[spec.color].notna(), "(missing)"), 8)
        d = d.dropna(subset=["x", "y"])
        if len(d) < 2:
            raise ChartError("Fewer than two rows have both values.")
        if d["x"].nunique() == 1 or d["y"].nunique() == 1:
            raise ChartError("One of the two columns has the same value in every row.")
        note = ""
        if len(d) > MAX_SCATTER_POINTS:
            d = d.sample(MAX_SCATTER_POINTS, random_state=0)
            note = f" (random sample of {MAX_SCATTER_POINTS:,} points)"
        fig = go.Figure()
        if spec.color:
            for i, (g, part) in enumerate(d.groupby("c", sort=False)):
                fig.add_trace(go.Scattergl(x=part["x"], y=part["y"], mode="markers", name=str(g), marker={"size": 5, "opacity": 0.65, "color": seq[i % len(seq)]}))
        else:
            fig.add_trace(go.Scattergl(x=d["x"], y=d["y"], mode="markers", marker={"size": 5, "opacity": 0.55, "color": main}, showlegend=False))
        fig.update_xaxes(title=spec.x)
        fig.update_yaxes(title=spec.y)
        return lay(fig, title + note)
    if k == "line":
        t = df[spec.x]
        if not pd.api.types.is_datetime64_any_dtype(t):
            t = pd.to_datetime(t, errors="coerce")
        d = pd.DataFrame({"t": t})
        d["y"] = pd.to_numeric(df[spec.y], errors="coerce") if spec.y else 1.0
        d = d.dropna(subset=["t"]).sort_values("t")
        if spec.y:
            d = d.dropna(subset=["y"])
        if len(d) < 2:
            raise ChartError("Fewer than two dated rows are available.")
        span = (d["t"].max() - d["t"].min()).days
        rule, label = ("D", "daily") if span <= 90 else (("W", "weekly") if span <= 730 else ("MS", "monthly"))
        agg = "count" if not spec.y else ("sum" if spec.agg == "sum" else ("median" if spec.agg == "median" else "mean"))
        series = d.set_index("t")["y"].resample(rule).agg("size" if agg == "count" else agg)
        series = series[series.notna()]
        if len(series) < 2:
            raise ChartError("All rows fall into a single time period.")
        fig = go.Figure(go.Scatter(x=series.index, y=series.values, mode="lines", line={"color": main, "width": 2}))
        fig.update_xaxes(title=spec.x)
        fig.update_yaxes(title=(f"{agg} of {spec.y}" if spec.y else "rows") + f" ({label})")
        return lay(fig, title)
    if k == "heatmap":
        cols = spec.columns or P.numeric_columns(df)
        cols = [c for c in cols if c in df.columns and P._is_numeric(df[c]) and df[c].dropna().nunique() > 1][:25]
        if len(cols) < 2:
            raise ChartError("At least two numeric columns that vary are needed for a correlation heatmap.")
        method = spec.agg if spec.agg in ("pearson", "spearman") else "pearson"
        corr = df[cols].astype(float).corr(method=method)
        z = corr.values
        fig = go.Figure(go.Heatmap(z=z, x=cols, y=cols, zmin=-1, zmax=1, colorscale=[[0, neg], [0.5, "#f8fafc"], [1, main]],
                                   text=np.round(z, 2), texttemplate="%{text}" if len(cols) <= 12 else None, hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
                                   colorbar={"title": "r"}))
        fig.update_yaxes(autorange="reversed")
        return lay(fig, title, height=max(380, 28 * len(cols) + 140))
    raise ChartError(f"Unknown chart type '{k}'.")


def default_title(s: ChartSpec) -> str:
    return {"histogram": f"Distribution of {s.x}", "box": f"{s.y}" + (f" by {s.x}" if s.x else " (spread and outliers)"),
            "bar": (f"{s.agg.title()} of {s.y} by {s.x}" if s.y and s.agg != "count" else f"Rows per {s.x}"),
            "scatter": f"{s.y} vs {s.x}", "line": (f"{s.y} over time" if s.y else "Rows over time"),
            "heatmap": "Correlation between numeric columns"}.get(s.kind, s.kind)


def recommend_charts(df: pd.DataFrame, target: str | None = None, max_charts: int = 14) -> list[ChartSpec]:
    """Suggest charts from column types only; every suggestion states why."""
    specs: list[ChartSpec] = []
    if len(df) == 0:
        return specs
    nums = P.numeric_columns(df)
    cats = [c for c in P.categorical_columns(df) if df[c].nunique() <= 30]
    dts = P.datetime_columns(df)
    if target and target in df.columns:
        if target in nums and df[target].nunique() > 10:
            specs.append(ChartSpec("histogram", x=target, reason="The value you want to predict: check its shape and range first."))
        elif df[target].nunique() <= 30:
            specs.append(ChartSpec("bar", x=target, reason="The classes you want to predict: check whether they are balanced."))
    for c in nums[:4]:
        if c != target:
            specs.append(ChartSpec("histogram", x=c, reason="Shows the shape and range of a numeric column."))
    osum = P.outlier_summary(df)
    if not osum.empty:
        for c in osum.sort_values("outliers", ascending=False).head(2)["column"]:
            specs.append(ChartSpec("box", y=c, reason="Shows spread and unusual values (review signals, not errors)."))
    for c in cats[:3]:
        if c != target:
            specs.append(ChartSpec("bar", x=c, reason="Compares how many rows fall in each category."))
    tc = P.top_correlations(P.correlation_matrix(df), 2)
    for _, r in tc[tc["correlation"].abs() >= 0.3].iterrows():
        specs.append(ChartSpec("scatter", x=r["column A"], y=r["column B"], reason=f"These columns are associated (r = {r['correlation']:.2f}); association is not cause."))
    if dts:
        vals = [c for c in nums if c != target][:2]
        if vals:
            for v in vals:
                specs.append(ChartSpec("line", x=dts[0], y=v, agg="mean", reason="Shows how the value changes over time."))
        else:
            specs.append(ChartSpec("line", x=dts[0], reason="Shows how many rows fall in each period."))
    if cats and nums:
        c = next((c for c in cats if df[c].nunique() <= 12 and c != target), None)
        n = next((n for n in nums if n != target), None)
        if c and n:
            specs.append(ChartSpec("box", x=c, y=n, reason="Compares a numeric column across categories."))
    if len(nums) >= 3:
        specs.append(ChartSpec("heatmap", columns=nums[:25], reason="Shows which numeric columns move together."))
    seen, out = set(), []
    for s in specs:
        if s.key() not in seen:
            seen.add(s.key())
            s.title = s.title or default_title(s)
            s.spec_id = f"c{len(out) + 1}"
            out.append(s)
    return out[:max_charts]


# ----------------------------------------------------------------------------
# Filters and dashboard export
# ----------------------------------------------------------------------------

def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
    """filters: {'column', 'type': 'category'|'range'|'date', 'values'|'min','max'|'start','end'}"""
    mask = pd.Series(True, index=df.index)
    for f in filters or []:
        c = f.get("column")
        if c not in df.columns:
            continue
        s = df[c]
        if f["type"] == "category" and f.get("values") is not None:
            mask &= s.astype(str).isin([str(v) for v in f["values"]])
        elif f["type"] == "range":
            num = pd.to_numeric(s, errors="coerce")
            mask &= num.between(f["min"], f["max"]).fillna(False)
        elif f["type"] == "date":
            t = pd.to_datetime(s, errors="coerce")
            mask &= ((t >= pd.Timestamp(f["start"])) & (t <= pd.Timestamp(f["end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))).fillna(False)
    return df[mask]


def describe_filters(filters: list[dict[str, Any]]) -> str:
    parts = []
    for f in filters or []:
        if f["type"] == "category":
            vals = list(f.get("values") or [])
            parts.append(f"{f['column']} in {', '.join(map(str, vals[:5]))}{'...' if len(vals) > 5 else ''}")
        elif f["type"] == "range":
            parts.append(f"{f['column']} between {f['min']:g} and {f['max']:g}")
        elif f["type"] == "date":
            parts.append(f"{f['column']} from {str(f['start'])[:10]} to {str(f['end'])[:10]}")
    return "; ".join(parts) or "none"


def dashboard_html(title: str, items: list[tuple[str, go.Figure, str]], dataset_note: str, filter_note: str) -> str:
    """Self-contained HTML dashboard. items = (title, figure, one-line interpretation)."""
    parts = []
    for i, (t, fig, note) in enumerate(items):
        div = fig.to_html(full_html=False, include_plotlyjs="inline" if i == 0 else False)
        parts.append(f"<section><h2>{html.escape(t)}</h2>{div}<p class='note'>{html.escape(note)}</p></section>")
    css = ("body{font-family:Segoe UI,Arial,sans-serif;color:#14213d;background:#fbfaf8;margin:0;padding:24px 32px}"
           "h1{font-size:22px;margin:0 0 4px}.meta{color:#475569;font-size:13px;margin-bottom:18px}"
           "section{background:#fff;border:1px solid #dbe2ea;border-radius:6px;padding:12px 16px;margin-bottom:16px}"
           "h2{font-size:15px;margin:4px 0}.note{color:#475569;font-size:12.5px;margin:2px 0 6px}")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>"
            f"<h1>{html.escape(title)}</h1><div class='meta'>{html.escape(dataset_note)}<br>Filters applied: {html.escape(filter_note)}. "
            "Generated by Nexora from the data shown. Correlation does not imply causation.</div>" + "".join(parts) + "</body></html>")
