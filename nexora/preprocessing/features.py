"""Automated feature engineering.

:class:`FeatureGenerator` is a scikit-learn transformer that takes the (coerced) feature DataFrame and returns it with
extra, clearly named columns. Anything that is *learned* from data (bin edges, frequency tables, whether a column
needs a signed log, missing-value indicators) is learned in ``fit`` on training rows and applied unchanged later.

Generated column names are known before fitting (:func:`plan_features`), so the rest of the pipeline can be built
around them, and the number of generated columns is capped to prevent feature explosion.
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MAX_GENERATED = 200
MAX_POLY_COLUMNS = 6
MAX_POLY_DEGREE = 3
DATE_PARTS = ["year", "quarter", "month", "week", "day", "dayofweek", "dayofyear", "hour", "minute", "weekend", "days_since_start"]
DEFAULT_DATE_PARTS = ["year", "month", "dayofweek", "dayofyear", "days_since_start"]
CYCLES = {"month": 12, "dayofweek": 7, "hour": 24, "dayofyear": 365.25}


class FeatureConfigError(ValueError):
    """The feature-engineering settings are not valid. The message is safe to show to the user."""


@dataclass
class FeatureConfig:
    log: list[str] = field(default_factory=list)
    square: list[str] = field(default_factory=list)
    poly_columns: list[str] = field(default_factory=list)
    poly_degree: int = 2
    interactions: list[list[str]] = field(default_factory=list)
    ratios: list[list[str]] = field(default_factory=list)
    differences: list[list[str]] = field(default_factory=list)
    bins: list[str] = field(default_factory=list)
    n_bins: int = 5
    quantile_rank: list[str] = field(default_factory=list)
    missing_indicators: list[str] = field(default_factory=list)
    frequency_encode: list[str] = field(default_factory=list)
    text_stats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "FeatureConfig":
        d = dict(d or {})
        return FeatureConfig(**{k: v for k, v in d.items() if k in FeatureConfig.__dataclass_fields__})

    def is_empty(self) -> bool:
        return not any([self.log, self.square, self.poly_columns, self.interactions, self.ratios, self.differences, self.bins,
                        self.quantile_rank, self.missing_indicators, self.frequency_encode, self.text_stats])


@dataclass
class GeneratedFeature:
    name: str
    kind: str          # numeric | categorical
    description: str


def _pairs(v: list) -> list[tuple[str, str]]:
    out = []
    for p in v or []:
        if not isinstance(p, (list, tuple)) or len(p) != 2 or p[0] == p[1]:
            raise FeatureConfigError(f"Each pair must name two different columns; got {p!r}.")
        out.append((str(p[0]), str(p[1])))
    return out


def plan_features(cfg: FeatureConfig, numeric: list[str], categorical: list[str], text: list[str], all_columns: list[str]) -> list[GeneratedFeature]:
    """Validate the configuration against the available columns and return the columns it will create (in order)."""
    nums, cats, txt = set(numeric), set(categorical), set(text)
    out: list[GeneratedFeature] = []

    def need(cols: list[str], allowed: set[str], what: str) -> None:
        bad = [c for c in cols if c not in allowed]
        if bad:
            raise FeatureConfigError(f"{what} needs {('numeric' if allowed is nums else 'categorical' if allowed is cats else 'text')} columns; "
                                     f"not usable: {', '.join(bad)}.")

    need(cfg.log, nums, "Log transform")
    out += [GeneratedFeature(f"log({c})", "numeric", f"log(1 + {c}); a signed log is used if training values are negative") for c in cfg.log]
    need(cfg.square, nums, "Squares")
    out += [GeneratedFeature(f"{c}^2", "numeric", f"{c} squared") for c in cfg.square]
    if cfg.poly_columns:
        need(cfg.poly_columns, nums, "Polynomial features")
        if len(cfg.poly_columns) > MAX_POLY_COLUMNS:
            raise FeatureConfigError(f"Polynomial features are limited to {MAX_POLY_COLUMNS} columns.")
        if not 2 <= int(cfg.poly_degree) <= MAX_POLY_DEGREE:
            raise FeatureConfigError(f"Polynomial degree must be between 2 and {MAX_POLY_DEGREE}.")
        for d in range(2, int(cfg.poly_degree) + 1):
            for combo in itertools.combinations_with_replacement(sorted(cfg.poly_columns), d):
                name = "*".join(combo)
                out.append(GeneratedFeature(f"poly({name})", "numeric", f"product of {', '.join(combo)}"))
    for a, b in _pairs(cfg.interactions):
        need([a, b], nums, "Interactions")
        out.append(GeneratedFeature(f"{a}*{b}", "numeric", f"{a} multiplied by {b}"))
    for a, b in _pairs(cfg.ratios):
        need([a, b], nums, "Ratios")
        out.append(GeneratedFeature(f"{a}/{b}", "numeric", f"{a} divided by {b}; missing where {b} is zero"))
    for a, b in _pairs(cfg.differences):
        need([a, b], nums, "Differences")
        out.append(GeneratedFeature(f"{a}-{b}", "numeric", f"{a} minus {b}"))
    if cfg.bins:
        need(cfg.bins, nums, "Binning")
        if not 2 <= int(cfg.n_bins) <= 20:
            raise FeatureConfigError("The number of bins must be between 2 and 20.")
        out += [GeneratedFeature(f"bin({c})", "categorical", f"{c} grouped into {cfg.n_bins} equal-count bins learned from training rows") for c in cfg.bins]
    need(cfg.quantile_rank, nums, "Quantile features")
    out += [GeneratedFeature(f"rank({c})", "numeric", f"position of {c} within the training distribution (0 to 1)") for c in cfg.quantile_rank]
    unknown = [c for c in cfg.missing_indicators if c not in all_columns]
    if unknown:
        raise FeatureConfigError(f"Missing-value indicators refer to unknown columns: {', '.join(unknown)}.")
    out += [GeneratedFeature(f"missing({c})", "numeric", f"1 if {c} is missing, else 0") for c in cfg.missing_indicators]
    need(cfg.frequency_encode, cats, "Frequency encoding")
    out += [GeneratedFeature(f"freq({c})", "numeric", f"share of training rows with the same {c} value (0 for unseen values)") for c in cfg.frequency_encode]
    need(cfg.text_stats, txt | cats, "Text statistics")
    for c in cfg.text_stats:
        out += [GeneratedFeature(f"len({c})", "numeric", f"number of characters in {c}"), GeneratedFeature(f"words({c})", "numeric", f"number of words in {c}")]
    names = [g.name for g in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise FeatureConfigError(f"These features would be created twice: {', '.join(sorted(dup))}.")
    clash = [n for n in names if n in all_columns]
    if clash:
        raise FeatureConfigError(f"Generated names clash with existing columns: {', '.join(clash)}.")
    if len(out) > MAX_GENERATED:
        raise FeatureConfigError(f"This would create {len(out)} features; the limit is {MAX_GENERATED}. Select fewer columns or a lower degree.")
    return out


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _finite(s: pd.Series) -> pd.Series:
    return s.where(np.isfinite(s), np.nan)


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """Adds the features described by a :class:`FeatureConfig` dictionary. Input and output are DataFrames."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config

    def fit(self, X, y=None):
        cfg = FeatureConfig.from_dict(self.config)
        X = pd.DataFrame(X)
        self.signed_log_ = {c: bool((_num(X[c]).dropna() < 0).any()) for c in cfg.log}
        self.bin_edges_ = {}
        for c in cfg.bins:
            v = _num(X[c]).dropna()
            edges = np.unique(np.quantile(v, np.linspace(0, 1, int(cfg.n_bins) + 1))) if len(v) else np.array([0.0, 1.0])
            self.bin_edges_[c] = edges[1:-1] if len(edges) > 2 else np.array([])
        self.sorted_ = {c: np.sort(_num(X[c]).dropna().values) for c in cfg.quantile_rank}
        self.freq_ = {}
        for c in cfg.frequency_encode:
            vc = X[c].astype(object).where(X[c].notna(), "(missing)").astype(str).value_counts(normalize=True)
            self.freq_[c] = vc.to_dict()
        self.n_rows_fit_ = len(X)
        return self

    def transform(self, X):
        cfg = FeatureConfig.from_dict(self.config)
        X = pd.DataFrame(X)
        new: dict[str, pd.Series] = {}
        for c in cfg.log:
            v = _num(X[c])
            new[f"log({c})"] = np.sign(v) * np.log1p(v.abs()) if self.signed_log_.get(c) else np.log1p(v.clip(lower=0))
        for c in cfg.square:
            new[f"{c}^2"] = _num(X[c]) ** 2
        if cfg.poly_columns:
            for d in range(2, int(cfg.poly_degree) + 1):
                for combo in itertools.combinations_with_replacement(sorted(cfg.poly_columns), d):
                    prod = pd.Series(1.0, index=X.index)
                    for c in combo:
                        prod = prod * _num(X[c])
                    new[f"poly({'*'.join(combo)})"] = prod
        for a, b in _pairs(cfg.interactions):
            new[f"{a}*{b}"] = _num(X[a]) * _num(X[b])
        for a, b in _pairs(cfg.ratios):
            den = _num(X[b])
            new[f"{a}/{b}"] = _num(X[a]) / den.where(den != 0)
        for a, b in _pairs(cfg.differences):
            new[f"{a}-{b}"] = _num(X[a]) - _num(X[b])
        for c in cfg.bins:
            v = _num(X[c])
            idx = np.digitize(v.fillna(0).values, self.bin_edges_[c]).astype(float)
            new[f"bin({c})"] = pd.Series(np.where(v.isna(), np.nan, idx), index=X.index).map(lambda x: np.nan if pd.isna(x) else f"bin{int(x)}").astype(object)
        for c in cfg.quantile_rank:
            v = _num(X[c])
            ref = self.sorted_[c]
            r = np.searchsorted(ref, v.fillna(0).values, side="right") / max(len(ref), 1)
            new[f"rank({c})"] = pd.Series(np.where(v.isna(), np.nan, r), index=X.index)
        for c in cfg.missing_indicators:
            new[f"missing({c})"] = X[c].isna().astype(float)
        for c in cfg.frequency_encode:
            key = X[c].astype(object).where(X[c].notna(), "(missing)").astype(str)
            new[f"freq({c})"] = key.map(self.freq_[c]).fillna(0.0).astype(float)
        for c in cfg.text_stats:
            t = X[c].astype(object).where(X[c].notna(), "").astype(str)
            new[f"len({c})"] = t.str.len().astype(float)
            new[f"words({c})"] = t.str.split().str.len().fillna(0).astype(float)
        if not new:
            return X
        add = pd.DataFrame({k: (_finite(v.astype(float)) if v.dtype != object else v) for k, v in new.items()}, index=X.index)
        return pd.concat([X, add], axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.array([])


def preview(df: pd.DataFrame, cfg: FeatureConfig, numeric: list[str], categorical: list[str], text: list[str], n: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the generator on ``df`` (preview only; model training refits on training rows) and return
    (sample of generated columns, quality table flagging constant, duplicate or mostly invalid columns)."""
    planned = plan_features(cfg, numeric, categorical, text, list(df.columns))
    gen = FeatureGenerator(cfg.to_dict()).fit(df)
    out = gen.transform(df)[[g.name for g in planned]]
    q = []
    seen: dict[str, str] = {}
    for g in planned:
        s = out[g.name]
        issue = ""
        valid = s.notna().mean()
        if valid < 0.5:
            issue = f"only {100 * valid:.0f}% valid values"
        elif s.nunique(dropna=True) <= 1:
            issue = "constant: it carries no information"
        else:
            sig = pd.util.hash_pandas_object(s.astype(str), index=False).sum()
            key = str(sig)
            if key in seen:
                issue = f"identical to {seen[key]}"
            seen.setdefault(key, g.name)
        q.append({"feature": g.name, "type": g.kind, "description": g.description, "valid %": round(100 * valid, 1),
                  "distinct": int(s.nunique(dropna=True)), "issue": issue})
    return out.head(n), pd.DataFrame(q)
