"""Serializable preprocessing. Everything learned (imputation values, scales, categories, date origin, vocabulary)
is learned in fit() on training data only; the same fitted object is then applied to held-out data."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer, RobustScaler, StandardScaler

from ..eda import profiling as P

HIGH_CARD = 50


class FrameCoercer(BaseEstimator, TransformerMixin):
    """Stateless: turn pandas nullable/boolean/string columns into plain float/object columns scikit-learn accepts."""

    def __init__(self, columns: list[str] | None = None, datetime_columns: list[str] | None = None):
        self.columns = columns
        self.datetime_columns = datetime_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        cols = self.columns if self.columns is not None else list(X.columns)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise ValueError(f"Columns missing from the data: {missing}")
        X = X[cols]
        out = {}
        for c in cols:
            s = X[c]
            if c in (self.datetime_columns or []):
                out[c] = pd.to_datetime(s, errors="coerce")
            elif pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
                out[c] = s.astype(object).map(lambda v: np.nan if pd.isna(v) else ("true" if bool(v) else "false")).astype(object)
            elif pd.api.types.is_numeric_dtype(s):
                out[c] = pd.to_numeric(s, errors="coerce").astype("float64")
            else:
                o = s.astype(object)
                out[c] = o.where(s.notna(), np.nan)
        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns or [])


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Calendar features of dates. ``parts=None`` keeps the original default (year, month, weekday, day of year,
    days since the training start; hour when any time of day exists). ``cyclical`` adds sine/cosine encodings of
    month, weekday, day of year and hour so that December sits next to January."""

    def __init__(self, columns: list[str] | None = None, parts: list[str] | None = None, cyclical: bool = False):
        self.columns = columns
        self.parts = parts
        self.cyclical = cyclical

    def _parts(self, c: str) -> list[str]:
        from .features import DATE_PARTS, DEFAULT_DATE_PARTS
        if self.parts is None:
            return list(DEFAULT_DATE_PARTS) + (["hour"] if self.use_hour_[c] else [])
        bad = [p for p in self.parts if p not in DATE_PARTS]
        if bad:
            raise ValueError(f"Unknown date part(s): {bad}. Choose from {DATE_PARTS}.")
        return [p for p in self.parts if p not in ("hour", "minute") or self.use_hour_[c]]

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.use_hour_ = {c: bool((pd.to_datetime(X[c], errors="coerce").dropna().dt.hour != 0).any()) for c in self.columns_}
        self.origin_ = {c: pd.to_datetime(X[c], errors="coerce").min() for c in self.columns_}   # learned from training data only
        return self

    @staticmethod
    def _part(t: pd.Series, part: str, origin) -> pd.Series:
        if part == "days_since_start":
            return (t - origin).dt.total_seconds() / 86400.0 if pd.notna(origin) else pd.Series(np.nan, index=t.index)
        if part == "week":
            return t.dt.isocalendar().week.astype("float64").where(t.notna())
        if part == "weekend":
            return (t.dt.day_of_week >= 5).astype(float).where(t.notna())
        return getattr(t.dt, _DT_ATTR.get(part, part)).astype("float64")

    def transform(self, X):
        from .features import CYCLES
        X = pd.DataFrame(X, columns=self.columns_)
        out = []
        for c in self.columns_:
            t = pd.to_datetime(X[c], errors="coerce")
            cols = [self._part(t, p, self.origin_[c]) for p in self._parts(c)]
            if self.cyclical:
                for p, period in CYCLES.items():
                    if p == "hour" and not self.use_hour_[c]:
                        continue
                    v = getattr(t.dt, _DT_ATTR.get(p, p)).astype("float64")
                    cols += [np.sin(2 * np.pi * v / period), np.cos(2 * np.pi * v / period)]
            out.append(np.column_stack([pd.Series(v).astype(float).values for v in cols]))
        return np.hstack(out) if out else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        from .features import CYCLES
        legacy = {"dayofweek": "weekday"}
        names = []
        for c in self.columns_:
            names += [f"{c}__{legacy.get(p, p)}" for p in self._parts(c)]
            if self.cyclical:
                for p in CYCLES:
                    if p == "hour" and not self.use_hour_[c]:
                        continue
                    names += [f"{c}__{p}_sin", f"{c}__{p}_cos"]
        return np.array(names)


class _FillEmpty(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(np.asarray(X).ravel()).astype(object).where(lambda s: s.notna(), "").astype(str).values

    def get_feature_names_out(self, input_features=None):
        return np.array(["text"])


# pandas 3.1 deprecates .dayofweek / .dayofyear; the feature names stay the same, only the attribute read changes
_DT_ATTR = {"dayofweek": "day_of_week", "dayofyear": "day_of_year"}


@dataclass
class PreprocessOptions:
    numeric_impute: str = "median"       # median | mean
    scaling: str = "auto"                # auto | standard | minmax | robust | none
    skew_transform: bool = False         # Yeo-Johnson on numeric features
    encoding: str = "onehot"             # onehot | ordinal
    max_categories: int = 30
    use_datetime: bool = True
    text_columns: list[str] = field(default_factory=list)
    date_parts: list[str] | None = None      # None = original default parts
    cyclical: bool = False
    outliers: dict[str, Any] | None = None   # {"method": "iqr", "treatment": "clip", "iqr_k": 1.5, ...} learned on training rows
    features: dict[str, Any] | None = None   # FeatureConfig dictionary (see features.py)
    reduction: dict[str, Any] | None = None  # {"method": "pca", "n_components": 5}
    selection: dict[str, Any] | None = None  # {"method": "kbest", "k": 10, ...}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureSet:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)

    @property
    def all(self) -> list[str]:
        return self.numeric + self.categorical + self.datetime + self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_features(df: pd.DataFrame, target: str | None = None, exclude: list[str] | None = None, include: list[str] | None = None,
                    options: PreprocessOptions | None = None) -> FeatureSet:
    """Decide which columns become features and why the others do not. Never silently drops a column."""
    options = options or PreprocessOptions()
    fs = FeatureSet()
    exclude = set(exclude or [])
    for c in df.columns:
        if c == target:
            continue
        if include is not None and c not in include:
            fs.excluded[c] = "Not selected"
            continue
        if c in exclude:
            fs.excluded[c] = "Excluded by you"
            continue
        role = P.column_role(df[c], c)
        if role == "empty":
            fs.excluded[c] = "No values"
        elif role == "constant":
            fs.excluded[c] = "Same value in every row"
        elif role == "identifier":
            fs.excluded[c] = "Looks like an identifier (unique per row); it would let a model memorise rows"
        elif role == "datetime":
            if options.use_datetime:
                fs.datetime.append(c)
            else:
                fs.excluded[c] = "Date features switched off"
        elif role == "numeric":
            fs.numeric.append(c)
        elif role == "text":
            if c in options.text_columns:
                fs.text.append(c)
            else:
                fs.excluded[c] = "Free text (not used unless you enable text features)"
        else:  # categorical / boolean
            if df[c].nunique() > HIGH_CARD * 4:
                fs.excluded[c] = f"{df[c].nunique()} distinct values: too many to encode reliably"
            else:
                fs.categorical.append(c)
    return fs


def generated_features(fs: FeatureSet, options: PreprocessOptions):
    """Features created by feature engineering for this feature set (empty list when none are configured)."""
    from .features import FeatureConfig, plan_features
    cfg = FeatureConfig.from_dict(options.features)
    if cfg.is_empty():
        return []
    return plan_features(cfg, fs.numeric, fs.categorical, fs.text, fs.all)


def build_pipeline_steps(fs: FeatureSet, options: PreprocessOptions, scale_sensitive: bool, task: str | None = None,
                         seed: int = 42, force_scaling: str | None = None) -> list[tuple[str, Any]]:
    """Steps ready to fit on training data only:

    coerce -> (feature engineering) -> prep [impute -> (outlier treatment) -> (power) -> scale | encode | dates | text]
    -> (dimensionality reduction) -> (feature selection)
    """
    from .features import FeatureGenerator
    scaling = force_scaling or options.scaling
    if scaling == "auto":
        needs = scale_sensitive or bool(options.reduction and options.reduction.get("method", "none") != "none")
        scaling = "standard" if needs else "none"
    scalers = {"standard": StandardScaler, "minmax": lambda: MinMaxScaler(clip=True), "robust": RobustScaler}
    gen = generated_features(fs, options)
    numeric = fs.numeric + [g.name for g in gen if g.kind == "numeric"]
    categorical = fs.categorical + [g.name for g in gen if g.kind == "categorical"]
    transformers: list[tuple[str, Any, Any]] = []
    if numeric:
        steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy=options.numeric_impute, keep_empty_features=True))]
        if options.outliers and options.outliers.get("treatment", "none") not in ("none", "report"):
            from .outliers import OutlierTreatment
            o = options.outliers
            steps.append(("outliers", OutlierTreatment(method=o.get("method", "iqr"), treatment=o.get("treatment", "clip"), iqr_k=float(o.get("iqr_k", 1.5)),
                                                       z_threshold=float(o.get("z_threshold", 3.0)), mz_threshold=float(o.get("mz_threshold", 3.5)),
                                                       winsor_limits=tuple(o.get("winsor_limits", (0.01, 0.99))))))
        if options.skew_transform:
            steps.append(("power", PowerTransformer(method="yeo-johnson", standardize=(scaling == "standard"))))
            if scaling in ("minmax", "robust"):
                steps.append(("scale", scalers[scaling]()))
        elif scaling in scalers:
            steps.append(("scale", scalers[scaling]()))
        transformers.append(("num", Pipeline(steps), numeric))
    if categorical:
        enc = (OneHotEncoder(handle_unknown="ignore", max_categories=options.max_categories, sparse_output=False)
               if options.encoding == "onehot" else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)), ("encode", enc)]), categorical))
    if fs.datetime:
        transformers.append(("date", Pipeline([("parts", DatetimeFeatures(parts=options.date_parts, cyclical=options.cyclical)),
                                               ("impute", SimpleImputer(strategy="median", keep_empty_features=True))]), fs.datetime))
    for i, c in enumerate(fs.text):
        transformers.append((f"text{i}", Pipeline([("fillna", _FillEmpty()), ("tfidf", TfidfVectorizer(max_features=200, min_df=2))]), c))
    if not transformers:
        raise ValueError("No usable feature columns remain. Select at least one column that varies.")
    ct = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False)
    out: list[tuple[str, Any]] = [("coerce", FrameCoercer(fs.all, fs.datetime))]
    if gen:
        out.append(("features", FeatureGenerator(options.features)))
    out.append(("prep", ct))
    red = options.reduction or {}
    if red.get("method", "none") != "none":
        from .reduction import Reducer
        nc = red.get("n_components", 5)
        out.append(("reduce", Reducer(red["method"], float(nc) if isinstance(nc, float) and nc < 1 else int(nc), seed)))
    sel = options.selection or {}
    if sel.get("method", "none") != "none":
        from .selection import FeatureSelector
        out.append(("select", FeatureSelector(method=sel["method"], task=task or "classification", k=int(sel.get("k", 20)),
                                              percentile=int(sel.get("percentile", 50)), threshold=float(sel.get("threshold", 0.0)),
                                              strength=float(sel.get("strength", 1.0)), seed=seed)))
    return out


def feature_names(pipe) -> list[str]:
    """Names of the columns the estimator receives, following any reduction or selection step."""
    try:
        names = [str(n) for n in pipe.named_steps["prep"].get_feature_names_out()]
    except Exception:  # names are a convenience; never fail a run for them
        return []
    steps = dict(pipe.named_steps)
    if "reduce" in steps and hasattr(steps["reduce"], "n_out_"):
        names = [str(n) for n in steps["reduce"].get_feature_names_out()]
    if "select" in steps and hasattr(steps["select"], "support_"):
        names = [n for n, keep in zip(names, steps["select"].support_) if keep]
    return names


def pipeline_reports(pipe) -> dict[str, Any]:
    """What the fitted optional steps did: reduction summary and feature-selection report."""
    out: dict[str, Any] = {}
    steps = dict(pipe.named_steps)
    try:
        prep_names = [str(n) for n in steps["prep"].get_feature_names_out()]
    except Exception:
        prep_names = []
    out["prepared_features"] = len(prep_names)
    if "reduce" in steps and hasattr(steps["reduce"], "n_out_"):
        out["reduction"] = steps["reduce"].summary(prep_names)
        prep_names = [str(n) for n in steps["reduce"].get_feature_names_out()]
    if "select" in steps and hasattr(steps["select"], "support_"):
        out["selection"] = steps["select"].report(prep_names)
    num = getattr(steps.get("prep"), "named_transformers_", {}).get("num")
    if num is not None and "outliers" in num.named_steps:
        o = num.named_steps["outliers"]
        nums = list(steps["prep"].transformers_[0][2])
        out["outliers"] = {"method": o.method, "treatment": o.treatment,
                           "bounds": {c: [None if not np.isfinite(lo) else round(float(lo), 6), None if not np.isfinite(hi) else round(float(hi), 6)]
                                      for c, lo, hi in zip(nums, o.lower_, o.upper_)}}
    return out
