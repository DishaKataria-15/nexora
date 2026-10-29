"""Outlier detection and treatment.

Outliers are observations far from the bulk of the data. They can be errors, rare but real events, or exactly the
records that matter, so Nexora never treats them automatically: detection is always reported first, and every
treatment is an explicit choice.

Two entry points share the same maths:

* :func:`detect` and :func:`apply_treatment` work on a DataFrame during data preparation (with row removal allowed,
  shown as a preview before it happens).
* :class:`OutlierTreatment` is a scikit-learn transformer for model pipelines. Its bounds are learned in ``fit`` on
  training rows only and applied unchanged to validation, test and new data. It never removes rows.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

METHODS = {
    "iqr": "Interquartile range (IQR)",
    "zscore": "Z-score",
    "modified_zscore": "Modified Z-score (median and MAD)",
    "isolation_forest": "Isolation Forest (multivariate)",
    "domain_bounds": "Domain limits from the profile",
}
METHOD_NOTES = {
    "iqr": "Flags values beyond Q1 - k x IQR or Q3 + k x IQR. Robust to a few extremes; on very skewed data it flags many values in the long tail.",
    "zscore": "Flags values more than the threshold number of standard deviations from the mean. Assumes a roughly bell-shaped distribution, "
              "and extreme values inflate the standard deviation themselves, which can hide them.",
    "modified_zscore": "Like the Z-score but uses the median and median absolute deviation, so extremes do not hide each other. "
                       "Fails when more than half the values are identical (the MAD is zero).",
    "isolation_forest": "Scores whole rows: a row is flagged because its combination of values is unusual, even if no single value is extreme. "
                        "Results depend on the contamination setting and the random seed. Only reporting or row removal is possible.",
    "domain_bounds": "Uses the valid ranges written in the domain profile. Only columns with configured limits are checked.",
}
TREATMENTS = {
    "report": "Report only (no change)",
    "flag": "Add a true/false flag column",
    "clip": "Clip values to the detection bounds",
    "winsorize": "Winsorize to percentile limits",
    "median": "Replace extreme values with the column median",
    "log": "Log-transform the column (non-negative columns only)",
    "remove": "Remove flagged rows",
}
ROW_LEVEL_TREATMENTS = {"report", "flag", "remove"}
PIPELINE_TREATMENTS = {"clip", "winsorize", "median"}


class OutlierConfigError(ValueError):
    """The outlier settings are not valid. The message is safe to show to the user."""


@dataclass
class OutlierConfig:
    method: str = "iqr"
    columns: list[str] = field(default_factory=list)
    iqr_k: float = 1.5
    z_threshold: float = 3.0
    mz_threshold: float = 3.5
    contamination: float | str = "auto"
    n_estimators: int = 100
    seed: int = 42
    winsor_limits: tuple[float, float] = (0.01, 0.99)
    bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["winsor_limits"] = list(self.winsor_limits)
        d["bounds"] = {k: list(v) for k, v in self.bounds.items()}
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "OutlierConfig":
        d = dict(d or {})
        if "winsor_limits" in d and d["winsor_limits"] is not None:
            d["winsor_limits"] = tuple(d["winsor_limits"])
        if "bounds" in d and d["bounds"]:
            d["bounds"] = {k: tuple(v) for k, v in d["bounds"].items()}
        return OutlierConfig(**{k: v for k, v in d.items() if k in OutlierConfig.__dataclass_fields__})


def validate_config(cfg: OutlierConfig, treatment: str = "report") -> None:
    if cfg.method not in METHODS:
        raise OutlierConfigError(f"Unknown outlier method '{cfg.method}'. Choose one of: {', '.join(METHODS)}.")
    if treatment not in TREATMENTS:
        raise OutlierConfigError(f"Unknown treatment '{treatment}'.")
    if not 0.5 <= float(cfg.iqr_k) <= 10:
        raise OutlierConfigError("The IQR multiplier must be between 0.5 and 10 (1.5 is common, 3 is conservative).")
    if not 1.0 <= float(cfg.z_threshold) <= 10:
        raise OutlierConfigError("The Z-score threshold must be between 1 and 10 (3 is common).")
    if not 1.0 <= float(cfg.mz_threshold) <= 10:
        raise OutlierConfigError("The modified Z-score threshold must be between 1 and 10 (3.5 is common).")
    if cfg.contamination != "auto" and not (0 < float(cfg.contamination) <= 0.5):
        raise OutlierConfigError("Contamination must be 'auto' or a share between 0 and 0.5.")
    if not 10 <= int(cfg.n_estimators) <= 2000:
        raise OutlierConfigError("The number of trees must be between 10 and 2000.")
    lo, hi = cfg.winsor_limits
    if not (0 <= lo < hi <= 1) or lo > 0.25 or hi < 0.75:
        raise OutlierConfigError("Winsorizing limits must satisfy 0 <= lower < upper <= 1, with lower <= 0.25 and upper >= 0.75.")
    if cfg.method == "isolation_forest" and treatment not in ROW_LEVEL_TREATMENTS:
        raise OutlierConfigError("Isolation Forest flags whole rows, so it cannot clip or replace individual values. "
                                 "Choose 'report', 'flag' or 'remove', or use a per-column method.")
    if cfg.method == "domain_bounds" and not cfg.bounds:
        raise OutlierConfigError("The selected domain profile has no numeric limits for these columns.")


def numeric_columns(df: pd.DataFrame, columns: list[str] | None = None) -> list[str]:
    """Numeric, non-boolean columns (optionally restricted to ``columns``). Categorical columns are never treated."""
    cols = columns if columns else list(df.columns)
    out = []
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            out.append(c)
    return out


def column_bounds(values: np.ndarray, cfg: OutlierConfig, column: str | None = None) -> tuple[float, float, str]:
    """(lower, upper, note) for one column. Bounds are infinite when they cannot be computed."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return -np.inf, np.inf, "too few values"
    m = cfg.method
    if m == "iqr":
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return -np.inf, np.inf, "interquartile range is zero"
        return float(q1 - cfg.iqr_k * iqr), float(q3 + cfg.iqr_k * iqr), ""
    if m == "zscore":
        sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        if sd == 0:
            return -np.inf, np.inf, "standard deviation is zero"
        mu = float(np.mean(v))
        return mu - cfg.z_threshold * sd, mu + cfg.z_threshold * sd, ""
    if m == "modified_zscore":
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad == 0:
            return -np.inf, np.inf, "median absolute deviation is zero (more than half the values are identical)"
        half = cfg.mz_threshold * mad / 0.6745
        return med - half, med + half, ""
    if m == "domain_bounds":
        lo, hi = cfg.bounds.get(column or "", (None, None))
        return (-np.inf if lo is None else float(lo)), (np.inf if hi is None else float(hi)), "" if (lo is not None or hi is not None) else "no limits configured"
    raise OutlierConfigError(f"'{m}' does not produce per-column bounds.")


@dataclass
class OutlierReport:
    method: str
    columns: list[str]
    row_mask: pd.Series                    # True = row flagged
    value_mask: pd.DataFrame               # True = value flagged (all False for multivariate methods)
    bounds: dict[str, tuple[float, float]]
    per_column: dict[str, int]
    notes: dict[str, str]
    multivariate: bool
    scores: pd.Series | None = None        # higher = more unusual (multivariate methods only)

    @property
    def n_rows(self) -> int:
        return len(self.row_mask)

    @property
    def n_flagged(self) -> int:
        return int(self.row_mask.sum())

    @property
    def pct_flagged(self) -> float:
        return round(100 * self.n_flagged / max(self.n_rows, 1), 2)

    def summary_table(self) -> pd.DataFrame:
        rows = []
        for c in self.columns:
            lo, hi = self.bounds.get(c, (np.nan, np.nan))
            rows.append({"column": c, "flagged values": self.per_column.get(c, 0),
                         "share %": round(100 * self.per_column.get(c, 0) / max(self.n_rows, 1), 2),
                         "lower bound": "none" if not np.isfinite(lo) else f"{lo:.6g}", "upper bound": "none" if not np.isfinite(hi) else f"{hi:.6g}",
                         "note": self.notes.get(c, "")})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "columns": self.columns, "n_rows": self.n_rows, "n_flagged": self.n_flagged, "pct_flagged": self.pct_flagged,
                "per_column": self.per_column, "bounds": {k: [None if not np.isfinite(a) else a for a in v] for k, v in self.bounds.items()},
                "notes": self.notes, "multivariate": self.multivariate}


def detect(df: pd.DataFrame, cfg: OutlierConfig) -> OutlierReport:
    """Detect outliers in the numeric columns of ``df``. ``df`` is not modified."""
    validate_config(cfg)
    cols = numeric_columns(df, cfg.columns)
    if cfg.method == "domain_bounds":
        cols = [c for c in cols if c in cfg.bounds]
    if not cols:
        raise OutlierConfigError("None of the selected columns is numeric. Outlier detection only applies to numbers.")
    value_mask = pd.DataFrame(False, index=df.index, columns=cols)
    bounds: dict[str, tuple[float, float]] = {}
    notes: dict[str, str] = {}
    if cfg.method == "isolation_forest":
        from sklearn.ensemble import IsolationForest
        X = df[cols].astype(float)
        filled = X.fillna(X.median())
        if len(filled) < 10:
            raise OutlierConfigError("Isolation Forest needs at least 10 rows.")
        model = IsolationForest(n_estimators=int(cfg.n_estimators), contamination=cfg.contamination, random_state=int(cfg.seed))
        pred = model.fit_predict(filled.values)
        scores = pd.Series(-model.score_samples(filled.values), index=df.index)
        row_mask = pd.Series(pred == -1, index=df.index)
        for c in cols:
            notes[c] = "rows are scored on all selected columns together"
        return OutlierReport(cfg.method, cols, row_mask, value_mask, {}, {c: 0 for c in cols}, notes, True, scores)
    for c in cols:
        s = df[c].astype(float)
        lo, hi, note = column_bounds(s.values, cfg, c)
        if np.isfinite(lo) and lo < 0 and s.notna().any() and float(s.min()) >= 0:
            # e.g. Q1 - 3 x IQR = -132.75 for prices: no value can be that low, so there is effectively no lower limit
            lo = -np.inf
            note = (note + "; " if note else "") + "no lower limit (the computed fence is below zero and the column has no negative values)"
        bounds[c] = (lo, hi)
        if note:
            notes[c] = note
        value_mask[c] = ((s < lo) | (s > hi)).fillna(False)
    per_col = {c: int(value_mask[c].sum()) for c in cols}
    return OutlierReport(cfg.method, cols, value_mask.any(axis=1), value_mask, bounds, per_col, notes, False)


def apply_treatment(df: pd.DataFrame, report: OutlierReport, cfg: OutlierConfig, treatment: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (new frame, summary). ``df`` is never modified. Summary row numbers are positions in ``df``'s index."""
    validate_config(cfg, treatment)
    out = df.copy()
    summary: dict[str, Any] = {"treatment": treatment, "rows_removed": [], "values_modified": {}, "columns_added": [], "warnings": []}
    if treatment == "report":
        return out, summary
    if treatment == "remove":
        removed = list(report.row_mask[report.row_mask].index)
        summary["rows_removed"] = removed
        if report.n_rows and len(removed) / report.n_rows > 0.05:
            summary["warnings"].append(f"{len(removed)} rows ({100 * len(removed) / report.n_rows:.1f}%) would be removed. "
                                       "That is a large share; check whether these are real events before deleting them.")
        return out.loc[~report.row_mask].copy(), summary
    if treatment == "flag":
        name = "row__outlier_flag" if report.multivariate else None
        if report.multivariate:
            out[name] = report.row_mask.astype(bool)
            summary["columns_added"].append(name)
        else:
            for c in report.columns:
                out[f"{c}__outlier_flag"] = report.value_mask[c].astype(bool)
                summary["columns_added"].append(f"{c}__outlier_flag")
        return out, summary
    for c in report.columns:
        s = df[c].astype(float)
        mask = report.value_mask[c]
        if treatment == "clip":
            lo, hi = report.bounds[c]
            new = s.clip(lower=lo if np.isfinite(lo) else None, upper=hi if np.isfinite(hi) else None)
            changed = mask & s.notna()
        elif treatment == "winsorize":
            lo_q, hi_q = cfg.winsor_limits
            lo, hi = s.quantile(lo_q), s.quantile(hi_q)
            new = s.clip(lower=lo, upper=hi)
            changed = s.notna() & ((s < lo) | (s > hi))
        elif treatment == "median":
            med = s[~mask].median()
            new = s.where(~mask, med)
            changed = mask & s.notna()
        elif treatment == "log":
            if (s.dropna() < 0).any():
                summary["warnings"].append(f"'{c}' has negative values, so it was not log-transformed.")
                continue
            new = np.log1p(s)
            changed = s.notna()
        else:  # pragma: no cover - guarded by validate_config
            raise OutlierConfigError(f"Unknown treatment '{treatment}'.")
        if str(df[c].dtype) == "Int64" and treatment in ("median", "clip", "winsorize", "log"):
            out[c] = new.astype("float64")
        else:
            out[c] = new
        summary["values_modified"][c] = int(changed.sum())
    return out, summary


class OutlierTreatment(BaseEstimator, TransformerMixin):
    """Pipeline step: learn per-column bounds on training rows, then clip / winsorize / replace with the training median.

    Works on numeric arrays (it sits after imputation in the numeric branch of the ColumnTransformer).
    """

    def __init__(self, method: str = "iqr", treatment: str = "clip", iqr_k: float = 1.5, z_threshold: float = 3.0,
                 mz_threshold: float = 3.5, winsor_limits: tuple[float, float] = (0.01, 0.99)):
        self.method = method
        self.treatment = treatment
        self.iqr_k = iqr_k
        self.z_threshold = z_threshold
        self.mz_threshold = mz_threshold
        self.winsor_limits = winsor_limits

    def _cfg(self) -> OutlierConfig:
        return OutlierConfig(method=self.method, iqr_k=self.iqr_k, z_threshold=self.z_threshold, mz_threshold=self.mz_threshold,
                             winsor_limits=tuple(self.winsor_limits))

    def fit(self, X, y=None):
        if self.treatment not in PIPELINE_TREATMENTS:
            raise OutlierConfigError(f"Inside a model pipeline the treatment must be one of {sorted(PIPELINE_TREATMENTS)}; rows cannot be removed there.")
        if self.method not in ("iqr", "zscore", "modified_zscore"):
            raise OutlierConfigError("Inside a model pipeline use the IQR, Z-score or modified Z-score method.")
        cfg = self._cfg()
        validate_config(cfg, self.treatment)
        X = np.asarray(X, dtype=float)
        self.lower_, self.upper_, self.median_ = [], [], []
        for j in range(X.shape[1]):
            col = X[:, j]
            if self.treatment == "winsorize":
                fin = col[np.isfinite(col)]
                lo, hi = (np.quantile(fin, self.winsor_limits[0]), np.quantile(fin, self.winsor_limits[1])) if len(fin) else (-np.inf, np.inf)
            else:
                lo, hi, _ = column_bounds(col, cfg)
            self.lower_.append(lo)
            self.upper_.append(hi)
            fin = col[np.isfinite(col)]
            inside = fin[(fin >= lo) & (fin <= hi)]
            self.median_.append(float(np.median(inside)) if len(inside) else (float(np.median(fin)) if len(fin) else 0.0))
        self.lower_, self.upper_, self.median_ = np.array(self.lower_), np.array(self.upper_), np.array(self.median_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} columns, got {X.shape[1]}.")
        if self.treatment in ("clip", "winsorize"):
            return np.clip(X, self.lower_, self.upper_)
        out_mask = (X < self.lower_) | (X > self.upper_)
        return np.where(out_mask, self.median_, X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features) if input_features is not None else np.array([f"x{i}" for i in range(self.n_features_in_)])
