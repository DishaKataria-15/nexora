"""Anomaly detection workflow: score every row for how unusual it is, flag the most unusual, and explain why.

Detection is unsupervised. If the data has a column that marks known anomalies, it can be used *only* to evaluate
the result (precision / recall), never to fit the detector, and it is excluded from the features.
Scores are oriented so that higher always means more unusual.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from ..preprocessing.pipeline import PreprocessOptions, build_pipeline_steps, select_features

METHODS = {
    "isolation_forest": ("Isolation Forest", "Isolates rows with random splits; unusual rows are isolated in fewer splits. A good general default.", None),
    "lof": ("Local Outlier Factor", "Compares each row's local density with its neighbours'; finds rows that are unusual for their area.", 50000),
    "one_class_svm": ("One-Class SVM", "Learns a boundary around normal rows. Sensitive to settings and slow on large data.", 10000),
    "elliptic": ("Elliptic Envelope", "Assumes normal rows form one elliptical cloud; flags rows far from its centre. Best for roughly bell-shaped data.", 50000),
    "robust_z": ("Robust z-score", "Flags rows with values far from the median of a column, measured in robust spreads (MAD). Simple and transparent.", None),
}


class AnomalyError(ValueError):
    """The anomaly run cannot be done. The message is safe to show to the user."""


@dataclass
class AnomalyConfig:
    methods: list[str] = field(default_factory=lambda: ["isolation_forest"])
    contamination: float | str = "auto"    # expected share of anomalies, or "auto" (method default / 1% for score-only methods)
    features_include: list[str] | None = None
    features_exclude: list[str] = field(default_factory=list)
    label_column: str | None = None        # known anomalies, used only for evaluation
    label_positive: str | None = None
    n_neighbors: int = 20
    seed: int = 42
    top_n: int = 50

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyResult:
    config: AnomalyConfig
    feature_set: dict[str, Any] = field(default_factory=dict)
    methods: list[dict[str, Any]] = field(default_factory=list)
    selected: str = ""
    top: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unit_warnings: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    _scores: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _flags: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_scores", None)
        d.pop("_flags", None)
        return d

    def assignments(self, method: str | None = None) -> pd.DataFrame:
        m = method or self.selected
        return pd.DataFrame({"row_in_dataset": np.arange(len(self._scores[m])), "anomaly_score": np.round(self._scores[m], 6),
                             "flagged": self._flags[m].astype(bool)})


UNIT_NAME = re.compile(r"(^|_)(unit|units|uom|currency|curr|scale|measure)$", re.I)


def mixed_unit_warnings(df: pd.DataFrame, exclude: list[str] | None = None) -> list[dict[str, Any]]:
    """Numeric columns whose readings come in more than one unit, recognised by a companion column such as
    'temperature_unit' or 'currency'. Rows in the less common unit would look unusual only because of the unit.

    Returns [{"value_cols", "unit_col", "units": {unit: rows}, "message"}]. Unit spellings are compared ignoring case.
    """
    exclude = set(exclude or [])
    out: list[dict[str, Any]] = []
    numeric = [c for c in df.columns if c not in exclude and pd.to_numeric(df[c], errors="coerce").notna().mean() >= 0.8]
    for uc in df.columns:
        if not UNIT_NAME.search(str(uc)):
            continue
        counts = df[uc].dropna().astype(str).str.strip().str.upper().replace("", np.nan).dropna().value_counts()   # 'usd' = 'USD'
        if len(counts) < 2 or len(counts) > 10:
            continue
        stem = UNIT_NAME.sub("", str(uc)).strip("_").lower()
        pairs = [c for c in numeric if c != uc and stem and str(c).lower().startswith(stem)]
        if not pairs and stem in ("", "currency", "curr"):
            pairs = [c for c in numeric if re.search(r"(amount|price|total|cost|balance|value)", str(c), re.I)]
        if not pairs:
            continue
        units = {str(k): int(v) for k, v in counts.items()}
        listing = ", ".join(f"{k} ({v:,} rows)" for k, v in units.items())
        cols = ", ".join(f"'{c}'" for c in pairs)
        out.append({"value_cols": pairs, "unit_col": uc, "units": units,
                    "message": f"{cols} {'is' if len(pairs) == 1 else 'are'} recorded in more than one unit according to '{uc}': {listing}. "
                               "Rows in the less common unit can look unusual only because of the unit. Convert them to one unit first "
                               "(Prepare Data), or leave " + ("that column" if len(pairs) == 1 else "those columns") + " out here."})
    return out


def _robust_z(Z: np.ndarray) -> np.ndarray:
    med = np.nanmedian(Z, axis=0)
    mad = np.nanmedian(np.abs(Z - med), axis=0) * 1.4826
    mad[~np.isfinite(mad) | (mad == 0)] = np.nan
    out = np.abs(Z - med) / mad
    return np.nan_to_num(out, nan=0.0)


def _contamination(cfg: AnomalyConfig) -> float | str:
    c = cfg.contamination
    if c == "auto":
        return "auto"
    try:
        c = float(c)
    except (TypeError, ValueError):
        raise AnomalyError("The expected share of anomalies must be a number between 0.1% and 50%, or 'auto'.") from None
    if not 0.001 <= c <= 0.5:
        raise AnomalyError("The expected share of anomalies must be between 0.1% and 50%.")
    return c


def _fit_method(key: str, Xt: np.ndarray, cont: float | str, cfg: AnomalyConfig) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (score, flag, note). Score: higher = more unusual."""
    share = 0.01 if cont == "auto" else float(cont)
    if key == "isolation_forest":
        m = IsolationForest(n_estimators=200, contamination="auto" if cont == "auto" else cont, random_state=cfg.seed, n_jobs=-1).fit(Xt)
        score = -m.score_samples(Xt)
        if cont != "auto":
            return score, m.predict(Xt) == -1, ""
        # scikit-learn's fixed 'auto' cut-off flags a large share of rows on many real datasets; instead flag scores that are
        # themselves outliers among the scores (above Q3 + 3 x IQR)
        q1, q3 = np.quantile(score, [0.25, 0.75])
        thr = q3 + 3 * (q3 - q1)
        return score, score > thr, "flagged when the score is far above the typical score (Q3 + 3 x IQR of the scores)"
    if key == "lof":
        k = int(min(max(2, cfg.n_neighbors), len(Xt) - 1))
        m = LocalOutlierFactor(n_neighbors=k, contamination=cont)
        flag = m.fit_predict(Xt) == -1
        return -m.negative_outlier_factor_, flag, f"{k} neighbours"
    if key == "one_class_svm":
        # features are robust-scaled, so a fixed width per feature is used; gamma="scale" would be inflated by the anomalies themselves
        nu = max(share, 0.05)          # very small nu lets the anomalies themselves become support vectors and hide
        m = OneClassSVM(nu=nu, gamma=1.0 / max(Xt.shape[1], 1)).fit(Xt)
        score = -m.decision_function(Xt)
        return score, score >= np.quantile(score, 1 - share), f"nu={nu:g}; the top {100 * share:g}% of scores are flagged"
    if key == "elliptic":
        m = EllipticEnvelope(contamination=share, random_state=cfg.seed, support_fraction=None).fit(Xt)
        return -m.score_samples(Xt), m.predict(Xt) == -1, ""
    if key == "robust_z":
        z = _robust_z(Xt).max(axis=1)
        thr = 3.5 if cont == "auto" else float(np.quantile(z, 1 - share))
        return z, z > thr, ("flagged when any value is more than 3.5 robust spreads from its median" if cont == "auto"
                            else f"top {100 * share:g}% by largest robust z-score")
    raise AnomalyError(f"Unknown anomaly method '{key}'.")


def run_anomaly(df: pd.DataFrame, cfg: AnomalyConfig, options: PreprocessOptions | None = None,
                progress: Callable[[float, str], None] | None = None, check: Callable[[], None] = lambda: None) -> AnomalyResult:
    t0 = time.time()
    if not cfg.methods:
        raise AnomalyError("Choose at least one detection method.")
    for m in cfg.methods:
        if m not in METHODS:
            raise AnomalyError(f"Unknown anomaly method '{m}'.")
    if len(df) < 20:
        raise AnomalyError("Anomaly detection needs at least 20 rows.")
    cont = _contamination(cfg)
    work = df.reset_index(drop=True)
    res = AnomalyResult(cfg)
    exclude = list(cfg.features_exclude) + ([cfg.label_column] if cfg.label_column else [])
    options = options or PreprocessOptions()
    if options.scaling == "auto":
        options.scaling = "robust"
    fs = select_features(work, None, exclude, cfg.features_include, options)
    if not fs.all:
        raise AnomalyError("No usable feature columns remain.")
    res.feature_set = fs.to_dict()
    res.unit_warnings = [w for w in mixed_unit_warnings(work, exclude) if set(w["value_cols"]) & set(fs.all)]
    prep = Pipeline(build_pipeline_steps(fs, options, True, task="clustering", seed=cfg.seed))
    Xt = np.asarray(prep.fit_transform(work[fs.all]), dtype=float)
    names = _names(prep, Xt.shape[1])
    labels = None
    if cfg.label_column:
        if cfg.label_column not in work:
            raise AnomalyError(f"Column '{cfg.label_column}' is not in the data.")
        lab = work[cfg.label_column]
        if lab.isna().any():
            res.warnings.append(f"{int(lab.isna().sum())} row(s) have no value in '{cfg.label_column}'; they are left out of the evaluation only.")
        pos = cfg.label_positive
        vals = lab.dropna().astype(str)
        if pos is None:
            pos = vals.value_counts().index[-1]
        labels = np.where(lab.isna(), np.nan, (lab.astype(str) == str(pos)).astype(float))
        res.warnings.append(f"'{cfg.label_column}' was used only to evaluate the results (value '{pos}' = anomaly). It was not a feature and did not train the detector.")
    zfeat = _robust_z(Xt)
    for i, key in enumerate(cfg.methods):
        check()
        name, about, cap = METHODS[key]
        if progress:
            progress(i / len(cfg.methods), f"Scoring with {name}")
        entry: dict[str, Any] = {"key": key, "name": name, "about": about, "status": "ok"}
        if cap and len(Xt) > cap:
            entry.update(status="skipped", error=f"{name} is limited to {cap:,} rows here; this data has {len(Xt):,}.")
            res.methods.append(entry)
            continue
        t1 = time.time()
        try:
            score, flag, note = _fit_method(key, Xt, cont, cfg)
        except Exception as e:
            entry.update(status="failed", error=f"{type(e).__name__}: {e}")
            res.methods.append(entry)
            continue
        score = np.asarray(score, dtype=float)
        entry.update(note=note, flagged=int(flag.sum()), flagged_pct=round(100 * float(flag.mean()), 2), seconds=round(time.time() - t1, 3))
        if labels is not None:
            ok = ~np.isnan(labels)
            yt = labels[ok].astype(int)
            if 0 < yt.sum() < len(yt):
                entry["evaluation"] = {"precision": float(precision_score(yt, flag[ok], zero_division=0)),
                                       "recall": float(recall_score(yt, flag[ok], zero_division=0)),
                                       "roc_auc": float(roc_auc_score(yt, score[ok])), "pr_auc": float(average_precision_score(yt, score[ok])),
                                       "known_anomalies": int(yt.sum()), "base_rate": float(yt.mean())}
            else:
                entry["evaluation"] = {"note": "The label column has only one class, so the detector cannot be evaluated."}
        res._scores[key], res._flags[key] = score, flag
        res.methods.append(entry)
    ok = [m for m in res.methods if m["status"] == "ok"]
    if not ok:
        raise AnomalyError("No method could score the data: " + "; ".join(f"{m['name']}: {m.get('error', '')}" for m in res.methods))
    if labels is not None and all("pr_auc" in m.get("evaluation", {}) for m in ok):
        res.selected = max(ok, key=lambda m: m["evaluation"]["pr_auc"])["key"]
    else:
        res.selected = ok[0]["key"]
    if len(ok) > 1:
        keys = [m["key"] for m in ok]
        ranks = pd.DataFrame({k: pd.Series(res._scores[k]).rank() for k in keys})
        corr = ranks.corr(method="pearson")
        for m in ok:
            m["agreement"] = {k: round(float(corr.loc[m["key"], k]), 3) for k in keys if k != m["key"]}
    score, flag = res._scores[res.selected], res._flags[res.selected]
    order = np.argsort(-score)[: cfg.top_n]
    for r in order:
        contrib = np.argsort(-zfeat[r])[:3]
        why = [f"{names[j]} ({zfeat[r, j]:.1f} robust spreads from typical)" for j in contrib if zfeat[r, j] >= 2]
        row = {"row_in_dataset": int(r), "anomaly_score": round(float(score[r]), 4), "flagged": bool(flag[r]),
               "why": "; ".join(why) or "an unusual combination of values rather than one extreme value"}
        for c in fs.all[:8]:
            v = work.at[r, c]
            if isinstance(v, pd.Timestamp) and v == v.normalize():
                v = v.strftime("%Y-%m-%d")                       # date-only values without 00:00:00
            row[c] = None if pd.isna(v) else (v if isinstance(v, (int, float, str, bool)) else str(v))
        res.top.append(row)
    res.seconds = round(time.time() - t0, 2)
    if progress:
        progress(1.0, "Done")
    return res


def _names(prep: Pipeline, n: int) -> list[str]:
    try:
        names = [str(x) for x in prep.named_steps["prep"].get_feature_names_out()]
        if len(names) == n:
            return [x.split("__", 1)[-1] for x in names]
    except Exception:
        pass
    return [f"feature {i}" for i in range(n)]
