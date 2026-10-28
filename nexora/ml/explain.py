"""Model explanations computed from training or validation rows (never the test rows).

* Native importance: impurity importance for tree models, coefficient size for linear models. These describe the
  prepared features the model actually receives (after encoding, scaling, reduction and selection).
* SHAP (optional package): mean absolute SHAP value per prepared feature for tree and linear models, on a bounded
  sample. Other model types are skipped with a stated reason rather than slow, approximate kernels.
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Any

import numpy as np
import pandas as pd

from ..preprocessing.pipeline import feature_names

log = logging.getLogger(__name__)
SHAP_AVAILABLE = importlib.util.find_spec("shap") is not None
SHAP_MAX_ROWS = 500

TREE_TYPES = ("DecisionTree", "RandomForest", "ExtraTrees", "HistGradientBoosting", "GradientBoosting", "XGB", "LGBM")
LINEAR_TYPES = ("LogisticRegression", "LinearRegression", "Ridge", "Lasso", "ElasticNet")


def _inner(model):
    from .model_registry import LabelEncodedClassifier
    return model.estimator_ if isinstance(model, LabelEncodedClassifier) else model


def prepared_matrix(pipe, X: pd.DataFrame) -> np.ndarray:
    """Apply every fitted step except resampling and the model."""
    out: Any = X
    for name, step in pipe.steps[:-1]:
        if name == "sampler":
            continue
        out = step.transform(out)
    if hasattr(out, "toarray"):
        out = out.toarray()
    return np.asarray(out, dtype=float)


def native_importance(pipe, top: int = 25) -> dict[str, Any] | None:
    model = _inner(pipe.steps[-1][1])
    names = feature_names(pipe)
    vals, kind = None, ""
    if hasattr(model, "feature_importances_"):
        vals, kind = np.asarray(model.feature_importances_, dtype=float), "impurity"
    elif hasattr(model, "coef_"):
        c = np.asarray(model.coef_, dtype=float)
        vals, kind = (np.abs(c).mean(axis=0) if c.ndim == 2 else np.abs(c)), "coefficient"
    if vals is None or not len(vals):
        return None
    if len(names) != len(vals):
        names = [f"feature {i}" for i in range(len(vals))]
    order = np.argsort(-vals)[:top]
    rows = [{"feature": names[i], "importance": float(vals[i])} for i in order]
    if kind == "coefficient":
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim == 1 or coef.shape[0] == 1:
            flat = coef.ravel()
            for r, i in zip(rows, order):
                r["direction"] = "raises the prediction" if flat[i] > 0 else "lowers the prediction" if flat[i] < 0 else "no effect"
    note = ("Impurity importance: how much each prepared feature reduced error inside the trees. It favours features with many distinct values."
            if kind == "impurity" else
            "Coefficient size on the prepared (encoded and scaled) features. Larger means a stronger effect per unit; correlated features share credit.")
    return {"kind": kind, "rows": rows, "note": note}


def shap_importance(pipe, X_ref: pd.DataFrame, seed: int = 42, max_rows: int = SHAP_MAX_ROWS, top: int = 25) -> dict[str, Any]:
    """Mean |SHAP| per prepared feature. Returns {"available": False, "reason": ...} when it cannot be computed."""
    if not SHAP_AVAILABLE:
        return {"available": False, "reason": "SHAP is an optional package. Install it with: pip install shap"}
    model = _inner(pipe.steps[-1][1])
    tname = type(model).__name__
    is_tree = any(tname.startswith(t) for t in TREE_TYPES)
    is_linear = any(tname.startswith(t) for t in LINEAR_TYPES)
    if not (is_tree or is_linear):
        return {"available": False, "reason": f"SHAP explanations are offered for tree and linear models; {tname} would need a slow approximate method."}
    import shap
    sample = X_ref if len(X_ref) <= max_rows else X_ref.sample(max_rows, random_state=seed)
    try:
        M = prepared_matrix(pipe, sample)
        if is_tree:
            sv = shap.TreeExplainer(model).shap_values(M, check_additivity=False)
        else:
            sv = shap.LinearExplainer(model, M).shap_values(M)
        arr = np.asarray(sv, dtype=float)
        if isinstance(sv, list):                         # older SHAP: one array per class
            arr = np.stack([np.asarray(s) for s in sv], axis=-1)
        if arr.ndim == 3:                                # rows x features x classes
            imp = np.abs(arr).mean(axis=(0, 2)) if arr.shape[1] == M.shape[1] else np.abs(arr).mean(axis=(0, 1))
        else:
            imp = np.abs(arr).mean(axis=0)
    except Exception as e:
        log.exception("SHAP failed")
        return {"available": False, "reason": f"SHAP could not explain this model: {type(e).__name__}: {e}"}
    names = feature_names(pipe)
    if len(names) != len(imp):
        names = [f"feature {i}" for i in range(len(imp))]
    order = np.argsort(-imp)[:top]
    return {"available": True, "rows": [{"feature": names[i], "mean_abs_shap": float(imp[i])} for i in order], "n_rows": int(len(sample)),
            "note": f"Average size of each prepared feature's contribution to individual predictions, on {len(sample):,} non-test rows."}
