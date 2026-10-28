"""Task-specific metrics. A metric that is not mathematically valid for the data is reported as unavailable, with the reason."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, calinski_harabasz_score, classification_report,
                             confusion_matrix, davies_bouldin_score, f1_score, mean_absolute_error, mean_squared_error, median_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.preprocessing import label_binarize

METRIC_HELP = {
    "accuracy": "Share of predictions that are correct. Misleading when one class dominates.",
    "balanced_accuracy": "Average of the per-class recall, so every class counts equally.",
    "precision": "Of the rows predicted as the class, how many truly belong to it. High precision means few false alarms.",
    "recall": "Of the rows that truly belong to the class, how many were found. High recall means few misses.",
    "f1": "Balance of precision and recall (their harmonic mean). 1.0 is perfect.",
    "roc_auc": "Chance that a random positive row is ranked above a random negative one. 0.5 is a coin flip, 1.0 is perfect.",
    "pr_auc": "Area under the precision-recall curve. More informative than ROC-AUC when the class of interest is rare.",
    "mae": "Average size of the error, in the units of the target. Easy to interpret.",
    "rmse": "Typical error in the units of the target, penalising large misses more than MAE does.",
    "r2": "Share of the target's variation the model explains. 1.0 is perfect, 0 is no better than predicting the average, negative is worse.",
    "mape": "Average error as a percentage of the true value. Undefined when any true value is zero.",
    "median_ae": "Middle value of the absolute errors; less affected by a few large misses.",
    "silhouette": "How well rows fit their own group compared with the nearest other group (-1 to 1; higher is better). An internal measure, not accuracy.",
    "davies_bouldin": "Average similarity between each group and its closest group (0 and up; LOWER is better). An internal measure, not accuracy.",
    "calinski_harabasz": "Ratio of between-group to within-group spread (HIGHER is better). Grows with k, so compare only on the same data. An internal measure, not accuracy.",
}
HIGHER_IS_BETTER = {"accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "r2", "silhouette", "calinski_harabasz"}
LOWER_IS_BETTER = {"mae", "rmse", "mape", "median_ae", "davies_bouldin"}
MIN_RELIABLE_ROWS = 100


def metric_direction(name: str) -> str:
    return "higher" if name in HIGHER_IS_BETTER else "lower"


def selection_metrics(task: str, n_classes: int = 2) -> dict[str, str]:
    if task == "classification":
        return {"f1": "F1 score" if n_classes == 2 else "F1 score (macro average)", "accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy",
                "recall": "Recall" if n_classes == 2 else "Recall (macro average)", "precision": "Precision" if n_classes == 2 else "Precision (macro average)",
                "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"}
    if task == "regression":
        return {"rmse": "RMSE", "mae": "MAE", "r2": "R²"}
    return {"silhouette": "Silhouette", "davies_bouldin": "Davies-Bouldin", "calinski_harabasz": "Calinski-Harabasz"}


def default_primary(task: str, imbalance_level: str = "balanced") -> str:
    return {"classification": "accuracy" if imbalance_level == "balanced" else "f1", "regression": "rmse", "clustering": "silhouette"}[task]


def classification_metrics(y_true, y_pred, proba: np.ndarray | None, classes: list, positive: Any = None) -> dict[str, Any]:
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
    classes = list(classes)
    binary = len(classes) == 2
    pos = positive if positive in classes else (classes[-1] if binary else None)
    avg = "binary" if binary else "macro"
    kw = {"pos_label": pos} if binary else {}
    m: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=avg, zero_division=0, labels=classes if not binary else None, **kw)),
        "recall": float(recall_score(y_true, y_pred, average=avg, zero_division=0, labels=classes if not binary else None, **kw)),
        "f1": float(f1_score(y_true, y_pred, average=avg, zero_division=0, labels=classes if not binary else None, **kw))}
    why: dict[str, str] = {}
    present = set(y_true.unique())
    if proba is None:
        why["roc_auc"] = why["pr_auc"] = "This model does not produce probabilities."
        m["roc_auc"] = m["pr_auc"] = None
    elif binary:
        if len(present) < 2:
            why["roc_auc"] = why["pr_auc"] = "Only one class is present in this part of the data, so ranking quality is undefined."
            m["roc_auc"] = m["pr_auc"] = None
        else:
            j = classes.index(pos)
            yb = (y_true == pos).astype(int)
            m["roc_auc"] = float(roc_auc_score(yb, proba[:, j]))
            m["pr_auc"] = float(average_precision_score(yb, proba[:, j]))
    else:
        if not set(classes) <= present:
            why["roc_auc"] = why["pr_auc"] = "Some classes are absent from this part of the data, so one-vs-rest AUC is undefined."
            m["roc_auc"] = m["pr_auc"] = None
        else:
            m["roc_auc"] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes))
            Y = label_binarize(y_true, classes=classes)
            m["pr_auc"] = float(np.mean([average_precision_score(Y[:, i], proba[:, i]) for i in range(len(classes))]))
    labels = [str(c) for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    rep = classification_report(y_true, y_pred, labels=classes, target_names=labels, output_dict=True, zero_division=0)
    per_class = pd.DataFrame([{"class": c, "precision": rep[c]["precision"], "recall": rep[c]["recall"], "f1": rep[c]["f1-score"], "rows": int(rep[c]["support"])} for c in labels])
    return {"metrics": m, "unavailable": why, "confusion_matrix": cm.tolist(), "labels": labels, "per_class": per_class,
            "positive_class": str(pos) if binary else None, "averaging": "positive class" if binary else "macro average over classes", "n": int(len(y_true))}


def regression_metrics(y_true, y_pred) -> dict[str, Any]:
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    m: dict[str, float | None] = {"mae": float(mean_absolute_error(yt, yp)), "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
                                  "r2": float(r2_score(yt, yp)) if len(yt) > 1 else None, "median_ae": float(median_absolute_error(yt, yp))}
    why: dict[str, str] = {}
    if len(yt) < 2:
        why["r2"] = "R² needs at least two rows."
    if (yt == 0).any():
        m["mape"] = None
        why["mape"] = f"Not defined: {int((yt == 0).sum())} true value(s) are exactly zero."
    elif (np.abs(yt) < 1e-9).any():
        m["mape"], why["mape"] = None, "Not defined: some true values are almost zero."
    else:
        m["mape"] = float(np.mean(np.abs((yt - yp) / yt)) * 100)
    if len(yt) > 1 and np.var(yt) == 0:
        m["r2"] = None
        why["r2"] = "Not defined: the true values are all identical."
    return {"metrics": m, "unavailable": why, "n": int(len(yt))}


def clustering_metrics(X: np.ndarray, labels: np.ndarray, seed: int = 42, sample_size: int = 5000) -> dict[str, Any]:
    labels = np.asarray(labels)
    noise = labels == -1
    core = ~noise
    n_clusters = len(np.unique(labels[core]))
    m: dict[str, float | None] = {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    why: dict[str, str] = {}
    if n_clusters < 2:
        for k in m:
            why[k] = f"Needs at least 2 groups; found {n_clusters}."
    elif core.sum() <= n_clusters:
        for k in m:
            why[k] = "Too few rows for the number of groups."
    else:
        Xc, lc = X[core], labels[core]
        m["silhouette"] = float(silhouette_score(Xc, lc, sample_size=min(sample_size, len(lc)) if len(lc) > sample_size else None, random_state=seed))
        m["davies_bouldin"] = float(davies_bouldin_score(Xc, lc))
        m["calinski_harabasz"] = float(calinski_harabasz_score(Xc, lc))
    sizes = pd.Series(labels).value_counts().sort_index()
    return {"metrics": m, "unavailable": why, "n_clusters": n_clusters, "noise_rows": int(noise.sum()), "noise_pct": round(100 * float(noise.mean()), 2),
            "sizes": {int(k): int(v) for k, v in sizes.items()}, "n": int(len(labels)),
            "note": "Computed on non-noise rows." if noise.any() else ""}


def interpret(task: str, result: dict[str, Any], baseline: dict[str, Any] | None, class_shares: dict[str, float] | None = None) -> list[str]:
    """Short interpretation and caveats built from the numbers; nothing hard-coded about the user's data."""
    out: list[str] = []
    m, n = result["metrics"], result.get("n", 0)
    if n < MIN_RELIABLE_ROWS:
        out.append(f"Only {n} rows were evaluated, so these scores can move a lot with a different split. Treat them as rough.")
    if task == "classification":
        out.append(f"Accuracy {100 * m['accuracy']:.1f}% and F1 {m['f1']:.2f} ({result['averaging']}).")
        if baseline:
            b = baseline["metrics"]
            out.append(f"A model that always predicts the most common class scores accuracy {100 * b['accuracy']:.1f}% and F1 {b['f1']:.2f}."
                       + (" This model does not clearly beat that baseline." if m["f1"] <= b["f1"] + 0.02 and m["accuracy"] <= b["accuracy"] + 0.02 else " This model beats that baseline."))
        if class_shares and max(class_shares.values()) >= 80:
            out.append("The classes are imbalanced, so accuracy alone overstates quality; rely on F1, recall, precision and PR-AUC.")
    elif task == "regression":
        out.append(f"Typical error: about {m['rmse']:.4g} (RMSE) and {m['mae']:.4g} on average (MAE), in the units of the target.")
        if m.get("r2") is not None:
            out.append(f"The model explains about {100 * max(m['r2'], 0):.0f}% of the variation (R² {m['r2']:.2f})." + (" Negative R² means it does worse than predicting the average." if m["r2"] < 0 else ""))
        if baseline and baseline["metrics"]["rmse"] > 0:
            out.append(f"Predicting the training average gives RMSE {baseline['metrics']['rmse']:.4g}; this model {'improves on' if m['rmse'] < baseline['metrics']['rmse'] else 'does not improve on'} it.")
    else:
        s = m.get("silhouette")
        if s is not None:
            out.append(f"Silhouette {s:.2f}: " + ("clear separation." if s >= 0.5 else "moderate structure." if s >= 0.25 else "weak structure; the groups overlap.") +
                       " These are internal measures of shape, not accuracy: there are no true labels to compare with.")
        if result.get("noise_rows"):
            out.append(f"{result['noise_rows']:,} row(s) ({result['noise_pct']}%) were labelled noise and are left out of these scores. That can make density-based results look better than "
                       "methods that assign every row to a group, so compare with K-Means before trusting the ranking.")
        out.append("Clusters are groups that are similar on the chosen features, not ground-truth categories. Check that the groups make business sense before using them.")
    return out
