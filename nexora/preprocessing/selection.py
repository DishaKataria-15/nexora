"""Automated feature selection as a pipeline step.

Because the selector is part of the fitted pipeline, it only ever sees the rows the pipeline is being fitted on:
the training part of each cross-validation fold, or the training split. Held-out targets are never inspected.
"""
from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .compat import l1_logistic

METHODS = {
    "none": "None",
    "variance": "Remove near-constant features (variance threshold)",
    "kbest": "Keep the k best features (univariate statistical test)",
    "percentile": "Keep the best percentage of features (univariate test)",
    "mutual_info": "Keep the k features with the highest mutual information",
    "rfe": "Recursive feature elimination (RFE)",
    "rfecv": "RFE with cross-validated number of features (RFECV)",
    "model": "Model-based (random-forest importance)",
    "l1": "Regularisation-based (L1 / Lasso)",
}
UNSUPERVISED_OK = {"none", "variance"}
METHOD_NOTES = {
    "variance": "Uses no target, so it is also valid for clustering.",
    "kbest": "Tests each feature on its own (ANOVA F-test or F-regression); misses features that only help in combination.",
    "percentile": "Same test as k best, but keeps a percentage.",
    "mutual_info": "Captures non-linear relationships one feature at a time; slower and slightly random (seeded).",
    "rfe": "Repeatedly fits a linear model and drops the weakest features; slower on wide data.",
    "rfecv": "Chooses how many features to keep with an inner 3-fold cross-validation inside the training rows; the slowest option.",
    "model": "Keeps features whose random-forest importance is above the median (or the top k).",
    "l1": "Keeps features with non-zero L1-penalised coefficients; a smaller C (or larger alpha) keeps fewer features.",
}


class SelectionError(ValueError):
    """Feature selection is not valid as configured. The message is safe to show to the user."""


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = "variance", task: str = "classification", k: int = 20, percentile: int = 50,
                 threshold: float = 0.0, strength: float = 1.0, seed: int = 42):
        self.method = method
        self.task = task
        self.k = k
        self.percentile = percentile
        self.threshold = threshold
        self.strength = strength
        self.seed = seed

    def _inner(self, n_features: int):
        from sklearn import feature_selection as FS
        m, clf = self.method, self.task == "classification"
        if m not in METHODS or m == "none":
            raise SelectionError(f"Unknown selection method '{m}'.")
        if self.task == "clustering" and m not in UNSUPERVISED_OK:
            raise SelectionError("Without a target only the variance threshold is valid; supervised scores cannot be computed for clustering.")
        k = max(1, min(int(self.k), n_features))
        if m == "variance":
            return FS.VarianceThreshold(threshold=float(self.threshold))
        if m == "kbest":
            return FS.SelectKBest(FS.f_classif if clf else FS.f_regression, k=k)
        if m == "percentile":
            if not 1 <= int(self.percentile) <= 100:
                raise SelectionError("The percentile must be between 1 and 100.")
            return FS.SelectPercentile(FS.f_classif if clf else FS.f_regression, percentile=int(self.percentile))
        if m == "mutual_info":
            score = partial(FS.mutual_info_classif if clf else FS.mutual_info_regression, random_state=self.seed)
            return FS.SelectKBest(score, k=k)
        if m in ("rfe", "rfecv"):
            from sklearn.linear_model import LogisticRegression, Ridge
            est = LogisticRegression(max_iter=2000) if clf else Ridge()
            step = max(1, n_features // 10)
            if m == "rfe":
                return FS.RFE(est, n_features_to_select=k, step=step)
            from sklearn.model_selection import KFold, StratifiedKFold
            cv = StratifiedKFold(3, shuffle=True, random_state=self.seed) if clf else KFold(3, shuffle=True, random_state=self.seed)
            return FS.RFECV(est, step=step, cv=cv, min_features_to_select=1)
        if m == "model":
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            est = (RandomForestClassifier if clf else RandomForestRegressor)(n_estimators=100, random_state=self.seed, n_jobs=1)
            return FS.SelectFromModel(est, threshold=-np.inf, max_features=k) if self.k and int(self.k) < n_features else FS.SelectFromModel(est, threshold="median")
        if m == "l1":
            if float(self.strength) <= 0:
                raise SelectionError("The regularisation strength must be positive.")
            from sklearn.linear_model import Lasso
            est = l1_logistic(C=float(self.strength), seed=self.seed) if clf else Lasso(alpha=float(self.strength), max_iter=10000)
            return FS.SelectFromModel(est)
        raise SelectionError(f"Unknown selection method '{m}'.")  # pragma: no cover

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.task == "clustering" and self.method not in UNSUPERVISED_OK:
            raise SelectionError("Without a target only the variance threshold is valid; supervised scores cannot be computed for clustering.")
        if self.method not in UNSUPERVISED_OK and y is None:
            raise SelectionError(f"'{METHODS.get(self.method, self.method)}' needs a target column.")
        self.selector_ = self._inner(X.shape[1])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.selector_.fit(X, y)
        self.support_ = np.asarray(self.selector_.get_support())
        if not self.support_.any():
            # never leave the model without inputs: keep the single best-scoring feature
            scores = getattr(self.selector_, "scores_", None)
            if scores is None:
                scores = getattr(self.selector_, "variances_", np.zeros(X.shape[1]))
            best = int(np.nanargmax(np.nan_to_num(np.asarray(scores, dtype=float), nan=-np.inf)))
            self.support_ = np.zeros(X.shape[1], dtype=bool)
            self.support_[best] = True
        self.n_features_in_ = X.shape[1]
        self.scores_ = getattr(self.selector_, "scores_", None)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]

    def get_support(self):
        return self.support_

    def get_feature_names_out(self, input_features=None):
        names = np.asarray(input_features) if input_features is not None else np.array([f"x{i}" for i in range(self.n_features_in_)])
        return names[self.support_]

    def report(self, input_features: list[str]) -> dict:
        names = list(input_features) if len(input_features) == self.n_features_in_ else [f"x{i}" for i in range(self.n_features_in_)]
        scores = self.scores_
        rows = []
        for i, n in enumerate(names):
            rows.append({"feature": n, "kept": bool(self.support_[i]),
                         "score": None if scores is None or not np.isfinite(scores[i]) else round(float(scores[i]), 6)})
        return {"method": self.method, "before": self.n_features_in_, "after": int(self.support_.sum()), "features": rows}
