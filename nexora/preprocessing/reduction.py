"""Dimensionality reduction.

* :class:`Reducer` is a pipeline step (PCA, truncated SVD, feature agglomeration). It is fitted on training rows and
  applied unchanged to validation, test and new data.
* :func:`project` produces 2-D or 3-D views for exploration, including t-SNE and (optionally) UMAP. Those two are
  visualisation-only: they cannot transform new rows, so they are never used inside a model pipeline.

Reducing dimensions does not automatically improve a model; compare scores with and without it.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

PIPELINE_METHODS = {"none": "None", "pca": "PCA (principal components)", "svd": "Truncated SVD", "agglomeration": "Feature agglomeration (merge similar features)"}
VIEW_METHODS = {"pca": "PCA", "svd": "Truncated SVD", "tsne": "t-SNE (visualisation only)", "umap": "UMAP (visualisation only, optional package)"}
UMAP_AVAILABLE = importlib.util.find_spec("umap") is not None
MAX_VIEW_ROWS = 5000


class ReductionError(ValueError):
    """Dimensionality reduction cannot be applied as configured. The message is safe to show to the user."""


class Reducer(BaseEstimator, TransformerMixin):
    def __init__(self, method: str = "pca", n_components: float | int = 2, seed: int = 42):
        self.method = method
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_feat = X.shape[1]
        nc = self.n_components
        if self.method not in ("pca", "svd", "agglomeration"):
            raise ReductionError(f"Unknown reduction method '{self.method}'.")
        if isinstance(nc, float) and not float(nc).is_integer():
            if self.method != "pca" or not 0 < nc < 1:
                raise ReductionError("A fractional number of components (share of variance to keep) is only valid for PCA, between 0 and 1.")
        else:
            nc = int(nc)
            if nc < 1:
                raise ReductionError("Keep at least 1 component.")
            if nc >= n_feat:
                raise ReductionError(f"There are only {n_feat} prepared features, so reducing to {nc} components would not reduce anything.")
        if self.method == "pca":
            from sklearn.decomposition import PCA
            self.model_ = PCA(n_components=nc, random_state=self.seed)
        elif self.method == "svd":
            from sklearn.decomposition import TruncatedSVD
            self.model_ = TruncatedSVD(n_components=int(nc), random_state=self.seed)
        else:
            from sklearn.cluster import FeatureAgglomeration
            self.model_ = FeatureAgglomeration(n_clusters=int(nc))
        self.model_.fit(X)
        self.n_features_in_ = n_feat
        out = self.model_.transform(X[:1])
        self.n_out_ = out.shape[1]
        return self

    def transform(self, X):
        return self.model_.transform(np.asarray(X, dtype=float))

    @property
    def explained_variance_ratio_(self):
        return getattr(self.model_, "explained_variance_ratio_", None)

    def get_feature_names_out(self, input_features=None):
        prefix = {"pca": "pc", "svd": "svd", "agglomeration": "group"}[self.method]
        return np.array([f"{prefix}{i + 1}" for i in range(self.n_out_)])

    def summary(self, input_features: list[str] | None = None) -> dict[str, Any]:
        d: dict[str, Any] = {"method": self.method, "inputs": self.n_features_in_, "outputs": self.n_out_}
        ev = self.explained_variance_ratio_
        if ev is not None:
            d["explained_variance"] = [round(float(v), 4) for v in ev]
            d["explained_total"] = round(float(np.sum(ev)), 4)
        if self.method == "agglomeration" and input_features is not None:
            groups: dict[str, list[str]] = {}
            for f, lab in zip(input_features, self.model_.labels_):
                groups.setdefault(f"group{int(lab) + 1}", []).append(f)
            d["groups"] = groups
        return d


@dataclass
class Projection:
    method: str
    points: pd.DataFrame                  # columns dim1..dimN, plus 'row' (position in the input)
    explained_variance: list[float] = field(default_factory=list)
    loadings: pd.DataFrame | None = None
    reusable: bool = False                 # True if new rows can be transformed with the fitted pipeline
    pipeline: Any = None
    note: str = ""
    sampled: bool = False


def project(df: pd.DataFrame, columns: list[str], method: str = "pca", n_components: int = 2, seed: int = 42,
            max_rows: int = MAX_VIEW_ROWS, perplexity: float = 30.0) -> Projection:
    """Low-dimensional view of the selected columns (numeric scaled, categories one-hot encoded, missing values imputed)."""
    from sklearn.pipeline import Pipeline

    from .pipeline import PreprocessOptions, build_pipeline_steps, feature_names, select_features
    if method not in VIEW_METHODS:
        raise ReductionError(f"Unknown method '{method}'.")
    if method == "umap" and not UMAP_AVAILABLE:
        raise ReductionError("UMAP needs the optional package 'umap-learn' (pip install umap-learn). PCA and t-SNE are available without it.")
    if not 2 <= n_components <= 3 and method in ("tsne", "umap"):
        raise ReductionError("t-SNE and UMAP views support 2 or 3 dimensions.")
    fs = select_features(df, None, None, columns)
    if not fs.all:
        raise ReductionError("None of the selected columns can be used (identifiers, constants and empty columns are skipped).")
    work = df
    sampled = False
    if len(df) > max_rows:
        work = df.sample(max_rows, random_state=seed)
        sampled = True
    if len(work) < 5:
        raise ReductionError("At least 5 rows are needed.")
    prep = Pipeline(build_pipeline_steps(fs, PreprocessOptions(scaling="standard"), True))
    Xt = np.asarray(prep.fit_transform(work[fs.all]), dtype=float)
    n_feat = Xt.shape[1]
    if method in ("pca", "svd") and n_components >= n_feat:
        raise ReductionError(f"The selected columns give {n_feat} prepared features; choose fewer than {n_feat} components.")
    explained: list[float] = []
    loadings = None
    reusable = False
    pipe = None
    note = ""
    if method in ("pca", "svd"):
        red = Reducer(method, int(n_components), seed).fit(Xt)
        pts = red.transform(Xt)
        explained = [round(float(v), 4) for v in (red.explained_variance_ratio_ if red.explained_variance_ratio_ is not None else [])]
        names = feature_names(prep) or [f"f{i}" for i in range(n_feat)]
        loadings = pd.DataFrame(red.model_.components_.T, index=names, columns=[f"dim{i + 1}" for i in range(pts.shape[1])])
        reusable = True
        pipe = Pipeline(list(prep.steps) + [("reduce", red)])
        note = "Fitted on the rows shown; it can transform new rows with the same columns."
    elif method == "tsne":
        from sklearn.manifold import TSNE
        perp = float(min(perplexity, max(5.0, (len(Xt) - 1) / 3)))
        pts = TSNE(n_components=int(n_components), random_state=seed, perplexity=perp, init="pca").fit_transform(Xt)
        note = "t-SNE preserves local neighbourhoods, not global distances, and cannot transform new rows. Use it to look, not to model."
    else:
        import umap  # type: ignore
        pts = umap.UMAP(n_components=int(n_components), random_state=seed).fit_transform(Xt)
        note = "UMAP is used here for visualisation only."
    points = pd.DataFrame(pts, columns=[f"dim{i + 1}" for i in range(pts.shape[1])])
    points.insert(0, "row", list(work.index))
    return Projection(method, points, explained, loadings, reusable, pipe, note, sampled)
