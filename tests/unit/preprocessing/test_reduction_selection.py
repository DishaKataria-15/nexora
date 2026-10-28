import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from nexora.preprocessing.pipeline import PreprocessOptions, build_pipeline_steps, feature_names, pipeline_reports, select_features
from nexora.preprocessing.reduction import ReductionError, Reducer, project
from nexora.preprocessing.selection import FeatureSelector, SelectionError


@pytest.fixture()
def clf():
    rng = np.random.RandomState(0)
    n = 300
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    X["const"] = 1.0
    y = (X["f0"] + 0.8 * X["f1"] + rng.normal(0, 0.3, n) > 0).astype(int)
    return X, y


def test_reducer_fits_on_training_rows_only(clf):
    X, _ = clf
    tr, te = X.iloc[:200].drop(columns="const").values, X.iloc[200:].drop(columns="const").values * 50
    r = Reducer("pca", 3).fit(tr)
    ref = PCA(3, random_state=42).fit(tr)
    assert np.allclose(np.abs(r.model_.components_), np.abs(ref.components_)) and np.allclose(r.transform(te), ref.transform(te))
    assert list(r.get_feature_names_out()) == ["pc1", "pc2", "pc3"] and 0 < r.summary()["explained_total"] <= 1


def test_reducer_options_and_validation(clf):
    X = clf[0].drop(columns="const").values
    assert Reducer("pca", 0.9).fit(X).n_out_ <= 8
    assert Reducer("svd", 2).fit(X).transform(X).shape == (300, 2)
    agg = Reducer("agglomeration", 3).fit(X)
    assert agg.transform(X).shape == (300, 3) and len(agg.summary([f"f{i}" for i in range(8)])["groups"]) == 3
    with pytest.raises(ReductionError, match="would not reduce"):
        Reducer("pca", 8).fit(X)
    with pytest.raises(ReductionError, match="only valid for PCA"):
        Reducer("svd", 0.5).fit(X)


@pytest.mark.parametrize("method,extra", [("variance", {}), ("kbest", {"k": 2}), ("percentile", {"percentile": 25}), ("mutual_info", {"k": 2}),
                                          ("rfe", {"k": 2}), ("rfecv", {}), ("model", {"k": 2}), ("l1", {"strength": 0.05})])
def test_selection_methods_find_the_informative_features(clf, method, extra):
    X, y = clf
    s = FeatureSelector(method=method, task="classification", **extra).fit(X.values, y.values)
    kept = set(np.array(X.columns)[s.get_support()])
    if method == "variance":
        assert kept == set(X.columns) - {"const"}
    else:
        assert {"f0", "f1"} <= kept or (method in ("rfecv", "l1", "percentile") and "f0" in kept)
        assert "const" not in kept
    rep = s.report(list(X.columns))
    assert rep["before"] == 9 and rep["after"] == len(kept)


def test_selection_rules_for_tasks_and_bounds(clf):
    X, y = clf
    with pytest.raises(SelectionError, match="Without a target"):
        FeatureSelector(method="kbest", task="clustering").fit(X.values)
    assert FeatureSelector(method="variance", task="clustering").fit(X.values).get_support().sum() == 8
    assert FeatureSelector(method="kbest", k=500).fit(X.values, y).get_support().sum() == 9          # k larger than features is clipped
    reg = FeatureSelector(method="kbest", task="regression", k=1).fit(X.values, X["f0"].values * 3)
    assert np.array(X.columns)[reg.get_support()].tolist() == ["f0"]


def test_selection_inside_pipeline_never_sees_held_out_targets(clf):
    """Fit on training rows; then corrupt the held-out targets: the fitted selection must not change."""
    X, y = clf
    opts = PreprocessOptions(selection={"method": "kbest", "k": 2})
    fs = select_features(X, None, options=opts)
    tr = slice(0, 200)
    p1 = Pipeline(build_pipeline_steps(fs, opts, True, task="classification") + [("m", LogisticRegression())]).fit(X.iloc[tr][fs.all], y.iloc[tr])
    y2 = y.copy()
    y2.iloc[200:] = 1 - y2.iloc[200:]
    p2 = Pipeline(build_pipeline_steps(fs, opts, True, task="classification") + [("m", LogisticRegression())]).fit(X.iloc[tr][fs.all], y2.iloc[tr])
    assert feature_names(p1) == feature_names(p2) == ["f0", "f1"]
    assert pipeline_reports(p1)["selection"]["after"] == 2


def test_reduction_and_selection_combine_in_pipeline(clf):
    X, y = clf
    opts = PreprocessOptions(reduction={"method": "pca", "n_components": 4}, selection={"method": "kbest", "k": 2})
    fs = select_features(X, None, options=opts)
    p = Pipeline(build_pipeline_steps(fs, opts, False, task="regression") + [("m", Ridge())]).fit(X[fs.all], X["f0"])
    rep = pipeline_reports(p)
    assert rep["reduction"]["outputs"] == 4 and rep["selection"]["after"] == 2 and all(n.startswith("pc") for n in feature_names(p))
    assert p.named_steps["prep"].named_transformers_["num"].named_steps["scale"].__class__.__name__ == "StandardScaler"   # PCA forces scaling


def test_projection_views(clf):
    X, y = clf
    d = X.assign(cat=np.where(y == 1, "a", "b"))
    pca = project(d, list(d.columns), "pca", 2)
    assert pca.points.shape == (300, 3) and pca.reusable and len(pca.explained_variance) == 2 and pca.loadings is not None
    assert pca.pipeline.transform(d[[c for c in d.columns if c != "const"]].iloc[:3]).shape == (3, 2)
    ts = project(d.iloc[:120], list(d.columns), "tsne", 2)
    assert not ts.reusable and ts.pipeline is None and "cannot transform new rows" in ts.note
    big = project(pd.concat([d] * 20, ignore_index=True), ["f0", "f1", "f2"], "pca", 2, max_rows=1000)
    assert big.sampled and len(big.points) == 1000
    with pytest.raises(ReductionError, match="choose fewer"):
        project(d, ["f0", "f1"], "pca", 2)
