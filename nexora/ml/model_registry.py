"""Model registry, model metadata, parameter validation, search spaces and data-driven recommendations.

Every model the application offers is described here once. Training, cross-validation, tuning, explainability,
persistence and the UI all read the same :class:`ModelSpec`, so adding a model means adding one entry.

Optional libraries (XGBoost, LightGBM) are registered but marked unavailable when the package is not installed;
the rest of the application keeps working and the UI shows the install command.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering, Birch, KMeans, MeanShift, SpectralClustering
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

XGB_AVAILABLE = importlib.util.find_spec("xgboost") is not None
LGBM_AVAILABLE = importlib.util.find_spec("lightgbm") is not None
INSTALL_HINT = {"xgboost": "pip install xgboost", "lightgbm": "pip install lightgbm"}


class ModelConfigError(ValueError):
    """A model setting is not valid. The message is safe to show to the user."""


@dataclass
class ParamSpec:
    name: str
    kind: str                      # int | float | choice | bool
    default: Any
    low: float | None = None
    high: float | None = None
    choices: list[Any] = field(default_factory=list)
    help: str = ""
    log: bool = False


@dataclass
class ModelSpec:
    key: str
    task: str                       # classification | regression | clustering
    name: str
    factory: Callable[..., Any]     # (seed, class_weight, device) -> estimator
    description: str
    tradeoff: str
    interpretable: bool
    scale_sensitive: bool
    supports_class_weight: bool = False
    gpu_capable: bool = False
    speed: str = "fast"             # fast | medium | slow
    proba: bool = True
    sample_weight: bool = True
    native_missing: bool = False
    predicts_new: bool = True
    tunable: bool = True
    params: list[ParamSpec] = field(default_factory=list)
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    max_rows: int | None = None
    warn_rows: int | None = None
    requires: str = ""              # "" | nonnegative | binary-ish
    force_scaling: str | None = None
    optional_package: str | None = None
    k_param: str | None = None      # clustering: the parameter that sets the number of groups
    param_prefix: str = ""          # when the estimator is wrapped, user settings go to the inner one (e.g. "estimator__")

    @property
    def available(self) -> bool:
        if self.optional_package == "xgboost":
            return XGB_AVAILABLE
        if self.optional_package == "lightgbm":
            return LGBM_AVAILABLE
        return True

    def metadata(self) -> dict[str, Any]:
        return {"task": self.task, "scaling": "required" if self.scale_sensitive else "not needed", "probabilities": self.proba,
                "sample_weights": self.sample_weight, "native_missing_values": self.native_missing, "predicts_new_rows": self.predicts_new,
                "tunable": self.tunable and bool(self.search_space), "class_weights": self.supports_class_weight, "gpu": self.gpu_capable,
                "speed": self.speed, "row_limit": self.max_rows, "requires": self.requires or "-", "optional_package": self.optional_package or "-",
                "available": self.available}


def _p(name, kind, default, low=None, high=None, choices=None, help="", log=False) -> ParamSpec:
    return ParamSpec(name, kind, default, low, high, list(choices or []), help, log)


def validate_params(spec: ModelSpec, params: dict[str, Any] | None) -> dict[str, Any]:
    """Check user-supplied parameters against the spec's declared ranges. Returns the cleaned values."""
    out: dict[str, Any] = {}
    known = {p.name: p for p in spec.params}
    for name, value in (params or {}).items():
        if name not in known:
            raise ModelConfigError(f"{spec.name} has no setting '{name}'. Available: {', '.join(known) or 'none'}.")
        p = known[name]
        if value is None and p.kind != "choice":
            continue
        if p.kind == "int":
            try:
                v = int(value)
            except (TypeError, ValueError):
                raise ModelConfigError(f"{spec.name}: '{name}' must be a whole number.") from None
        elif p.kind == "float":
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ModelConfigError(f"{spec.name}: '{name}' must be a number.") from None
        elif p.kind == "bool":
            v = bool(value)
        else:
            if value not in p.choices:
                raise ModelConfigError(f"{spec.name}: '{name}' must be one of {p.choices}.")
            v = value
        if p.kind in ("int", "float"):
            if p.low is not None and v < p.low or p.high is not None and v > p.high:
                raise ModelConfigError(f"{spec.name}: '{name}' must be between {p.low} and {p.high} (got {value}).")
        out[name] = v
    return out


def build_estimator(spec: ModelSpec, seed: int, class_weight: bool, device: str, params: dict[str, Any] | None = None):
    """Create the estimator with validated user parameters applied."""
    if not spec.available:
        raise ModelConfigError(f"{spec.name} needs the optional package '{spec.optional_package}' ({INSTALL_HINT.get(spec.optional_package or '', '')}).")
    est = spec.factory(seed, class_weight, device)
    clean = validate_params(spec, params)
    if clean:
        est.set_params(**{spec.param_prefix + k: v for k, v in clean.items()})
    return est


def search_space_for(spec: ModelSpec) -> dict[str, list[Any]]:
    """The bounded search space, with parameter names as the fitted pipeline sees them."""
    return {f"model__{spec.param_prefix}{k}": list(v) for k, v in spec.search_space.items()}


class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    """Lets estimators that need integer labels (XGBoost) work with text class labels."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, **kw):
        self.le_ = LabelEncoder().fit(y)
        self.estimator_ = clone(self.estimator).fit(X, self.le_.transform(y), **kw)
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X):
        return self.le_.inverse_transform(np.asarray(self.estimator_.predict(X)).astype(int))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.estimator_.feature_importances_


REGISTRY: dict[str, ModelSpec] = {}


def _reg(spec: ModelSpec) -> None:
    REGISTRY[spec.key] = spec


def _cw(flag: bool):
    return "balanced" if flag else None


# ---------------------------------------------------------------- classification
_reg(ModelSpec("logistic_regression", "classification", "Logistic Regression",
               lambda s, c, d: LogisticRegression(max_iter=2000, class_weight=_cw(c), random_state=s),
               "A linear model that estimates class probabilities from a weighted sum of the features.",
               "Easy to explain and fast, but cannot capture curved or interacting effects unless you add them.", True, True, True,
               params=[_p("C", "float", 1.0, 1e-4, 1e4, help="Inverse regularisation strength: smaller values give a simpler model.", log=True),
                       _p("max_iter", "int", 2000, 100, 20000)],
               search_space={"C": [0.01, 0.1, 1.0, 10.0]}))
_reg(ModelSpec("decision_tree", "classification", "Decision Tree",
               lambda s, c, d: DecisionTreeClassifier(max_depth=6, min_samples_leaf=5, class_weight=_cw(c), random_state=s),
               "Splits the data with a sequence of yes/no questions. The result can be read as rules.",
               "Very readable, but a single tree is unstable and can overfit small data.", True, False, True,
               params=[_p("max_depth", "int", 6, 1, 50), _p("min_samples_leaf", "int", 5, 1, 500), _p("min_samples_split", "int", 2, 2, 500)],
               search_space={"max_depth": [3, 5, 8, 12], "min_samples_leaf": [1, 5, 20]}))
_reg(ModelSpec("random_forest", "classification", "Random Forest",
               lambda s, c, d: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=-1, class_weight=_cw(c), random_state=s),
               "Averages many decision trees trained on random parts of the data.",
               "Usually accurate and robust with little tuning; harder to explain and slower than one tree.", False, False, True, speed="medium",
               params=[_p("n_estimators", "int", 200, 10, 2000), _p("max_depth", "int", None, 1, 100), _p("min_samples_leaf", "int", 2, 1, 200),
                       _p("max_features", "choice", "sqrt", choices=["sqrt", "log2", None])],
               search_space={"n_estimators": [100, 300], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 3], "max_features": ["sqrt", "log2"]}))
_reg(ModelSpec("gradient_boosting", "classification", "Gradient Boosted Trees",
               lambda s, c, d: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight=_cw(c), random_state=s),
               "Builds trees one after another, each correcting the previous errors (scikit-learn HistGradientBoosting).",
               "Often the most accurate on tabular data; more settings to tune and harder to explain.", False, False, True, speed="medium", native_missing=True,
               params=[_p("learning_rate", "float", 0.08, 0.001, 1.0, log=True), _p("max_iter", "int", 200, 10, 2000), _p("max_depth", "int", None, 1, 50),
                       _p("min_samples_leaf", "int", 20, 1, 500), _p("l2_regularization", "float", 0.0, 0.0, 100.0)],
               search_space={"learning_rate": [0.03, 0.08, 0.2], "max_depth": [None, 4, 8], "min_samples_leaf": [10, 30]}))
_reg(ModelSpec("extra_trees", "classification", "Extra Trees",
               lambda s, c, d: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, class_weight=_cw(c), random_state=s),
               "Like a random forest, but split points are chosen at random, which reduces variance further.",
               "Fast to train and robust; slightly less accurate than a tuned forest on some data and hard to explain.", False, False, True, speed="medium",
               params=[_p("n_estimators", "int", 300, 10, 2000), _p("max_depth", "int", None, 1, 100), _p("min_samples_leaf", "int", 2, 1, 200)],
               search_space={"n_estimators": [200, 500], "max_depth": [None, 12], "min_samples_leaf": [1, 3]}))
_reg(ModelSpec("svc", "classification", "Support Vector Machine",
               lambda s, c, d: CalibratedClassifierCV(SVC(C=1.0, kernel="rbf", gamma="scale", class_weight=_cw(c), random_state=s), ensemble=False, cv=3),
               "Finds the boundary between classes with the widest margin; a kernel lets it draw curved boundaries.",
               "Strong on small and medium data, but training time grows quickly with rows, and probabilities come from an extra internal calibration fit.",
               False, True, True, speed="slow", max_rows=20000, warn_rows=5000, param_prefix="estimator__",
               params=[_p("C", "float", 1.0, 1e-3, 1e4, log=True, help="Larger C fits the training data more closely."),
                       _p("kernel", "choice", "rbf", choices=["rbf", "linear", "poly", "sigmoid"]), _p("gamma", "choice", "scale", choices=["scale", "auto"])],
               search_space={"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"], "gamma": ["scale", "auto"]}))
_reg(ModelSpec("knn", "classification", "K-Nearest Neighbours",
               lambda s, c, d: KNeighborsClassifier(n_neighbors=7),
               "Predicts the majority class among the most similar training rows.",
               "No training phase and easy to explain, but prediction is slow on large data and every feature must be on a comparable scale.",
               True, True, False, sample_weight=False, max_rows=200000, warn_rows=50000,
               params=[_p("n_neighbors", "int", 7, 1, 200), _p("weights", "choice", "uniform", choices=["uniform", "distance"]),
                       _p("metric", "choice", "minkowski", choices=["minkowski", "manhattan", "chebyshev"])],
               search_space={"n_neighbors": [3, 7, 15, 31], "weights": ["uniform", "distance"]}))
_reg(ModelSpec("gaussian_nb", "classification", "Gaussian Naive Bayes",
               lambda s, c, d: GaussianNB(),
               "Assumes each feature follows a bell curve within each class and that features are independent.",
               "Extremely fast and works with few rows, but the independence assumption is rarely true, so probabilities are often over-confident.",
               True, False, False,
               params=[_p("var_smoothing", "float", 1e-9, 1e-12, 1.0, log=True)], search_space={"var_smoothing": [1e-10, 1e-9, 1e-8, 1e-6]}))
_reg(ModelSpec("multinomial_nb", "classification", "Multinomial Naive Bayes",
               lambda s, c, d: MultinomialNB(alpha=1.0),
               "Naive Bayes for counts and frequencies, such as word weights from text.",
               "Very fast and suited to text; needs non-negative inputs, so numbers are min-max scaled to 0-1 and dimensionality reduction cannot be used.",
               True, False, False, requires="nonnegative", force_scaling="minmax",
               params=[_p("alpha", "float", 1.0, 0.0, 100.0, help="Smoothing for unseen combinations.")], search_space={"alpha": [0.1, 0.5, 1.0, 2.0]}))
_reg(ModelSpec("bernoulli_nb", "classification", "Bernoulli Naive Bayes",
               lambda s, c, d: BernoulliNB(alpha=1.0, binarize=0.0),
               "Naive Bayes for yes/no features. Each prepared feature is treated as present when it is above its training average.",
               "Fast and suited to binary indicators; loses information from continuous numbers.", True, True, False,
               params=[_p("alpha", "float", 1.0, 0.0, 100.0)], search_space={"alpha": [0.1, 0.5, 1.0, 2.0]}))
_reg(ModelSpec("adaboost", "classification", "AdaBoost",
               lambda s, c, d: AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=s),
               "Boosts many shallow trees, giving more weight to rows earlier trees got wrong.",
               "Simple boosting that rarely needs much tuning; sensitive to noisy labels and outliers.", False, False, False, speed="medium",
               params=[_p("n_estimators", "int", 100, 10, 2000), _p("learning_rate", "float", 0.5, 0.001, 5.0, log=True)],
               search_space={"n_estimators": [50, 100, 300], "learning_rate": [0.1, 0.5, 1.0]}))


def _xgb_c(seed, cw, dev):
    from xgboost import XGBClassifier
    return LabelEncodedClassifier(XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=5, tree_method="hist", random_state=seed, n_jobs=-1,
                                                device="cuda" if dev == "gpu" else "cpu", verbosity=0))


def _lgbm_c(seed, cw, dev):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight=_cw(cw), random_state=seed, n_jobs=1, verbose=-1)


_reg(ModelSpec("xgboost", "classification", "XGBoost", _xgb_c, "Optimised gradient boosted trees (optional package).",
               "Strong accuracy and optional GPU use; an extra dependency, and class weighting is not applied.", False, False, False, True, "medium",
               native_missing=True, optional_package="xgboost", param_prefix="estimator__",
               params=[_p("n_estimators", "int", 300, 10, 3000), _p("learning_rate", "float", 0.08, 0.001, 1.0, log=True), _p("max_depth", "int", 5, 1, 16),
                       _p("subsample", "float", 1.0, 0.3, 1.0), _p("colsample_bytree", "float", 1.0, 0.3, 1.0)],
               search_space={"n_estimators": [200, 500], "learning_rate": [0.03, 0.1], "max_depth": [3, 6]}))
_reg(ModelSpec("lightgbm", "classification", "LightGBM", _lgbm_c, "Fast leaf-wise gradient boosted trees (optional package).",
               "Very fast on large data and usually accurate; can overfit small data unless leaves and minimum samples are limited.", False, False, True,
               speed="medium", native_missing=True, optional_package="lightgbm",
               params=[_p("n_estimators", "int", 300, 10, 3000), _p("learning_rate", "float", 0.05, 0.001, 1.0, log=True), _p("num_leaves", "int", 31, 2, 512),
                       _p("min_child_samples", "int", 20, 1, 500)],
               search_space={"n_estimators": [200, 500], "learning_rate": [0.03, 0.1], "num_leaves": [15, 31, 63]}))

# ---------------------------------------------------------------- regression
_reg(ModelSpec("linear_regression", "regression", "Linear Regression", lambda s, c, d: LinearRegression(),
               "Fits a straight-line (weighted sum) relationship between the features and the number to predict.",
               "Simple and explainable, but misses curved or interacting effects and is sensitive to extreme values.", True, False, tunable=False))
_reg(ModelSpec("ridge", "regression", "Ridge Regression", lambda s, c, d: Ridge(alpha=1.0),
               "Linear regression with a penalty that shrinks coefficients, which stabilises it when features are correlated.",
               "Explainable and robust to correlated features; still linear.", True, True,
               params=[_p("alpha", "float", 1.0, 1e-6, 1e6, log=True)], search_space={"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}))
_reg(ModelSpec("lasso", "regression", "Lasso Regression", lambda s, c, d: Lasso(alpha=0.1, max_iter=20000),
               "Linear regression whose penalty can set coefficients to exactly zero, so it also selects features.",
               "Produces a simpler model; with strongly correlated features it keeps one and drops the others somewhat arbitrarily.", True, True,
               params=[_p("alpha", "float", 0.1, 1e-6, 1e4, log=True), _p("max_iter", "int", 20000, 1000, 200000)], search_space={"alpha": [0.001, 0.01, 0.1, 1.0]}))
_reg(ModelSpec("elastic_net", "regression", "Elastic Net", lambda s, c, d: ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
               "Combines the Ridge and Lasso penalties.", "Balances stability and feature selection; two settings to tune.", True, True,
               params=[_p("alpha", "float", 0.1, 1e-6, 1e4, log=True), _p("l1_ratio", "float", 0.5, 0.0, 1.0)],
               search_space={"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}))
_reg(ModelSpec("decision_tree_regressor", "regression", "Decision Tree Regressor",
               lambda s, c, d: DecisionTreeRegressor(max_depth=7, min_samples_leaf=5, random_state=s),
               "Splits the data into groups and predicts each group's average.", "Readable rules, but predictions are step-like and a single tree overfits easily.", True, False,
               params=[_p("max_depth", "int", 7, 1, 50), _p("min_samples_leaf", "int", 5, 1, 500)],
               search_space={"max_depth": [3, 5, 8, 12], "min_samples_leaf": [1, 5, 20]}))
_reg(ModelSpec("random_forest_regressor", "regression", "Random Forest Regressor",
               lambda s, c, d: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=s),
               "Averages many regression trees.", "Robust and accurate with little tuning; cannot predict beyond the range seen in training and is harder to explain.",
               False, False, speed="medium",
               params=[_p("n_estimators", "int", 200, 10, 2000), _p("max_depth", "int", None, 1, 100), _p("min_samples_leaf", "int", 2, 1, 200),
                       _p("max_features", "choice", 1.0, choices=[1.0, "sqrt", "log2"])],
               search_space={"n_estimators": [100, 300], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3]}))
_reg(ModelSpec("gradient_boosting_regressor", "regression", "Gradient Boosted Trees",
               lambda s, c, d: HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08, random_state=s),
               "Builds trees sequentially, each correcting the previous errors (scikit-learn HistGradientBoosting).",
               "Often the most accurate on tabular data; more settings and harder to explain.", False, False, speed="medium", native_missing=True,
               params=[_p("learning_rate", "float", 0.08, 0.001, 1.0, log=True), _p("max_iter", "int", 250, 10, 3000), _p("max_depth", "int", None, 1, 50),
                       _p("min_samples_leaf", "int", 20, 1, 500), _p("l2_regularization", "float", 0.0, 0.0, 100.0)],
               search_space={"learning_rate": [0.03, 0.08, 0.2], "max_depth": [None, 4, 8], "min_samples_leaf": [10, 30]}))
_reg(ModelSpec("extra_trees_regressor", "regression", "Extra Trees Regressor",
               lambda s, c, d: ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=s),
               "Randomised tree ensemble for numbers.", "Fast and robust; cannot extrapolate beyond the training range.", False, False, speed="medium",
               params=[_p("n_estimators", "int", 300, 10, 2000), _p("max_depth", "int", None, 1, 100), _p("min_samples_leaf", "int", 2, 1, 200)],
               search_space={"n_estimators": [200, 500], "max_depth": [None, 12], "min_samples_leaf": [1, 3]}))
_reg(ModelSpec("svr", "regression", "Support Vector Regression", lambda s, c, d: SVR(C=1.0, epsilon=0.1, kernel="rbf", gamma="scale"),
               "Fits a function that keeps most training points within a margin; kernels allow curved relationships.",
               "Good on small and medium data with scaled features; slow on large data and sensitive to C and epsilon, which depend on the target's scale.",
               False, True, speed="slow", max_rows=20000, warn_rows=5000,
               params=[_p("C", "float", 1.0, 1e-3, 1e5, log=True), _p("epsilon", "float", 0.1, 0.0, 1e4), _p("kernel", "choice", "rbf", choices=["rbf", "linear", "poly"]),
                       _p("gamma", "choice", "scale", choices=["scale", "auto"])],
               search_space={"C": [1.0, 10.0, 100.0], "epsilon": [0.01, 0.1, 1.0], "kernel": ["rbf", "linear"]}))
_reg(ModelSpec("knn_regressor", "regression", "K-Nearest Neighbours Regressor", lambda s, c, d: KNeighborsRegressor(n_neighbors=7),
               "Predicts the average of the most similar training rows.", "Simple and flexible; slow predictions on large data and needs scaled features.",
               True, True, sample_weight=False, max_rows=200000, warn_rows=50000,
               params=[_p("n_neighbors", "int", 7, 1, 200), _p("weights", "choice", "uniform", choices=["uniform", "distance"]),
                       _p("metric", "choice", "minkowski", choices=["minkowski", "manhattan"])],
               search_space={"n_neighbors": [3, 7, 15, 31], "weights": ["uniform", "distance"]}))
_reg(ModelSpec("adaboost_regressor", "regression", "AdaBoost Regressor", lambda s, c, d: AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=s),
               "Boosts shallow trees, focusing on rows with large errors.", "Simple boosting; sensitive to outliers in the target.", False, False, speed="medium",
               params=[_p("n_estimators", "int", 100, 10, 2000), _p("learning_rate", "float", 0.5, 0.001, 5.0, log=True),
                       _p("loss", "choice", "linear", choices=["linear", "square", "exponential"])],
               search_space={"n_estimators": [50, 150], "learning_rate": [0.1, 0.5, 1.0], "loss": ["linear", "square"]}))


def _xgb_r(seed, cw, dev):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, learning_rate=0.08, max_depth=5, tree_method="hist", random_state=seed, n_jobs=-1,
                        device="cuda" if dev == "gpu" else "cpu", verbosity=0)


def _lgbm_r(seed, cw, dev):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=seed, n_jobs=1, verbose=-1)


_reg(ModelSpec("xgboost_regressor", "regression", "XGBoost Regressor", _xgb_r, "Optimised gradient boosting for numbers (optional package).",
               "Strong accuracy and optional GPU use; an extra dependency.", False, False, False, True, "medium", native_missing=True, optional_package="xgboost",
               params=[_p("n_estimators", "int", 300, 10, 3000), _p("learning_rate", "float", 0.08, 0.001, 1.0, log=True), _p("max_depth", "int", 5, 1, 16)],
               search_space={"n_estimators": [200, 500], "learning_rate": [0.03, 0.1], "max_depth": [3, 6]}))
_reg(ModelSpec("lightgbm_regressor", "regression", "LightGBM Regressor", _lgbm_r, "Fast leaf-wise gradient boosting for numbers (optional package).",
               "Very fast and usually accurate; limit leaves on small data.", False, False, speed="medium", native_missing=True, optional_package="lightgbm",
               params=[_p("n_estimators", "int", 300, 10, 3000), _p("learning_rate", "float", 0.05, 0.001, 1.0, log=True), _p("num_leaves", "int", 31, 2, 512),
                       _p("min_child_samples", "int", 20, 1, 500)],
               search_space={"n_estimators": [200, 500], "learning_rate": [0.03, 0.1], "num_leaves": [15, 31, 63]}))

# ---------------------------------------------------------------- clustering
_reg(ModelSpec("kmeans", "clustering", "K-Means", lambda s, c, d: KMeans(n_clusters=3, n_init=10, random_state=s),
               "Assigns each row to the nearest of k centres. You (or the search) choose k.",
               "Fast and simple, but assumes round, similar-sized groups and every row is forced into a group.", True, True, k_param="n_clusters"))
_reg(ModelSpec("dbscan", "clustering", "DBSCAN", lambda s, c, d: DBSCAN(eps=0.5, min_samples=5),
               "Finds dense regions and marks isolated rows as noise. The number of groups is not chosen in advance.",
               "Handles odd shapes and outliers, but is sensitive to its distance setting and struggles with groups of different density.", False, True,
               predicts_new=False))
_reg(ModelSpec("agglomerative", "clustering", "Hierarchical (Ward)", lambda s, c, d: AgglomerativeClustering(n_clusters=3, linkage="ward"),
               "Repeatedly merges the closest groups until k remain.", "Good for small or medium data; memory use grows quickly with rows, so it is capped at 5,000 rows here.",
               True, True, speed="slow", predicts_new=False, max_rows=5000, k_param="n_clusters",
               params=[_p("linkage", "choice", "ward", choices=["ward", "average", "complete", "single"])]))
_reg(ModelSpec("gaussian_mixture", "clustering", "Gaussian Mixture", lambda s, c, d: GaussianMixture(n_components=3, covariance_type="full", random_state=s),
               "Models the data as a mix of overlapping bell-shaped groups and gives each row a membership probability.",
               "Allows elliptical, differently sized groups; can be unstable with many features or few rows.", False, True, speed="medium", k_param="n_components",
               params=[_p("covariance_type", "choice", "full", choices=["full", "tied", "diag", "spherical"])]))
_reg(ModelSpec("birch", "clustering", "BIRCH", lambda s, c, d: Birch(n_clusters=3, threshold=0.5),
               "Builds a compact summary tree of the data and clusters it; designed for large datasets.",
               "Scales well; sensitive to the threshold setting and best for compact, round groups.", False, True, k_param="n_clusters",
               params=[_p("threshold", "float", 0.5, 0.01, 10.0)]))
_reg(ModelSpec("mean_shift", "clustering", "Mean Shift", lambda s, c, d: MeanShift(),
               "Moves towards the densest areas of the data; the number of groups follows from a bandwidth setting.",
               "No need to choose k, but slow on large data (capped at 10,000 rows here) and the bandwidth controls the result.", False, True,
               speed="slow", max_rows=10000))
_reg(ModelSpec("spectral", "clustering", "Spectral Clustering", lambda s, c, d: SpectralClustering(n_clusters=3, affinity="nearest_neighbors", random_state=s),
               "Clusters a similarity graph of the rows, which can separate groups with complex shapes.",
               "Finds non-convex groups, but memory and time grow quickly (capped at 3,000 rows) and new rows cannot be assigned.", False, True,
               speed="slow", predicts_new=False, max_rows=3000, k_param="n_clusters"))
_reg(ModelSpec("optics", "clustering", "OPTICS", lambda s, c, d: OPTICS(min_samples=10),
               "A density method like DBSCAN that copes with groups of different density and marks isolated rows as noise.",
               "No fixed distance setting needed; slower than DBSCAN (capped at 10,000 rows) and new rows cannot be assigned.", False, True,
               speed="slow", predicts_new=False, max_rows=10000, params=[_p("min_samples", "int", 10, 2, 200), _p("xi", "float", 0.05, 0.001, 0.5)]))

AGGLOMERATIVE_MAX_ROWS = 5000


def models_for(task: str, include_unavailable: bool = False) -> list[ModelSpec]:
    return [m for m in REGISTRY.values() if m.task == task and (include_unavailable or m.available)]


def check_compatibility(spec: ModelSpec, n_rows: int, n_features: int = 0, options: dict[str, Any] | None = None) -> tuple[bool, list[str], list[str]]:
    """(usable, blocking reasons, warnings) for running ``spec`` on data of this size with these preprocessing options."""
    options = options or {}
    reasons: list[str] = []
    warns: list[str] = []
    if not spec.available:
        reasons.append(f"Needs the optional package '{spec.optional_package}'. Install it with: {INSTALL_HINT.get(spec.optional_package or '', '')}")
    if spec.max_rows and n_rows > spec.max_rows:
        reasons.append(f"{spec.name} is limited to {spec.max_rows:,} rows here (this data has {n_rows:,}); it would be too slow or use too much memory.")
    elif spec.warn_rows and n_rows > spec.warn_rows:
        warns.append(f"{spec.name} may be slow on {n_rows:,} rows.")
    red = (options.get("reduction") or {}).get("method", "none")
    if spec.requires == "nonnegative" and red in ("pca", "svd"):
        reasons.append(f"{spec.name} needs non-negative inputs, but {red.upper()} produces negative components. Remove dimensionality reduction or pick another model.")
    if spec.requires == "nonnegative" and options.get("skew_transform"):
        warns.append(f"{spec.name}: the skew transform is followed by 0-1 scaling so inputs stay non-negative.")
    if spec.key in ("knn", "knn_regressor") and n_features > 100:
        warns.append("Nearest-neighbour methods work poorly with many features (distances become similar); consider feature selection or reduction.")
    return (not reasons), reasons, warns


@dataclass
class Recommendation:
    key: str
    rank: int
    reason: str
    tradeoff: str


def recommend_models(task: str, n_rows: int, n_numeric: int, n_categorical: int, *, imbalance_ratio: float | None = None,
                     prefer: str = "balanced") -> list[Recommendation]:
    """Rank candidate models from data characteristics. A starting point, not a verdict: training decides."""
    avail = {m.key: m for m in models_for(task) if check_compatibility(m, n_rows)[0]}
    feats = n_numeric + n_categorical
    order: list[tuple[str, str]] = []
    if task == "classification":
        if prefer == "interpretable":
            order = [("logistic_regression", "you asked for explainable results"), ("decision_tree", "its rules can be read directly")]
        elif prefer == "speed":
            order = [("logistic_regression", "it trains in well under a second"), ("gaussian_nb", "it trains almost instantly"), ("decision_tree", "it is very fast")]
        elif n_rows < 500:
            order = [("logistic_regression", f"with only {n_rows:,} rows a simple model is less likely to overfit"),
                     ("svc", f"support vector machines work well on small data such as these {n_rows:,} rows"),
                     ("random_forest", "it copes with small data better than boosting")]
        else:
            order = [("gradient_boosting", f"with {n_rows:,} rows and {feats} features, boosted trees usually score best on tabular data"),
                     ("random_forest", "a robust second opinion that needs little tuning")]
        order += [("logistic_regression", "a transparent baseline to compare against"), ("random_forest", "robust default"),
                  ("decision_tree", "readable rules"), ("gradient_boosting", "strong on tabular data")]
        if imbalance_ratio is not None and imbalance_ratio < 0.25:
            order = [(k, r + (f"; class weighting is supported (smallest class is {100 * imbalance_ratio:.0f}% of the largest)"
                              if k in avail and avail[k].supports_class_weight and i == 0 else ""))
                     for i, (k, r) in enumerate(order)]
    elif task == "regression":
        if prefer == "interpretable":
            order = [("linear_regression", "its coefficients show each feature's effect"), ("ridge", "a linear model that stays stable with correlated features"),
                     ("decision_tree_regressor", "its rules can be read directly")]
        elif prefer == "speed":
            order = [("linear_regression", "it fits instantly"), ("ridge", "it fits instantly and is more stable")]
        elif n_rows < 300:
            order = [("linear_regression", f"with only {n_rows:,} rows a simple model is less likely to overfit"), ("ridge", "regularisation helps with few rows"),
                     ("random_forest_regressor", "robust to noise")]
        else:
            order = [("gradient_boosting_regressor", f"with {n_rows:,} rows boosted trees usually give the lowest error on tabular data"),
                     ("random_forest_regressor", "a robust second opinion")]
        order += [("linear_regression", "a transparent baseline"), ("random_forest_regressor", "robust default"),
                  ("decision_tree_regressor", "readable rules"), ("gradient_boosting_regressor", "strong on tabular data")]
    else:
        order = [("kmeans", "fast and a standard first choice when you expect compact groups"),
                 ("gaussian_mixture", "allows overlapping, elliptical groups with membership probabilities"),
                 ("agglomerative", "shows how groups merge; practical for up to 5,000 rows"), ("dbscan", "useful if groups are irregular or there are outliers")]
    out, seen = [], set()
    for k, why in order:
        if k in avail and k not in seen:
            seen.add(k)
            out.append(Recommendation(k, len(out) + 1, why[0].upper() + why[1:] + ".", avail[k].tradeoff))
    return out
