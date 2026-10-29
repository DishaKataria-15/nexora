"""Experiment orchestration: selection on validation/CV, one-time test evaluation, clustering, artifacts.

Test-set discipline is enforced here, not left to the user interface:
* ``run_selection`` never reads test rows.
* ``finalize_experiment`` evaluates the selected model on the test rows exactly once per experiment.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from ..cleaning.validation import frame_fingerprint
from ..preprocessing import leakage_checks as L
from ..preprocessing.pipeline import PreprocessOptions, build_pipeline_steps, pipeline_reports, select_features
from . import explain as XP
from . import evaluation as E
from . import imbalance as IMB
from .compute import ComputeError, detect_compute, resolve_device
from .cv import CVConfig, CVError, make_folds
from .model_registry import REGISTRY, ModelConfigError, build_estimator, check_compatibility, search_space_for, validate_params
from .tuning import TuningConfig, TuningError, tune
from .splitting import SplitConfig, SplitError, make_split

log = logging.getLogger(__name__)


class ExperimentError(ValueError):
    """The experiment cannot run as configured. The message is safe to show to the user."""


class TestSetAlreadyUsed(RuntimeError):
    """The test rows were already used to evaluate this experiment's final model."""
    __test__ = False


class TrainingCancelled(RuntimeError):
    pass


class CancelToken:
    def __init__(self) -> None:
        self._e = threading.Event()

    def cancel(self) -> None:
        self._e.set()

    @property
    def cancelled(self) -> bool:
        return self._e.is_set()

    def check(self) -> None:
        if self._e.is_set():
            raise TrainingCancelled("Training was cancelled.")


@dataclass
class ExperimentConfig:
    task: str
    target: str | None = None
    name: str = ""
    dataset_name: str = ""
    dataset_version: str = ""
    features_include: list[str] | None = None
    features_exclude: list[str] = field(default_factory=list)
    split: dict[str, Any] = field(default_factory=lambda: asdict(SplitConfig()))
    imbalance: str = "none"
    models: list[str] = field(default_factory=list)
    primary_metric: str = ""
    compute: str = "auto"
    seed: int = 42
    preprocess: dict[str, Any] = field(default_factory=lambda: PreprocessOptions().to_dict())
    cv_folds: int = 5
    positive_label: str | None = None
    drop_missing_target: bool = False
    cluster_k_min: int = 2
    cluster_k_max: int = 8
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)   # {model key: {setting: value}}, validated against the registry
    cv: dict[str, Any] | None = None          # CVConfig fields; None means k-fold with cv_folds folds chosen to match the split
    tuning: dict[str, Any] = field(default_factory=lambda: TuningConfig().to_dict())
    shap: bool = False                        # compute SHAP values for the selected model (optional package)
    extra_metrics: list[str] = field(default_factory=list)   # shown side by side in the comparison; the primary metric alone picks the model

    def cv_config(self) -> CVConfig:
        return CVConfig(**self.cv) if self.cv else CVConfig(folds=self.cv_folds)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    key: str
    name: str
    status: str = "ok"                 # ok | failed | skipped
    score: float | None = None
    score_std: float | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    seconds: float = 0.0
    device: str = "cpu"
    params: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    n_clusters: int | None = None
    noise_pct: float | None = None
    note: str = ""
    fold_scores: list[float] = field(default_factory=list)
    tuning: dict[str, Any] | None = None


@dataclass
class Experiment:
    config: ExperimentConfig
    created_at: str = ""
    status: str = "created"            # created | selection_complete | finalized | failed
    selection_basis: str = ""
    split_info: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    leakage: list[dict[str, Any]] = field(default_factory=list)
    feature_set: dict[str, Any] = field(default_factory=dict)
    class_distribution: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    selected_key: str = ""
    baseline_selection: dict[str, Any] | None = None
    test_result: dict[str, Any] | None = None
    test_baseline: dict[str, Any] | None = None
    test_evaluated: bool = False
    interpretation: list[str] = field(default_factory=list)
    importance: list[dict[str, Any]] = field(default_factory=list)
    cluster_profile: list[dict[str, Any]] = field(default_factory=list)
    projection: list[dict[str, Any]] = field(default_factory=list)
    compute_note: str = ""
    seconds: float = 0.0
    environment: dict[str, str] = field(default_factory=dict)
    dataset_fingerprint: str = ""
    prior_test_uses: int = 0
    cv_info: dict[str, Any] = field(default_factory=dict)
    native_importance: dict[str, Any] | None = None
    shap: dict[str, Any] | None = None
    pipeline_report: dict[str, Any] = field(default_factory=dict)
    _state: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def selected(self) -> Candidate | None:
        return next((c for c in self.candidates if c.key == self.selected_key), None)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_state", None)
        return _clean(d)


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return _clean(o.to_dict("records"))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def environment() -> dict[str, str]:
    env = {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}
    if IMB.IMBLEARN:
        import imblearn
        env["imbalanced-learn"] = imblearn.__version__
    return env


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pipeline(steps: list, sampler, model) -> Pipeline:
    if sampler is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline(steps + [("sampler", sampler), ("model", model)])
    return Pipeline(steps + [("model", model)])


def _proba(pipe, X, classes: list) -> np.ndarray | None:
    if not hasattr(pipe, "predict_proba"):
        return None
    p = pipe.predict_proba(X)
    fitted = list(pipe.classes_) if hasattr(pipe, "classes_") else list(pipe.named_steps["model"].classes_)
    if fitted == classes:
        return p
    out = np.zeros((len(X), len(classes)))
    for j, c in enumerate(fitted):
        out[:, classes.index(c)] = p[:, j]
    return out


def _score_split(task: str, pipe, X, y, classes, positive) -> dict[str, Any]:
    pred = pipe.predict(X)
    if task == "classification":
        return E.classification_metrics(y, pred, _proba(pipe, X, classes), classes, positive)
    return E.regression_metrics(y, pred)


def _mean_metrics(parts: list[dict[str, float | None]]) -> dict[str, float | None]:
    keys = set().union(*[p.keys() for p in parts])
    out = {}
    for k in keys:
        vals = [p[k] for p in parts if p.get(k) is not None]
        out[k] = float(np.mean(vals)) if len(vals) == len(parts) else None
    return out


def _better(a: float, b: float, metric: str) -> bool:
    return a > b if E.metric_direction(metric) == "higher" else a < b


def _check_model_params(cfg: ExperimentConfig) -> None:
    for k, params in (cfg.model_params or {}).items():
        if k not in REGISTRY:
            raise ExperimentError(f"Settings were given for unknown model '{k}'.")
        try:
            validate_params(REGISTRY[k], params)
        except ModelConfigError as e:
            raise ExperimentError(str(e)) from e


def _make_pipeline(spec, fs, options: PreprocessOptions, cfg: ExperimentConfig, sampler, use_cw: bool, device: str, params: dict[str, Any] | None = None):
    steps = build_pipeline_steps(fs, options, spec.scale_sensitive, task=cfg.task, seed=cfg.seed, force_scaling=spec.force_scaling)
    model = build_estimator(spec, cfg.seed, use_cw, device, cfg.model_params.get(spec.key) if params is None else params)
    return _pipeline(steps, clone(sampler) if sampler is not None else None, model)


def validate_target(df: pd.DataFrame, cfg: ExperimentConfig) -> list[str]:
    """Checks that must pass before any supervised run. Returns warnings; raises ExperimentError for blockers."""
    warns: list[str] = []
    if not cfg.target or cfg.target not in df.columns:
        raise ExperimentError("Choose the column you want to predict.")
    y = df[cfg.target]
    n_missing = int(y.isna().sum())
    if n_missing and not cfg.drop_missing_target:
        raise ExperimentError(f"{n_missing:,} row(s) have no value in '{cfg.target}'. Nexora never invents labels. "
                              "Confirm that these rows should be left out of training and evaluation to continue.")
    if n_missing:
        warns.append(f"{n_missing:,} row(s) without a '{cfg.target}' value were left out (labels are never invented).")
    yy = y.dropna()
    if cfg.task == "classification":
        vc = yy.astype(str).value_counts()
        if len(vc) < 2:
            raise ExperimentError("The target has only one class, so there is nothing to predict.")
        if len(vc) > max(30, int(0.2 * len(yy))):
            raise ExperimentError(f"'{cfg.target}' has {len(vc)} distinct values, which looks like a number or an identifier rather than a category. "
                                  "Choose 'predict a number' or a different target.")
        if vc.min() < 2:
            raise ExperimentError(f"Class '{vc.index[-1]}' has only one row, so it cannot appear in both training and test data. Merge or remove it first.")
    elif cfg.task == "regression":
        if not pd.api.types.is_numeric_dtype(yy) or pd.api.types.is_bool_dtype(yy):
            raise ExperimentError(f"'{cfg.target}' is not numeric, so it cannot be predicted as a number.")
        if yy.nunique() < 2:
            raise ExperimentError("The target has the same value in every row.")
        if not np.isfinite(yy.astype(float)).all():
            raise ExperimentError("The target contains infinite values.")
    return warns


def run_selection(df: pd.DataFrame, cfg: ExperimentConfig, *, cancel: CancelToken | None = None,
                  progress: Callable[[float, str], None] | None = None, prior_test_uses: int = 0) -> Experiment:
    """Train candidates and pick one using validation data or cross-validation. Never touches the test rows."""
    cancel = cancel or CancelToken()
    t0 = time.time()
    exp = Experiment(config=cfg, created_at=_now(), environment=environment(), prior_test_uses=prior_test_uses,
                     dataset_fingerprint=frame_fingerprint(df.reset_index(drop=True)) if len(df) <= 200_000 else "")
    if cfg.task == "clustering":
        return _run_clustering(df, cfg, exp, cancel, progress, t0)
    if cfg.task not in ("classification", "regression"):
        raise ExperimentError(f"Unknown task '{cfg.task}'.")
    exp.warnings += validate_target(df, cfg)
    work = df.loc[df[cfg.target].notna()].reset_index(drop=True)
    y = work[cfg.target].astype(str) if cfg.task == "classification" else work[cfg.target].astype(float)
    options = PreprocessOptions(**cfg.preprocess)
    fs = select_features(work, cfg.target, cfg.features_exclude, cfg.features_include, options)
    if not fs.all:
        raise ExperimentError("No usable feature columns remain. Select at least one column that varies and is not an identifier.")
    exp.feature_set = fs.to_dict()
    X = work[fs.all]

    sc = SplitConfig(**cfg.split)
    sc.seed = cfg.seed
    try:
        split = make_split(len(work), sc, y=y if cfg.task == "classification" else None,
                           groups=work[sc.group_col] if sc.group_col else None, times=work[sc.time_col] if sc.time_col else None)
    except SplitError as e:
        raise ExperimentError(str(e)) from e
    exp.split_info = split.to_dict()
    exp.warnings += split.warnings
    cvc = cfg.cv_config()
    tcfg = TuningConfig(**cfg.tuning)
    try:
        tcfg.validate()
    except TuningError as e:
        raise ExperimentError(str(e)) from e
    folds = None
    if split.val.size == 0 or tcfg.method != "none":
        try:
            folds = make_folds(len(split.train), cvc, task=cfg.task, split_strategy=split.effective_strategy,
                               y=y.iloc[split.train] if cfg.task == "classification" else None,
                               groups=work[sc.group_col].iloc[split.train] if sc.group_col else None,
                               times=work[sc.time_col].iloc[split.train] if sc.time_col else None, seed=cfg.seed)
        except CVError as e:
            raise ExperimentError(str(e)) from e
        exp.warnings += folds.warnings
        exp.cv_info = {"method": folds.method, "description": folds.description, "folds": folds.summary(), "config": cvc.to_dict()}
    exp.selection_basis = "the validation set" if split.val.size else f"{folds.description} on the training set"
    if tcfg.method != "none":
        exp.warnings.append(f"Settings were tuned with {folds.description} on the training rows only."
                            + ("" if split.val.size else " Because the same folds also rank the models, tuned scores are slightly optimistic; "
                               "the one-time test result is the unbiased estimate."))

    # leakage and safety screening (warnings only; nothing is fitted here)
    leaks = (L.check_target_leakage(work, cfg.target, cfg.task, fs.all) + L.check_post_outcome_names(fs.all, cfg.target)
             + L.check_duplicates(work, fs.all, cfg.target)
             + L.check_split_strategy(work, split.effective_strategy, fs.datetime, L.group_candidates(work)))
    dup_across = L.duplicates_across_splits(X.iloc[split.train], X.iloc[split.test])
    if dup_across:
        leaks.append(L.LeakageWarning("notice", "duplicate_leakage", [], f"{dup_across:,} test row(s) have identical features to a training row."))
    exp.leakage = [asdict(w) for w in leaks]

    classes = sorted(y.unique().tolist()) if cfg.task == "classification" else []
    if cfg.task == "classification":
        exp.class_distribution = IMB.class_distribution(y.iloc[split.train]).table.to_dict("records")
    if not cfg.models:
        raise ExperimentError("Choose at least one model.")
    for k in cfg.models:
        if k not in REGISTRY or REGISTRY[k].task != cfg.task:
            raise ExperimentError(f"Model '{k}' is not available for {cfg.task}.")
    _check_model_params(cfg)
    metric = cfg.primary_metric or E.default_primary(cfg.task)
    if metric not in E.selection_metrics(cfg.task):
        raise ExperimentError(f"'{metric}' is not a valid selection metric for {cfg.task}.")
    positive = cfg.positive_label
    if cfg.task == "classification" and len(classes) == 2 and positive not in classes:
        positive = str(y.iloc[split.train].value_counts().index[-1])   # default: the smaller class
    exp.config.positive_label = positive
    exp.config.primary_metric = metric
    if cfg.imbalance != "none" and cfg.task != "classification":
        raise ExperimentError("Class balancing only applies to classification.")

    compute = detect_compute()
    Xtr, ytr = X.iloc[split.train], y.iloc[split.train]
    Xva, yva = (X.iloc[split.val], y.iloc[split.val]) if split.val.size else (None, None)
    fitted: dict[str, Any] = {}
    n = len(cfg.models)
    for i, key in enumerate(cfg.models):
        cancel.check()
        spec = REGISTRY[key]
        if progress:
            progress(i / (n + 1), f"Training {spec.name} ({i + 1} of {n})")
        cand = Candidate(key, spec.name, params=dict(cfg.model_params.get(key) or {}))
        t1 = time.time()
        ok_c, reasons, cwarn = check_compatibility(spec, len(Xtr), len(fs.all), cfg.preprocess)
        exp.warnings += cwarn
        if not ok_c:
            cand.status, cand.error = "skipped", " ".join(reasons)
            exp.candidates.append(cand)
            continue
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                device, note = resolve_device(cfg.compute, spec.gpu_capable, compute)
                cand.device = device
                exp.compute_note = note
                use_cw = cfg.imbalance == "class_weight" and spec.supports_class_weight
                if cfg.imbalance == "class_weight" and not spec.supports_class_weight:
                    exp.warnings.append(f"{spec.name} does not support class weights; it was trained without them.")
                sampler = IMB.build_sampler(cfg.imbalance, ytr, cfg.seed) if cfg.task == "classification" else None

                chosen = dict(validate_params(spec, cfg.model_params.get(key)))
                space = {k: v for k, v in search_space_for(spec).items() if k.split("__")[-1] not in chosen}
                if tcfg.method != "none" and spec.tunable and space:
                    def make_t(p, chosen=chosen):
                        pipe = _make_pipeline(spec, fs, options, cfg, sampler, use_cw, device, chosen)
                        return pipe.set_params(**p)

                    def score_t(pipe, Xs, ys):
                        return _score_split(cfg.task, pipe, Xs, ys, classes, positive)["metrics"].get(metric)
                    tr = tune(space, tcfg, make_pipe=make_t, score_fn=score_t, X=Xtr.reset_index(drop=True), y=ytr.reset_index(drop=True),
                              folds=folds.folds, higher_is_better=E.metric_direction(metric) == "higher", seed=cfg.seed, check=cancel.check,
                              progress=(lambda m, name=spec.name, i=i: progress(i / (n + 1), f"Tuning {name}: {m}")) if progress else None,
                              passthrough=(TrainingCancelled,))
                    cand.tuning = tr.to_dict()
                    chosen.update({k.split("__")[-1]: v for k, v in tr.best_params.items()})
                    cand.params = dict(chosen)
                elif tcfg.method != "none":
                    cand.note = "Nothing to tune for this model; default settings were used."

                def make(chosen=chosen):
                    return _make_pipeline(spec, fs, options, cfg, sampler, use_cw, device, chosen)

                if Xva is not None:
                    pipe = make().fit(Xtr, ytr)
                    res = _score_split(cfg.task, pipe, Xva, yva, classes, positive)
                    cand.metrics = res["metrics"]
                    score = res["metrics"].get(metric)
                    if score is None:
                        raise ExperimentError(res["unavailable"].get(metric, f"{metric} is not defined for this data."))
                    cand.score, cand.score_std = float(score), None
                    fitted[key] = pipe
                else:
                    parts, scores = [], []
                    for a, b in folds.folds:
                        cancel.check()
                        p = make().fit(Xtr.iloc[a], ytr.iloc[a])      # resampling happens inside this fit, on the fold's training part only
                        r = _score_split(cfg.task, p, Xtr.iloc[b], ytr.iloc[b], classes, positive)
                        parts.append(r["metrics"])
                        if r["metrics"].get(metric) is not None:
                            scores.append(r["metrics"][metric])
                    if len(scores) != len(parts):
                        raise ExperimentError(f"{metric} could not be computed in every fold (a fold is missing a class). Choose another metric or fewer folds.")
                    cand.metrics = _mean_metrics(parts)
                    cand.score, cand.score_std = float(np.mean(scores)), float(np.std(scores))
                    cand.fold_scores = [float(v) for v in scores]
            except TrainingCancelled:
                raise
            except (ComputeError, ExperimentError, ModelConfigError, TuningError) as e:
                cand.status, cand.error = "failed", str(e)
            except Exception as e:  # a failing model must not abort the comparison
                log.exception("Model %s failed", key)
                cand.status, cand.error = "failed", f"{type(e).__name__}: {e}"
        if any("onverge" in str(w.message) or "onverge" in w.category.__name__ for w in caught):
            cand.note = "Did not fully converge; scores may improve with more iterations or scaled features."
        cand.seconds = round(time.time() - t1, 2)
        exp.candidates.append(cand)

    ok = [c for c in exp.candidates if c.status == "ok"]
    if not ok:
        exp.status = "failed"
        raise ExperimentError("Every model failed: " + "; ".join(f"{c.name}: {c.error}" for c in exp.candidates))
    order = {k: i for i, k in enumerate(REGISTRY)}
    best = ok[0]
    for c in ok[1:]:
        if _better(c.score, best.score, metric) or (abs(c.score - best.score) < 1e-9 and order.get(c.key, 99) < order.get(best.key, 99)):
            best = c
    exp.selected_key = best.key

    cancel.check()
    dummy = DummyClassifier(strategy="most_frequent") if cfg.task == "classification" else DummyRegressor(strategy="mean")
    base_opts = PreprocessOptions(**{**cfg.preprocess, "reduction": None, "selection": None})
    dpipe = Pipeline(build_pipeline_steps(fs, base_opts, False, task=cfg.task, seed=cfg.seed) + [("model", dummy)]).fit(Xtr, ytr)
    if Xva is not None:
        exp.baseline_selection = _score_split(cfg.task, dpipe, Xva, yva, classes, positive)

    # final model: fitted on training rows only (test rows untouched)
    if progress:
        progress(n / (n + 1), f"Fitting the selected model ({best.name})")
    spec = REGISTRY[best.key]
    if best.key in fitted:
        final = fitted[best.key]
    else:
        use_cw = cfg.imbalance == "class_weight" and spec.supports_class_weight
        sampler = IMB.build_sampler(cfg.imbalance, ytr, cfg.seed) if cfg.task == "classification" else None
        final = _make_pipeline(spec, fs, options, cfg, sampler, use_cw, best.device, best.params).fit(Xtr, ytr)
    exp._state = {"pipeline": final, "X": X, "y": y, "split": split, "classes": classes, "positive": positive, "fs": fs, "dummy_pipe": dpipe}
    # explanation from non-test data only (validation if present, else a training sample)
    ref_X, ref_y = (Xva, yva) if Xva is not None else (Xtr, ytr)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pi = permutation_importance(final, ref_X.iloc[:1000], ref_y.iloc[:1000], n_repeats=3, random_state=cfg.seed,
                                        scoring="f1_macro" if cfg.task == "classification" else "neg_root_mean_squared_error")
        imp = pd.DataFrame({"feature": ref_X.columns, "importance": pi.importances_mean, "spread": pi.importances_std}).sort_values("importance", ascending=False)
        exp.importance = imp.head(20).to_dict("records")
    except Exception:
        log.exception("importance failed")
    try:
        exp.pipeline_report = pipeline_reports(final)
        exp.native_importance = XP.native_importance(final)
    except Exception:
        log.exception("pipeline report failed")
    if cfg.shap:
        exp.shap = XP.shap_importance(final, ref_X, seed=cfg.seed)
    exp.status = "selection_complete"
    exp.seconds = round(time.time() - t0, 2)
    if progress:
        progress(1.0, "Done")
    return exp


def finalize_experiment(exp: Experiment) -> Experiment:
    """Evaluate the selected model on the held-out test rows. Allowed exactly once per experiment."""
    if exp.config.task == "clustering":
        raise ExperimentError("Clustering has no held-out test set; its metrics are already final.")
    if exp.test_evaluated:
        raise TestSetAlreadyUsed("The test set was already used for this experiment. Create a new experiment instead of re-testing.")
    if exp.status != "selection_complete" or not exp._state:
        raise ExperimentError("Run model selection first. (A reloaded experiment keeps its results but not its fitted model.)")
    st = exp._state
    X, y, split = st["X"], st["y"], st["split"]
    Xte, yte = X.iloc[split.test], y.iloc[split.test]
    task = exp.config.task
    exp.test_evaluated = True            # set first: once the rows are about to be seen, they count as used
    res = _score_split(task, st["pipeline"], Xte, yte, st["classes"], st["positive"])
    base = _score_split(task, st["dummy_pipe"], Xte, yte, st["classes"], st["positive"])
    exp.test_result, exp.test_baseline = res, base
    shares = {r["class"]: r["share %"] for r in exp.class_distribution} if task == "classification" else None
    exp.interpretation = E.interpret(task, res, base, shares)
    pred = st["pipeline"].predict(Xte)
    out = pd.DataFrame({"row_in_dataset": split.test, "actual": yte.values, "predicted": pred})
    if task == "classification":
        pr = _proba(st["pipeline"], Xte, st["classes"])
        if pr is not None:
            for j, c in enumerate(st["classes"]):
                out[f"probability_{c}"] = pr[:, j].round(4)
    else:
        out["error"] = out["predicted"] - out["actual"]
    st["test_predictions"] = out
    exp.status = "finalized"
    return exp


# ----------------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------------

def describe_groups(work: pd.DataFrame, labels, numeric: list[str], categorical: list[str], top: int = 2,
                    dates: list[str] | None = None) -> dict[int, str]:
    """A short plain-language name per group, from what sets it apart from all rows.

    Numeric columns: the group average compared with the overall spread (at least half a standard deviation counts as
    "high" or "low"). Categorical columns: a value that is much more common in the group than overall. Noise (-1) is
    named "Noise (no group)". Date columns count as "earlier" or "later". When nothing clears those bars, the strongest
    weaker difference is used ("slightly ..."), and only a group with no difference at all is called "Typical rows".
    Names that would repeat get the group number added.
    """
    labels = np.asarray(labels)
    out: dict[int, str] = {}
    num = {c: pd.to_numeric(work[c], errors="coerce") for c in numeric}
    for c in dates or []:
        d = pd.to_datetime(work[c], errors="coerce")
        if d.notna().sum() > 1:
            num[f"\x00{c}"] = (d - d.min()).dt.total_seconds()
    stats = {c: (float(s.mean()), float(s.std())) for c, s in num.items() if s.notna().sum() > 1 and float(s.std() or 0) > 0}

    def word(c: str, z: float) -> str:
        if c.startswith("\x00"):
            return f"{'later' if z > 0 else 'earlier'} {c[1:]}"
        return f"{'high' if z > 0 else 'low'} {c}"

    cat_share = {c: work[c].dropna().astype(str).value_counts(normalize=True) for c in categorical[:8]}
    for g in sorted(set(labels.tolist())):
        if g == -1:
            out[-1] = "Noise (no group)"
            continue
        mask = labels == g
        traits: list[tuple[float, str]] = []
        weak: list[tuple[float, str]] = []
        for c, (mu, sd) in stats.items():
            v = num[c][mask].mean()
            if pd.notna(v):
                z = (float(v) - mu) / sd
                (traits if abs(z) >= 0.5 else weak).append((abs(z), word(c, z)))
        for c, overall in cat_share.items():
            vc = work.loc[mask, c].dropna().astype(str).value_counts(normalize=True)
            if len(vc):
                val, share = vc.index[0], float(vc.iloc[0])
                base = float(overall.get(val, 0.0))
                if share >= 0.5 and share - base >= 0.2:
                    traits.append((2 * (share - base), f"mostly {c} = {val}"))
                elif share - base >= 0.05:
                    weak.append((2 * (share - base), f"more {c} = {val}"))
        traits.sort(key=lambda t: -t[0])
        weak.sort(key=lambda t: -t[0])
        if traits:
            name = ", ".join(t for _, t in traits[:top])
        elif weak and weak[0][0] >= 0.1:
            name = "slightly " + weak[0][1]
        else:
            name = "typical rows"
        out[int(g)] = name[0].upper() + name[1:]
    seen: dict[str, int] = {}
    for n in out.values():
        seen[n] = seen.get(n, 0) + 1
    return {g: (f"{n} ({g})" if seen[n] > 1 and g != -1 else n) for g, n in out.items()}


def _run_clustering(df: pd.DataFrame, cfg: ExperimentConfig, exp: Experiment, cancel: CancelToken, progress, t0: float) -> Experiment:
    options = PreprocessOptions(**cfg.preprocess)
    if options.scaling == "auto":
        options.scaling = "standard"
    work = df.reset_index(drop=True)
    fs = select_features(work, None, cfg.features_exclude, cfg.features_include, options)
    if not fs.all:
        raise ExperimentError("No usable feature columns remain.")
    exp.feature_set = fs.to_dict()
    if len(work) < 20:
        raise ExperimentError("Clustering needs at least 20 rows.")
    if not cfg.models:
        raise ExperimentError("Choose at least one clustering method.")
    metric = cfg.primary_metric or "silhouette"
    if metric not in E.selection_metrics("clustering"):
        raise ExperimentError(f"'{metric}' is not a valid selection metric for clustering.")
    exp.config.primary_metric = metric
    exp.selection_basis = "internal clustering metrics on all rows (clustering has no held-out test set)"
    exp.warnings.append("Clustering has no ground truth: scores describe how compact and separated the groups are, not whether they are 'correct'.")
    prep = Pipeline(build_pipeline_steps(fs, options, True, task="clustering", seed=cfg.seed))
    Xt = np.asarray(prep.fit_transform(work[fs.all]), dtype=float)
    if not np.isfinite(Xt).all():
        raise ExperimentError("The prepared features contain non-finite values; check for infinite numbers.")
    cands: list[tuple[Candidate, np.ndarray, Any]] = []
    kr = range(max(2, cfg.cluster_k_min), min(cfg.cluster_k_max, len(work) - 1) + 1)
    jobs: list[tuple[str, str, dict, Callable[[], Any]]] = []
    _check_model_params(cfg)
    for key in cfg.models:
        spec = REGISTRY.get(key)
        if spec is None or spec.task != "clustering":
            raise ExperimentError(f"Model '{key}' is not a clustering method.")
        ok_c, reasons, cwarn = check_compatibility(spec, len(work), Xt.shape[1], cfg.preprocess)
        exp.warnings += cwarn
        if not ok_c:
            exp.candidates.append(Candidate(key, spec.name, "skipped", error=" ".join(reasons)))
            continue
        user = validate_params(spec, cfg.model_params.get(key))
        label = ", ".join(f"{k}={v}" for k, v in user.items())

        def mk(extra: dict, spec=spec, user=user):
            m = spec.factory(cfg.seed, False, "cpu")
            m.set_params(**{**user, **extra})
            return m
        if key == "dbscan":
            ms = int(user.get("min_samples") or max(5, min(2 * Xt.shape[1], 20)))
            sample = Xt if len(Xt) <= 5000 else Xt[np.random.RandomState(cfg.seed).choice(len(Xt), 5000, replace=False)]
            d, _ = NearestNeighbors(n_neighbors=min(ms, len(sample) - 1)).fit(sample).kneighbors(sample)
            kd = np.sort(d[:, -1])
            for q in (0.80, 0.90, 0.95):
                eps = float(np.quantile(kd, q))
                if eps > 0:
                    jobs.append((key, f"DBSCAN (eps={eps:.2f}, min_samples={ms})", {"eps": round(eps, 4), "min_samples": ms},
                                 (lambda eps=eps, ms=ms, mk=mk: mk({"eps": eps, "min_samples": ms}))))
        elif spec.k_param:
            short = spec.name.split(" (")[0]
            jobs += [(key, f"{short} (k={k}{', ' + label if label else ''})", {"k": k, **user}, (lambda k=k, mk=mk, kp=spec.k_param: mk({kp: k}))) for k in kr]
        else:
            jobs.append((key, spec.name + (f" ({label})" if label else ""), dict(user), (lambda mk=mk: mk({}))))
    for i, (key, name, params, factory) in enumerate(jobs):
        cancel.check()
        if progress:
            progress(i / max(len(jobs), 1), f"Fitting {name}")
        cand, t1 = Candidate(f"{key}|{name}", name, params=params), time.time()
        try:
            model = factory()
            labels = model.fit_predict(Xt)
            r = E.clustering_metrics(Xt, labels, cfg.seed)
            cand.metrics, cand.n_clusters, cand.noise_pct = r["metrics"], r["n_clusters"], r["noise_pct"]
            if r["metrics"].get(metric) is None:
                cand.status, cand.error = "failed", r["unavailable"].get(metric, "Metric not defined.")
            else:
                cand.score = float(r["metrics"][metric])
                cands.append((cand, labels, model))
        except TrainingCancelled:
            raise
        except Exception as e:
            cand.status, cand.error = "failed", f"{type(e).__name__}: {e}"
        cand.seconds = round(time.time() - t1, 2)
        exp.candidates.append(cand)
    if not cands:
        exp.status = "failed"
        raise ExperimentError("No clustering produced at least 2 groups. " + "; ".join(f"{c.name}: {c.error}" for c in exp.candidates if c.error))
    best = cands[0]
    for c in cands[1:]:
        if _better(c[0].score, best[0].score, metric):
            best = c
    cand, labels, model = best
    exp.selected_key = cand.key
    exp._state = {"prep": prep, "model": model, "labels": labels, "fs": fs, "chosen": cand}
    exp.test_result = E.clustering_metrics(Xt, labels, cfg.seed)
    exp.test_evaluated = False
    exp.interpretation = E.interpret("clustering", exp.test_result, None)
    prof = work.assign(cluster=labels)
    names = describe_groups(work, labels, fs.numeric, fs.categorical, dates=fs.datetime)
    rows = []
    for g, part in prof.groupby("cluster"):
        r: dict[str, Any] = {"cluster": "noise" if g == -1 else int(g), "name": names.get(int(g), ""), "rows": int(len(part)),
                             "share %": round(100 * len(part) / len(prof), 1)}
        for c in fs.numeric:
            r[f"mean {c}"] = round(float(part[c].mean()), 3) if part[c].notna().any() else None
        for c in fs.categorical[:4]:
            vc = part[c].dropna().astype(str).value_counts()
            r[f"top {c}"] = f"{vc.index[0]} ({100 * vc.iloc[0] / len(part):.0f}%)" if len(vc) else ""
        rows.append(r)
    exp.cluster_profile = rows
    try:
        pts = PCA(n_components=2, random_state=cfg.seed).fit_transform(Xt) if Xt.shape[1] >= 2 else np.column_stack([Xt[:, 0], np.zeros(len(Xt))])
        idx = np.arange(len(pts)) if len(pts) <= 3000 else np.random.RandomState(cfg.seed).choice(len(pts), 3000, replace=False)
        exp.projection = [{"x": float(pts[i, 0]), "y": float(pts[i, 1]), "cluster": "noise" if labels[i] == -1 else str(int(labels[i]))} for i in idx]
    except Exception:
        log.exception("projection failed")
    exp.status = "finalized"
    exp.seconds = round(time.time() - t0, 2)
    if progress:
        progress(1.0, "Done")
    return exp


# ----------------------------------------------------------------------------
# Artifacts and background jobs
# ----------------------------------------------------------------------------

def save_artifact(exp: Experiment, path: str) -> str:
    """Write the fitted pipeline plus the metadata needed to use it. Load only files you created (pickle-based)."""
    st = exp._state
    if not st:
        raise ExperimentError("There is no fitted model to save in this session.")
    payload = {"kind": exp.config.task, "config": exp.config.to_dict(), "environment": exp.environment, "created_at": exp.created_at,
               "selected_model": exp.selected_key, "feature_columns": st["fs"].all, "dataset_fingerprint": exp.dataset_fingerprint}
    if exp.config.task == "clustering":
        payload.update({"prep": st["prep"], "model": st["model"], "labels": st["labels"]})
    else:
        payload.update({"pipeline": st["pipeline"], "classes": st["classes"], "target": exp.config.target})
    joblib.dump(payload, path)
    return path


def load_artifact(path: str) -> dict[str, Any]:
    return joblib.load(path)


class TrainingJob:
    """Runs a training function in a background thread so the UI can show real status and offer cancellation.

    Cancellation takes effect between models, folds and clustering fits; a single scikit-learn fit cannot be interrupted.
    """

    def __init__(self, fn: Callable[[Callable[[float, str], None], CancelToken], Experiment]):
        self._fn = fn
        self.token = CancelToken()
        self.state = "pending"        # pending | running | done | failed | cancelled
        self.fraction = 0.0
        self.message = "Waiting to start"
        self.error = ""
        self.result: Experiment | None = None
        self.started: float | None = None
        self.ended: float | None = None
        self._thread: threading.Thread | None = None

    def _progress(self, frac: float, msg: str) -> None:
        self.fraction, self.message = float(frac), msg

    def _run(self) -> None:
        try:
            self.result = self._fn(self._progress, self.token)
            self.state = "done"
            self.fraction, self.message = 1.0, "Finished"
        except TrainingCancelled:
            self.state, self.message = "cancelled", "Cancelled before finishing; no results were saved."
        except Exception as e:
            log.exception("Training job failed")
            self.state, self.error, self.message = "failed", str(e), "Failed"
        finally:
            self.ended = time.time()

    def start(self) -> "TrainingJob":
        self.state, self.started = "running", time.time()
        self.message = "Starting"
        self._thread = threading.Thread(target=self._run, daemon=True, name="nexora-training")
        self._thread.start()
        return self

    def cancel(self) -> None:
        self.token.cancel()
        if self.state == "running":
            self.message = "Cancelling after the current step finishes"

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    @property
    def elapsed(self) -> float:
        if self.started is None:
            return 0.0
        return (self.ended or time.time()) - self.started
