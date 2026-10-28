"""Time-series forecasting with honest, rolling-origin backtesting.

Workflow:
1. Order rows by the date column and check the series (duplicates, gaps, regular frequency).
2. Backtest each method with an expanding window: fit only on data before each cut-off, forecast the next
   ``horizon`` steps, compare with what actually happened. No future value is ever used to predict the past.
3. Compare against naive and seasonal-naive baselines; a method that cannot beat them is reported as such.
4. Refit the chosen method on the full history and forecast the future.

Methods: naive, seasonal naive, moving average, lag-feature regression (recursive, scikit-learn models), and
Holt-Winters exponential smoothing when the optional statsmodels package is installed.
"""
from __future__ import annotations

import importlib.util
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

STATSMODELS_AVAILABLE = importlib.util.find_spec("statsmodels") is not None
METHODS = {
    "naive": "Naive (repeat the last value)",
    "seasonal_naive": "Seasonal naive (repeat the value from one season ago)",
    "moving_average": "Moving average of the last season",
    "ridge_lags": "Linear model on lags and calendar features",
    "forest_lags": "Random forest on lags and calendar features",
    "boosting_lags": "Gradient boosting on lags and calendar features",
    "holt_winters": "Holt-Winters exponential smoothing (needs statsmodels)",
}
BASELINES = ("naive", "seasonal_naive")
MAX_HORIZON = 365
MAX_FOLDS = 10


class ForecastError(ValueError):
    """The forecast cannot be made. The message is safe to show to the user."""


@dataclass
class ForecastConfig:
    date_col: str
    target: str
    horizon: int = 14
    season: int | None = None             # season length in steps; None = inferred from the frequency
    methods: list[str] = field(default_factory=lambda: ["naive", "seasonal_naive", "ridge_lags", "boosting_lags"])
    folds: int = 3
    lags: list[int] | None = None         # None = 1..season plus season multiples
    exog: list[str] = field(default_factory=list)   # known-in-advance numeric columns (e.g. a planned promotion flag)
    aggregate: str = "sum"                # how to combine duplicate timestamps: sum | mean | last
    fill: str = "interpolate"            # how to fill gaps in the regular calendar: interpolate | zero | ffill | none
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ForecastResult:
    config: ForecastConfig
    frequency: str = ""
    n_points: int = 0
    checks: list[str] = field(default_factory=list)
    backtest: list[dict[str, Any]] = field(default_factory=list)
    fold_windows: list[dict[str, str]] = field(default_factory=list)
    selected: str = ""
    beats_baseline: bool = False
    forecast: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    backtest_predictions: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SEASON_FOR = {"D": 7, "B": 5, "W": 52, "MS": 12, "ME": 12, "M": 12, "QS": 4, "QE": 4, "Q": 4, "h": 24, "H": 24, "min": 60}


def prepare_series(df: pd.DataFrame, cfg: ForecastConfig) -> tuple[pd.Series, pd.DataFrame, str, list[str]]:
    """Return a regular, date-indexed target series, aligned exogenous columns, the frequency and data notes."""
    for c in [cfg.date_col, cfg.target, *cfg.exog]:
        if c not in df.columns:
            raise ForecastError(f"Column '{c}' is not in the data.")
    notes: list[str] = []
    d = pd.DataFrame({"t": pd.to_datetime(df[cfg.date_col], errors="coerce"), "y": pd.to_numeric(df[cfg.target], errors="coerce")})
    for c in cfg.exog:
        d[c] = pd.to_numeric(df[c], errors="coerce")
    bad = int(d["t"].isna().sum())
    if bad:
        notes.append(f"{bad} row(s) without a valid date were left out.")
        d = d[d["t"].notna()]
    if d["y"].isna().all():
        raise ForecastError(f"'{cfg.target}' has no numeric values to forecast.")
    dup = int(d["t"].duplicated().sum())
    if dup:
        agg = {"sum": "sum", "mean": "mean", "last": "last"}.get(cfg.aggregate)
        if agg is None:
            raise ForecastError("Duplicate dates can be combined by sum, mean or last.")
        notes.append(f"{dup} row(s) shared a date with another row and were combined by {cfg.aggregate}.")
        d = d.groupby("t", as_index=False).agg({c: (agg if c == "y" else "mean") for c in d.columns if c != "t"})
    d = d.sort_values("t").set_index("t")
    if len(d) < 10:
        raise ForecastError("Forecasting needs at least 10 dated observations.")
    freq = pd.infer_freq(d.index[: min(len(d), 200)]) if len(d) >= 3 else None
    if freq is None:
        step = pd.Series(d.index).diff().dropna().median()
        freq = {pd.Timedelta(days=1): "D", pd.Timedelta(hours=1): "h", pd.Timedelta(weeks=1): "W"}.get(step)
        if freq is None and pd.Timedelta(days=28) <= step <= pd.Timedelta(days=31):
            freq = "MS"
        if freq is None:
            raise ForecastError("The dates are not evenly spaced (daily, weekly, monthly, ...), so a step-by-step forecast is not possible.")
        notes.append(f"The dates are not perfectly regular; a {freq} calendar was assumed.")
    full = pd.date_range(d.index.min(), d.index.max(), freq=freq)
    if freq in ("MS", "ME", "M"):
        d.index = d.index.to_period("M").to_timestamp(how="start") if freq == "MS" else d.index
        full = pd.date_range(d.index.min(), d.index.max(), freq=freq)
    d = d[~d.index.duplicated()].reindex(full)
    gaps = int(d["y"].isna().sum())
    if gaps:
        if cfg.fill == "none":
            raise ForecastError(f"{gaps} step(s) in the calendar have no value. Choose how to fill them (interpolate, zero, or carry forward).")
        if cfg.fill == "zero":
            d["y"] = d["y"].fillna(0.0)
        elif cfg.fill == "ffill":
            d["y"] = d["y"].ffill().bfill()
        else:
            d["y"] = d["y"].interpolate(limit_direction="both")
        notes.append(f"{gaps} missing step(s) ({100 * gaps / len(d):.1f}%) were filled by {cfg.fill}.")
    for c in cfg.exog:
        d[c] = d[c].ffill().bfill().fillna(0.0)
    d.index.freq = pd.tseries.frequencies.to_offset(freq)
    return d["y"].astype(float), d[cfg.exog], freq, notes


def infer_season(freq: str, n: int) -> int:
    base = "".join(ch for ch in freq if not ch.isdigit()).split("-")[0]
    s = SEASON_FOR.get(base, 1)
    return s if n >= 3 * s else 1


# ----------------------------------------------------------------------------- methods
def _calendar(idx: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"dow": idx.day_of_week, "month": idx.month, "dom": idx.day, "hour": idx.hour, "doy": idx.day_of_year}, index=idx)


def _regressor(key: str, seed: int):
    if key == "ridge_lags":
        return Ridge(alpha=1.0)
    if key == "forest_lags":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=seed)


def _lag_forecast(key: str, y: pd.Series, exog: pd.DataFrame, future_exog: pd.DataFrame, h: int, lags: list[int], seed: int,
                  future_idx: pd.DatetimeIndex) -> np.ndarray:
    """Recursive multi-step forecast: predictions become the lags for later steps."""
    maxlag = max(lags)
    if len(y) <= maxlag + 5:
        raise ForecastError(f"Not enough history ({len(y)} points) for lags up to {maxlag}.")
    vals = y.values
    rows = []
    for t in range(maxlag, len(vals)):
        rows.append([vals[t - L] for L in lags])
    Xl = pd.DataFrame(rows, columns=[f"lag{L}" for L in lags], index=y.index[maxlag:])
    X = pd.concat([Xl, _calendar(y.index[maxlag:]), exog.iloc[maxlag:]], axis=1)
    model = _regressor(key, seed)
    if key == "ridge_lags":
        mu, sd = X.mean(), X.std().replace(0, 1)
        model.fit((X - mu) / sd, vals[maxlag:])
    else:
        model.fit(X, vals[maxlag:])
    hist = list(vals)
    out = []
    cal = _calendar(future_idx)
    for i in range(h):
        row = pd.DataFrame([[hist[-L] for L in lags]], columns=Xl.columns)
        row = pd.concat([row, cal.iloc[[i]].reset_index(drop=True), future_exog.iloc[[i]].reset_index(drop=True)], axis=1)
        p = float(model.predict((row - mu) / sd if key == "ridge_lags" else row)[0])
        out.append(p)
        hist.append(p)
    return np.asarray(out)


def _forecast(key: str, y: pd.Series, exog: pd.DataFrame, future_exog: pd.DataFrame, h: int, season: int, lags: list[int], seed: int) -> np.ndarray:
    future_idx = pd.date_range(y.index[-1], periods=h + 1, freq=y.index.freq)[1:]
    if key == "naive":
        return np.repeat(y.values[-1], h)
    if key == "seasonal_naive":
        s = max(1, season)
        last = y.values[-s:]
        return np.array([last[i % s] for i in range(h)])
    if key == "moving_average":
        return np.repeat(float(np.mean(y.values[-max(season, 3):])), h)
    if key in ("ridge_lags", "forest_lags", "boosting_lags"):
        return _lag_forecast(key, y, exog, future_exog, h, lags, seed, future_idx)
    if key == "holt_winters":
        if not STATSMODELS_AVAILABLE:
            raise ForecastError("Holt-Winters needs the optional package statsmodels (pip install statsmodels).")
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = "add" if season > 1 and len(y) >= 2 * season + 2 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y.values, trend="add", damped_trend=True, seasonal=seasonal,
                                       seasonal_periods=season if seasonal else None, initialization_method="estimated").fit()
        return np.asarray(fit.forecast(h), dtype=float)
    raise ForecastError(f"Unknown forecasting method '{key}'.")


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float | None]:
    err = pred - actual
    out: dict[str, float | None] = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}
    out["mape"] = float(np.mean(np.abs(err / actual)) * 100) if np.all(actual != 0) else None
    den = np.abs(actual) + np.abs(pred)
    out["smape"] = float(np.mean(np.where(den == 0, 0.0, 2 * np.abs(err) / np.where(den == 0, 1, den))) * 100)
    return out


def run_forecast(df: pd.DataFrame, cfg: ForecastConfig, progress: Callable[[float, str], None] | None = None,
                 check: Callable[[], None] = lambda: None, future_exog: pd.DataFrame | None = None) -> ForecastResult:
    t0 = time.time()
    if not 1 <= int(cfg.horizon) <= MAX_HORIZON:
        raise ForecastError(f"The forecast horizon must be between 1 and {MAX_HORIZON} steps.")
    if not 1 <= int(cfg.folds) <= MAX_FOLDS:
        raise ForecastError(f"Choose between 1 and {MAX_FOLDS} backtest windows.")
    if not cfg.methods:
        raise ForecastError("Choose at least one forecasting method.")
    for m in cfg.methods:
        if m not in METHODS:
            raise ForecastError(f"Unknown forecasting method '{m}'.")
    if cfg.target == cfg.date_col:
        raise ForecastError("The value to forecast and the date column must be different columns.")
    y, exog, freq, notes = prepare_series(df, cfg)
    res = ForecastResult(cfg, frequency=freq, n_points=len(y), checks=notes)
    season = int(cfg.season) if cfg.season else infer_season(freq, len(y))
    if season < 1:
        raise ForecastError("The season length must be at least 1.")
    lags = sorted(set(cfg.lags)) if cfg.lags else sorted(set(list(range(1, min(season, 14) + 1)) + [season, 2 * season] if season > 1 else [1, 2, 3, 7]))
    if any(L < 1 for L in lags):
        raise ForecastError("Lags must be whole numbers of at least 1.")
    h, k = int(cfg.horizon), int(cfg.folds)
    min_train = max(max(lags) + 10, 2 * season + 2, 20)
    if len(y) < min_train + h * k:
        raise ForecastError(f"{len(y)} points are too few for {k} backtest window(s) of {h} steps (at least {min_train + h * k} are needed). "
                            "Shorten the horizon or use fewer windows.")
    if cfg.exog:
        res.warnings.append("Extra columns are treated as known in advance (for example a planned promotion). Their future values must be supplied; "
                            "if they are not, the last known value is repeated, which may be wrong.")
    cutoffs = [len(y) - h * (k - i) for i in range(k)]
    for c in cutoffs:
        res.fold_windows.append({"train_until": str(y.index[c - 1].date()), "test_from": str(y.index[c].date()), "test_until": str(y.index[c + h - 1].date())})
    methods = list(dict.fromkeys(list(cfg.methods) + [b for b in BASELINES if b not in cfg.methods]))
    preds_rows = []
    for j, key in enumerate(methods):
        check()
        if progress:
            progress(j / (len(methods) + 1), f"Backtesting {METHODS[key]}")
        entry: dict[str, Any] = {"key": key, "name": METHODS[key], "baseline": key in BASELINES, "requested": key in cfg.methods, "status": "ok"}
        t1 = time.time()
        try:
            fold_m = []
            for fi, c in enumerate(cutoffs):
                check()
                ytr, ex_tr = y.iloc[:c], exog.iloc[:c]
                ex_fu = exog.iloc[c:c + h]              # known-in-advance columns for the window (actual planned values)
                p = _forecast(key, ytr, ex_tr, ex_fu, h, season, lags, cfg.seed)
                act = y.values[c:c + h]
                fold_m.append(_metrics(act, p))
                for t, a, q in zip(y.index[c:c + h], act, p):
                    preds_rows.append({"method": key, "window": fi + 1, "date": str(t.date()), "actual": float(a), "predicted": float(q)})
            entry["folds"] = fold_m
            entry["metrics"] = {m: (float(np.mean([f[m] for f in fold_m])) if all(f[m] is not None for f in fold_m) else None) for m in fold_m[0]}
        except ForecastError as e:
            entry.update(status="failed", error=str(e))
        except Exception as e:
            entry.update(status="failed", error=f"{type(e).__name__}: {e}")
        entry["seconds"] = round(time.time() - t1, 3)
        res.backtest.append(entry)
    res.backtest_predictions = preds_rows
    ok = [e for e in res.backtest if e["status"] == "ok" and e["requested"]]
    if not ok:
        raise ForecastError("No requested method could be backtested: " + "; ".join(f"{e['name']}: {e.get('error', '')}" for e in res.backtest if e["requested"]))
    best = min(ok, key=lambda e: e["metrics"]["mae"])
    res.selected = best["key"]
    base = [e for e in res.backtest if e["baseline"] and e["status"] == "ok"]
    best_base = min(base, key=lambda e: e["metrics"]["mae"]) if base else None
    res.beats_baseline = bool(best_base is None or best["metrics"]["mae"] < best_base["metrics"]["mae"])
    if best_base is not None and not best["baseline"] and not res.beats_baseline:
        res.warnings.append(f"{best['name']} did not beat the simple baseline ({best_base['name']}); the baseline is the more honest forecast here.")
    if progress:
        progress(len(methods) / (len(methods) + 1), "Forecasting the future")
    fut_idx = pd.date_range(y.index[-1], periods=h + 1, freq=y.index.freq)[1:]
    if cfg.exog:
        if future_exog is not None and len(future_exog) >= h:
            fe = future_exog[cfg.exog].iloc[:h].reset_index(drop=True).astype(float)
        else:
            fe = pd.DataFrame([exog.iloc[-1].values] * h, columns=cfg.exog)
    else:
        fe = pd.DataFrame(index=range(h))
    fe.index = fut_idx
    future = _forecast(res.selected, y, exog, fe, h, season, lags, cfg.seed)
    # a simple, honest interval: the backtest absolute errors at each step ahead
    bt = pd.DataFrame([r for r in preds_rows if r["method"] == res.selected])
    bt["step"] = bt.groupby("window").cumcount()
    q = bt.assign(err=(bt["predicted"] - bt["actual"]).abs()).groupby("step")["err"].quantile(0.8)
    for i, (t, v) in enumerate(zip(fut_idx, future)):
        w = float(q.get(i, q.iloc[-1]))
        res.forecast.append({"date": str(t.date()) if t.hour == 0 else str(t), "forecast": float(v), "lower_80": float(v - w), "upper_80": float(v + w)})
    if k < 3:
        res.warnings.append("The range is based on fewer than 3 backtest windows, so it is a rough guide only.")
    res.history = [{"date": str(t.date()) if t.hour == 0 else str(t), "value": float(v)} for t, v in y.iloc[-min(len(y), 500):].items()]
    res.seconds = round(time.time() - t0, 2)
    if progress:
        progress(1.0, "Done")
    return res
