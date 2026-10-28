"""Class-imbalance analysis and resampling. Resampling happens only inside training data / training folds."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field

import pandas as pd

IMBLEARN = importlib.util.find_spec("imblearn") is not None

STRATEGIES = {
    "none": "Continue without balancing",
    "class_weight": "Class weights (penalise mistakes on small classes more)",
    "random_over": "Random oversampling (repeat small-class rows in training data)",
    "random_under": "Random undersampling (drop large-class rows from training data)",
    "smote": "SMOTE (create synthetic small-class rows in training data)",
}


@dataclass
class ImbalanceInfo:
    table: pd.DataFrame
    ratio: float                 # smallest class / largest class
    level: str                   # balanced | mild | moderate | severe
    smallest_class: str
    smallest_count: int
    n_classes: int


@dataclass
class StrategyOption:
    key: str
    label: str
    available: bool
    recommended: bool = False
    why: str = ""
    warnings: list[str] = field(default_factory=list)


def class_distribution(y: pd.Series) -> ImbalanceInfo:
    vc = y.dropna().value_counts()
    total = int(vc.sum())
    table = pd.DataFrame({"class": vc.index.astype(str), "rows": vc.values.astype(int), "share %": (100 * vc.values / max(total, 1)).round(1)})
    ratio = float(vc.min() / vc.max()) if len(vc) > 1 else 1.0
    level = "balanced" if ratio >= 0.5 else ("mild" if ratio >= 0.2 else ("moderate" if ratio >= 0.1 else "severe"))
    return ImbalanceInfo(table, ratio, level, str(vc.index[-1]), int(vc.min()), len(vc))


def metric_advice(info: ImbalanceInfo) -> str:
    if info.level == "balanced":
        return "Classes are fairly even, so accuracy is a reasonable headline number alongside F1."
    top = float(info.table["share %"].iloc[0])
    return (f"Accuracy can mislead here: always predicting the largest class would already be right {top:.0f}% of the time. "
            "Prefer F1, recall or precision for the class you care about, or PR-AUC; compare against the baseline shown with the results.")


def strategy_options(info: ImbalanceInfo, all_numeric_features: bool, model_supports_class_weight: bool = True) -> list[StrategyOption]:
    opts: list[StrategyOption] = []
    imbalanced = info.level != "balanced"
    min_n = info.smallest_count
    opts.append(StrategyOption("none", STRATEGIES["none"], True, recommended=not imbalanced,
                               why="Classes are even enough." if not imbalanced else "Allowed, but the model may ignore the small class."))
    opts.append(StrategyOption("class_weight", STRATEGIES["class_weight"], True, recommended=imbalanced,
                               why="Adds no synthetic data and keeps all rows; the usual first choice." if imbalanced else "Rarely needed for balanced classes.",
                               warnings=([] if model_supports_class_weight else ["Some selected models ignore class weights; they are trained unweighted."])))
    over_warn = ["Copies of small-class rows can be memorised, which inflates training scores; judge only on validation and test results."]
    if min_n < 10:
        over_warn.append(f"The smallest class has only {min_n} row(s); any resampling is unreliable.")
    opts.append(StrategyOption("random_over", STRATEGIES["random_over"], IMBLEARN, why="Keeps every row." if IMBLEARN else "Needs the optional 'imbalanced-learn' package.", warnings=over_warn))
    opts.append(StrategyOption("random_under", STRATEGIES["random_under"], IMBLEARN,
                               why="Fast, but throws away data." if IMBLEARN else "Needs the optional 'imbalanced-learn' package.",
                               warnings=["Discards information from the large class; avoid on small datasets."] + ([f"Only {info.table['rows'].min()} rows per class would remain."] if info.table["rows"].min() < 30 else [])))
    smote_ok = IMBLEARN and all_numeric_features and min_n >= 2
    why = ("Interpolates between neighbouring small-class rows." if smote_ok else
           "Needs the optional 'imbalanced-learn' package." if not IMBLEARN else
           "Not available: your features include categories or dates, and SMOTE would invent meaningless in-between values for them." if not all_numeric_features else
           "Not available: the smallest class needs at least 2 rows.")
    w = ["Synthetic rows are estimates, not observations."] + ([f"With only {min_n} small-class rows the synthetic rows will be very similar to each other."] if min_n < 15 else [])
    opts.append(StrategyOption("smote", STRATEGIES["smote"], smote_ok, why=why, warnings=w))
    return opts


def build_sampler(strategy: str, y_train: pd.Series, seed: int):
    """Return an imbalanced-learn sampler or None. Used as a pipeline step, so it only ever sees training data."""
    if strategy in ("none", "class_weight"):
        return None
    if not IMBLEARN:
        raise RuntimeError("Resampling needs the optional package 'imbalanced-learn' (pip install imbalanced-learn).")
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    if strategy == "random_over":
        return RandomOverSampler(random_state=seed)
    if strategy == "random_under":
        return RandomUnderSampler(random_state=seed)
    if strategy == "smote":
        k = max(1, min(5, int(y_train.value_counts().min()) - 1))
        return SMOTE(k_neighbors=k, random_state=seed)
    raise ValueError(f"Unknown balancing strategy '{strategy}'.")
