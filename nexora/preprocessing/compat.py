"""Small helpers that keep Nexora working across the supported scikit-learn versions (>=1.5)."""
from __future__ import annotations

import sklearn
from sklearn.linear_model import LogisticRegression

_SK = tuple(int(p) for p in sklearn.__version__.split(".")[:2] if p.isdigit())


def l1_logistic(C: float = 1.0, seed: int = 0, class_weight=None) -> LogisticRegression:
    """L1-penalised logistic regression without deprecated arguments."""
    if _SK >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, solver="saga", C=C, max_iter=5000, random_state=seed, class_weight=class_weight)
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000, random_state=seed, class_weight=class_weight)
