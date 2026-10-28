"""Use a saved model on new rows. The saved pipeline contains every preparation step, fitted on training rows."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

MAX_PREDICT_ROWS = 500_000


class PredictionError(ValueError):
    """The rows cannot be scored with this model. The message is safe to show to the user."""


def predict_frame(payload: dict[str, Any], df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return (rows with predictions, notes). Missing feature columns stop the run; extra columns are ignored."""
    notes: list[str] = []
    cols = list(payload.get("feature_columns") or [])
    if not cols:
        raise PredictionError("This model file does not list its input columns, so it cannot be used safely.")
    if len(df) == 0:
        raise PredictionError("There are no rows to score.")
    if len(df) > MAX_PREDICT_ROWS:
        raise PredictionError(f"At most {MAX_PREDICT_ROWS:,} rows can be scored at once.")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise PredictionError(f"The data is missing column(s) the model needs: {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}.")
    extra = [c for c in df.columns if c not in cols]
    if extra:
        notes.append(f"{len(extra)} column(s) not used by the model were ignored" + (f" (including the target '{payload.get('target')}')" if payload.get("target") in extra else "") + ".")
    X = df[cols].reset_index(drop=True)
    out = df.reset_index(drop=True).copy()
    kind = payload.get("kind")
    if kind == "clustering":
        model = payload["model"]
        if not hasattr(model, "predict"):
            raise PredictionError(f"{type(model).__name__} cannot assign new rows to groups; only the rows it was trained on have groups.")
        Xt = payload["prep"].transform(X)
        out["predicted_group"] = model.predict(np.asarray(Xt, dtype=float))
        return out, notes
    pipe = payload["pipeline"]
    out[f"predicted_{payload.get('target') or 'value'}"] = pipe.predict(X)
    if kind == "classification" and hasattr(pipe, "predict_proba"):
        try:
            p = pipe.predict_proba(X)
            classes = list(getattr(pipe, "classes_", payload.get("classes") or []))
            for j, c in enumerate(classes):
                out[f"probability_{c}"] = np.round(p[:, j], 4)
        except Exception:  # probabilities are optional
            notes.append("This model does not provide probabilities.")
    nmiss = int(X.isna().any(axis=1).sum())
    if nmiss:
        notes.append(f"{nmiss:,} row(s) had missing inputs; they were filled the same way as during training.")
    return out, notes
