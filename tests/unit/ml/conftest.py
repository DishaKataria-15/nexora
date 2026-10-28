import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def clf_df():
    rng = np.random.RandomState(0)
    n = 400
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    cat = rng.choice(["a", "b", "c"], n)
    y = ((x1 + 0.8 * (cat == "a") + rng.normal(0, 0.6, n)) > 0.9).astype(int)          # ~25% positives
    return pd.DataFrame({"x1": x1, "x2": x2, "cat": cat, "y": np.where(y == 1, "yes", "no")})


@pytest.fixture()
def reg_df():
    rng = np.random.RandomState(1)
    n = 300
    a, b = rng.normal(size=n), rng.uniform(0, 10, n)
    g = rng.choice(["u", "v"], n)
    return pd.DataFrame({"a": a, "b": b, "g": g, "t": 3 * a + 0.5 * b + (g == "u") * 2 + rng.normal(0, 0.5, n)})


@pytest.fixture()
def blobs_df():
    rng = np.random.RandomState(2)
    parts = [pd.DataFrame({"p": rng.normal(c[0], 0.5, 80), "q": rng.normal(c[1], 0.5, 80)}) for c in [(0, 0), (6, 6), (0, 7)]]
    return pd.concat(parts, ignore_index=True)
