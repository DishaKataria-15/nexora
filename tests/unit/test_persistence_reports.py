import io
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

from nexora.persistence import SqliteRepository
from nexora.reporting import eda_summary_html, experiment_report_html, zip_bytes
from nexora.reporting import html as H


@pytest.fixture()
def repo(tmp_path):
    return SqliteRepository(tmp_path / "d")


def _is_root() -> bool:
    return getattr(os, "geteuid", lambda: 1)() == 0


def test_dataset_raw_bytes_are_stored_unchanged_and_read_only(repo):
    did = repo.add_dataset("t", "t.csv", b"a,b\n1,2\n", "abc", 1, 2, "upload")
    assert repo.read_raw_bytes(did) == b"a,b\n1,2\n" and repo.get_dataset(did)["name"] == "t" and len(repo.list_datasets()) == 1
    path = repo.get_dataset(did)["raw_path"]
    if os.name == "posix":
        assert oct(os.stat(path).st_mode & 0o777) == "0o400"
    assert not os.access(path, os.W_OK) or _is_root()
    with pytest.raises(KeyError):
        repo.read_raw_bytes("nope")
    assert repo.get_dataset("nope") is None


def test_filename_cannot_escape_the_data_folder(repo):
    did = repo.add_dataset("t", "../../evil.csv", b"a\n1\n", "x", 1, 1, "upload")
    p = repo.get_dataset(did)["raw_path"]
    assert p.startswith(str(repo.root)) and ".." not in p


def test_version_roundtrip_restores_types_and_missing(repo):
    df = pd.DataFrame({"i": pd.array([1, None, 3], dtype="Int64"), "f": [1.5, np.nan, 3.0],
                       "d": pd.to_datetime(["2026-01-01", None, "2026-03-01 10:00:00"], format="mixed"),
                       "b": pd.array([True, None, False], dtype="boolean"), "s": pd.Series(["x", None, "z"], dtype=object)})
    did = repo.add_dataset("t", "t.csv", b"x", "h", 3, 5, "upload")
    vid = repo.save_version(did, df, {"kind": "prepared", "note": "n"})
    back, meta = repo.load_version(vid)
    assert meta["note"] == "n" and str(back["i"].dtype) == "Int64" and back["i"].isna().tolist() == [False, True, False]
    assert pd.api.types.is_datetime64_any_dtype(back["d"]) and back["d"].iloc[2] == pd.Timestamp("2026-03-01 10:00:00") and str(back["b"].dtype) == "boolean"
    assert back["f"].tolist()[0] == 1.5 and pd.isna(back["s"].iloc[1]) and repo.list_versions(did)[0]["rows"] == 3
    assert repo.version_csv_path(vid).endswith(f"{vid}.csv") and repo.version_csv_path("zzz") == ""
    with pytest.raises(KeyError):
        repo.load_version("zzz")


def test_experiments_store_update_and_count_test_reuse(repo, tmp_path):
    split = {"strategy": "random", "train": 0.7, "val": 0.15, "test": 0.15, "group_col": None, "time_col": None, "seed": 1}
    res = {"config": {"task": "classification", "seed": 1, "split": split, "name": "e1"}, "status": "selection_complete", "test_evaluated": False}
    art = tmp_path / "m.joblib"
    art.write_bytes(b"model")
    eid = repo.save_experiment("v1", res, str(art))
    assert repo.count_test_uses("v1", 1, split) == 0
    repo.update_experiment(eid, {**res, "status": "finalized", "test_evaluated": True})
    assert repo.count_test_uses("v1", 1, split) == 1 and repo.count_test_uses("v1", 2, split) == 0 and repo.count_test_uses("v2", 1, split) == 0
    assert repo.count_test_uses("v1", 1, {**split, "test": 0.2, "val": 0.0, "train": 0.8}) == 0   # a different split looks at different test rows
    got = repo.get_experiment(eid)
    assert got["status"] == "finalized" and os.path.exists(got["artifact_path"]) and got["result"]["config"]["name"] == "e1"
    assert repo.list_experiments("v1")[0]["id"] == eid and repo.get_experiment("none") is None


def test_clear_all_removes_everything_even_read_only_files(repo):
    did = repo.add_dataset("t", "t.csv", b"x", "h", 1, 1, "upload")
    repo.save_version(did, pd.DataFrame({"a": [1]}), {})
    repo.clear_all()
    assert repo.list_datasets() == [] and repo.list_versions() == [] and not list(repo.raw_dir.iterdir())


# ---- reports ---------------------------------------------------------------------------------------------------------------------
def test_html_helpers_escape_hostile_content():
    t = H.table(pd.DataFrame({"<img src=x onerror=alert(1)>": ["<script>alert(1)</script>"]}))
    assert "<script>" not in t and "&lt;script&gt;" in t and "<img" not in t
    assert H.fmt(float("nan")) == "n/a" and H.fmt(None) == "n/a" and H.fmt(1234.5678) == "1,235"
    assert "<title>A &amp; B</title>" in H.page("A & B", "s", "")
    assert "None." in H.table(pd.DataFrame())


def test_experiment_report_contains_the_numbers_and_honest_statements():
    exp = {"config": {"task": "classification", "target": "y", "seed": 4, "primary_metric": "f1", "imbalance": "none", "compute": "cpu", "preprocess": {}, "name": "n<b>"},
           "created_at": "t", "status": "finalized", "selection_basis": "the validation set", "split_info": {"effective_strategy": "random", "sizes": {"train": 7}},
           "feature_set": {"numeric": ["a"], "excluded": {"id": "identifier"}}, "warnings": ["careful"], "leakage": [{"message": "leak <x>"}],
           "candidates": [{"key": "m", "name": "Model M", "status": "ok", "score": 0.8123, "score_std": None, "seconds": 1.0, "error": ""}], "selected_key": "m",
           "test_result": {"metrics": {"f1": 0.77, "roc_auc": None}, "unavailable": {"roc_auc": "one class"}, "confusion_matrix": [[1, 0], [0, 1]], "labels": ["a", "b"],
                           "per_class": [{"class": "a", "precision": 1.0}]}, "test_baseline": {"metrics": {"f1": 0.0}}, "interpretation": ["beats baseline"],
           "importance": [{"feature": "a", "importance": 0.1}], "prior_test_uses": 2, "environment": {"python": "3"}, "dataset_fingerprint": "abcdef" * 4}
    html = experiment_report_html(exp, "My data")
    for needle in ("0.8123", "0.77", "one class", "Confusion matrix", "beats baseline", "2 earlier experiment", "used once", "not a general ranking", "&lt;b&gt;", "leak &lt;x&gt;"):
        assert needle in html, needle
    assert "<b>" not in html.split("<h1>")[1].split("</h1>")[0]
    exp["test_result"] = None
    assert "not been used yet" in experiment_report_html(exp, "d")


def test_eda_summary_and_zip():
    df = pd.DataFrame({"column": ["a"], "role": ["numeric"]})
    h = eda_summary_html("d", {"rows": 1, "columns": 1, "missing_cells": 0, "missing_pct": 0, "duplicate_rows": 0}, df, ["obs <i>"], pd.DataFrame(), pd.DataFrame())
    assert "obs &lt;i&gt;" in h and "does not show that one causes" in h
    z = zipfile.ZipFile(io.BytesIO(zip_bytes({"a.txt": "hi", "b.bin": b"\x00"})))
    assert z.read("a.txt") == b"hi" and sorted(z.namelist()) == ["a.txt", "b.bin"]
