"""Local persistence (SQLite + files) behind a small repository interface.

Raw uploads are written once and made read-only. Prepared data are stored as new versions; nothing is overwritten.
"""
from __future__ import annotations

import abc
import contextlib
import json
import os
import shutil
import sqlite3
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from ..cleaning.validation import schema_of, to_text_frame, typed_from_schema
from ..validation.ingestion import safe_filename


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _id() -> str:
    return uuid.uuid4().hex[:12]


class Repository(abc.ABC):
    """What the application needs from storage. Swap the implementation without touching the UI."""

    @abc.abstractmethod
    def add_dataset(self, name: str, filename: str, data: bytes, sha256: str, rows: int, cols: int, source: str, sheet: str | None = None) -> str: ...
    @abc.abstractmethod
    def list_datasets(self) -> list[dict[str, Any]]: ...
    @abc.abstractmethod
    def get_dataset(self, dataset_id: str) -> dict[str, Any] | None: ...
    @abc.abstractmethod
    def read_raw_bytes(self, dataset_id: str) -> bytes: ...
    @abc.abstractmethod
    def save_version(self, dataset_id: str, prepared: pd.DataFrame, meta: dict[str, Any]) -> str: ...
    @abc.abstractmethod
    def list_versions(self, dataset_id: str | None = None) -> list[dict[str, Any]]: ...
    @abc.abstractmethod
    def load_version(self, version_id: str) -> tuple[pd.DataFrame, dict[str, Any]]: ...
    @abc.abstractmethod
    def version_csv_path(self, version_id: str) -> str: ...
    @abc.abstractmethod
    def save_experiment(self, version_id: str | None, result: dict[str, Any], artifact_src: str | None = None) -> str: ...
    @abc.abstractmethod
    def update_experiment(self, experiment_id: str, result: dict[str, Any], artifact_src: str | None = None) -> None: ...
    @abc.abstractmethod
    def list_experiments(self, version_id: str | None = None) -> list[dict[str, Any]]: ...
    @abc.abstractmethod
    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None: ...
    @abc.abstractmethod
    def count_test_uses(self, version_id: str, seed: int, split: dict[str, Any]) -> int: ...
    @abc.abstractmethod
    def rename_dataset(self, dataset_id: str, name: str) -> None: ...
    @abc.abstractmethod
    def delete_dataset(self, dataset_id: str) -> dict[str, int]: ...
    @abc.abstractmethod
    def save_run(self, kind: str, dataset_name: str, version_id: str | None, summary: dict[str, Any], files: dict[str, bytes]) -> str: ...
    @abc.abstractmethod
    def list_runs(self, kind: str | None = None) -> list[dict[str, Any]]: ...
    @abc.abstractmethod
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...
    @abc.abstractmethod
    def delete_run(self, run_id: str) -> None: ...
    @abc.abstractmethod
    def clear_all(self) -> None: ...


class SqliteRepository(Repository):
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.raw_dir, self.ver_dir, self.art_dir, self.run_dir = (self.root / d for d in ("raw", "versions", "artifacts", "runs"))
        self.db_path = self.root / "nexora.db"
        for d in (self.raw_dir, self.ver_dir, self.art_dir, self.run_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._init()

    @contextlib.contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=15, check_same_thread=False)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY, name TEXT, filename TEXT, sha256 TEXT, rows INTEGER, cols INTEGER,
                source TEXT, sheet TEXT, raw_path TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS versions(id TEXT PRIMARY KEY, dataset_id TEXT, created_at TEXT, csv_path TEXT, rows INTEGER, cols INTEGER,
                schema_json TEXT, meta_json TEXT);
            CREATE TABLE IF NOT EXISTS experiments(id TEXT PRIMARY KEY, version_id TEXT, name TEXT, task TEXT, status TEXT, created_at TEXT,
                test_evaluated INTEGER, seed INTEGER, split_json TEXT, result_json TEXT, artifact_path TEXT);
            CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, kind TEXT, dataset_name TEXT, version_id TEXT, created_at TEXT,
                summary_json TEXT, files_dir TEXT);
            """)

    # datasets -----------------------------------------------------------------------------------
    def add_dataset(self, name, filename, data, sha256, rows, cols, source, sheet=None) -> str:
        did = _id()
        d = self.raw_dir / did
        d.mkdir(parents=True)
        path = d / safe_filename(filename)
        path.write_bytes(data)
        os.chmod(path, stat.S_IREAD)       # the raw source is immutable on disk
        with self._conn() as c:
            c.execute("INSERT INTO datasets VALUES (?,?,?,?,?,?,?,?,?,?)", (did, name, safe_filename(filename), sha256, rows, cols, source, sheet, str(path), _now()))
        return did

    def list_datasets(self):
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM datasets ORDER BY created_at DESC, rowid DESC")]

    def get_dataset(self, dataset_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return dict(r) if r else None

    def read_raw_bytes(self, dataset_id):
        ds = self.get_dataset(dataset_id)
        if not ds:
            raise KeyError(f"Dataset {dataset_id} not found")
        return Path(ds["raw_path"]).read_bytes()

    def rename_dataset(self, dataset_id, name) -> None:
        name = " ".join(str(name or "").split())[:120]
        if not name:
            raise ValueError("The name cannot be empty.")
        with self._conn() as c:
            if not c.execute("UPDATE datasets SET name=? WHERE id=?", (name, dataset_id)).rowcount:
                raise KeyError(f"Dataset {dataset_id} not found")

    def delete_dataset(self, dataset_id) -> dict[str, int]:
        """Remove a dataset with its raw file, its prepared versions and the experiments trained on those versions.

        Returns counts of what was removed. Experiments trained on the unprepared data keep their results in Reports
        (they do not record which dataset they came from beyond its name).
        """
        ds = self.get_dataset(dataset_id)
        if not ds:
            raise KeyError(f"Dataset {dataset_id} not found")
        with self._conn() as c:
            vers = [dict(r) for r in c.execute("SELECT id, csv_path FROM versions WHERE dataset_id=?", (dataset_id,))]
            vids = [v["id"] for v in vers]
            exps = [dict(r) for r in c.execute(f"SELECT id, artifact_path FROM experiments WHERE version_id IN ({','.join('?' * len(vids)) or 'NULL'})", vids)]
            c.execute(f"DELETE FROM experiments WHERE version_id IN ({','.join('?' * len(vids)) or 'NULL'})", vids)
            c.execute("DELETE FROM versions WHERE dataset_id=?", (dataset_id,))
            c.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
        for path in [v["csv_path"] for v in vers] + [e["artifact_path"] for e in exps if e["artifact_path"]]:
            with contextlib.suppress(OSError):
                Path(path).unlink()
        raw = self.raw_dir / dataset_id
        for p in raw.rglob("*"):
            with contextlib.suppress(OSError):
                os.chmod(p, stat.S_IRWXU)            # the raw copy is read-only on disk
        shutil.rmtree(raw, ignore_errors=True)
        return {"versions": len(vers), "experiments": len(exps)}

    # versions -----------------------------------------------------------------------------------
    def save_version(self, dataset_id, prepared, meta) -> str:
        vid = _id()
        path = self.ver_dir / f"{vid}.csv"
        to_text_frame(prepared).to_csv(path, index=False)
        with self._conn() as c:
            c.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?)", (vid, dataset_id, _now(), str(path), len(prepared), prepared.shape[1],
                                                                          json.dumps(schema_of(prepared)), json.dumps(meta, default=str)))
        return vid

    def list_versions(self, dataset_id=None):
        q, a = "SELECT id, dataset_id, created_at, rows, cols, meta_json FROM versions", ()
        if dataset_id:
            q, a = q + " WHERE dataset_id=?", (dataset_id,)
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q + " ORDER BY created_at DESC, rowid DESC", a)]
        for r in rows:
            r["meta"] = json.loads(r.pop("meta_json") or "{}")
        return rows

    def load_version(self, version_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM versions WHERE id=?", (version_id,)).fetchone()
        if not r:
            raise KeyError(f"Version {version_id} not found")
        text = pd.read_csv(r["csv_path"], dtype=str, keep_default_na=False)
        return typed_from_schema(text, json.loads(r["schema_json"])), json.loads(r["meta_json"] or "{}")

    def version_csv_path(self, version_id: str) -> str:
        with self._conn() as c:
            r = c.execute("SELECT csv_path FROM versions WHERE id=?", (version_id,)).fetchone()
        return r["csv_path"] if r else ""

    # experiments --------------------------------------------------------------------------------
    def _store_artifact(self, eid: str, src: str | None) -> str | None:
        if not src:
            return None
        dst = self.art_dir / f"{eid}.joblib"
        shutil.copyfile(src, dst)
        return str(dst)

    def save_experiment(self, version_id, result, artifact_src=None) -> str:
        eid = _id()
        cfg = result.get("config", {})
        art = self._store_artifact(eid, artifact_src)
        with self._conn() as c:
            c.execute("INSERT INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                eid, version_id, cfg.get("name") or f"{cfg.get('task', '')} {_now()[:16]}", cfg.get("task"), result.get("status"), _now(),
                int(bool(result.get("test_evaluated"))), cfg.get("seed"), json.dumps(cfg.get("split", {}), sort_keys=True), json.dumps(result, default=str), art))
        return eid

    def update_experiment(self, experiment_id, result, artifact_src=None) -> None:
        art = self._store_artifact(experiment_id, artifact_src)
        with self._conn() as c:
            c.execute("UPDATE experiments SET status=?, test_evaluated=?, result_json=?, artifact_path=COALESCE(?, artifact_path) WHERE id=?",
                      (result.get("status"), int(bool(result.get("test_evaluated"))), json.dumps(result, default=str), art, experiment_id))

    def list_experiments(self, version_id=None):
        q, a = "SELECT id, version_id, name, task, status, created_at, test_evaluated, seed, artifact_path FROM experiments", ()
        if version_id:
            q, a = q + " WHERE version_id=?", (version_id,)
        with self._conn() as c:
            return [dict(r) for r in c.execute(q + " ORDER BY created_at DESC, rowid DESC", a)]

    def get_experiment(self, experiment_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["result"] = json.loads(d.pop("result_json"))
        d["split"] = json.loads(d.pop("split_json"))
        return d

    def count_test_uses(self, version_id, seed, split) -> int:
        """How many finalized experiments already looked at the same test rows (same data version, seed and split settings)."""
        with self._conn() as c:
            rows = c.execute("SELECT split_json FROM experiments WHERE version_id=? AND seed=? AND test_evaluated=1", (version_id, seed)).fetchall()
        return sum(1 for r in rows if _same_test_rows(json.loads(r["split_json"]), split))

    # anomaly and forecast runs ----------------------------------------------------------------
    def save_run(self, kind, dataset_name, version_id, summary, files) -> str:
        rid = _id()
        d = self.run_dir / rid
        d.mkdir(parents=True)
        for fn, data in files.items():
            (d / safe_filename(fn)).write_bytes(data)
        with self._conn() as c:
            c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)", (rid, kind, dataset_name, version_id, _now(), json.dumps(summary, default=str), str(d)))
        return rid

    def list_runs(self, kind=None):
        q, a = "SELECT id, kind, dataset_name, version_id, created_at, summary_json FROM runs", ()
        if kind:
            q, a = q + " WHERE kind=?", (kind,)
        with self._conn() as c:
            rows = [dict(r) for r in c.execute(q + " ORDER BY created_at DESC, rowid DESC", a)]
        for r in rows:
            r["summary"] = json.loads(r.pop("summary_json") or "{}")
        return rows

    def get_run(self, run_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["summary"] = json.loads(d.pop("summary_json") or "{}")
        folder = Path(d.pop("files_dir"))
        d["files"] = {p.name: p.read_bytes() for p in sorted(folder.glob("*")) if p.is_file()} if folder.is_dir() else {}
        return d

    def delete_run(self, run_id) -> None:
        with self._conn() as c:
            r = c.execute("SELECT files_dir FROM runs WHERE id=?", (run_id,)).fetchone()
            c.execute("DELETE FROM runs WHERE id=?", (run_id,))
        if r:
            shutil.rmtree(r["files_dir"], ignore_errors=True)

    def clear_all(self) -> None:
        for d in (self.raw_dir, self.ver_dir, self.art_dir, self.run_dir):
            for p in d.rglob("*"):
                with contextlib.suppress(OSError):
                    os.chmod(p, stat.S_IRWXU)
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            for t in ("datasets", "versions", "experiments", "runs"):
                c.execute(f"DELETE FROM {t}")


def _same_test_rows(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(a.get(k) == b.get(k) for k in ("strategy", "train", "val", "test", "group_col", "time_col"))
