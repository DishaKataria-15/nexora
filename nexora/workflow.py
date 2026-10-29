"""Application service layer: the only place the UI talks to ingestion, preparation, ML and storage.

Keeping this separate from Streamlit means every workflow is testable without a browser.
"""
from __future__ import annotations

import json
from urllib.parse import urlparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

from .cleaning import GOALS, SAFE, Plan, PreparationResult, build_plan, default_mapping, prepare, resolve_actions
from .cleaning.transformations import ExecContext, run_actions
from .cleaning.validation import to_text_frame
from .config import Settings, get_settings
from .ml import training as T
from .persistence import Repository, SqliteRepository
from .reporting import experiment_report_html, preparation_report_html, preparation_report_json, transformation_log_csv
from .validation.ingestion import IngestResult, load_upload
from .validation.profiles import Profile, ProfileError, load_builtin_profiles, profile_from_dict

log = logging.getLogger(__name__)

SAMPLES = [
    {"file": "retail_orders_messy.csv", "label": "Retail orders (messy)", "domain": "retail", "goal": "analytics",
     "about": "228 order lines with typical real-world problems: mixed date layouts, placeholder text, currency symbols, repeated rows."},
    {"file": "finance_transactions_messy.csv", "label": "Finance transactions (messy)", "domain": "finance", "goal": "analytics",
     "about": "154 ledger postings with accounting-style negatives, thousand separators, repeated rows and balance mismatches."},
    {"file": "manufacturing_sensors_messy.csv", "label": "Manufacturing sensor readings (messy)", "domain": "manufacturing", "goal": "analytics",
     "about": "Machine readings in mixed units (C and F), repeated rows, blanks and a vibration spike."},
    {"file": "customer_churn.csv", "label": "Customer churn (classification demo)", "domain": "generic", "goal": "supervised",
     "about": "1,200 customers; predict who left. Includes an intentionally leaky column to demonstrate leakage warnings."},
    {"file": "house_prices.csv", "label": "House prices (regression demo)", "domain": "generic", "goal": "supervised",
     "about": "900 listings; predict the price from size, age, neighbourhood and more."},
    {"file": "customer_segments.csv", "label": "Customer segments (clustering demo)", "domain": "generic", "goal": "unsupervised",
     "about": "600 customers with spending and visit patterns; discover groups."},
    {"file": "daily_demand.csv", "label": "Daily demand (time-ordered demo)", "domain": "generic", "goal": "supervised",
     "about": "420 days of units sold; use a chronological split to predict the future from the past."},
]


@dataclass
class DatasetHandle:
    dataset_id: str
    name: str
    raw: pd.DataFrame
    ingest: IngestResult | None = None
    source: str = "upload"


class Workspace:
    def __init__(self, settings: Settings | None = None, repo: Repository | None = None):
        self.settings = settings or get_settings()
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.repo: Repository = repo or SqliteRepository(self.settings.data_dir)
        self._profiles = load_builtin_profiles(self.settings.profiles_dir)
        self.custom_profiles: dict[str, Profile] = {}

    # ---- profiles -------------------------------------------------------------------------------------
    @property
    def profiles(self) -> dict[str, Profile]:
        return {**self._profiles, **self.custom_profiles}

    def add_custom_profile(self, yaml_text: str) -> Profile:
        """Register a user-supplied domain profile (YAML). Validation errors are raised with a readable message."""
        try:
            raw = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            raise ProfileError(f"The file is not valid YAML: {e}") from e
        if not isinstance(raw, dict):
            raise ProfileError("The profile file must contain a YAML mapping (profile_id, fields, rules, ...).")
        prof = profile_from_dict(raw, source="custom")
        if prof.profile_id in self._profiles:
            raise ProfileError(f"'{prof.profile_id}' is a built-in profile id. Choose a different profile_id.")
        self.custom_profiles[prof.profile_id] = prof
        return prof

    @staticmethod
    def profile_depth(profile: Profile) -> str:
        if profile.profile_id in ("retail", "finance", "manufacturing"):
            return "Fully supported: domain checks and domain-aware cleaning, tested end to end"
        if profile.source != "custom" and (profile.settings or {}).get("cleaning"):
            return "Domain checks and domain-aware cleaning (record key, spellings, units, outlier policy); rules still need expert review"
        if profile.profile_id == "generic":
            return "Generic: representation clean-up only, no domain rules"
        if profile.source == "custom":
            return "Custom profile: your rules; cleaning is generic"
        return "Basic: domain checks only; cleaning is generic"

    @staticmethod
    def required_reference_tables(profile: Profile) -> dict[str, str]:
        """External tables a profile's referential rules need: {table name: column that must exist in it}."""
        out: dict[str, str] = {}
        for r in profile.rules:
            if r.rule_type == "referential" and r.params.get("reference_table"):
                out[str(r.params["reference_table"])] = str(r.params.get("reference_column", ""))
        return out

    # ---- ingestion ------------------------------------------------------------------------------------
    def read_upload(self, filename: str, data: bytes, sheet: str | int = 0) -> IngestResult:
        s = self.settings
        return load_upload(data, filename, max_bytes=s.max_upload_bytes, max_rows=s.max_rows, max_cols=s.max_cols, sheet=sheet)

    def ingest(self, filename: str, data: bytes, *, sheet: str | int = 0, source: str = "upload", name: str | None = None) -> DatasetHandle:
        res = self.read_upload(filename, data, sheet)
        did = self.repo.add_dataset(name or Path(res.filename).stem, res.filename, data, res.sha256, res.n_rows, res.n_cols, source,
                                    None if sheet == 0 else str(sheet))
        return DatasetHandle(did, name or Path(res.filename).stem, res.dataframe, res, source)

    def ingest_frame(self, df: pd.DataFrame, name: str, source: str) -> DatasetHandle:
        """Store a table read from a database or an API as an immutable CSV source, then load it like an upload."""
        from .validation.ingestion import IngestError
        if df.empty:
            raise IngestError("No rows were returned, so there is nothing to add.")
        safe = "".join(ch if ch.isalnum() or ch in "-_ " else "_" for ch in name).strip() or "imported"
        data = df.to_csv(index=False).encode("utf-8")
        return self.ingest(f"{safe[:80]}.csv", data, source=source, name=name[:120])

    def combine_datasets(self, sources: dict[str, str], base: str, steps: list, name: str) -> tuple[DatasetHandle, list, dict]:
        """Combine saved datasets ({label: dataset_id}) and save the result as a new dataset. Sources are not changed."""
        from .combine import combine
        frames = {lab: self.open_dataset(did).raw for lab, did in sources.items()}
        out, reports = combine(frames, base, steps, self.settings.max_rows)
        recipe = {"base": base, "sources": sources, "steps": [s.to_dict() for s in steps], "rows": len(out), "columns": int(out.shape[1])}
        handle = self.ingest_frame(out, name, "combined")
        return handle, reports, recipe

    def connect_database(self, cfg):
        """Open a read-only database session (see nexora.connectors.database)."""
        from .connectors.database import DatabaseConnection
        cfg.timeout_s = cfg.timeout_s or self.settings.db_timeout_s
        return DatabaseConnection(cfg)

    def ingest_database(self, conn, *, table: str | None = None, sql: str | None = None, params: dict | None = None,
                        limit: int | None = None, full: bool = False, confirm_full: bool = False, name: str | None = None):
        limit = limit or self.settings.db_row_limit
        if table:
            res = conn.read_table(table, limit=limit, full=full, confirm_full=confirm_full)
        elif sql:
            res = conn.run_query(sql, params, limit=limit)
        else:
            raise ValueError("Choose a table or enter a query.")
        handle = self.ingest_frame(res.frame, name or (table or "query result"), "database")
        return handle, res

    def ingest_api(self, cfg, name: str | None = None):
        from .connectors.web_api import fetch_api
        res = fetch_api(cfg, allow_private=self.settings.allow_private_api_hosts)
        if not name:
            last = res.source.rstrip("/").rsplit("/", 1)[-1]
            name = (last.rsplit(".", 1)[0] if "." in last else last) or urlparse(res.source).hostname or "web data"
        handle = self.ingest_frame(res.frame, name, "web_api")
        return handle, res

    def sample_bytes(self, filename: str) -> bytes:
        return (self.settings.sample_dir / filename).read_bytes()

    def load_sample(self, filename: str) -> DatasetHandle:
        meta = next(s for s in SAMPLES if s["file"] == filename)
        return self.ingest(filename, self.sample_bytes(filename), source="sample", name=meta["label"])

    def open_dataset(self, dataset_id: str) -> DatasetHandle:
        ds = self.repo.get_dataset(dataset_id)
        if not ds:
            raise KeyError("That dataset is no longer available.")
        sheet: str | int = ds["sheet"] if ds.get("sheet") else 0
        res = self.read_upload(ds["filename"], self.repo.read_raw_bytes(dataset_id), sheet)
        return DatasetHandle(dataset_id, ds["name"], res.dataframe, res, ds["source"])

    # ---- preparation ----------------------------------------------------------------------------------
    def make_plan(self, handle: DatasetHandle, profile_id: str, goal: str, mapping: dict[str, str | None] | None = None,
                  as_of: str | None = None, reference_tables: dict[str, pd.DataFrame] | None = None) -> Plan:
        prof = self.profiles[profile_id]
        mp = mapping if mapping is not None else default_mapping(list(handle.raw.columns), prof)
        return build_plan(handle.raw, prof, mp, goal, dataset_name=handle.name, reference_tables=reference_tables, as_of=as_of)

    def run_preparation(self, handle: DatasetHandle, plan: Plan, approvals: dict[str, bool] | None, choices: dict[str, str] | None,
                        *, as_of: str | None = None, seed: int | None = None, progress: Callable[[float, str], None] | None = None,
                        reference_tables: dict[str, pd.DataFrame] | None = None,
                        overrides: dict[str, dict] | None = None) -> tuple[PreparationResult, str]:
        prof = self.profiles[plan.profile_id]
        res = prepare(handle.raw, prof, plan.mapping, plan, approvals=approvals, choices=choices, overrides=overrides, as_of=as_of,
                      reference_tables=reference_tables,
                      seed=self.settings.default_seed if seed is None else seed, progress=progress)
        rep = preparation_report_json(res, handle.name, prof.profile_id, prof.version, plan.goal, plan.mapping)
        vid = self.repo.save_version(handle.dataset_id, res.prepared, {"kind": "prepared", "profile_id": prof.profile_id, "profile_version": prof.version,
                                                                         "goal": plan.goal, "report": rep, "reconciliation_ok": res.reconciliation["ok"],
                                                                         "dataset_name": handle.name})
        return res, vid

    def preparation_exports(self, res: PreparationResult, handle: DatasetHandle, plan: Plan) -> dict[str, bytes]:
        prof = self.profiles[plan.profile_id]
        rep = preparation_report_json(res, handle.name, prof.profile_id, prof.version, plan.goal, plan.mapping)
        return {"prepared_data.csv": to_text_frame(res.prepared).to_csv(index=False).encode("utf-8"),
                "preparation_report.html": preparation_report_html(res, handle.name, prof.name, prof.version, GOALS.get(plan.goal, plan.goal)).encode("utf-8"),
                "preparation_report.json": json.dumps(rep, indent=2, default=str).encode("utf-8"),
                "transformation_log.csv": transformation_log_csv(res).encode("utf-8")}

    def autotype(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Best-effort typing for 'explore only': safe representation fixes, no cleaning decisions, no row removal."""
        prof = self.profiles["generic"]
        plan = build_plan(raw, prof, {}, "explore", dataset_name="autotype")
        run, _ = resolve_actions(plan, {}, {})
        run = [a for a in run if a.kind == SAFE and a.op != "remove_exact_duplicates"]
        out, _ = run_actions(raw.reset_index(drop=True).astype(object), run, ExecContext(set(prof.null_tokens)))
        return out

    def working_frame(self, version_id: str | None, handle: DatasetHandle | None) -> tuple[pd.DataFrame, str]:
        """The data used for exploration and modelling, plus a plain-language description of where it came from."""
        if version_id:
            df, meta = self.repo.load_version(version_id)
            return df, f"Prepared version ({len(df):,} rows) of {meta.get('dataset_name', 'the dataset')}"
        if handle is not None:
            return self.autotype(handle.raw), "Unprepared data: types were detected automatically, nothing was cleaned or removed"
        raise ValueError("No dataset is open.")

    # ---- machine learning -----------------------------------------------------------------------------
    def run_experiment_job(self, df: pd.DataFrame, cfg: T.ExperimentConfig, version_id: str | None) -> T.TrainingJob:
        prior = 0
        if version_id and cfg.task != "clustering":
            prior = self.repo.count_test_uses(version_id, cfg.seed, cfg.split)

        def fn(progress, token):
            return T.run_selection(df, cfg, cancel=token, progress=progress, prior_test_uses=prior)
        return T.TrainingJob(fn).start()

    def save_experiment(self, exp: T.Experiment, version_id: str | None, existing_id: str | None = None) -> str:
        art = None
        if exp._state:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                art = T.save_artifact(exp, f.name)
        try:
            if existing_id:
                self.repo.update_experiment(existing_id, exp.to_dict(), art)
                return existing_id
            return self.repo.save_experiment(version_id, exp.to_dict(), art)
        finally:
            if art:
                Path(art).unlink(missing_ok=True)

    def predict_saved(self, experiment_id: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Score new rows with a model saved by this application (pickle files from elsewhere are never loaded)."""
        from .ml.predict import PredictionError, predict_frame
        rec = self.repo.get_experiment(experiment_id)
        if not rec or not rec.get("artifact_path") or not Path(rec["artifact_path"]).exists():
            raise PredictionError("This experiment has no saved model file.")
        return predict_frame(T.load_artifact(rec["artifact_path"]), df)

    def experiment_exports(self, exp: T.Experiment, dataset_name: str) -> dict[str, bytes]:
        d = exp.to_dict()
        files: dict[str, bytes] = {"experiment_report.html": experiment_report_html(d, dataset_name).encode("utf-8"),
                                   "experiment_config.json": json.dumps({"config": d["config"], "split": d["split_info"], "environment": d["environment"],
                                                                        "dataset_fingerprint": d["dataset_fingerprint"]}, indent=2).encode("utf-8")}
        st = exp._state
        if st and st.get("test_predictions") is not None:
            files["test_predictions.csv"] = st["test_predictions"].to_csv(index=False).encode("utf-8")
        if exp.config.task == "clustering" and st:
            lab = pd.DataFrame({"row_in_dataset": range(len(st["labels"])), "cluster": ["noise" if v == -1 else int(v) for v in st["labels"]]})
            files["cluster_assignments.csv"] = lab.to_csv(index=False).encode("utf-8")
        if st:
            with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
                p = T.save_artifact(exp, f.name)
            files["model_pipeline.joblib"] = Path(p).read_bytes()
            Path(p).unlink(missing_ok=True)
        return files
