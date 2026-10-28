# Architecture notes

## Overview

Nexora is a modular Python application. Streamlit provides the interface; everything else is plain Python that can be tested without a browser.

```
Streamlit pages (nexora/ui)  ->  Workspace service (nexora/workflow.py)  ->  domain modules
                                                                              validation  cleaning  eda  preprocessing  ml  reporting
                                                                                                   |
                                                                              Repository interface -> SQLite + files (nexora/persistence)
```

The project started from DataSentinel, an existing validation-only application. Its rule engine (20 rule types), 18 YAML domain profiles,
ingestion, profiler and scoring were kept unchanged in `nexora/validation/` together with their 342 tests. DataSentinel's own UI, database
and review/export modules were not carried over; Nexora replaces them. Nexora adds everything after "found a problem": preparation, EDA, ML and reporting.

## Data flow

1. **Ingest** (`validation/ingestion.py`). Reads bytes safely (encoding and delimiter detection, size/row/column limits, header clean-up,
   filename sanitising). Every cell is kept as text so nothing is silently reinterpreted. The original bytes are stored read-only.
2. **Plan** (`cleaning/planner.py`). Builds `Action` objects from the raw text frame, the domain profile, the column mapping and the goal.
   Each safe action is dry-run on a scratch copy, so the counts and examples shown are measured, and later proposals see the effect of earlier ones.
   Domain validation is run on the scratch result to predict what will remain.
3. **Confirm** (UI). One confirmation. Safe actions default on, optional actions off, decisions default to "leave unchanged".
4. **Execute** (`cleaning/transformations.py`, `cleaning/engine.py`). Actions run in a fixed order on a copy. Each executor computes its change first and assigns it last,
   so a failure cannot leave a half-changed frame; a failing step is logged and skipped.
5. **Validate and reconcile** (`cleaning/validation.py`). The prepared frame is rendered back to text exactly as it would be exported, validated with the same rule engine, and measured
   with domain-independent indicators. Reconciliation checks: row counts add up and chain step by step, no rows invented, columns accounted for, identifiers not invented,
   raw fingerprint unchanged, no step failed.
6. **Persist** (`persistence/repository.py`). The prepared version is stored as CSV plus a schema, so types survive a reload.
7. **Explore / model** use the prepared version (or a lightly typed copy of the raw data, labelled unprepared).

## Configuration-driven domains

A domain profile is a YAML file. Validation rules work as in DataSentinel. A `settings.cleaning` block drives domain-aware preparation:

```yaml
settings:
  cleaning:
    normalize_case: {order_type: upper, currency: upper}
    missing:   {discount_amount: {strategy: constant, value: "0", rationale: "..."}}
    duplicates: {key: [order_id, sku], rationale: "..."}
    units:
      - {label: temperature, value_field: temperature, unit_field: temperature_unit, target_unit: C,
         conversions: {C: [1, 0], F: [0.5556, -17.78], K: [1, -273.15]}}   # y = x*scale + offset
         # optional: also_fields: [other value columns measured in the same unit column]
    outliers: {policy: review | report_only, rationale: "..."}
```

`normalize_case` accepts either a mode (`upper`, `lower`, `title`) or `{mode: upper, map: {"IN TRANSIT": IN_TRANSIT}}` for domain spellings.
The whole section is validated when a profile loads (unknown sections, undeclared fields, malformed conversions or a target unit without a
conversion entry are errors), so a typo fails loudly instead of silently doing nothing.

No domain logic lives in application code. To extend a domain, edit its YAML. To add a new kind of transformation, register an executor in
`transformations.EXECUTORS`, give it a position in `ORDER`, and have the planner propose it. Referential rules can name an external table
(`reference_table`); the UI asks for that table and passes it through.

### Domain coverage

| Level | Profiles | What works |
|---|---|---|
| Fully supported | Retail, Finance, Manufacturing | Domain checks plus domain-aware cleaning; tested end to end against known injected problems and recorded ground truth |
| Domain cleaning | 15 other bundled profiles | Domain checks plus domain-aware cleaning (record key, spellings, exact unit conversions where safe, outlier policy); tested with synthetic data generated from each profile's field list. Rules are prototypes that need expert review |
| Generic / custom | `generic`, uploaded YAML | Representation clean-up; your own rules |

## Safety model for preparation

| Principle | How it is enforced |
|---|---|
| Raw source immutable | Stored read-only; frames are copied before every step; a content fingerprint is compared before and after and checked in reconciliation |
| Nothing guessed | Ambiguous dates, unreadable numbers, missing required identifiers and keyless duplicates become *decisions* that default to "leave unchanged" |
| Exact vs similar duplicates | Only rows identical in every column *and* sharing a non-missing record key are removed. Same key with different values is listed for review |
| Outliers are signals | Detected and reported by default; flag, clip, winsorize, median replacement or row removal only when the user chooses it, after a preview. Profiles can recommend report-only |
| Identifiers and labels never invented | Imputation excludes identifier-like columns; missing targets are excluded only with explicit confirmation |
| Complete log | Operation, columns, reason, parameters, rows in/out, values changed, status, skipped and removed row numbers; actions that were *not* applied are logged too |

## Machine learning discipline

* **Fit on training data only.** Imputers, scalers, encoders, date origin and text vocabularies live inside a scikit-learn `Pipeline` and are fitted on training rows.
  A test asserts the learned statistics equal training-only statistics.
* **Resampling only inside training data.** The sampler is a pipeline step, so it runs during `fit` only: on the training rows (validation protocol) or on each fold's
  training part (cross-validation). A test records the sizes seen by a spy sampler.
* **Test set used once.** `run_selection` never touches test rows: a test scrambles the test rows' features and targets and requires identical selection results.
  `finalize_experiment` raises `TestSetAlreadyUsed` the second time. Experiments on the same data version, seed and split are counted, and the count is shown as a warning.
* **Leakage checks.** Single-feature predictors (including "is it filled in?"), near-copies of a regression target, post-outcome column names, duplicate rows across splits,
  random splits on dated data, and repeated entities under a non-group split.
* **Fair comparison.** All candidates share one split, one preprocessing configuration and one metric. "Best" always means best on that protocol.
* **Compute.** CPU/GPU detection with `nvidia-smi`. GPU is offered only if a device **and** a GPU-capable model exist; only the optional XGBoost models are GPU-capable.

### Pipeline order

`coerce -> (feature generation) -> ColumnTransformer [numeric: impute -> (outlier treatment) -> (power) -> scale | categories | dates | text]
-> (reduction) -> (selection) -> (sampler) -> model`. Every bracketed step is fitted inside `Pipeline.fit`, so in cross-validation and tuning it is
re-learned on each fold's training part. Tests corrupt held-out rows and require identical learned statistics, selected features, tuning trials and
explanations.

### Model registry

`ml/model_registry.py` holds one `ModelSpec` per model: task, factory, plain-language description and trade-off, whether scaling is needed, probability,
sample-weight and native-missing-value support, whether new rows can be predicted (not for DBSCAN, OPTICS, spectral or hierarchical clustering), row
limits, input requirements (Multinomial NB needs non-negative input, so it forces 0-1 scaling and refuses PCA/SVD), optional package, declared settings
with ranges (`ParamSpec`) and a small, fixed search space. `check_compatibility` explains why a model cannot run; `validate_params` rejects out-of-range
settings with a readable message. Support vector classification gets probabilities from `CalibratedClassifierCV(ensemble=False)` (scikit-learn 1.9
deprecates `SVC(probability=True)`); XGBoost is wrapped so text class labels work.

### Cross-validation and tuning

`ml/cv.py` builds folds over training rows only: k-fold, stratified, repeated (stratified), group (stratified-group when possible) and forward-chaining
time-series folds with an optional gap. *Automatic* matches the split strategy (chronological split -> time-series folds; group split -> group folds).
`ml/tuning.py` runs grid, random and successive-halving searches as its own loop so it can be cancelled between fits, respect a time budget and a trial
cap (at most 60), and record every trial. A failing combination is recorded and the search continues. When models are compared by cross-validation, the
tuned scores reuse the tuning folds and are slightly optimistic; the interface says so, and the one-time test result remains the unbiased estimate.

### Explanations

`ml/explain.py`: impurity importance or coefficient size on the prepared features; optional SHAP (tree and linear models only, at most 500 non-test rows).
Permutation importance on the original columns is kept from the first release.

### Anomaly detection and forecasting

`ml/anomaly.py` fits unsupervised detectors on robust-scaled prepared features and orients every score so that higher is more unusual. A label column,
if given, is excluded from the features and used only for precision / recall / PR-AUC; a test checks scores are identical with and without it.
`ml/forecasting.py` regularises the series (duplicates combined, gaps filled or refused), infers the season, backtests every method with an expanding
window (a test corrupts everything after the first window and requires the first window's errors to be unchanged), always includes naive and
seasonal-naive baselines, and builds a rough range from backtest errors per step ahead.

## Connectors

`connectors/database.py` (SQLAlchemy): SQLite opened with `mode=ro`; PostgreSQL sessions with `default_transaction_read_only` and a statement timeout;
MySQL with `TRANSACTION READ ONLY` and `MAX_EXECUTION_TIME`; SQL Server with `ApplicationIntent=ReadOnly`. Table and column names are checked against
the database catalogue and the query is built with SQLAlchemy constructs; free SQL must be a single SELECT/WITH statement without write or
side-effect keywords (checked after removing comments and quoted text) and values are bound parameters. Reads stop at a row limit and report
truncation; whole-table reads need explicit confirmation; SQLite queries are interrupted at the timeout. Passwords are held in memory (or read from an
environment variable), excluded from `repr`, and redacted from errors and logs.

`connectors/web_api.py` (requests): http/https only; credentials only over https (or to localhost when private hosts are allowed); host names are
resolved and loopback, private, link-local (including cloud metadata), carrier-grade NAT, multicast and reserved addresses are refused unless
`NEXORA_ALLOW_PRIVATE_API=1`. Redirects are followed manually (at most 3) and re-checked; credentials are never sent to another host, including
pagination links. Timeouts, bounded retries with exponential backoff (429 / 5xx / connection errors; `Retry-After` honoured up to 30 s), a 20 MB
per-response and 100 MB total limit, and at most 50 pages. Request logs contain no headers or query strings, and urllib3's debug request logging is
silenced because it would include query strings.

Both connectors hand the result to `Workspace.ingest_frame`, which stores it as an immutable CSV source like any upload.

## Persistence

`Repository` is an abstract interface; `SqliteRepository` stores metadata in SQLite and files under `data/` (`raw/`, `versions/`, `artifacts/`).
A different backend can replace it without touching the UI.

## Background training

`TrainingJob` runs an experiment in a daemon thread and exposes state, progress, elapsed time and `cancel()`. Progress values are reported by the training loop after
real events (model started, fold finished), never simulated. The UI polls with a Streamlit fragment. Cancellation is cooperative: it is honoured between models,
folds and clustering fits.

## Assumptions

* Single user on one computer; no authentication.
* Files fit in memory (default limits: 50 MB, 300,000 rows, 300 columns).
* Each row of an uploaded file is one record.
* A column named like an identifier and mostly unique is treated as an identifier.
* Placeholder tokens (`N/A`, `null`, `none`, `nan`, blank, ...) are configurable per profile. "None" can be a real category in some data, which is why the plan shows the count and examples.

## Limitations

* The 15 profiles beyond Retail, Finance and Manufacturing have domain-aware cleaning, but their rules and policies were written without a domain expert and
  are tested only on synthetic data generated from their field lists.
* The interface was exercised through Streamlit's AppTest (real widgets, buttons, navigation and state) on Linux. It has not been viewed in a browser on Windows,
  and visual layout (spacing, contrast, overflow on small screens) has not been reviewed on a real screen.
* Cancellation cannot interrupt a single model fit. Training uses scikit-learn estimators on CPU only.
* Hierarchical clustering is capped at 5,000 rows. DBSCAN's silhouette ignores noise rows, which can flatter it; the interface says so.
* PNG chart export is not included (it needs an extra rendering package); charts and dashboards export as interactive HTML.
* Forecasting handles one series at a time (no per-entity panels), uses recursive lag models, optional Holt-Winters, and no ARIMA. Ranges are empirical
  from a few backtest windows, not calibrated intervals.
* Database connections were tested against SQLite only. PostgreSQL, MySQL and SQL Server code paths (read-only session settings, timeouts) were
  not run against real servers in this environment. The SELECT-only check is conservative and can refuse unusual but harmless queries.
* Web API host checks resolve the name before the request; a DNS answer that changes between the check and the request (DNS rebinding) is not fully
  prevented.
* Free-text features are simple word weights (TF-IDF, 200 terms) and off by default.
* Exported CSVs are not protected against spreadsheet formula injection: a value beginning with `=` is written as-is.
* Model files use pickle and must only be loaded from trusted sources.
* Preparation fixes representation problems; it does not decide business truth. Domain checks that fail (for example totals that do not match) are reported, not corrected.
