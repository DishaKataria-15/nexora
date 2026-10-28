# Nexora

**Connect data, models and decisions.**

Nexora is a local application that takes a spreadsheet from upload to a trained, evaluated model in one guided flow:
upload a CSV or Excel file, review a short automatic preparation plan, confirm once, explore the cleaned data, then
predict a number, predict a category, or discover groups. Every number it shows is computed from your data.

It runs on your own computer. Nothing is sent anywhere and no paid service is used.

## What it does

| Area | What you get |
|---|---|
| **Datasets** | CSV / TSV / Excel upload with validation and preview; read-only **database** connections (SQLite, PostgreSQL, MySQL, SQL Server) and **web API** ingestion (JSON / CSV, pagination, authentication); clearly labelled synthetic sample data; **Combine datasets (advanced)** joins, summarises or stacks two to five saved datasets into one; every source is stored read-only and never modified. |
| **Prepare Data** | Profile, choose a domain and goal, review a plan where every step has a plain-language reason and expandable details, confirm once. Safe steps run automatically; ambiguous ones (for example `03/04/2026`) wait for your decision. Before/after report with reconciliation checks and a complete step log. |
| **Visualize & Dashboard** | Computed summaries, per-column views, correlations, recommended charts, a chart builder, a filterable dashboard you can export as HTML, and a **Dimensions** view (PCA, SVD, t-SNE, optional UMAP). |
| **ML Studio** | Regression, classification and clustering with 34 registered models (XGBoost and LightGBM optional). Target and leakage checks, reproducible splits, class-imbalance handling, in-pipeline outlier treatment, feature engineering, dimensionality reduction and feature selection (all learned on training rows only), k-fold / stratified / group / repeated / time-series cross-validation, bounded grid, random and successive-halving tuning, background training with cancellation, one-time test-set evaluation, native importance and optional SHAP. |
| **Anomalies** | Unsupervised anomaly scoring (Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope, robust z-score), the most unusual rows with the reason, and optional evaluation against a column of known anomalies. |
| **Forecasting** | Time-series forecasts with rolling-origin backtesting against naive and seasonal-naive baselines; lag-feature models, optional Holt-Winters, known-in-advance columns, rough prediction ranges. |
| **Help assistant** | A *Help: ask about Nexora* panel in the sidebar answers typed questions about the interface from about 60 built-in answers, with suggested questions for the page you are on and a *Go to* button. It works offline and says so when it has no answer. |
| **Reports** | Every experiment and preparation run is saved with its settings, seed and results, and can be reopened, compared and downloaded. Saved models can score new rows. |

Domain depth: all 18 domain profiles have domain-aware cleaning (record keys, required spellings, exact unit conversions where they
are safe, an outlier policy). **Retail, Finance and Manufacturing** are additionally tested end to end against recorded ground truth;
the other 15 are tested with synthetic data built from their field lists, and their rules still need review by a domain expert. See
`docs/ARCHITECTURE.md` for exactly what each level means.

## Quick start (Windows PowerShell)

Requires Python 3.10 or newer.

```powershell
cd path\to\nexora
.\scripts\setup.ps1      # creates .venv and installs requirements
.\scripts\run.ps1        # starts Nexora; your browser opens at http://localhost:8501
```

If PowerShell refuses to run scripts, allow them for this window only, then retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

The same steps by hand:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
streamlit run app.py
```

Optional extras (XGBoost, LightGBM, SHAP, Holt-Winters, UMAP, database drivers): `pip install -r requirements-optional.txt`.
Nexora works without them; each feature shows its install command when the package is missing.

### macOS / Linux

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
streamlit run app.py
```

## Try it in two minutes

1. Open **Datasets > Sample data > Retail orders (messy)** and click *Open sample*.
2. **Prepare Data**: *Next*, then *Build preparation plan*. Read the plan, choose how to read the ambiguous ship dates, and click *Confirm and prepare data*.
3. Read the before/after report and download the cleaned CSV and report.
4. **Visualize & Dashboard**: add charts to the dashboard and filter them.
5. Open **Datasets > Sample data > Customer churn**, go to **ML Studio**, choose *Predict a category*, and watch Nexora flag the leaky column.

Sample files are in `sample_data/synthetic/` and can be regenerated with `python sample_data/make_samples.py`. They are invented data.

## Run the tests

```powershell
.\scripts\test.ps1          # or:  python -m pytest -q
```

The suite has 540 tests and takes about two to three minutes. See `docs/TESTING.md` for what is covered and the last recorded result.

## Where things are

```
app.py                  Streamlit entry point
config/domains/*.yaml   Domain profiles (rules and cleaning configuration)
nexora/
  validation/           Rule engine, ingestion, profiler (from DataSentinel)
  cleaning/             Planner, transformations, engine, reconciliation
  eda/                  Statistics, observations, charts
  preprocessing/        Pipelines, leakage checks, outliers, feature engineering, reduction, selection
  ml/                   Splits, CV, tuning, registry, training, evaluation, explain, anomaly, forecasting, predict
  connectors/           Read-only database and web API ingestion
  reporting/            HTML / JSON / CSV reports
  persistence/          SQLite repository
  ui/                   Streamlit pages
  workflow.py           Service layer used by the UI and the tests
sample_data/            Synthetic datasets and their generator
tests/                  Unit and integration tests
docs/                   User guide, architecture notes, test report
scripts/                Windows PowerShell setup, run and test scripts
```

Your data lives in `data/` (created on first run). Set `NEXORA_DATA_DIR` to move it. Other settings: `NEXORA_MAX_UPLOAD_MB` (50),
`NEXORA_MAX_ROWS` (300000), `NEXORA_MAX_COLS` (300), `NEXORA_SEED` (42), `NEXORA_DB_TIMEOUT_S` (30), `NEXORA_DB_ROW_LIMIT` (100000),
`NEXORA_ALLOW_PRIVATE_API` (0; set to 1 only to let web API ingestion reach local or private network addresses).

## Troubleshooting

| Symptom | What to do |
|---|---|
| `running scripts is disabled on this system` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then rerun. |
| `python` is not recognised | Install Python from python.org with *Add python.exe to PATH*, or use `py -3`. |
| Port 8501 is busy | `streamlit run app.py --server.port 8502` |
| `pyarrow requires NumPy 2.0 or newer` | `pip install -U "numpy>=2.0"` (as pinned in `requirements.txt`). |
| Upload rejected: too large or too many rows | Raise `NEXORA_MAX_UPLOAD_MB` / `NEXORA_MAX_ROWS` before starting, or split the file. |
| Over-/under-sampling and SMOTE show as unavailable | `pip install imbalanced-learn`. SMOTE is also withheld when features include categories or dates, by design. |
| The GPU option never appears | Intended: it is offered only when a compatible GPU **and** a GPU-capable model exist. Nexora's built-in models run on CPU. |
| Preparing a very large file feels slow | Planning and preparing scale with rows and columns; 150,000 rows by 5 columns took about 20 seconds (plan 11 s, prepare 8 s) in testing. |
| A model, SHAP, Holt-Winters or UMAP option says a package is needed | `pip install -r requirements-optional.txt` (or just the package named). |
| Database tab: "needs the driver package" | Install the driver it names (for example `pip install psycopg2-binary`). |
| Web API: "points to a private or internal address" | Intended protection. For an API on your own machine or network, start Nexora with `NEXORA_ALLOW_PRIVATE_API=1`. |
| A page shows "Something went wrong" | Go back one step. Details are in the terminal. Your original file is never changed. |
| Dataset list is empty after moving the folder | Keep `data/` with the project, or set `NEXORA_DATA_DIR` to its location. |

## Safety notes

* Model files (`.joblib`) are Python pickles. Only load model files you created yourself.
* Nexora has no sign-in. It is designed for one person on one computer, not as a shared server.
* Exported CSV files contain your values exactly. If a value starts with `=`, `+`, `-` or `@`, a spreadsheet program may treat it as a formula when opened.
* Database access is read-only (read-only sessions plus a SELECT-only query check). Use a database account with read permission only for full protection.
  Passwords are kept in memory for the session only, or read from an environment variable; they are never written to disk or logs.
* Web API credentials are only sent over https to the host you entered, never to redirect or pagination targets on other hosts.
* Preparation never invents identifiers or target labels, never deletes rows that merely look similar, and never changes your original file.
