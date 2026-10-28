# Nexora: remaining-features completion report

Date: 28 October 2026. Base: Nexora as first delivered (540 tests passing). All work was done in place on that repository, in the phase order of the
implementation prompt, with one commit per phase.

## A. Summary

Every phase in the prompt (2 to 10) is implemented, wired into the interface and covered by tests. The full suite now has **753 tests, all passing**
(540 original + 213 new). No existing feature was removed. Three original test expectations were changed, each because a feature changed on purpose
(listed in section D). Things that could not be verified here are listed in section F.

## B. What was implemented

| Phase | Delivered | Main files |
|---|---|---|
| 2 Outliers | IQR, z-score, modified z-score, Isolation Forest and profile-limit detection. Treatments: report, flag, clip, winsorize, median, log, remove. Preparation: a DECISION step that defaults to *report only* and has a preview. Model pipeline: an `OutlierTreatment` step whose bounds are learned on training rows | `preprocessing/outliers.py`, `cleaning/planner.py`, `cleaning/transformations.py` |
| 2 Feature engineering | Log, square, polynomial, product, ratio, difference, bins, percentile rank, missing indicators, frequency encoding, text length and word count, extra date parts and cyclical encodings. Capped at 200 generated columns, with a preview and a quality check | `preprocessing/features.py`, `preprocessing/pipeline.py` |
| 2 Dimensionality reduction | PCA, SVD and feature agglomeration inside the pipeline. PCA, SVD, t-SNE and optional UMAP as an exploration view | `preprocessing/reduction.py` |
| 3 Models | 34 registered models. **Classification:** logistic, tree, random forest, HistGB, extra trees, SVC, KNN, Gaussian/Multinomial/Bernoulli NB, AdaBoost, plus optional XGBoost and LightGBM. **Regression:** linear, ridge, lasso, elastic net, tree, random forest, HistGB, extra trees, SVR, KNN, AdaBoost, plus optional XGBoost and LightGBM. **Clustering:** k-means, DBSCAN, hierarchical, Gaussian mixture, BIRCH, mean shift, spectral, OPTICS. Each model has metadata, validated settings, a bounded search space and compatibility checks | `ml/model_registry.py`, `ml/training.py` |
| 4 CV and tuning | Fold types: k-fold, stratified, repeated, group, and time-series with a gap; *automatic* matches the split. Grid, random and successive-halving search with trial caps, a time budget and cancellation. Every trial is recorded | `ml/cv.py`, `ml/tuning.py` |
| 5 Selection and explainability | Variance, k-best, percentile, mutual information, RFE, RFECV, model-based and L1 selection inside the pipeline. Impurity and coefficient importance; optional SHAP for tree and linear models | `preprocessing/selection.py`, `ml/explain.py` |
| 6 Anomalies and forecasting | **Anomalies:** five detectors, the reason each top row was flagged, agreement between methods, and evaluation against known labels used only for scoring. **Forecasting:** rolling-origin backtests against naive and seasonal-naive baselines; lag models (ridge, forest, boosting); optional Holt-Winters; inputs known in advance; rough prediction ranges | `ml/anomaly.py`, `ml/forecasting.py` |
| 7 Connectors | Read-only SQLite, PostgreSQL, MySQL and SQL Server connections. Web API ingestion (JSON and CSV) with four authentication types and four pagination types | `connectors/database.py`, `connectors/web_api.py`, `workflow.py` |
| 8 Domain cleaning | All 15 remaining profiles now have `settings.cleaning`: record key, required spellings, safe exact unit conversions (energy, environment, logistics, real estate, transportation, agriculture), and an outlier policy (report-only for healthcare, insurance, cybersecurity, telecom and aviation). The section is validated when the profile loads | `config/domains/*.yaml`, `validation/profiles.py` |
| 9 UI | **Prepare:** outlier decision with method, threshold and preview. **ML Studio:** optional pipeline steps, model settings, compatibility messages, validation and tuning, plus new result sections (tuning trials, fold scores, learned steps, importance, SHAP). **New pages:** Anomalies, Forecasting. **Explore:** Dimensions tab. **Datasets:** Database and Web API tabs. **Reports:** *Use a saved model* | `ui/pages/*.py` |
| 10 Verification | Full test run, launch smoke test, headless-browser screenshots, documentation and this report | `docs/` |

## C. Design decisions worth knowing

* **Pipeline order:** coerce, generate features, then column transforms (impute, outliers, power, scale), then reduction, selection, sampler and model. Every learned step is fitted inside `Pipeline.fit`, so cross-validation and tuning re-learn it in every fold.
* **Tuning:** the search loop is Nexora's own rather than scikit-learn's `*SearchCV`. That allows cancellation between fits, a time budget, Nexora's own metrics (including the positive-class F1), and a record of every trial.
* **Optimistic scores:** when models are compared by cross-validation after tuning, the reused folds make the scores slightly optimistic. The interface says so, and the one-time test result remains the unbiased number.
* **SVC probabilities** come from `CalibratedClassifierCV(ensemble=False)`, because scikit-learn 1.9 deprecates `SVC(probability=True)`.
* **XGBoost** is wrapped so that text class labels work.
* **Isolation Forest *automatic* mode** flags scores above Q3 + 3×IQR of the score distribution. scikit-learn's fixed cut-off flagged 39% of the daily-demand sample when it was viewed in the browser.
* **One-Class SVM** uses `nu ≥ 0.05`. Smaller values let the anomalies become support vectors and hide themselves (found by a failing test).
* **Default compare set:** *Compare models* starts with the four core models per task rather than all 13. That keeps runs short, and the other models can be added.

## D. Tests and results

* Command: `python -m pytest -q`. Result: **753 passed, 4 warnings, 233.93 s**. The warnings come from scikit-learn about real data behaviour in clustering tests.
* Breakdown by area and the mapping of requirements to tests: `docs/TESTING.md`.
* **Leakage tests** corrupt held-out rows and require training-only results to stay identical. They cover:
  * outlier bounds
  * generated features
  * reduction
  * selection
  * tuning trials
  * explanations
  * forecast backtests
  * anomaly labels (scores must be identical with and without the label column)
* **Original tests changed on purpose (3):**
  * `test_ui_flow::test_explore_dashboard_...`: the Explore tab list gained *Dimensions*.
  * `test_training::test_selection_never_reads_test_rows`: compares scores with `rel=1e-9` instead of exact equality. Multi-threaded tree ensembles differ in the last floating-point digit between runs; any tampering changes scores by orders of magnitude.
  * `test_training::test_experiment_to_dict_is_json_safe`: checks for a `"pipeline":` key instead of the word "pipeline", because the new `pipeline_report` field contains that word.
  * Kept unchanged on purpose: the UI test that counts three "Choose" decisions still passes, because the new outlier decision has its own label ("Outlier treatment"). *Compare models* still starts with the same four core models, so that test is also unchanged.
* **Outside pytest:**
  * `streamlit run app.py` returned HTTP 200 and `/_stcore/health` returned ok.
  * Headless Chromium drove Forecasting and Anomalies with real clicks, and the screenshots were inspected.

## E. Security (connectors)

| Requirement | How it is met | Test |
|---|---|---|
| Read-only by default | SQLite opened with `mode=ro`. PostgreSQL uses a read-only transaction default, MySQL `TRANSACTION READ ONLY`, SQL Server `ApplicationIntent=ReadOnly`. There is no write mode | `test_sqlite_is_opened_read_only_even_if_validation_were_bypassed` |
| No credentials in logs or errors | Passwords are left out of `repr` and redacted from errors. Request logs omit headers and query strings, and urllib3's debug logging is silenced | `test_passwords_never_appear...`, `test_secrets_are_not_logged` |
| No plain-text password storage | Passwords are kept in memory for the session or read from an environment variable, and never saved with a dataset | `test_password_from_environment_variable` |
| Parameterised values, validated identifiers | Values are bound as `:name` parameters. Table and column names are checked against the catalogue and quoted by SQLAlchemy | `test_parameterised_queries...`, `test_identifiers_are_checked...` |
| No modification statements | Exactly one SELECT/WITH statement is allowed, after comments and quoted text are stripped. Write, side-effect and sleep keywords are refused | `test_anything_but_one_select_is_refused` (13 cases) |
| Confirmation before full-table reads | The row count is shown and a confirm flag is required | `test_sqlite_catalogue_preview_and_limits` |
| Timeouts and limits | Connection and statement timeouts; SQLite queries are interrupted; row limits report truncation | `test_slow_queries_are_stopped_by_the_timeout` |
| URL scheme and SSRF | http(s) only. Loopback, private, link-local and metadata, carrier-grade NAT, multicast and reserved addresses are refused, including after DNS resolution and on every redirect | `test_unsafe_urls_are_refused` (9 cases), `test_dns_names_pointing_inside...`, `test_redirect_to_metadata_address_is_blocked...` |
| No credentials to other domains | Credentials go only to the original host over https, never to redirect or pagination hosts | `test_pagination_never_follows_credentials...`, `test_credentials_require_https...` |
| Timeouts, bounded retries with backoff | Retries only on 429, 5xx and connection errors, at most 3, with exponential backoff and `Retry-After` | `test_retries_with_backoff...` |
| Size and pagination limits | 20 MB per response, 100 MB in total, 50 pages | `test_size_limit_and_errors`, `test_cursor_link_csv_and_page_limit` |

## F. Not verified / known limitations

* **PostgreSQL, MySQL and SQL Server** were never connected to. Only SQLite was exercised, so their read-only session options and timeouts are untested.
* **DNS rebinding** (the name resolving differently between the check and the request) is not fully prevented.
* **Domain profiles:** the rules and policies for the 15 non-flagship profiles were written without a domain expert. They are tested on synthetic data only.
* **GPU paths** (XGBoost `device="cuda"`) and **UMAP** were not run.
* **Windows** was not tested; everything ran on Linux.
* **Visual review:** only Forecasting and Anomalies were viewed in a browser.
* **Forecasting scope:** one series at a time; no ARIMA, no per-entity panels; prediction ranges are empirical, not calibrated.
* **Forecast selection:** the selected forecast is always one of the requested methods. When a baseline wins, the page warns but still forecasts with the selected model. On the daily-demand sample the naive baseline won, and the page says so.
* **Text features** remain TF-IDF only.
* **Clustering in ML Studio** has no Preparation step, so the optional pipeline steps are not offered there. They are available for classification and regression.

## G. How to run

```powershell
pip install -r requirements-dev.txt          # core
pip install -r requirements-optional.txt     # optional: XGBoost, LightGBM, SHAP, statsmodels, UMAP, DB drivers
streamlit run app.py
python -m pytest -q
```

Environment settings added in this phase:

* `NEXORA_DB_TIMEOUT_S` (default 30)
* `NEXORA_DB_ROW_LIMIT` (default 100000)
* `NEXORA_ALLOW_PRIVATE_API` (default 0; set to 1 only for APIs on your own machine or network)

## H. Remaining work (suggested next steps)

1. Run the connector tests against real PostgreSQL, MySQL and SQL Server instances, for example in Docker.
2. Have a domain expert review the 15 newly configured profiles.
3. Do a visual pass over every page on Windows in a real browser.
4. Add the optional pipeline steps to the clustering path in ML Studio.
5. Offer to forecast with the winning baseline when no model beats it.
6. Pin the resolved IP for web API requests to close the DNS-rebinding gap.
