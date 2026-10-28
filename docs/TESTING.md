# Test report

## Last recorded run

| Item | Value |
|---|---|
| Date | 28 October 2026 |
| Command | `python -m pytest -q` |
| Result | **753 passed, 4 warnings in 233.93s (0:03:53)** |
| Environment | Python 3.13.17, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1, Streamlit 1.66.1, Plotly 7.1.0, imbalanced-learn 0.14.2, SQLAlchemy 2.1.6, requests 2.34.2; optional: XGBoost 3.4.1, LightGBM 4.7.0, SHAP 0.54.1, statsmodels 0.15.0 (UMAP not installed); Linux |

Nothing was skipped and nothing failed. The 4 warnings are scikit-learn reporting genuine data behaviour inside tests (BIRCH finding 3 groups
when 4 were requested on three-blob data; spectral clustering's neighbour graph not fully connected). Tests for optional packages are skipped
automatically when the package is missing; in this run all were installed except UMAP, which has no test that requires it. The slowest tests drive the Streamlit UI through full training runs (about 10 to 20 seconds each).

## What the 753 tests cover

| Area | Tests |
|---|---|
| Validation core (ported from DataSentinel, unchanged behaviour) | 342 |
| Preparation rules | 57 |
| EDA and charts | 23 |
| ML: splits, leakage, imbalance, metrics, preprocessing, training, compute | 62 |
| Persistence and reports | 8 |
| Integration: preparation flow | 27 |
| Integration: end-to-end ML workflows | 8 |
| Integration: Streamlit UI (AppTest) | 13 |
| *Added in the second phase:* | |
| Outliers, feature engineering, reduction, selection | 41 |
| Model registry (every model trains through the real pipeline) | 43 |
| Cross-validation and tuning | 22 |
| Explanations (native importance, SHAP) | 11 |
| Anomaly detection and forecasting | 21 |
| Scoring with a saved model | 3 |
| Database and web API connectors | 41 |
| Integration: domain-aware cleaning for all 15 remaining profiles | 22 |
| Integration: Streamlit UI for the new features | 9 |
| **Total** | **753** |

### Requirements traced to tests

| Requirement | Where it is tested |
|---|---|
| Each cleaning rule: valid, invalid, boundary, missing, malformed | `tests/unit/cleaning/test_transformations.py` |
| upload, profile, plan, confirm, transform, validate, export | `tests/integration/test_prepare_flow.py`, `test_ui_flow.py::test_prepare_flow_...` |
| Source data unchanged; logs accurate | `test_source_is_immutable_everywhere`, `test_log_is_complete_and_consistent`, `test_retail_normalisations_match_independent_counts` |
| Findings checked against known injected problems (not against Nexora's own output) | `sample_data/synthetic/expected_results.json` used throughout `test_prepare_flow.py` |
| Disabled rules, reproducibility | `test_disabled_action_leaves_data_untouched_and_is_logged`, `test_untick_a_safe_step...`, `test_preparation_is_reproducible`, `test_experiments_are_reproducible` |
| Empty files, malformed dates, mixed types, duplicate columns, large files, unsupported encodings | `test_empty_and_header_only_files`, `test_mixed_types_and_malformed_dates...`, `test_duplicate_column_names...`, `test_large_file_completes_and_reconciles` (40,000 rows), `test_unsupported_type_and_encoding` |
| EDA charts on missing, constant, high-cardinality and mixed fields | `tests/unit/eda/test_eda.py` |
| Classification, regression, clustering workflows | `tests/unit/ml/test_training.py`, `tests/integration/test_ml_flow.py`, `test_ui_flow.py` |
| Split strategies | `tests/unit/ml/test_splitting_leakage_imbalance.py` |
| Leakage safeguards | same file, plus `test_classification_with_leakage_detected_excluded_and_score_dropping` |
| Preprocessing fitted on training data only | `test_scaler_and_imputer_statistics_come_from_training_rows_only`, `test_datetime_features_learn_origin...`, `test_text_column_is_used_only_when_enabled...` |
| Resampling only inside training data and folds | `test_resampling_sees_only_training_rows` (spy sampler records every fit size) |
| Test set never used for selection; used once | `test_selection_never_reads_test_rows` (test rows scrambled, selection identical), `TestSetAlreadyUsed` checks, `test_repeated_testing_is_surfaced` |
| Task-specific metrics, validity rules | `tests/unit/ml/test_evaluation_preprocessing.py` (hand-calculated values) |
| Missing dependency | `test_missing_dependency_is_reported_not_crashed` |
| Unsupported GPU | `test_gpu_request_without_support...`, `test_compute_options_never_offer_unavailable_gpu` |
| Training failures | `test_one_failing_model_does_not_stop_the_comparison`, `test_all_models_failing_is_a_clear_error`, `test_training_job_surfaces_errors...` |
| Export failures | `test_export_failure_is_shown_as_a_message_not_a_crash` |
| Cancellation | `test_training_job_can_be_cancelled_between_models` |
| Every page, empty states, navigation, buttons, filters, downloads, confirmations | `tests/integration/test_ui_flow.py` |
| Outlier bounds and treatments learned on training rows only | `test_outliers.py::...training_only...`, `test_prepare_outlier_decision_preview_and_override` |
| Feature generation, reduction and selection never see held-out rows | `test_features.py`, `test_reduction_selection.py` (held-out targets corrupted, selection unchanged), `test_explain.py::test_explanations_use_no_test_rows` |
| Every registered model trains, scores and saves | `test_registry.py::test_every_classifier_...`, `..._regressor_...`, `..._clustering_method_runs` (parametrised over all available models) |
| Model settings validated; incompatible models explained | `test_validate_params_...`, `test_row_limits_skip_a_model...`, `test_multinomial_nb_gets_non_negative_inputs_and_blocks_pca`, `test_unavailable_optional_package...` |
| Time-series folds never validate on the past | `test_time_series_folds_always_validate_on_later_rows_and_respect_the_gap`, `test_chronological_experiment_uses_forward_folds` |
| Tuning is bounded, reproducible and leak-free | `test_grid_search_...` (score re-computed by hand), `test_time_budget_stops_the_search...`, `test_tuning_never_reads_test_or_validation_rows` |
| Anomaly labels used only for evaluation | `test_labels_only_evaluate_and_never_become_features` (scores identical with and without the label column) |
| Forecast backtests use only the past | `test_backtest_never_uses_future_values` (everything after the first window corrupted; first window unchanged) |
| Database: read-only, catalogue-checked names, bound values, SELECT only, timeouts, redaction | `tests/unit/test_connectors.py` (real SQLite file) |
| Web API: SSRF blocking, https for credentials, no credential leaks to other hosts, retries, size and page limits, no secrets in logs | `tests/unit/test_connectors.py` (local HTTP server) |
| Domain cleaning for every profile | `tests/integration/test_domain_cleaning.py` (expected unit conversions computed in the test from the factors) |

## Checks run outside pytest

| Check | Result |
|---|---|
| `streamlit run app.py` on a clean data folder | Started; page returned HTTP 200; `/_stcore/health` returned `ok`; no errors in the log |
| Headless Chromium (Playwright) | Opened the sample daily-demand data, ran Forecasting and Anomalies through real clicks, screenshots inspected: results tables, warning that no model beat the naive baseline, flag rates per method |
| 150,000 rows x 5 columns (4.5 MB CSV) | Ingest 1.1 s, plan 11.2 s, prepare 8.3 s; reconciliation passed; all 150,000 rows kept |
| Static check (pyflakes) | No undefined names; only intentional package re-exports reported |

## Not verified

* Running on Windows itself. The PowerShell scripts and commands were written for Windows but executed only on Linux.
* Visual review of every page in a real browser. Only the Forecasting and Anomalies pages were screenshotted (desktop width, Linux Chromium).
* A GPU machine. GPU detection is tested with simulated devices; XGBoost's GPU path (`device="cuda"`) was never executed.
* PostgreSQL, MySQL and SQL Server connections: only SQLite was exercised. The read-only session settings for the other servers are untested here.
* UMAP (not installed in the test environment).
