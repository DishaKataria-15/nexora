# Nexora user guide

Nexora is built for two kinds of people: analysts who want clean data and charts, and machine-learning users who want a trustworthy model.
The default path asks you for a few decisions and does the technical work for you.

## Getting help inside the app

Open **Help: ask about Nexora** at the bottom of the sidebar. Type a question in your own words (for example *how do I combine two files?*
or *what is leakage?*) or click one of the common questions for the page you are on. Answers come from Nexora's built-in help, so it works
offline and nothing is sent anywhere. If a question is not about using Nexora, it says it has no answer and offers the closest topics.

## 1. Load a dataset (Datasets)

* **Upload**: CSV, TSV or Excel (.xlsx). For Excel files with several worksheets, pick the sheet. Nexora checks size, rows, columns, encoding and
  delimiter and shows a preview. Problems are explained in plain language (for example an empty file or an unsupported type).
* **Sample data**: invented files marked *SAMPLE DATA*. They are safe to experiment with.
* **Database**: connect to SQLite (a file), PostgreSQL, MySQL / MariaDB or SQL Server. Access is read-only. Pick a table (or view) or write one
  `SELECT` query; put values as `:name` placeholders and fill them in below the query, so they are never pasted into the SQL. Every read has a row limit;
  reading a whole table asks you to confirm after showing its row count. The password is typed for this session only or read from an environment
  variable; it is never saved. Use a database account that can only read.
* **Web API**: fetch JSON or CSV from an https address. Choose authentication (bearer token, API key in a header or the URL, or user and password),
  where the records are in the JSON (for example `data.items`), and pagination (page number, offset, cursor or next link) with a page limit.
  Internal and private network addresses are blocked unless an administrator allows them.
* **My datasets**: reopen anything you used before, optionally with a prepared version.
* **Combine datasets (advanced)**: put two to five saved datasets into one table. Pick the main table, then for each other table choose
  *Join* (add its columns to matching rows), *Summarise* (add per-key statistics such as the average rating and number of ratings per movie)
  or *Stack* (append rows with the same columns). Nexora suggests key columns, compares keys after trimming spaces (optionally ignoring case),
  and **Preview** reports matched rows, rows with no match (with examples), unused keys and repeated keys. A join that would multiply the
  main table's rows more than 20 times, or exceed the row limit, is refused with an explanation. The result is saved as a new dataset.
* After a dataset opens, **What would you like to do next?** offers Prepare, Visualize, Build a model, Find unusual rows and Forecast.
  Every working page also has a dataset bar with a short summary and **Change dataset**.

Your original file is saved read-only. Nothing in Nexora edits it.

## 2. Prepare the data (Prepare Data)

Four steps, shown at the top of the page.

1. **Profile** shows rows, columns, how much is missing, repeated rows and likely identifiers.
2. **Goal and domain**. Choose what you want to do (analytics, supervised ML, unsupervised ML, or explore only) and a domain profile.
   Every built-in profile knows its record key, required spellings, safe unit conversions and outlier policy. Retail, Finance and Manufacturing are
   additionally tested end to end; the other profiles' rules still need review by a domain expert.
   *Generic* only fixes representation problems. *Review column mapping* lets you correct which column plays which role. If the profile has cross-checks
   against another table (for example "every department code exists"), you can upload that table here. *Advanced* lets you upload your own YAML profile.
3. **Review plan**. Nexora tests every step on a copy of your data and shows the real counts.
   * **Runs automatically**: safe, meaning-preserving steps such as treating `N/A` as missing, trimming spaces, merging spelling variants,
     reading numbers and dates, converting units where the profile defines it, and removing rows that are identical copies of a record with the same key.
     Untick any step to skip it. *View details* shows affected columns, counts, the reason, expected impact, alternatives and examples with row numbers.
   * **Needs your decision**: nothing changes unless you choose. Examples: day-first or month-first dates; values that are not numbers;
     rows with no required identifier; identical rows when there is no record key; **unusual values (outliers)**. Optional extras (filling blanks with a typical value) are off by default.
   * **Unusual values**: the default is *Report only*. You can instead add a flag column, clip to the detection bounds, winsorize, replace with the median,
     or remove the rows. *Detection settings and preview* lets you change the method (IQR, z-score, modified z-score, Isolation Forest, or the profile's
     own limits), the threshold and the columns, and **Preview effect** shows how many values or rows would change before anything happens.
     Some profiles (for example healthcare, insurance, cybersecurity) recommend reporting only, because extreme values there are often the important ones.
   * **For your review**: problems Nexora reports but never changes, such as totals that do not match quantity times price.

   Press **Confirm and prepare data** once.
4. **Result**. Before/after measures, reconciliation checks, every step with rows in/out and values changed, rows removed, what still needs attention, and downloads
   (prepared CSV, HTML report, step log CSV, JSON, or everything as a ZIP).

What Nexora will not do: invent identifiers or labels, delete near-duplicates, change outliers unless you choose to, or guess an ambiguous date.
For ML goals, missing values are not filled here; they are filled inside the model using training data only.

## 3. Visualize & Dashboard

Uses the prepared data (or, if you skipped preparation, a lightly typed copy, clearly labelled as unprepared).

* **Overview**: what stands out (each statement is computed from your data), column table, missing values, unusual values, and a downloadable summary.
* **Columns**: one column at a time with the right chart for its type.
* **Relationships**: correlation heatmap (Pearson or Spearman), strongest pairs, scatter and box plots. Correlation is association, not cause.
* **Charts**: recommended charts with the reason for each, and a builder.
* **Dimensions**: pick columns and draw them as points in two dimensions with PCA, SVD, t-SNE or (if installed) UMAP. Colour points by any column.
  PCA shows how much of the variation each dimension holds and which columns drive it. t-SNE and UMAP are for looking only.
* **Colour palette** (top of the page): six palettes, including a colour-blind-safe one, for every chart and the exported dashboard; each dashboard chart also has its own colour swatch.
* **Dashboard**: *Add to dashboard* under any chart. Filter by category, number range or dates; reorder or remove charts; download the dashboard as a standalone HTML file.

## 4. Build a model (ML Studio)

Steps are shown at the top and adapt to your goal.

1. **Goal**: predict a number (regression), predict a category (classification), discover groups (clustering), or explore only.
2. **Target and features** (not for clustering). Nexora suggests likely targets; you confirm. Rows with no target value are left out only if you confirm.
   Identifier-like columns are excluded automatically with the reason shown. **Leakage warnings** appear when a column gives the answer away
   (for example a "cancellation reason" that is only filled in for customers who left); one click excludes them.
3. **Split**: a recommended split by dataset size, or choose 70/15/15, 80/20 or custom. Strategies: random, stratified, group-aware, chronological.
   The seed is saved so you can reproduce the split.
4. **Preparation**: imputation, scaling, encoding, skew reduction, date features and optional text features, all learned from training rows only.
   **Optional steps inside the model** (each learned from training rows only, and re-learned inside every cross-validation fold):
   calendar parts and cyclical date encodings; outlier clipping / winsorizing / median replacement; feature engineering (log, square, polynomial terms,
   products, ratios, differences, bins, percentile ranks, missing-value indicators, frequency encoding, text length) with a preview of the generated
   columns; dimensionality reduction (PCA, SVD, feature agglomeration); feature selection (variance, k best, percentile, mutual information, RFE, RFECV,
   model-based, L1). For clustering only target-free selection is offered.
   For classification you also see class balance and options (class weights, random over/under-sampling, SMOTE) with reasons when an option is unavailable.
   Balancing is applied to training data only, and inside each fold during cross-validation.
5. **Models**: *Compare models* (the default: starts with the top 3 suggestions; same split, same preparation, same metric), *Recommended model* (trains only the top suggestion), or *Choose a model*.
   Models that cannot run on your data are explained (for example a support vector machine above its row limit, or Multinomial Naive Bayes with PCA).
   Optional models (XGBoost, LightGBM) appear when installed. *Model settings* lets you change each model's main settings within safe ranges.
   *Validation and tuning*: fold strategy (automatic, k-fold, stratified, repeated, group, time-series with a gap), folds and repeats; tuning by grid,
   random search or successive halving, with a trial limit and a time budget per model; optional SHAP explanations.
   Pick the primary metric; expand "What do these metrics mean?" for definitions. Compute: Automatic or CPU (GPU only if a compatible GPU and model exist).
6. **Train**: runs in the background with real progress. **Cancel** takes effect between models and folds.
7. **Results**: the comparison table shows scores from validation data or cross-validation. Then press **Evaluate selected model on the test set** to see the final,
   honest result. This can be done **once per experiment**. If earlier experiments already used the same test split, you are warned.

Reading the results

| Task | Metrics shown | Notes |
|---|---|---|
| Classification | accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, per-class table | ROC/PR-AUC shown only when mathematically valid. Precision/recall/F1 use the positive class (binary) or macro average. Accuracy can mislead when classes are imbalanced. |
| Regression | MAE, RMSE, R², median absolute error, MAPE | MAPE is shown only when no true value is zero. Errors are in the units of the target. |
| Clustering | silhouette, Davies-Bouldin, Calinski-Harabasz | Internal measures of group shape, not accuracy. Clusters are not ground-truth labels. |

With tuning, the results show every combination tried, its score and spread, and the chosen settings. A box plot shows the score in each fold.
If you used reduction, selection or outlier treatment, the results show what those steps learned. Tree models show impurity importance and linear
models coefficient sizes; SHAP (optional) shows each prepared feature's average contribution, computed on non-test rows.

Every result is compared with a simple baseline (always predict the most common class, or the average) so you can see whether the model adds anything.

## 5. Anomalies

Choose one or more methods (Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope, robust z-score), let each decide how many rows to
flag or set the expected share, and leave out columns such as identifiers. If you have a column marking known anomalies, choose it: it is used **only**
to measure precision, recall and PR-AUC, never to train the detector. The page lists the most unusual rows with the columns that make them unusual,
shows how much the methods agree, and exports every row's score. A flagged row is a candidate for review, not proof of an error.

## 6. Forecasting

Choose the date column, the value to forecast, the number of steps and the number of backtest windows. Nexora orders the rows, combines rows that share a
date, fills gaps in the calendar (or stops, if you prefer), and backtests each method: it trains only on data before each cut-off and scores the next
steps. Naive and seasonal-naive baselines are always included; if no model beats them, the page says so. The chosen method is refitted on all history to
forecast the future, with a rough range from the backtest errors. *Columns known in advance* (for example a planned promotion) can be added; only use
columns whose future values you will really know.

## 7. Reports and Settings

* **Reports**: all experiments (settings, seed, split, environment, results), side-by-side comparison, and preparation runs. Reports can be downloaded again.
  **Use a saved model** scores an uploaded file or the open dataset with any model you trained here, and downloads the predictions.
* **Settings**: default seed, limits, detected compute, software versions, and deletion of all local data.

## Downloads

| From | File | Contents |
|---|---|---|
| Prepare Data | `prepared_data.csv` | The cleaned data exactly as validated |
| | `preparation_report.html` / `.json` | Before/after, reconciliation, steps, unresolved issues |
| | `transformation_log.csv` | Every step: operation, columns, reason, parameters, rows in/out, values changed, skipped and removed rows |
| Visualize & Dashboard | `dashboard.html`, `data_summary.html` | Interactive, self-contained |
| ML Studio | `experiment_report.html`, `experiment_config.json` | Settings, seed, split sizes, comparison, final results, caveats |
| | `test_predictions.csv`, `cluster_assignments.csv` | Row-level output |
| | `model_pipeline.joblib` | Full fitted pipeline (preparation and model). Only load files you created. |
| Anomalies | `anomaly_scores.csv`, `anomaly_summary.json` | Every row's score and flag; methods, settings and evaluation |
| Forecasting | `forecast.csv`, `backtest_predictions.csv`, `forecast_summary.json` | Forecast with range; every backtest prediction; settings and metrics |
| Reports | `predictions.csv` | New rows scored with a saved model |

## Glossary

* **Leakage**: information in the features that would not be available when making a real prediction, which makes scores look better than they are.
* **Validation set / cross-validation**: data used to compare models. The **test set** is kept aside for one final check.
* **Baseline**: a trivial model used as a yardstick.
* **Reconciliation**: automatic checks that row and column counts add up and that the original is unchanged.
