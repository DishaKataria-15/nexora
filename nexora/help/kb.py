"""Built-in help: questions and answers about using Nexora. Works offline; nothing is sent anywhere.

Each entry: id, page it belongs to (or "any"), several ways people ask the question, the answer (markdown), and
optionally the page the "Go to" button opens. Answers describe the interface as it is; keep them in step with the UI.
"""
from __future__ import annotations

ANY = "any"
P_HOME, P_DATA, P_PREP, P_VIS, P_ML, P_AN, P_FC, P_REP, P_SET = ("Home", "Datasets", "Prepare Data", "Visualize & Dashboard", "ML Studio",
                                                                "Anomalies", "Forecasting", "Reports", "Settings")

KB: list[dict] = [
    # ---------------------------------------------------------------- getting started
    {"id": "start", "page": ANY, "goto": P_DATA,
     "q": ["How do I start?", "where do i begin", "what should I do first", "getting started", "how to use nexora", "first steps"],
     "a": "1. Open **Datasets** and upload a CSV/Excel file, or open a **Sample data** file.\n"
          "2. Under **What would you like to do next?** pick a card: usually **Prepare this data** first.\n"
          "3. Then visualise it (**Visualize & Dashboard**), build a model (**ML Studio**), find unusual rows (**Anomalies**) or forecast (**Forecasting**)."},
    {"id": "what_is", "page": ANY,
     "q": ["What is Nexora?", "what does this app do", "what can nexora do", "features", "overview of the app"],
     "a": "Nexora takes a spreadsheet from upload to an evaluated model: **load** data (files, samples, databases, web APIs), **prepare** it "
          "(a reviewed cleaning plan), **visualise** it (charts and dashboards), and **model** it (prediction, clustering, anomalies, forecasting). "
          "It runs on your computer; your original files are never changed."},
    {"id": "workflow", "page": ANY,
     "q": ["What is the order of the pages?", "which page next", "workflow", "what comes after preparing", "steps of the process", "pipeline order"],
     "a": "Datasets → Prepare Data → then any of Visualize & Dashboard, ML Studio, Anomalies, Forecasting → Reports.\n"
          "After you open a dataset or finish preparing, **What would you like to do next?** cards take you to the next page, and the "
          "recommended card is highlighted."},
    {"id": "nav", "page": ANY,
     "q": ["Where is the sidebar?", "menu disappeared", "how to navigate", "sidebar hidden", "where are the pages"],
     "a": "Pages are listed in the left **sidebar**. If it is hidden (narrow window), click **>>** at the top left to open it."},
    # ---------------------------------------------------------------- datasets
    {"id": "upload", "page": P_DATA, "goto": P_DATA,
     "q": ["How do I upload a file?", "upload csv", "import excel", "load my data", "add a dataset", "upload xlsx"],
     "a": "Go to **Datasets → Upload**, choose a **CSV, TSV or Excel (.xlsx)** file, check the preview and click **Use this dataset**. "
          "For Excel files with several sheets, pick the sheet first. Your original file is stored read-only."},
    {"id": "limits", "page": P_DATA,
     "q": ["What is the file size limit?", "file too large", "upload rejected", "maximum rows", "how big can the file be", "too many rows"],
     "a": "Default limits: **50 MB**, **300,000 rows**, **300 columns**. They are shown in **Settings → Limits**. To raise them, start Nexora with "
          "`NEXORA_MAX_UPLOAD_MB`, `NEXORA_MAX_ROWS` or `NEXORA_MAX_COLS` set. Very large files are better reduced first (a sample, or per-group summaries)."},
    {"id": "samples", "page": P_DATA, "goto": P_DATA,
     "q": ["Is there sample data?", "demo dataset", "example files", "try without my data", "sample datasets"],
     "a": "Yes: **Datasets → Sample data** has 7 invented datasets marked *SAMPLE DATA*: retail orders, finance transactions, manufacturing sensors "
          "(all messy, for Prepare Data), customer churn (classification), house prices (regression), customer segments (clustering) and daily demand "
          "(forecasting)."},
    {"id": "reopen", "page": P_DATA,
     "q": ["How do I open a dataset I used before?", "previous dataset", "saved datasets", "switch dataset", "change dataset", "open prepared version"],
     "a": "Use **Change dataset** in the dataset bar at the top of any working page, or **Datasets → My datasets**. You can pick the raw file or any "
          "prepared version. Switching resets unfinished work on that page; saved experiments stay in Reports."},
    {"id": "combine", "page": P_DATA, "goto": P_DATA,
     "q": ["Can I use two files?", "merge datasets", "join tables", "combine files", "multiple datasets", "vlookup", "add columns from another file"],
     "a": "Yes: **Datasets → Combine datasets (advanced)**. Upload each file first, pick 2 to 5 datasets and the **main table**, then for each other table "
          "choose **Join** (add its columns), **Summarise** (per-key statistics such as the average rating per movie) or **Stack** (append rows). "
          "Nexora suggests key columns; **Preview** shows matched and unmatched rows before **Combine and save** creates a new dataset."},
    {"id": "combine_dupes", "page": P_DATA,
     "q": ["join made too many rows", "too many rows after combining", "Why did my join create more rows?", "repeated keys warning", "join duplicates rows", "rows multiplied after join"],
     "a": "If the other table has several rows with the same key, each matching main row is repeated once per match. If you want one row per key, "
          "use **Summarise** instead of **Join**. Joins that would multiply the rows more than 20 times are refused."},
    {"id": "database", "page": P_DATA, "goto": P_DATA,
     "q": ["Can I connect a database?", "sql", "postgres", "mysql", "sql server", "sqlite", "read from database"],
     "a": "**Datasets → Database** connects read-only to SQLite, PostgreSQL, MySQL/MariaDB or SQL Server. Pick a table or write one `SELECT` query "
          "(values as `:name` placeholders). Every read has a row limit; reading a whole table asks you to confirm. Needs SQLAlchemy and the "
          "database's driver (the tab tells you what to install). Passwords are never saved."},
    {"id": "api", "page": P_DATA, "goto": P_DATA,
     "q": ["Can I load data from an API?", "web api", "json url", "rest api", "fetch from url"],
     "a": "**Datasets → Web API** fetches JSON or CSV from an https address, with optional token / API key / password and pagination. Internal and "
          "private network addresses are blocked unless an administrator sets `NEXORA_ALLOW_PRIVATE_API=1`."},
    # ---------------------------------------------------------------- prepare
    {"id": "prep_what", "page": P_PREP, "goto": P_PREP,
     "q": ["What does Prepare Data do?", "how does cleaning work", "clean my data", "data cleaning", "preprocessing"],
     "a": "Four steps: **Profile** (what is in the file), **Goal and domain**, **Review plan** (every proposed step with its reason, tested on a copy), "
          "and **Result** (before/after checks, step log, downloads). Safe steps run automatically after you confirm; anything ambiguous waits for "
          "your decision. Nothing is changed until you press **Confirm and prepare data**."},
    {"id": "prep_kinds", "page": P_PREP,
     "q": ["What is the difference between automatic steps and decisions?", "needs your decision", "for your review", "safe steps"],
     "a": "**Runs automatically**: meaning-preserving fixes (placeholders to missing, trimming spaces, consistent spelling, reading numbers and dates, "
          "unit conversion, removing exact copies of a record). Untick any to skip it.\n**Needs your decision**: cases with more than one sensible "
          "answer (ambiguous dates, unreadable numbers, outliers). The default is always to leave the data unchanged.\n**For your review**: problems "
          "Nexora reports but never changes."},
    {"id": "domain", "page": P_PREP,
     "q": ["Which domain should I choose?", "domain profile", "what is generic", "retail finance manufacturing profile", "custom profile"],
     "a": "Pick the domain closest to your data (retail, finance, healthcare, ...). A profile knows that domain's record keys, spellings, safe unit "
          "conversions and checks. If none fits, use **generic**, which only fixes representation problems. You can also upload your own YAML profile "
          "under *Advanced*."},
    {"id": "goal", "page": P_PREP,
     "q": ["What goal should I pick?", "analytics vs supervised", "intended use", "goal and domain"],
     "a": "**Analytics and dashboards** for charts and reports (optional filling of blanks is offered). **Supervised ML** when you will predict a "
          "column, and **Unsupervised ML** for clustering: missing values are then left for the model pipeline, which fills them from training rows "
          "only to avoid leakage. **Explore and visualize only** is for a first look."},
    {"id": "dates", "page": P_PREP,
     "q": ["Why does it ask day-first or month-first?", "ambiguous dates", "03/04/2026", "date format question", "dmy mdy"],
     "a": "A value like 03/04/2026 can be 3 April or 4 March. If the file never proves which (e.g. with 25/12/2026), Nexora will not guess. "
          "Choose the order you know is right, or leave the column as text."},
    {"id": "outliers_prep", "page": P_PREP,
     "q": ["How do I handle outliers?", "unusual values", "outlier treatment", "remove outliers", "clip outliers", "preview effect"],
     "a": "Under **Needs your decision**, the outlier item defaults to **Report only**. Pick flag, clip, winsorize, median or remove, then open "
          "**Detection settings and preview** to change the method (IQR, z-score, modified z-score, Isolation Forest, profile limits) and click "
          "**Preview effect** to see how many values or rows would change. Some domains recommend reporting only."},
    {"id": "duplicates", "page": P_PREP,
     "q": ["How are duplicates handled?", "duplicate rows", "remove duplicates", "near duplicates", "same record twice"],
     "a": "Only rows identical in **every** column are removed, and only automatically when the profile defines a record key. Rows that share a key "
          "but differ elsewhere are listed for review, never deleted. Without a key, removing identical rows is your decision."},
    {"id": "missing", "page": P_PREP,
     "q": ["How are missing values filled?", "impute", "blank cells", "fill missing", "null values", "NaN"],
     "a": "Placeholders such as N/A or null become missing. Filling with typical values (median / most common) is an **optional** step for the "
          "analytics goal and is off by default because it invents values. For ML goals, the model pipeline fills them from training rows only. "
          "Identifiers and labels are never invented."},
    {"id": "prep_result", "page": P_PREP,
     "q": ["What does reconciliation checks passed mean?", "result page", "not applied", "before after", "step log"],
     "a": "Reconciliation checks confirm the row and column counts add up, every removal is logged and the original file is unchanged. "
          "**Not applied** lists decisions you left on their default (so nothing changed there). The step log shows rows in/out and values changed for "
          "every step."},
    {"id": "prep_download", "page": P_PREP,
     "q": ["How do I download the cleaned data?", "export prepared data", "download csv", "preparation report"],
     "a": "On the **Result** step: **Prepared data (CSV)**, **Report (HTML)**, **Step log (CSV)**, **Report data (JSON)** or **Everything (ZIP)**. "
          "Later you can download them again from **Reports → Preparation reports**."},
    # ---------------------------------------------------------------- visualise
    {"id": "vis_where", "page": P_VIS, "goto": P_VIS,
     "q": ["Where are the charts?", "visualization", "graphs", "plots", "where is the dashboard", "make a chart"],
     "a": "**Visualize & Dashboard** has tabs: **Overview** (key facts), **Columns** (one column at a time), **Relationships** (correlations, scatter, "
          "box plots), **Charts** (recommended charts and a chart builder), **Dashboard** and **Dimensions** (PCA / t-SNE views)."},
    {"id": "dashboard", "page": P_VIS, "goto": P_VIS,
     "q": ["How do I build a dashboard?", "add to dashboard", "dashboard filters", "reorder charts", "export dashboard"],
     "a": "Click **Add to dashboard** under any chart, then open the **Dashboard** tab. Add filters (category, number range, dates), move charts with "
          "**Up/Down**, remove them, and **Download dashboard (HTML)**: a single interactive file that opens in any browser."},
    {"id": "colours", "page": P_VIS,
     "q": ["Can I change chart colours?", "colour palette", "color theme", "colorblind", "change color of a chart"],
     "a": "Choose a **Colour palette** at the top of the page (six options, including **Colour-blind safe** and **Greyscale**). It applies to all charts "
          "and the exported dashboard. On the Dashboard tab each chart also has its own colour swatch, with **Use palette colour** to reset."},
    {"id": "chart_image", "page": P_VIS,
     "q": ["Can I save a chart as an image?", "download chart png", "export graph picture", "save plot", "screenshot of chart"],
     "a": "Hover over a chart and click the **camera icon** in its toolbar (top right) to download it as a PNG. The whole dashboard downloads as an "
          "interactive HTML file from the **Dashboard** tab."},
    {"id": "correlation", "page": P_VIS,
     "q": ["What does correlation mean?", "pearson vs spearman", "heatmap", "correlated columns"],
     "a": "Correlation (from -1 to 1) measures how two numeric columns move together. **Pearson** looks for straight-line relationships; **Spearman** for "
          "any steadily rising or falling one. Correlation is association, not cause."},
    {"id": "dimensions", "page": P_VIS,
     "q": ["What is the Dimensions tab?", "pca", "tsne", "umap", "dimensionality reduction view"],
     "a": "It draws many columns as points in two dimensions so similar rows sit close together. **PCA/SVD** also show how much variation each "
          "dimension keeps and which columns drive it; **t-SNE/UMAP** are for looking only. Colour the points by any column."},
    {"id": "not_prepared", "page": ANY,
     "q": ["What does Not prepared mean?", "unprepared data warning", "types detected automatically", "should I prepare first"],
     "a": "The page is using the raw file with column types detected automatically: nothing was cleaned or removed. Results can be misleading on "
          "messy data, so run **Prepare Data** first when you can."},
    # ---------------------------------------------------------------- ML studio
    {"id": "ml_start", "page": P_ML, "goto": P_ML,
     "q": ["How do I train a model?", "build a model", "machine learning", "predict", "ml studio steps"],
     "a": "**ML Studio**: choose a **Goal** (predict a number, predict a category, discover groups), the **Target**, a **Split**, **Preparation** options, "
          "**Models**, then **Start training**. Results compare models on validation data; finally evaluate the chosen model **once** on the test set."},
    {"id": "task", "page": P_ML,
     "q": ["Regression or classification?", "which goal", "predict a number vs category", "what is clustering"],
     "a": "**Regression** predicts a number (price, quantity). **Classification** predicts a label (yes/no, type). **Clustering** finds groups of similar "
          "rows without a target; groups are patterns, not ground truth."},
    {"id": "leakage", "page": P_ML,
     "q": ["What is leakage?", "red warning may leak the answer", "exclude leaky column", "cancellation reason warning", "too good to be true"],
     "a": "Leakage is information a model would not have when making a real prediction (for example a *cancellation reason* that only exists after a "
          "customer left). It makes scores look perfect and fail in practice. Click **Exclude suspected leaky column(s)** unless the column is truly "
          "known before the prediction."},
    {"id": "split", "page": P_ML,
     "q": ["Which split should I use?", "train test split", "chronological split", "stratified", "group split", "validation set"],
     "a": "**Stratified** keeps class shares equal (classification). **Random** for independent rows. **Chronological** trains on earlier dates and tests "
          "on later ones (use it for predicting the future). **Group** keeps all rows of one entity (e.g. a customer) on one side. Small datasets use "
          "cross-validation instead of a separate validation set."},
    {"id": "imbalance", "page": P_ML,
     "q": ["My classes are imbalanced, what should I do?", "class imbalance", "smote", "class weights", "oversampling", "balancing"],
     "a": "In **Preparation → Class imbalance** choose class weights (usually recommended), random over/under-sampling or SMOTE (only when all features "
          "are numbers). Balancing is applied to training rows only, never to validation or test rows. Expect higher recall and lower precision; "
          "judge with F1, recall, precision and PR-AUC rather than accuracy."},
    {"id": "optional_steps", "page": P_ML,
     "q": ["What are the optional steps inside the model?", "feature engineering", "feature selection", "pca in model", "outliers in model"],
     "a": "In **Preparation → Optional steps inside the model**: calendar date parts, outlier clipping, feature engineering (log, products, ratios, "
          "bins, ...), dimensionality reduction (PCA, SVD) and feature selection (k best, mutual information, RFE, L1, ...). All are learned from "
          "training rows only and re-learned in every fold."},
    {"id": "models", "page": P_ML,
     "q": ["Which model should I choose?", "compare models", "recommended model", "model list", "xgboost", "random forest"],
     "a": "**Recommended model** trains the top suggestion; **Choose a model** trains one; **Compare models** trains several under the same split, "
          "preparation and metric. Models that cannot run on your data say why. XGBoost and LightGBM appear when installed."},
    {"id": "metric", "page": P_ML,
     "q": ["Which metric should I use?", "primary metric", "f1 vs accuracy", "multiple metrics", "also compare models on", "pr-auc"],
     "a": "The **primary metric** picks the winning model. Add up to 4 more under **Also compare models on** to see trade-offs as extra columns. For "
          "imbalanced classes prefer F1, recall, precision or PR-AUC over accuracy; for regression RMSE or MAE (errors in the target's units)."},
    {"id": "metric_defs", "page": P_ML,
     "q": ["What is accuracy?", "what is precision", "what is recall", "what is f1", "roc auc meaning", "what is rmse", "what is r2", "r squared", "metrics glossary"],
     "a": "**Accuracy**: share of all predictions that are right (misleading when one class dominates). **Precision**: of the rows predicted positive, "
          "the share that really are. **Recall**: of the real positives, the share found. **F1**: balance of precision and recall. **ROC-AUC**: chance a "
          "random positive is ranked above a random negative (0.5 = guessing). **PR-AUC**: precision/recall trade-off, useful for rare classes. "
          "**RMSE / MAE**: typical error in the target's units. **R²**: share of the target's variation explained (1 = perfect, 0 = no better than the average)."},
    {"id": "tuning", "page": P_ML,
     "q": ["How do I tune hyperparameters?", "grid search", "random search", "cross validation folds", "validation and tuning"],
     "a": "Open **Models → Validation and tuning**: choose the fold strategy (automatic, k-fold, stratified, repeated, group, time-series) and a "
          "search (grid, random, successive halving) with a trial limit and time budget. Results list every combination tried and the chosen settings. "
          "**Model settings** lets you fix values by hand."},
    {"id": "test_once", "page": P_ML,
     "q": ["Why can I evaluate on the test set only once?", "evaluate selected model", "test set button disappeared", "retest"],
     "a": "The test rows are an honest final check. Looking at them repeatedly and changing settings would make the score optimistic, so each "
          "experiment allows **one** test evaluation. To try again, start a new experiment; Nexora warns if the same test split was used before."},
    {"id": "results", "page": P_ML,
     "q": ["How do I read the results?", "confusion matrix", "baseline", "precision recall", "what do the metrics mean", "interpretation"],
     "a": "The **comparison table** shows validation (or cross-validation) scores; the **baseline** always predicts the majority class or the average, so a "
          "useful model beats it. The **confusion matrix** counts correct and wrong predictions per class. **Precision**: share of predicted positives "
          "that are right; **recall**: share of real positives found. Expand **What do these metrics mean?** on the Models step for all definitions."},
    {"id": "importance", "page": P_ML,
     "q": ["Which features matter?", "feature importance", "shap", "permutation importance", "coefficients"],
     "a": "Results show **permutation importance** (drop in score when a column is shuffled), **model-specific importance** (trees: impurity; linear "
          "models: coefficient size) and optional **SHAP** (tick it under Validation and tuning; needs the shap package). All use non-test rows. "
          "Importance shows what the model uses, not what causes the outcome."},
    {"id": "low_scores", "page": P_ML,
     "q": ["Why are my scores low?", "bad accuracy", "model performs poorly", "f1 is low", "how to improve the model"],
     "a": "Common reasons: the features carry little signal, classes are imbalanced, or a leaky column was rightly removed. Try: preparing the data, "
          "class weights, a chronological split for time data, feature engineering, comparing more models, or tuning. Change one thing per run and "
          "compare runs in **Reports**."},
    {"id": "saved_model", "page": P_REP, "goto": P_REP,
     "q": ["How do I use a trained model on new data?", "predict new rows", "score new data", "use saved model", "deploy model"],
     "a": "**Reports → Use a saved model**: pick a model you trained, upload a file (or use the open dataset) with the same input columns, and click "
          "**Score rows**. Download the predictions as CSV. Model files are pickles: only load ones you created."},
    # ---------------------------------------------------------------- anomalies and forecasting
    {"id": "anomalies", "page": P_AN, "goto": P_AN,
     "q": ["How do I find anomalies?", "unusual rows", "outlier detection", "fraud detection", "isolation forest"],
     "a": "**Anomalies**: choose methods (Isolation Forest, Local Outlier Factor, One-Class SVM, Elliptic Envelope, robust z-score) and click **Find "
          "unusual rows**. You get flag counts, a score chart and the most unusual rows with **why**. A flagged row is a candidate for review, not proof "
          "of an error."},
    {"id": "anomaly_zero", "page": P_AN,
     "q": ["no anomalies found", "found no anomalies", "flagged 0 rows", "Why were no rows flagged?", "zero anomalies", "nothing flagged", "flag more rows", "expected share"],
     "a": "In automatic mode only clearly extreme rows are flagged, so clean data can show none. Untick **Let each method decide how many rows to flag** "
          "and set an expected share (e.g. 2%) to flag the top rows. The most unusual rows are listed either way."},
    {"id": "anomaly_labels", "page": P_AN,
     "q": ["What is the column with known anomalies?", "evaluate anomaly detection", "labelled anomalies", "precision recall anomalies"],
     "a": "If a column marks known anomalies (e.g. confirmed fraud), choose it: Nexora uses it **only** to measure precision, recall and PR-AUC. It is "
          "never used to train the detector."},
    {"id": "forecast", "page": P_FC, "goto": P_FC,
     "q": ["How does forecasting work?", "time series", "forecast sales", "predict future values", "backtest"],
     "a": "Pick the **date column**, the **value to forecast** and the number of steps, then **Backtest and forecast**. Each method is trained only on data "
          "before a cut-off and scored on the next steps, several times (backtest windows). Naive baselines are always included. The best method then "
          "forecasts the future with a rough range."},
    {"id": "forecast_baseline", "page": P_FC,
     "q": ["Why did the model not beat the baseline?", "naive baseline wins", "did not beat the simple baseline", "forecast is bad"],
     "a": "Sometimes repeating the last value is hard to beat, especially for noisy daily data. Nexora says so instead of hiding it. Adding a column "
          "known in advance (e.g. a planned promotion) under **More settings** often helps."},
    {"id": "forecast_mae", "page": P_FC,
     "q": ["What does MAE mean in forecasting?", "rmse mape", "forecast error", "lower is better"],
     "a": "**MAE** is the average size of the error in the value's units (MAE 14 = off by about 14 units per step). **RMSE** punishes big misses more; "
          "**MAPE** is the error in percent. Lower is better for all of them."},
    {"id": "forecast_nodate", "page": P_FC,
     "q": ["Forecasting says no date column", "dates not recognised", "forecast disabled", "needs a date column"],
     "a": "Forecasting needs a column of real dates. Run **Prepare Data** so dates are read as dates (for ambiguous formats you choose day-first or "
          "month-first), then try again."},
    # ---------------------------------------------------------------- reports, settings, troubleshooting
    {"id": "reports", "page": P_REP, "goto": P_REP,
     "q": ["Where are my past results?", "experiment history", "compare experiments", "reports page", "download report"],
     "a": "**Reports → Experiments** lists every training run with settings, seed and results; select two or more to compare, open one to download its "
          "report, settings and model. **Preparation reports** lists every preparation run with its downloads."},
    {"id": "reproduce", "page": ANY,
     "q": ["Will I get the same results again?", "random seed", "reproducible", "results changed"],
     "a": "Splits and models use a **seed** (default 42, set in **Settings**; each experiment can set its own on the Split step). The same data, settings "
          "and seed give the same results. Tree models using several CPU cores can differ in the last decimal places."},
    {"id": "privacy", "page": ANY,
     "q": ["is my data uploaded to the cloud", "does it send data online", "cloud", "Is my data sent anywhere?", "privacy", "offline", "internet", "is it safe", "where is my data stored"],
     "a": "Nexora runs on your computer and sends nothing anywhere (this help assistant works offline too). Data is stored in the `data` folder "
          "(or `NEXORA_DATA_DIR`). Only the Database and Web API tabs contact the source you enter. **Settings** can delete all local data."},
    {"id": "optional_pkgs", "page": ANY,
     "q": ["A feature says a package is needed", "install xgboost", "shap not installed", "optional packages", "needs the optional package"],
     "a": "Some extras are optional. Install them in the activated environment, for example `pip install xgboost lightgbm shap statsmodels sqlalchemy`, "
          "then restart Nexora (Ctrl + C, then `streamlit run app.py`)."},
    {"id": "error", "page": ANY,
     "q": ["Something went wrong on this page", "error message", "app crashed", "page shows an error", "bug"],
     "a": "Go back one step and try again; the technical details are in the terminal where Nexora runs (tick **Show technical details** to see them on "
          "the page). Your original file is never changed. If it keeps happening, note what you clicked and the message."},
    {"id": "restart", "page": ANY,
     "q": ["How do I start Nexora again?", "restart the app", "run the app", "streamlit run", "stop the app"],
     "a": "In a terminal: activate the environment (`C:\\venvs\\nexora\\Scripts\\Activate.ps1` in this setup), then `streamlit run app.py`. Stop it with "
          "**Ctrl + C** in that terminal."},
    {"id": "settings", "page": P_SET, "goto": P_SET,
     "q": ["What can I change in Settings?", "default seed", "delete all data", "software versions", "compute gpu"],
     "a": "**Settings**: default random seed, the current limits, detected compute (CPU/GPU), software versions, and deleting all local data."},
]

# Questions shown as quick buttons on each page (ids from KB, in order)
SUGGESTED = {
    P_HOME: ["start", "workflow", "samples", "privacy"],
    P_DATA: ["upload", "combine", "limits", "database"],
    P_PREP: ["prep_kinds", "domain", "outliers_prep", "dates"],
    P_VIS: ["dashboard", "colours", "vis_where", "correlation"],
    P_ML: ["leakage", "metric", "imbalance", "test_once"],
    P_AN: ["anomalies", "anomaly_zero", "anomaly_labels"],
    P_FC: ["forecast", "forecast_baseline", "forecast_mae"],
    P_REP: ["saved_model", "reports", "reproduce"],
    P_SET: ["settings", "optional_pkgs", "privacy"],
}
