# Sample data (synthetic)

Every file here is **invented** by `make_samples.py` (fixed seed). None of it describes a real person, customer, bank or machine.
Regenerate with `python sample_data/make_samples.py`.

| File | Rows | Use | Problems deliberately included |
|---|---|---|---|
| `retail_orders_messy.csv` | 228 | Preparation (Retail profile) | Placeholders, stray spaces, case variants, currency symbols, day-first and ambiguous dates, a non-numeric quantity, missing order IDs, total mismatches, negative prices, 5 exact repeats, 3 near-duplicates |
| `finance_transactions_messy.csv` | 154 | Preparation (Finance profile) | Accounting negatives `(12.50)`, thousand separators, mixed date layouts, case variants, missing transaction IDs, balance mismatches, 4 exact repeats |
| `manufacturing_sensors_messy.csv` | 180 | Preparation (Manufacturing profile) | Fahrenheit readings among Celsius, missing readings, a vibration spike, a time gap, case variants, 4 exact repeats |
| `customer_churn.csv` | 1,200 | Classification | Imbalanced target (about 1 in 5 leave) and a deliberately leaky `cancellation_reason` column |
| `house_prices.csv` | 900 | Regression | A few missing ages |
| `customer_segments.csv` | 600 | Clustering | Four planted groups |
| `daily_demand.csv` | 420 | Time-ordered regression | Seasonality, weekends and promotions; use a chronological split |

`expected_results.json` records exactly which rows carry which injected problem, so the tests compare Nexora's findings with known truth.
