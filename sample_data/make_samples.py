"""Generate Nexora's synthetic sample datasets (deterministic).

ALL DATA IS INVENTED. Nothing here comes from a real person, customer, bank or machine.
Run:  python sample_data/make_samples.py
The injected problems are recorded in sample_data/synthetic/expected_results.json so tests can check the
application's findings against known truth rather than against its own output.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "synthetic"
SEED = 20261026


def write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)


# ----------------------------------------------------------------------------------------------
# Messy domain datasets (used for the preparation workflow)
# ----------------------------------------------------------------------------------------------

def make_retail(rng: np.random.RandomState, truth: dict) -> pd.DataFrame:
    n = 220
    skus = {"SKU-A": 24.5, "SKU-B": 15.0, "SKU-C": 62.0, "SKU-D": 9.9, "SKU-E": 38.0, "SKU-F": 71.5, "SKU-G": 12.25, "SKU-H": 45.0}
    names = list(skus)
    rows = []
    start = pd.Timestamp("2026-08-01")
    for i in range(n):
        sku = names[rng.randint(len(names))]
        qty = int(rng.randint(1, 7))
        price = skus[sku]
        disc = float(rng.choice([0, 0, 0, 0, 5.0, 2.5]))
        is_ret = rng.rand() < 0.08
        od = start + pd.Timedelta(days=int(rng.randint(0, 60)))
        rows.append({
            "order_id": f"O{2001 + i}", "order_date": od.strftime("%Y-%m-%d"), "customer_id": f"C{rng.randint(1, 60):03d}",
            "sku": sku, "quantity": str(qty), "unit_price": f"{price:.2f}",
            "discount_amount": "" if disc == 0 else f"{disc:.2f}", "order_total": f"{qty * price - disc:.2f}",
            "order_type": "RETURN" if is_ret else "SALE", "return_reason": "damaged" if is_ret else "",
            "currency": "USD", "stock_on_hand": str(int(rng.randint(5, 120))),
            "ship_date": (od + pd.Timedelta(days=int(rng.randint(2, 6)))).strftime("%Y-%m-%d"),
        })
    df = pd.DataFrame(rows)
    idx = list(rng.permutation(n))

    def take(k):
        out, idx[:] = idx[:k], idx[k:]
        return out

    r = take(12)
    df.loc[r[:6], "order_type"] = "sale"
    df.loc[r[6:], "order_type"] = "Sale"
    df.loc[take(6), "currency"] = "usd"
    r = take(10)
    df.loc[r, "unit_price"] = ["$" + v for v in df.loc[r, "unit_price"]]
    df.loc[take(9), "customer_id"] = "N/A"
    df.loc[take(4), "stock_on_hand"] = "null"
    r = take(7)
    df.loc[r, "sku"] = df.loc[r, "sku"] + " "
    # Only the *layout* changes below; the dates themselves stay consistent (ship date never before order date).
    r = [i for i in idx if pd.Timestamp(df.loc[i, "order_date"]).day > 12][:15]      # day > 12 proves day-first from the data
    idx[:] = [i for i in idx if i not in r]
    df.loc[r, "order_date"] = [pd.Timestamp(v).strftime("%d/%m/%Y") for v in df.loc[r, "order_date"]]
    r = [i for i in idx if (pd.Timestamp(df.loc[i, "ship_date"]).day <= 12 and pd.Timestamp(df.loc[i, "ship_date"]).day != pd.Timestamp(df.loc[i, "ship_date"]).month)][:6]
    idx[:] = [i for i in idx if i not in r]                                          # both parts <= 12: genuinely ambiguous
    df.loc[r, "ship_date"] = [pd.Timestamp(v).strftime("%d/%m/%Y") for v in df.loc[r, "ship_date"]]
    truth["retail"] = {"ambiguous_ship_date_rows": sorted(int(i) + 2 for i in r)}
    r = take(2)
    df.loc[r, "quantity"] = "two"
    truth["retail"]["unreadable_quantity_rows"] = sorted(int(i) + 2 for i in r)
    r = take(3)
    df.loc[r, "order_id"] = ""
    truth["retail"]["missing_order_id_rows"] = sorted(int(i) + 2 for i in r)
    r = take(4)   # arithmetic mismatch -> review item, never auto-fixed
    df.loc[r, "order_total"] = ["999.99", "1.00", "250.00", "3.50"]
    truth["retail"]["total_mismatch_rows"] = sorted(int(i) + 2 for i in r)
    r = take(2)
    df.loc[r, "unit_price"] = ["-5.00", "-12.50"]
    truth["retail"]["negative_price_rows"] = sorted(int(i) + 2 for i in r)
    # exact duplicates (appended copies) and near duplicates (same key, different quantity)
    dup_src = take(5)
    near_src = take(3)
    near = df.loc[near_src].copy()
    near["quantity"] = "9"
    near["order_total"] = "999.00"
    dups = df.loc[dup_src].copy()
    out = pd.concat([df, dups, near], ignore_index=True)  # a repeated load appends rows; positions stay traceable
    truth["retail"]["exact_duplicate_copies"] = 5
    truth["retail"]["near_duplicate_rows"] = 3
    truth["retail"]["rows"] = len(out)
    return out


def make_finance(rng: np.random.RandomState, truth: dict) -> pd.DataFrame:
    accounts = [f"A{100 + i}" for i in range(10)]
    bal = {a: float(rng.randint(500, 5000)) for a in accounts}
    rows = []
    date = pd.Timestamp("2026-08-01")
    n = 150
    for i in range(n):
        acct = accounts[rng.randint(len(accounts))]
        date += pd.Timedelta(hours=int(rng.randint(1, 20)))
        credit = rng.rand() < 0.4
        amt = float(np.round(rng.uniform(200, 3200), 2)) if credit else -float(np.round(rng.uniform(5, 400), 2))
        before = bal[acct]
        after = round(before + amt, 2)
        bal[acct] = after
        rows.append({"transaction_id": f"T{1001 + i}", "account_id": acct, "posting_date": date.strftime("%Y-%m-%d"),
                     "amount": f"{amt:.2f}", "currency": "USD", "txn_type": "CREDIT" if credit else "DEBIT",
                     "balance_before": f"{before:.2f}", "balance_after": f"{after:.2f}", "original_transaction_id": ""})
    df = pd.DataFrame(rows)
    idx = list(rng.permutation(n))

    def take(k):
        out, idx[:] = idx[:k], idx[k:]
        return out

    big = df.index[df["amount"].astype(float) >= 1000]
    r = list(big[:8])
    df.loc[r, "amount"] = [f"{float(v):,.2f}" for v in df.loc[r, "amount"]]
    idx = [i for i in idx if i not in r]
    neg = [i for i in df.index[df["amount"].astype(str).str.startswith("-")] if i in idx][:5]
    df.loc[neg, "amount"] = [f"({abs(float(v)):.2f})" for v in df.loc[neg, "amount"]]
    idx = [i for i in idx if i not in neg]
    df.loc[take(7), "currency"] = "usd"
    df.loc[take(6), "txn_type"] = "debit"
    r = take(9)
    df.loc[r, "posting_date"] = [pd.Timestamp(v).strftime("%Y/%m/%d") for v in df.loc[r, "posting_date"]]
    df.loc[take(10), "original_transaction_id"] = "N/A"
    r = take(5)
    df.loc[r, "account_id"] = df.loc[r, "account_id"] + " "
    r = take(2)
    df.loc[r, "transaction_id"] = ""
    truth["finance"] = {"missing_transaction_id_rows": sorted(int(i) + 2 for i in r)}
    r = take(3)
    df.loc[r, "balance_after"] = ["1.00", "2.00", "3.00"]
    truth["finance"]["balance_mismatch_rows"] = sorted(int(i) + 2 for i in r)
    dup_src = take(4)
    out = pd.concat([df, df.loc[dup_src]], ignore_index=True)
    truth["finance"]["exact_duplicate_copies"] = 4
    truth["finance"]["rows"] = len(out)
    return out


def make_manufacturing(rng: np.random.RandomState, truth: dict) -> pd.DataFrame:
    rows = []
    t0 = pd.Timestamp("2026-08-03T06:00:00")
    for dev in ("M1", "M2", "M3"):
        for k in range(60):
            temp_c = 72 + 4 * np.sin(k / 8) + rng.normal(0, 0.6)
            rows.append({"device_id": dev, "timestamp": (t0 + pd.Timedelta(seconds=60 * k)).strftime("%Y-%m-%dT%H:%M:%S"),
                         "temperature": f"{temp_c:.1f}", "temperature_unit": "C", "vibration": f"{abs(rng.normal(2.0, 0.2)):.2f}",
                         "machine_state": "RUNNING", "output_count": str(int(rng.randint(8, 13))),
                         "last_calibration_date": "2026-07-01"})
    df = pd.DataFrame(rows)
    f_rows = df.index[(df["device_id"] == "M2")][5:17]          # Fahrenheit readings: value and unit both recorded in F
    df.loc[f_rows, "temperature"] = [f"{float(v) * 9 / 5 + 32:.1f}" for v in df.loc[f_rows, "temperature"]]
    df.loc[f_rows, "temperature_unit"] = "F"
    truth["manufacturing"] = {"fahrenheit_rows": sorted(int(i) + 2 for i in f_rows)}
    lower = df.index[(df["device_id"] == "M1")][20:26]
    df.loc[lower, "machine_state"] = "running"
    truth["manufacturing"]["state_case_variants"] = 6
    miss_t = df.index[(df["device_id"] == "M3")][10:15]
    df.loc[miss_t, "temperature"] = ""
    df.loc[df.index[(df["device_id"] == "M1")][40:43], "vibration"] = "null"
    truth["manufacturing"]["missing_temperature_rows"] = sorted(int(i) + 2 for i in miss_t)
    spike = df.index[(df["device_id"] == "M1")][33]
    df.loc[spike, "vibration"] = "25.00"
    truth["manufacturing"]["vibration_spike_row"] = int(spike) + 2
    gap = df.index[(df["device_id"] == "M3")][30:34]
    dup_src = list(df.index[(df["device_id"] == "M2")][40:44])
    df = pd.concat([df.drop(index=gap), df.loc[dup_src]], ignore_index=True)
    truth["manufacturing"]["exact_duplicate_copies"] = 4
    truth["manufacturing"]["rows"] = len(df)
    return df


# ----------------------------------------------------------------------------------------------
# ML demonstration datasets (genuine signal, so metrics are meaningful)
# ----------------------------------------------------------------------------------------------

def make_churn(rng: np.random.RandomState) -> pd.DataFrame:
    n = 1200
    tenure = rng.randint(1, 73, n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tickets = rng.poisson(1.2, n)
    monthly = np.round(rng.normal(65, 20, n).clip(18, 120), 2)
    usage = np.round(rng.gamma(2.0, 40.0, n), 1)
    age = rng.randint(19, 80, n)
    region = rng.choice(["North", "South", "East", "West", "Central"], n)
    payment = rng.choice(["Card", "Bank transfer", "Wallet", "Cash"], n, p=[0.4, 0.3, 0.2, 0.1])
    partner = rng.choice(["yes", "no"], n)
    z = -1.9 - 0.045 * tenure + 1.1 * (contract == "Month-to-month") - 0.8 * (contract == "Two year") + 0.38 * tickets + 0.012 * (monthly - 65)
    p = 1 / (1 + np.exp(-z))
    churn = rng.rand(n) < p
    reason = np.where(churn, rng.choice(["price", "service", "moved", "competitor"], n), "")
    signup = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.randint(0, 1800, n), unit="D")
    df = pd.DataFrame({
        "customer_id": [f"CU{10000 + i}" for i in range(n)], "signup_date": signup.strftime("%Y-%m-%d"), "age": age,
        "region": region, "contract": contract, "payment_method": payment, "has_partner": partner, "tenure_months": tenure.astype(object),
        "monthly_charges": monthly.astype(object), "usage_gb": usage, "support_tickets": tickets,
        "cancellation_reason": reason, "churned": np.where(churn, "yes", "no")})
    for col, frac in (("tenure_months", 0.03), ("monthly_charges", 0.04), ("region", 0.02)):
        m = rng.rand(n) < frac
        df[col] = df[col].astype(object)
        df.loc[m, col] = ""
    return df


def make_houses(rng: np.random.RandomState) -> pd.DataFrame:
    n = 900
    area = rng.normal(1650, 520, n).clip(450, 4200).round(0)
    beds = np.clip((area / 650 + rng.normal(0, 0.7, n)).round(), 1, 6).astype(int)
    baths = np.clip((beds * 0.6 + rng.normal(0, 0.5, n)).round(), 1, 5).astype(int)
    age = rng.randint(0, 60, n)
    hood = rng.choice(["Riverside", "Old Town", "Hillcrest", "Lakeside", "Industrial"], n, p=[0.25, 0.2, 0.2, 0.2, 0.15])
    hood_eff = pd.Series(hood).map({"Riverside": 40, "Old Town": 15, "Hillcrest": 60, "Lakeside": 75, "Industrial": -25}).values
    garage = rng.choice(["yes", "no"], n, p=[0.6, 0.4])
    dist = np.round(rng.gamma(3.0, 3.0, n), 1)
    price = 55 + 0.145 * area + 9 * baths + 3 * beds - 0.9 * age - 2.2 * dist + hood_eff + 14 * (garage == "yes") + rng.normal(0, 14, n)
    listed = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.randint(0, 600, n), unit="D")
    df = pd.DataFrame({"house_id": [f"H{5000 + i}" for i in range(n)], "listed_date": listed.strftime("%Y-%m-%d"), "area_sqft": area.astype(int),
                       "bedrooms": beds, "bathrooms": baths, "age_years": age, "neighborhood": hood, "has_garage": garage,
                       "distance_to_center_km": dist, "price_thousands_usd": price.round(1)})
    m = rng.rand(n) < 0.03
    df["age_years"] = df["age_years"].astype(object)
    df.loc[m, "age_years"] = ""
    return df


def make_segments(rng: np.random.RandomState) -> pd.DataFrame:
    centers = [(28, 82, 31, 8), (62, 25, 52, 2), (95, 78, 40, 6), (45, 50, 66, 4)]
    parts = []
    for cx in centers:
        k = 150
        parts.append(pd.DataFrame({"annual_income_k": rng.normal(cx[0], 6, k), "spending_score": rng.normal(cx[1], 8, k).clip(1, 100),
                                   "age": rng.normal(cx[2], 5, k).clip(18, 80), "visits_per_month": rng.normal(cx[3], 1.2, k).clip(0, 20)}))
    df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=3).reset_index(drop=True).round(1)
    df.insert(0, "customer_id", [f"S{7000 + i}" for i in range(len(df))])
    return df


def make_demand(rng: np.random.RandomState) -> pd.DataFrame:
    n = 420
    d = pd.date_range("2025-01-01", periods=n, freq="D")
    temp = 15 + 10 * np.sin(2 * np.pi * (np.arange(n) - 80) / 365) + rng.normal(0, 2, n)
    promo = (rng.rand(n) < 0.15).astype(int)
    dow = d.dayofweek.values
    demand = 200 + 0.04 * np.arange(n) + 18 * (dow >= 5) + 2.4 * temp + 45 * promo + rng.normal(0, 9, n)
    return pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "temperature_c": temp.round(1), "promo_active": promo, "units_sold": demand.round(0).astype(int)})


def main() -> None:
    rng = np.random.RandomState(SEED)
    truth: dict = {}
    write(make_retail(rng, truth), "retail_orders_messy.csv")
    write(make_finance(rng, truth), "finance_transactions_messy.csv")
    write(make_manufacturing(rng, truth), "manufacturing_sensors_messy.csv")
    write(make_churn(rng), "customer_churn.csv")
    write(make_houses(rng), "house_prices.csv")
    write(make_segments(rng), "customer_segments.csv")
    write(make_demand(rng), "daily_demand.csv")
    (OUT / "expected_results.json").write_text(json.dumps(truth, indent=2, sort_keys=True))
    print("Wrote sample data to", OUT)


if __name__ == "__main__":
    main()
