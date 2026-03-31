"""
generate_data.py
Generates a synthetic personal expense dataset with realistic patterns
and seeded anomalies for testing the anomaly detection pipeline.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Normal spending profiles per category ──────────────────────────────────
CATEGORIES = {
    "Groceries":     {"mean": 55,  "std": 18, "freq_per_month": 8},
    "Restaurants":   {"mean": 28,  "std": 10, "freq_per_month": 10},
    "Transport":     {"mean": 18,  "std": 7,  "freq_per_month": 20},
    "Utilities":     {"mean": 110, "std": 20, "freq_per_month": 2},
    "Entertainment": {"mean": 35,  "std": 15, "freq_per_month": 6},
    "Healthcare":    {"mean": 60,  "std": 25, "freq_per_month": 2},
    "Shopping":      {"mean": 75,  "std": 30, "freq_per_month": 5},
}

# ── Anomaly types injected into the dataset ────────────────────────────────
ANOMALY_TEMPLATES = [
    {"category": "Groceries",     "amount": 420,  "description": "Unusually large grocery bill"},
    {"category": "Entertainment", "amount": 380,  "description": "Expensive night out"},
    {"category": "Shopping",      "amount": 650,  "description": "Large impulse purchase"},
    {"category": "Restaurants",   "amount": 290,  "description": "Very expensive dinner"},
    {"category": "Transport",     "amount": 310,  "description": "Unexpected travel cost"},
    {"category": "Healthcare",    "amount": 480,  "description": "Emergency medical expense"},
    {"category": "Utilities",     "amount": 390,  "description": "Abnormally high utility bill"},
    {"category": "Shopping",      "amount": 900,  "description": "Possible fraudulent charge"},
]


def _random_time(date):
    """Attach a plausible time-of-day to a date."""
    hour = int(np.clip(np.random.normal(14, 4), 6, 23))
    minute = np.random.randint(0, 60)
    return date.replace(hour=hour, minute=minute)


def generate_expenses(months: int = 6, anomaly_count: int = 8) -> pd.DataFrame:
    """
    Build a DataFrame of synthetic expense transactions.

    Parameters
    ----------
    months        : Number of months of history to generate.
    anomaly_count : Number of anomalous transactions to inject.

    Returns
    -------
    pd.DataFrame with columns:
        date, category, amount, is_anomaly, description
    """
    records = []
    start_date = datetime.today() - timedelta(days=30 * months)

    # ── Normal transactions ────────────────────────────────────────────────
    for month_offset in range(months):
        month_start = start_date + timedelta(days=30 * month_offset)
        for category, profile in CATEGORIES.items():
            n = max(1, int(np.random.normal(profile["freq_per_month"], 1)))
            for _ in range(n):
                day_offset = np.random.randint(0, 30)
                txn_date = _random_time(month_start + timedelta(days=day_offset))
                amount = max(1.0, round(np.random.normal(profile["mean"], profile["std"]), 2))
                records.append({
                    "date":        txn_date,
                    "category":    category,
                    "amount":      amount,
                    "is_anomaly":  False,
                    "description": f"Regular {category.lower()} expense",
                })

    # ── Anomalous transactions ─────────────────────────────────────────────
    templates = np.random.choice(ANOMALY_TEMPLATES, size=anomaly_count, replace=False)
    total_days = 30 * months
    anomaly_days = np.random.choice(range(total_days), size=anomaly_count, replace=False)

    for template, day_offset in zip(templates, anomaly_days):
        txn_date = _random_time(start_date + timedelta(days=int(day_offset)))
        # Add a small random jitter so no two anomalies are identical
        amount = round(template["amount"] * np.random.uniform(0.9, 1.1), 2)
        records.append({
            "date":        txn_date,
            "category":    template["category"],
            "amount":      amount,
            "is_anomaly":  True,
            "description": template["description"],
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    df = generate_expenses(months=6, anomaly_count=8)
    out_path = os.path.join(os.path.dirname(__file__), "expenses.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} transactions ({df['is_anomaly'].sum()} anomalies) → {out_path}")
    print(df.tail(10).to_string())
