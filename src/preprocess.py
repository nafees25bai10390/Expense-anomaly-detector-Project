"""
src/preprocess.py
Feature engineering pipeline for the expense anomaly detector.
Converts raw transaction records into a numeric feature matrix
suitable for clustering and anomaly detection algorithms.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Features used by the ML models
FEATURE_COLS = [
    "amount",
    "category_encoded",
    "day_of_week",
    "hour_of_day",
    "amount_vs_cat_mean",   # how far this txn deviates from the category average
    "is_weekend",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to the raw transaction DataFrame.

    Parameters
    ----------
    df : DataFrame with at least columns: date, category, amount

    Returns
    -------
    DataFrame with additional feature columns appended.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Temporal features ──────────────────────────────────────────────────
    df["day_of_week"] = df["date"].dt.dayofweek          # 0 = Monday
    df["hour_of_day"] = df["date"].dt.hour
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df["month"]       = df["date"].dt.month

    # ── Category encoding ──────────────────────────────────────────────────
    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])

    # ── Amount deviation from per-category mean ────────────────────────────
    cat_means = df.groupby("category")["amount"].transform("mean")
    cat_stds  = df.groupby("category")["amount"].transform("std").clip(lower=1)
    df["amount_vs_cat_mean"] = (df["amount"] - cat_means) / cat_stds

    return df


def get_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """
    Extract and scale the feature matrix from an engineered DataFrame.

    Parameters
    ----------
    df : DataFrame that has already been passed through engineer_features()

    Returns
    -------
    X_scaled : numpy array of shape (n_samples, n_features), scaled
    scaler   : fitted StandardScaler (keep for inverse-transforming later)
    """
    X = df[FEATURE_COLS].values.astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
