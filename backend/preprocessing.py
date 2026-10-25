"""
preprocessing.py

Shared preprocessing / feature-engineering logic for RapidRoute.

IMPORTANT: This module is imported by BOTH train_model.py and predict.py so that
the exact same transformations are applied at training time and at inference
time. This avoids a common bug class (train/serve skew) where the scaler or
feature order used in production silently differs from training.

Dataset facts this module encodes (verified by direct inspection of
backend/data/bengaluru_traffic.csv on 2024 data pull, see results/dataset_report.txt):
    - 8,936 rows, 16 columns, 0 missing values, 0 duplicate rows.
    - Granularity: ONE ROW PER (Date, Road/Intersection Name). This is a
      DAILY, per-road dataset -- NOT a high-frequency (e.g. 5-minute) feed.
    - Sampling per road is IRREGULAR: gaps between consecutive observations
      for the same road range from 1 to 10 days (most commonly 1-4 days).
      There is no fixed sampling frequency to rely on.
    - 16 distinct roads/intersections across 8 Bengaluru areas.
    - Date range: 2022-01-01 to 2024-08-09.

Because sampling is irregular, sequences are built as "the previous N
observations recorded for this road", NOT "the previous N minutes/hours".
This is documented rather than assumed -- see training_metadata.json for the
exact sequence length chosen and the reasoning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns used as LSTM input features, in a FIXED order. This exact order
# must be preserved between training and inference.
FEATURE_COLUMNS = [
    "Traffic Volume",
    "Average Speed",
    "Congestion Level",
    "Road Capacity Utilization",
    "Incident Reports",
    "Public Transport Usage",
    "Traffic Signal Compliance",
    "Parking Usage",
    "Pedestrian and Cyclist Count",
    "is_weekend",
    "is_rush_hour_proxy",
    "Roadwork_flag",
    "Weather_Clear",
    "Weather_Overcast",
    "Weather_Fog",
    "Weather_Rain",
    "Weather_Windy",
]

# The model predicts this column, one step ahead, for a given road.
TARGET_COLUMN = "Average Speed"

WEATHER_CATEGORIES = ["Clear", "Overcast", "Fog", "Rain", "Windy"]


def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal + encoded features. Only creates features that are
    actually meaningful for THIS dataset's granularity.

    NOTE: The dataset has no time-of-day component (one row per road per
    day), so an hour-of-day / rush-hour feature cannot be computed from a
    timestamp. We do NOT fabricate one. Instead we keep only the temporal
    features that are genuinely derivable from `Date`:
        - day_of_week
        - is_weekend
    `is_rush_hour_proxy` is intentionally NOT a real rush-hour indicator; it
    is fixed at 0 for every row and kept only as an explicit placeholder so
    the feature contract stays stable if hourly data is added later. This is
    documented here rather than silently invented.
    """
    df = df.sort_values(["Road/Intersection Name", "Date"]).reset_index(drop=True)

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["Date"].dt.month

    # Explicit placeholder - see docstring. Not derived from real hourly data.
    df["is_rush_hour_proxy"] = 0

    df["Roadwork_flag"] = (df["Roadwork and Construction Activity"] == "Yes").astype(int)

    for cat in WEATHER_CATEGORIES:
        df[f"Weather_{cat}"] = (df["Weather Conditions"] == cat).astype(int)

    return df


def build_sequences(
    df: pd.DataFrame,
    seq_len: int,
    feature_cols: list[str] = FEATURE_COLUMNS,
    target_col: str = TARGET_COLUMN,
):
    """Build (X, y, meta) sequences PER ROAD, in chronological order.

    For each road, sequence i is:
        X[i] = observations [t-seq_len ... t-1] (in recorded order, gaps allowed)
        y[i] = target value at observation t (the NEXT recorded observation)

    Sequences never cross road boundaries (a sequence is always built from a
    single road's own history).

    Returns:
        X: np.ndarray shape (num_sequences, seq_len, num_features)
        y: np.ndarray shape (num_sequences,)
        meta: pd.DataFrame with columns [Road/Intersection Name, Date] for
              the TARGET row of each sequence (useful for chronological
              splitting and for tracing predictions back to real rows).
    """
    X_list, y_list, meta_rows = [], [], []

    for road, g in df.groupby("Road/Intersection Name", sort=False):
        g = g.sort_values("Date").reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g[target_col].to_numpy(dtype=np.float32)
        dates = g["Date"].to_numpy()

        if len(g) <= seq_len:
            continue  # not enough history for this road yet

        for t in range(seq_len, len(g)):
            X_list.append(feats[t - seq_len : t])
            y_list.append(targets[t])
            meta_rows.append({"Road/Intersection Name": road, "Date": dates[t]})

    X = np.stack(X_list) if X_list else np.empty((0, seq_len, len(feature_cols)))
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def chronological_split(meta: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    """Split sequence indices chronologically using the GLOBAL date of each
    sequence's target row (not a per-road split), so that no split contains
    information from the future relative to the split(s) before it.

    Returns boolean masks (train_mask, val_mask, test_mask) aligned to `meta`.
    """
    order = np.argsort(meta["Date"].to_numpy())
    n = len(order)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask
