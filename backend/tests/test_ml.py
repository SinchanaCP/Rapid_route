"""Tests for preprocessing + prediction (Phase 39: ML tests)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing import load_raw, engineer_features, build_sequences, chronological_split, FEATURE_COLUMNS
from predict import TrafficPredictor, InsufficientHistoryError, ModelNotLoadedError

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bengaluru_traffic.csv"


@pytest.fixture(scope="module")
def raw_df():
    return load_raw(str(DATA_PATH))


def test_load_raw_has_no_missing_values(raw_df):
    assert raw_df.isnull().sum().sum() == 0


def test_engineer_features_adds_expected_columns(raw_df):
    df = engineer_features(raw_df.copy())
    for col in ["is_weekend", "Roadwork_flag", "Weather_Clear"]:
        assert col in df.columns
    assert set(df["is_weekend"].unique()) <= {0, 1}


def test_build_sequences_shapes(raw_df):
    df = engineer_features(raw_df.copy())
    X, y, meta = build_sequences(df, seq_len=10)
    assert X.shape[0] == y.shape[0] == len(meta)
    assert X.shape[1] == 10
    assert X.shape[2] == len(FEATURE_COLUMNS)


def test_build_sequences_never_crosses_road_boundary(raw_df):
    df = engineer_features(raw_df.copy())
    # Every sequence's target road must have >= seq_len+1 rows in the source.
    counts = df["Road/Intersection Name"].value_counts()
    X, y, meta = build_sequences(df, seq_len=10)
    for road, n in counts.items():
        expected = max(0, n - 10)
        actual = (meta["Road/Intersection Name"] == road).sum()
        assert actual == expected


def test_chronological_split_no_leakage(raw_df):
    df = engineer_features(raw_df.copy())
    X, y, meta = build_sequences(df, seq_len=10)
    train_mask, val_mask, test_mask = chronological_split(meta)
    max_train_date = meta.loc[train_mask, "Date"].max()
    min_val_date = meta.loc[val_mask, "Date"].min()
    min_test_date = meta.loc[test_mask, "Date"].min()
    assert max_train_date <= min_val_date
    assert meta.loc[val_mask, "Date"].max() <= min_test_date
    # masks partition the data with no overlap
    assert not (train_mask & val_mask).any()
    assert not (val_mask & test_mask).any()


def test_predictor_raises_before_load():
    p = TrafficPredictor()
    with pytest.raises(ModelNotLoadedError):
        p.predict_traffic("100 Feet Road", pd.DataFrame())


def test_predictor_insufficient_history(raw_df):
    p = TrafficPredictor()
    p.load()
    short_hist = raw_df[raw_df["Road/Intersection Name"] == "100 Feet Road"].head(3)
    with pytest.raises(InsufficientHistoryError):
        p.predict_traffic("100 Feet Road", short_hist)


def test_predictor_returns_sane_prediction(raw_df):
    p = TrafficPredictor()
    p.load()
    hist = raw_df[raw_df["Road/Intersection Name"] == "100 Feet Road"]
    result = p.predict_traffic("100 Feet Road", hist)
    assert 0 <= result.predicted_avg_speed_kmh <= 150
    assert result.congestion_level in {"LOW", "MODERATE", "HIGH", "SEVERE"}
    assert result.source == "lstm_prediction"


def test_congestion_levels_follow_predicted_speed_quartiles():
    p = TrafficPredictor()
    p._speed_quartiles = (30.0, 40.0, 50.0)

    assert p.congestion_for_speed(25.0) == "SEVERE"
    assert p.congestion_for_speed(35.0) == "HIGH"
    assert p.congestion_for_speed(45.0) == "MODERATE"
    assert p.congestion_for_speed(55.0) == "LOW"
