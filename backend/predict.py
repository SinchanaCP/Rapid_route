"""
predict.py

Loads the trained LSTM + scalers exactly once, and exposes:
    predict_traffic(road_name, history_df) -> PredictionResult

`history_df` must be the most recent recorded observations for that road
(at least SEQ_LEN rows), with the same raw columns as the training CSV.
This module applies the EXACT SAME preprocessing as train_model.py by
importing from preprocessing.py, so there is no risk of train/serve skew.

Congestion categorization (Phase 9):
    RapidRoute does NOT invent arbitrary speed thresholds. Instead it derives
    LOW/MODERATE/HIGH/SEVERE cut points from the ACTUAL distribution of
    `Average Speed` in the training data (quartiles), computed once at
    training time and stored in training_metadata.json. This keeps the
    category boundaries grounded in this dataset rather than in assumptions
    borrowed from another city or dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from preprocessing import (
    engineer_features,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"


class ModelNotLoadedError(RuntimeError):
    pass


class InsufficientHistoryError(ValueError):
    pass


@dataclass
class PredictionResult:
    road_name: str
    predicted_avg_speed_kmh: float
    congestion_level: str  # LOW | MODERATE | HIGH | SEVERE
    based_on_observations: int
    source: str = "lstm_prediction"  # always explicit, never confused with live data


class TrafficPredictor:
    """Holds the loaded model/scaler/metadata as singletons."""

    def __init__(self):
        self.model = None
        self.scalers = None
        self.metadata = None
        self.seq_len = None
        self._speed_quartiles = None  # (q25, q50, q75) computed from raw train data
        self.loaded = False

    def load(self):
        model_path = MODEL_DIR / "model.keras"
        scaler_path = MODEL_DIR / "scaler.pkl"
        meta_path = MODEL_DIR / "training_metadata.json"

        missing = [p.name for p in [model_path, scaler_path, meta_path] if not p.exists()]
        if missing:
            raise ModelNotLoadedError(
                f"Missing model artifact(s): {missing}. Run backend/model/train_model.py first."
            )

        self.model = keras.models.load_model(str(model_path))
        self.scalers = joblib.load(scaler_path)
        with open(meta_path) as f:
            self.metadata = json.load(f)
        self.seq_len = self.metadata["sequence_length"]

        # Compute congestion thresholds from the ACTUAL raw dataset's Average
        # Speed distribution, not invented values.
        raw = pd.read_csv(BASE_DIR / "data" / "bengaluru_traffic.csv")
        self._speed_quartiles = tuple(
            raw["Average Speed"].quantile([0.25, 0.50, 0.75]).tolist()
        )
        self.loaded = True

    def health(self) -> dict:
        return {
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scalers is not None,
            "metadata_loaded": self.metadata is not None,
        }

    def congestion_for_speed(self, speed_kmh: float) -> str:
        # Lower speed => higher congestion. Thresholds are the 25th/50th/75th
        # percentiles of Average Speed measured across the whole raw dataset.
        q25, q50, q75 = self._speed_quartiles
        if speed_kmh >= q75:
            return "LOW"
        elif speed_kmh >= q50:
            return "MODERATE"
        elif speed_kmh >= q25:
            return "HIGH"
        else:
            return "SEVERE"

    def predict_traffic(self, road_name: str, history_df: pd.DataFrame) -> PredictionResult:
        """
        history_df: raw rows (same columns as bengaluru_traffic.csv) for a
        SINGLE road, containing AT LEAST seq_len rows, sorted or unsorted
        (this function sorts by Date). Only the most recent seq_len rows are
        used.
        """
        if not self.loaded:
            raise ModelNotLoadedError("Call .load() before predicting.")

        if history_df.empty:
            raise InsufficientHistoryError(f"No history provided for road '{road_name}'.")

        df = history_df.copy()
        if "Date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")

        if len(df) < self.seq_len:
            raise InsufficientHistoryError(
                f"Road '{road_name}' has only {len(df)} historical observations; "
                f"need at least {self.seq_len}."
            )

        df = engineer_features(df)
        recent = df.tail(self.seq_len)

        feats = recent[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        feats_scaled = self.scalers["feature_scaler"].transform(feats)
        X = feats_scaled.reshape(1, self.seq_len, len(FEATURE_COLUMNS))

        pred_scaled = self.model.predict(X, verbose=0).ravel()
        pred = self.scalers["target_scaler"].inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
        pred = float(np.clip(pred, 0, 150))  # sanity clamp; speeds can't be negative

        return PredictionResult(
            road_name=road_name,
            predicted_avg_speed_kmh=round(pred, 2),
            congestion_level=self.congestion_for_speed(pred),
            based_on_observations=len(recent),
        )


# Module-level singleton, loaded lazily by the FastAPI app on startup.
predictor = TrafficPredictor()
