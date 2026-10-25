"""
train_model.py

Trains the RapidRoute traffic-prediction LSTM.

Target: Average Speed (km/h), predicted one step ahead per road.
Why Average Speed and not Congestion Level / Travel Time Index / Road
Capacity Utilization: direct inspection of the dataset (see
results/dataset_report.txt) showed those three alternative columns are
heavily saturated -- 43.5%, 58.5% and 73.7% of all rows sit exactly at the
column's maximum value respectively. That kind of clipping makes them poor
regression targets (a large fraction of "signal" is just the ceiling).
Average Speed is continuous across its full range (20.0 - 89.8 km/h, 0% of
rows at the max) and translates directly into a travel-time-per-edge figure
for the routing graph (time = distance / speed), which is exactly what the
routing module needs.

Run:
    python train_model.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
from preprocessing import (  # noqa: E402
    load_raw,
    engineer_features,
    build_sequences,
    chronological_split,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "bengaluru_traffic.csv"
MODEL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 10  # see reasoning in training_metadata.json / README
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def main():
    print("[1/7] Loading raw dataset...")
    df = load_raw(str(DATA_PATH))
    print(f"    rows={len(df)}, roads={df['Road/Intersection Name'].nunique()}")

    print("[2/7] Engineering features...")
    df = engineer_features(df)

    print(f"[3/7] Building sequences (seq_len={SEQ_LEN}, per-road, chronological)...")
    X, y, meta = build_sequences(df, seq_len=SEQ_LEN)
    print(f"    total sequences={len(X)}, feature dim={X.shape[2]}")

    print("[4/7] Chronological train/val/test split (70/15/15)...")
    train_mask, val_mask, test_mask = chronological_split(meta, train_frac=0.70, val_frac=0.15)
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    print(f"    train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"    train date range: {meta.loc[train_mask,'Date'].min()} -> {meta.loc[train_mask,'Date'].max()}")
    print(f"    val   date range: {meta.loc[val_mask,'Date'].min()} -> {meta.loc[val_mask,'Date'].max()}")
    print(f"    test  date range: {meta.loc[test_mask,'Date'].min()} -> {meta.loc[test_mask,'Date'].max()}")

    print("[5/7] Fitting scalers on TRAIN data only...")
    n_features = X_train.shape[2]
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(X_train.reshape(-1, n_features))

    def scale_X(arr):
        shp = arr.shape
        return feature_scaler.transform(arr.reshape(-1, n_features)).reshape(shp)

    X_train_s = scale_X(X_train)
    X_val_s = scale_X(X_val)
    X_test_s = scale_X(X_test)

    target_scaler = MinMaxScaler()
    target_scaler.fit(y_train.reshape(-1, 1))
    y_train_s = target_scaler.transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = target_scaler.transform(y_test.reshape(-1, 1)).ravel()

    print("[6/7] Building and training LSTM...")
    model = keras.Sequential([
        layers.Input(shape=(SEQ_LEN, n_features)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(16, activation="relu"),
        layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    model.summary()

    # NOTE: saved in Keras 3's native ".keras" format rather than legacy
    # ".h5". During development, loading an H5-saved model back with this
    # Keras/TensorFlow version raised a deserialization error for the
    # built-in "mse" loss/metric aliases (a known Keras 3 / legacy-H5
    # interoperability issue). Native ".keras" format loads reliably, so it
    # is used here instead -- documented rather than silently worked around.
    ckpt_path = MODEL_DIR / "model.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_val_s, y_val_s),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    print("[7/7] Evaluating on held-out chronological test set...")
    y_pred_s = model.predict(X_test_s, verbose=0).ravel()
    y_pred = target_scaler.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    print(f"    TEST MAE={mae:.4f} km/h  RMSE={rmse:.4f} km/h  R2={r2:.4f}")

    # Baseline for comparison: naive "persistence" model (predict last observed speed)
    naive_pred = X_test[:, -1, FEATURE_COLUMNS.index("Average Speed")]
    naive_mae = float(mean_absolute_error(y_test, naive_pred))
    naive_rmse = float(np.sqrt(mean_squared_error(y_test, naive_pred)))
    print(f"    NAIVE PERSISTENCE BASELINE MAE={naive_mae:.4f} km/h RMSE={naive_rmse:.4f} km/h")

    # Save artifacts
    model.save(str(ckpt_path))
    joblib.dump({"feature_scaler": feature_scaler, "target_scaler": target_scaler}, MODEL_DIR / "scaler.pkl")

    metadata = {
        "target_column": TARGET_COLUMN,
        "target_reasoning": (
            "Average Speed chosen over Congestion Level / Travel Time Index / "
            "Road Capacity Utilization because those columns are heavily "
            "saturated at their max value (43.5%/58.5%/73.7% of rows respectively), "
            "making them poor continuous regression targets. Average Speed is "
            "continuous, unclipped, and converts directly to edge travel time "
            "for routing (time = distance / speed)."
        ),
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": SEQ_LEN,
        "sequence_length_reasoning": (
            "Dataset is DAILY and IRREGULARLY sampled per road (gaps of 1-10 days "
            "between consecutive observations for the same road; no fixed 5-minute "
            "or hourly frequency exists in this data). Sequences are therefore built "
            "from the previous N RECORDED observations for a road, not N fixed time "
            "units. seq_len=10 was chosen as a balance: large enough to give the LSTM "
            "meaningful history, small enough that even the road with the fewest "
            "observations (275) still yields hundreds of usable training sequences."
        ),
        "split": {
            "method": "chronological, global (70/15/15) by target-row date, no shuffling",
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "train_date_range": [str(meta.loc[train_mask, "Date"].min()), str(meta.loc[train_mask, "Date"].max())],
            "val_date_range": [str(meta.loc[val_mask, "Date"].min()), str(meta.loc[val_mask, "Date"].max())],
            "test_date_range": [str(meta.loc[test_mask, "Date"].min()), str(meta.loc[test_mask, "Date"].max())],
        },
        "architecture": "LSTM(64, return_sequences=True) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(16, relu) -> Dense(1)",
        "optimizer": "Adam(lr=1e-3)",
        "loss": "mse",
        "epochs_trained": len(history.history["loss"]),
        "test_metrics": {"mae_kmh": mae, "rmse_kmh": rmse, "r2": r2},
        "naive_persistence_baseline": {"mae_kmh": naive_mae, "rmse_kmh": naive_rmse},
        "random_seed": RANDOM_SEED,
    }
    with open(MODEL_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Plots
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("MSE (scaled)"); plt.title("Training vs Validation Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_val_loss.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    idx = np.arange(len(y_test))
    order = np.argsort(meta.loc[test_mask, "Date"].to_numpy())
    plt.plot(idx, y_test[order], label="actual", alpha=0.7)
    plt.plot(idx, y_pred[order], label="predicted", alpha=0.7)
    plt.xlabel("test sequence (chronological)"); plt.ylabel("Average Speed (km/h)")
    plt.title(f"Actual vs Predicted Average Speed (MAE={mae:.2f}, RMSE={rmse:.2f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "actual_vs_predicted.png", dpi=120)
    plt.close()

    print("\nSaved: model.keras, scaler.pkl, training_metadata.json")
    print(f"Saved plots to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
