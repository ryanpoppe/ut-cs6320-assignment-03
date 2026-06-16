"""Shared helpers for CS 6320 Assignment 3 Part A.

This module reuses the local-script organization from my Assignment 2
(`adult_common.py`): one place for data loading, the train/validation split,
preprocessing, regression metrics, and the JSON/CSV/metadata writers. The
training-loop script and the baseline script both import from here so the
preprocessing and metric code are identical across models (apples-to-apples).

Assignment 3 differs from Assignment 2 in two ways that live in this file:

1. The task is **regression** (predict ``target_trip_seconds``), so the metrics
   are MAE / RMSE / R2 instead of the classification metrics from Assignment 2.
2. The Week 3 dataset ships as a single CSV, so this module performs the
   train/validation split here rather than reading pre-split files.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Task definition (from the Week 3 data_dictionary.md / dataset_audit.md)
# --------------------------------------------------------------------------- #
TARGET_COLUMN = "target_trip_seconds"

# Per the data dictionary: time-of-day fields are used as numeric features and
# pickup/dropoff areas + shared-trip flag are treated as categorical.
NUMERIC_FEATURES = [
    "trip_miles",
    "start_hour",
    "start_day_of_week",
    "start_month",
    "trips_pooled",
]
CATEGORICAL_FEATURES = [
    "pickup_community_area",
    "dropoff_community_area",
    "shared_trip_authorized",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SPLIT_NAMES = ("train", "validation")
RANDOM_STATE = 6320  # same seed convention as my Assignment 2 work

CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = (
    CODE_DIR
    / "datasets"
    / "student_visible"
    / "week03_tnp_trips"
    / "week03_tnp_trip_duration.csv"
)
DEFAULT_PREP_DIR = CODE_DIR / "artifacts" / "tnp_prep"


# --------------------------------------------------------------------------- #
# Data loading + split
# --------------------------------------------------------------------------- #
def load_raw(dataset_path: Path) -> pd.DataFrame:
    """Read the cleaned Week 3 CSV and keep only the modeling columns."""
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")
    frame = pd.read_csv(dataset_path)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in frame.columns]
    if missing:
        raise SystemExit(f"Dataset is missing expected columns: {missing}")
    # shared_trip_authorized arrives as the strings "true"/"false"; normalize to
    # a clean string category so the one-hot encoder is deterministic.
    frame["shared_trip_authorized"] = frame["shared_trip_authorized"].astype(str).str.lower()
    return frame[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()


def make_split(
    frame: pd.DataFrame, val_fraction: float, seed: int
) -> dict[str, pd.DataFrame]:
    """Single random train/validation split.

    A random split is acceptable for this assignment (see dataset_audit.md);
    validation MAE is a training-diagnostic, not a deployment claim.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    n_val = int(round(len(frame) * val_fraction))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    return {
        "train": frame.iloc[train_idx].reset_index(drop=True),
        "validation": frame.iloc[val_idx].reset_index(drop=True),
    }


def load_splits(prep_dir: Path) -> dict[str, pd.DataFrame]:
    data_dir = prep_dir / "data"
    return {name: pd.read_csv(data_dir / f"{name}.csv") for name in SPLIT_NAMES}


def split_xy(splits: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    return {
        name: (frame[FEATURE_COLUMNS].copy(), frame[TARGET_COLUMN].astype(float).copy())
        for name, frame in splits.items()
    }


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Standardize numeric features, one-hot encode categoricals.

    Standardizing the numeric inputs matters for a neural network: it keeps the
    gradients on a similar scale across features so training is stable.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# --------------------------------------------------------------------------- #
# Metrics (regression)
# --------------------------------------------------------------------------- #
def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / R2 in the original units of the target (seconds)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "mae_seconds": float(mean_absolute_error(y_true, y_pred)),
        "rmse_seconds": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(y_true.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Output writers (same habits as my Assignment 2 run artifacts)
# --------------------------------------------------------------------------- #
def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics_summary(
    path: Path, model_name: str, all_metrics: dict[str, dict[str, float]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "split", "mae_seconds", "rmse_seconds", "r2", "n"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in SPLIT_NAMES:
            if split_name not in all_metrics:
                continue
            row = {"model": model_name, "split": split_name}
            row.update({key: all_metrics[split_name].get(key) for key in fieldnames[2:]})
            writer.writerow(row)


def write_run_metadata(path: Path, prep_dir: Path, extras: dict[str, Any]) -> None:
    payload = {
        "argv": sys.argv,
        "prep_dir": str(prep_dir),
        "target": TARGET_COLUMN,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    payload.update(extras)
    write_json(path, payload)


def default_run_dir(prep_dir: Path, run_name: str) -> Path:
    return prep_dir / "runs" / run_name
