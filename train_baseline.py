"""Context baselines for the TNP trip-duration regression task.

Two simple reference points (both suggested in the Week 3 dataset notes):

1. **Mean predictor** - always predict the training-set mean duration. This is
   the regression analogue of the majority-class floor I used in Assignment 2:
   any real model must beat it.
2. **Ridge regression** - a linear model on the same preprocessed features. This
   tells me how much signal a linear model captures before any neural network.

Both use the identical preprocessing as the neural network (`tnp_common`), so
the comparison in the writeup is apples-to-apples.

Run:
    python train_baseline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

from tnp_common import (
    DEFAULT_PREP_DIR,
    build_preprocessor,
    default_run_dir,
    evaluate_predictions,
    load_splits,
    split_xy,
    write_json,
    write_metrics_summary,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep-dir", default=str(DEFAULT_PREP_DIR))
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge L2 strength.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prep_dir = Path(args.prep_dir).resolve()

    split_pairs = split_xy(load_splits(prep_dir))
    x_train_df, y_train = split_pairs["train"]
    x_val_df, y_val = split_pairs["validation"]

    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(x_train_df).astype(np.float32)
    x_val = preprocessor.transform(x_val_df).astype(np.float32)

    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()

    # ---- Baseline 1: mean predictor --------------------------------------- #
    mean_seconds = float(y_train_np.mean())
    mean_run = default_run_dir(prep_dir, "baseline_mean")
    mean_run.mkdir(parents=True, exist_ok=True)
    mean_metrics = {
        "train": evaluate_predictions(y_train_np, np.full_like(y_train_np, mean_seconds)),
        "validation": evaluate_predictions(y_val_np, np.full_like(y_val_np, mean_seconds)),
    }
    write_json(mean_run / "metrics.json", mean_metrics)
    write_metrics_summary(mean_run / "metrics_summary.csv", "baseline_mean", mean_metrics)
    write_run_metadata(
        mean_run / "run_metadata.json",
        prep_dir,
        {"model_type": "mean_predictor", "predicted_seconds": mean_seconds},
    )

    # ---- Baseline 2: Ridge regression ------------------------------------- #
    ridge = Ridge(alpha=args.alpha)
    ridge.fit(x_train, y_train_np)
    ridge_run = default_run_dir(prep_dir, "baseline_ridge")
    ridge_run.mkdir(parents=True, exist_ok=True)
    ridge_metrics = {
        "train": evaluate_predictions(y_train_np, ridge.predict(x_train)),
        "validation": evaluate_predictions(y_val_np, ridge.predict(x_val)),
    }
    joblib.dump({"preprocessor": preprocessor, "model": ridge}, ridge_run / "ridge.joblib")
    write_json(ridge_run / "metrics.json", ridge_metrics)
    write_metrics_summary(ridge_run / "metrics_summary.csv", "baseline_ridge", ridge_metrics)
    write_run_metadata(
        ridge_run / "run_metadata.json",
        prep_dir,
        {"model_type": "ridge_regression", "alpha": args.alpha, "input_dim": int(x_train.shape[1])},
    )

    print("Baselines (validation MAE in seconds):")
    print(f"  mean predictor : MAE={mean_metrics['validation']['mae_seconds']:.1f}  "
          f"RMSE={mean_metrics['validation']['rmse_seconds']:.1f}  "
          f"R2={mean_metrics['validation']['r2']:.3f}")
    print(f"  ridge          : MAE={ridge_metrics['validation']['mae_seconds']:.1f}  "
          f"RMSE={ridge_metrics['validation']['rmse_seconds']:.1f}  "
          f"R2={ridge_metrics['validation']['r2']:.3f}")


if __name__ == "__main__":
    main()
