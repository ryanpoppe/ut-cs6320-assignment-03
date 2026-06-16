"""Prepare the Week 3 Chicago TNP trip-duration dataset for Part A.

Reads the cleaned course CSV, makes a single random train/validation split
(seed 6320), and writes the split CSVs plus a manifest. This mirrors the
data-prep step from my Assignment 2 (`prepare_adult_data.py`): a small, separate
script so the training scripts only ever read prepared splits.

Run:
    python prepare_tnp_data.py
    python prepare_tnp_data.py --val-fraction 0.2 --output-dir artifacts/tnp_prep
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tnp_common import (
    CATEGORICAL_FEATURES,
    DEFAULT_DATASET,
    DEFAULT_PREP_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    load_raw,
    make_split,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Path to the Week 3 CSV.")
    parser.add_argument("--output-dir", default=str(DEFAULT_PREP_DIR), help="Where to write splits.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction.")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="Split seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir).resolve()
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    frame = load_raw(dataset_path)
    splits = make_split(frame, val_fraction=args.val_fraction, seed=args.seed)

    for name, split_frame in splits.items():
        split_frame.to_csv(data_dir / f"{name}.csv", index=False)

    target = frame[TARGET_COLUMN]
    manifest = {
        "dataset": str(dataset_path),
        "target": TARGET_COLUMN,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "n_total": int(len(frame)),
        "n_train": int(len(splits["train"])),
        "n_validation": int(len(splits["validation"])),
        "target_mean_seconds": float(target.mean()),
        "target_std_seconds": float(target.std()),
        "target_min_seconds": float(target.min()),
        "target_max_seconds": float(target.max()),
    }
    write_json(output_dir / "manifest.json", manifest)

    print(f"Prepared splits written to: {output_dir}")
    print(f"  train      : {manifest['n_train']:,} rows")
    print(f"  validation : {manifest['n_validation']:,} rows")
    print(
        f"  target '{TARGET_COLUMN}': mean={manifest['target_mean_seconds']:.1f}s "
        f"min={manifest['target_min_seconds']:.0f}s max={manifest['target_max_seconds']:.0f}s"
    )


if __name__ == "__main__":
    main()
