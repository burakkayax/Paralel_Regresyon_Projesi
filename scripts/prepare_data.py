from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parallel_regression.data import prepare_train_test_data, save_processed_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe preprocessing for the housing regression dataset."
    )
    parser.add_argument("--data-path", default="data/raw/housing.csv")
    parser.add_argument("--target-column", default="median_house_value")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_train_test_data(
        data_path=args.data_path,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    train_path, test_path = save_processed_splits(prepared, args.output_dir)
    print(f"Train processed CSV: {train_path}")
    print(f"Test processed CSV: {test_path}")
    print(f"Feature count: {prepared.X_train.shape[1]}")
    print("Target scale: median_house_value / 100000")


if __name__ == "__main__":
    main()
