from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parallel_regression.benchmark import run_benchmark_suite, save_benchmark_outputs
from parallel_regression.data import prepare_train_test_data, save_processed_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run serial, parallel and scikit-learn linear regression benchmarks."
    )
    parser.add_argument("--data-path", default="data/raw/housing.csv")
    parser.add_argument("--target-column", default="median_house_value")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--processes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Save leakage-safe processed train/test CSV files.",
    )
    parser.add_argument("--processed-output-dir", default="data/processed")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_train_test_data(
        data_path=args.data_path,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    if args.save_processed:
        train_path, test_path = save_processed_splits(
            prepared,
            args.processed_output_dir,
        )
        print(f"Processed train CSV: {train_path}")
        print(f"Processed test CSV: {test_path}")

    results, loss_histories = run_benchmark_suite(
        X_train=prepared.X_train,
        y_train=prepared.y_train,
        X_test=prepared.X_test,
        y_test=prepared.y_test,
        epochs=args.epochs,
        learning_rate=args.lr,
        alpha=args.alpha,
        processes=args.processes,
        repeats=args.repeats,
        random_state=args.random_state,
        early_stopping=args.early_stopping,
        tol=args.tol,
        patience=args.patience,
        include_sklearn=True,
    )
    paths = save_benchmark_outputs(results, loss_histories, args.output_dir)

    print(results.to_string(index=False))
    print("\nSaved outputs:")
    for name, path in paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
