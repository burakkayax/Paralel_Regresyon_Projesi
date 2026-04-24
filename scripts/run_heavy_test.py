from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parallel_regression.benchmark import (
    make_synthetic_regression,
    run_benchmark_suite,
    save_benchmark_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic compute-load benchmark for parallel regression."
    )
    parser.add_argument("--n-samples", type=int, default=1_000_000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--processes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="results/heavy")
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, X_test, y_train, y_test, _ = make_synthetic_regression(
        n_samples=args.n_samples,
        n_features=args.n_features,
        random_state=args.random_state,
        test_size=args.test_size,
    )

    results, loss_histories = run_benchmark_suite(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
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
