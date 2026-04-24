"""Benchmark orchestration for custom and scikit-learn regression models."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.model_selection import train_test_split

from parallel_regression.metrics import evaluate_model
from parallel_regression.models import (
    TrainingResult,
    predict,
    train_parallel_gradient_descent,
    train_serial_gradient_descent,
)
from parallel_regression.plots import (
    plot_efficiency,
    plot_loss_curve,
    plot_speedup,
    plot_training_time,
)


def run_benchmark_suite(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    learning_rate: float,
    alpha: float,
    processes: list[int],
    repeats: int,
    random_state: int,
    early_stopping: bool,
    tol: float,
    patience: int,
    include_sklearn: bool = True,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Run custom serial/parallel benchmarks and optional scikit-learn baselines."""

    if repeats < 1:
        raise ValueError("repeats en az 1 olmalıdır.")

    records: list[dict[str, Any]] = []
    loss_histories: dict[str, list[float]] = {}

    serial_record, serial_history = _benchmark_custom_model(
        label="Custom GD",
        implementation="serial_numpy",
        processes_value=1,
        train_fn=lambda: train_serial_gradient_descent(
            X_train,
            y_train,
            epochs=epochs,
            learning_rate=learning_rate,
            alpha=alpha,
            random_state=random_state,
            early_stopping=early_stopping,
            tol=tol,
            patience=patience,
        ),
        X_test=X_test,
        y_test=y_test,
        repeats=repeats,
    )
    records.append(serial_record)
    loss_histories["serial_numpy"] = serial_history
    serial_median_time = serial_record["median_time"]

    for process_count in processes:
        parallel_record, parallel_history = _benchmark_custom_model(
            label="Custom GD",
            implementation="multiprocessing",
            processes_value=process_count,
            train_fn=lambda process_count=process_count: train_parallel_gradient_descent(
                X_train,
                y_train,
                epochs=epochs,
                learning_rate=learning_rate,
                num_processes=process_count,
                alpha=alpha,
                random_state=random_state,
                early_stopping=early_stopping,
                tol=tol,
                patience=patience,
            ),
            X_test=X_test,
            y_test=y_test,
            repeats=repeats,
        )
        _add_parallel_ratios(parallel_record, serial_median_time, process_count)
        records.append(parallel_record)
        if process_count == max(processes):
            loss_histories[f"multiprocessing_{process_count}p"] = parallel_history

    records[0]["speedup"] = 1.0
    records[0]["efficiency"] = 1.0

    if include_sklearn:
        records.extend(
            _benchmark_sklearn_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                epochs=epochs,
                learning_rate=learning_rate,
                alpha=alpha,
                repeats=repeats,
                random_state=random_state,
            )
        )

    return pd.DataFrame(records), loss_histories


def save_benchmark_outputs(
    results: pd.DataFrame,
    loss_histories: dict[str, list[float]],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save CSV, JSON and benchmark plots."""

    output_path = Path(output_dir)
    plots_dir = output_path / "plots"
    output_path.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "benchmark_results.csv"
    json_path = output_path / "benchmark_results.json"
    results.to_csv(csv_path, index=False)
    serializable_results = results.astype(object).where(pd.notna(results), None)
    json_path.write_text(
        json.dumps(serializable_results.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    training_time_path = plots_dir / "training_time.png"
    speedup_path = plots_dir / "speedup.png"
    efficiency_path = plots_dir / "efficiency.png"
    loss_curve_path = plots_dir / "loss_curve.png"

    plot_training_time(results, training_time_path)
    plot_speedup(results, speedup_path)
    plot_efficiency(results, efficiency_path)
    plot_loss_curve(loss_histories, loss_curve_path)

    return {
        "csv": csv_path,
        "json": json_path,
        "training_time": training_time_path,
        "speedup": speedup_path,
        "efficiency": efficiency_path,
        "loss_curve": loss_curve_path,
    }


def make_synthetic_regression(
    n_samples: int,
    n_features: int,
    noise: float = 0.1,
    random_state: int = 42,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic linear regression problem for compute-load tests."""

    if n_samples < 2:
        raise ValueError("n_samples en az 2 olmalıdır.")
    if n_features < 1:
        raise ValueError("n_features en az 1 olmalıdır.")

    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    true_weights = rng.normal(size=n_features + 1)
    y = true_weights[0] + X @ true_weights[1:] + rng.normal(scale=noise, size=n_samples)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test, true_weights


def _benchmark_custom_model(
    label: str,
    implementation: str,
    processes_value: int,
    train_fn: Callable[[], TrainingResult],
    X_test: np.ndarray,
    y_test: np.ndarray,
    repeats: int,
) -> tuple[dict[str, Any], list[float]]:
    times: list[float] = []
    final_result: TrainingResult | None = None

    for _ in range(repeats):
        start = time.perf_counter()
        final_result = train_fn()
        times.append(time.perf_counter() - start)

    if final_result is None:
        raise RuntimeError("Benchmark sonucu üretilemedi.")

    predictions = predict(X_test, final_result.weights)
    record = _base_record(label, implementation, processes_value, times)
    record.update(evaluate_model(y_test, predictions))
    return record, final_result.loss_history


def _benchmark_sklearn_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    learning_rate: float,
    alpha: float,
    repeats: int,
    random_state: int,
) -> list[dict[str, Any]]:
    sklearn_specs: list[tuple[str, Callable[[], Any]]] = [
        ("sklearn_linear_regression", lambda: LinearRegression()),
        ("sklearn_ridge", lambda: Ridge(alpha=alpha, random_state=random_state)),
        (
            "sklearn_sgd_regressor",
            lambda: SGDRegressor(
                max_iter=epochs,
                eta0=learning_rate,
                learning_rate="constant",
                penalty="l2" if alpha > 0 else None,
                alpha=alpha,
                tol=None,
                random_state=random_state,
            ),
        ),
    ]

    records: list[dict[str, Any]] = []
    for implementation, factory in sklearn_specs:
        times: list[float] = []
        model = None
        for _ in range(repeats):
            model = factory()
            start = time.perf_counter()
            model.fit(X_train, y_train)
            times.append(time.perf_counter() - start)
        if model is None:
            continue
        predictions = model.predict(X_test)
        record = _base_record("sklearn", implementation, np.nan, times)
        record.update(evaluate_model(y_test, predictions))
        record["speedup"] = np.nan
        record["efficiency"] = np.nan
        records.append(record)
    return records


def _base_record(
    model: str,
    implementation: str,
    processes_value: int | float,
    times: list[float],
) -> dict[str, Any]:
    median_time = float(statistics.median(times))
    return {
        "model": model,
        "implementation": implementation,
        "processes": processes_value,
        "training_time": median_time,
        "median_time": median_time,
        "min_time": float(min(times)),
        "max_time": float(max(times)),
        "std_time": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
        "speedup": np.nan,
        "efficiency": np.nan,
    }


def _add_parallel_ratios(
    record: dict[str, Any],
    serial_median_time: float,
    processes_value: int,
) -> None:
    speedup = serial_median_time / record["median_time"]
    record["speedup"] = float(speedup)
    record["efficiency"] = float(speedup / processes_value)
