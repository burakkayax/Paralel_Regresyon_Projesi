"""Parallel gradient calculation helpers."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterable

import numpy as np


def add_intercept_column(X: np.ndarray) -> np.ndarray:
    """Return X with a leading intercept column of ones."""

    X_array = np.asarray(X, dtype=float)
    if X_array.ndim != 2:
        raise ValueError(f"X iki boyutlu olmalıdır; gelen shape: {X_array.shape}")
    intercept = np.ones((X_array.shape[0], 1), dtype=float)
    return np.hstack((intercept, X_array))


def compute_gradient_chunk(
    X_chunk: np.ndarray,
    y_chunk: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return the gradient sum and row count for one data chunk."""

    X_with_intercept = add_intercept_column(X_chunk)
    y_vector = np.asarray(y_chunk, dtype=float).reshape(-1)
    weight_vector = np.asarray(weights, dtype=float).reshape(-1)

    if X_with_intercept.shape[0] != y_vector.shape[0]:
        raise ValueError("X_chunk ve y_chunk aynı sayıda örnek içermelidir.")
    if X_with_intercept.shape[1] != weight_vector.shape[0]:
        raise ValueError(
            "weights uzunluğu, intercept dahil feature sayısıyla eşleşmelidir."
        )

    errors = X_with_intercept @ weight_vector - y_vector
    gradient_sum = 2.0 * X_with_intercept.T @ errors
    return gradient_sum, y_vector.shape[0]


def aggregate_chunk_gradients(
    chunk_results: Iterable[tuple[np.ndarray, int]],
) -> np.ndarray:
    """Aggregate gradient sums using chunk sizes, not a naive mean of chunks."""

    total_size = 0
    total_gradient: np.ndarray | None = None

    for gradient_sum, chunk_size in chunk_results:
        if chunk_size <= 0:
            continue
        if total_gradient is None:
            total_gradient = np.zeros_like(gradient_sum, dtype=float)
        total_gradient += gradient_sum
        total_size += chunk_size

    if total_gradient is None or total_size == 0:
        raise ValueError("Gradient hesaplamak için boş olmayan chunk gerekli.")

    return total_gradient / total_size


def split_arrays(
    X: np.ndarray,
    y: np.ndarray,
    num_chunks: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split arrays into non-empty chunks."""

    if num_chunks < 1:
        raise ValueError("num_chunks en az 1 olmalıdır.")
    indices = np.array_split(np.arange(X.shape[0]), num_chunks)
    return [(X[idx], y[idx]) for idx in indices if idx.size > 0]


def compute_parallel_gradient(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    num_processes: int,
) -> np.ndarray:
    """Compute the full gradient through multiprocessing chunk aggregation."""

    chunks = split_arrays(np.asarray(X, dtype=float), np.asarray(y, dtype=float), num_processes)
    tasks = [(X_chunk, y_chunk, weights) for X_chunk, y_chunk in chunks]

    with mp.Pool(processes=num_processes) as pool:
        chunk_results = pool.starmap(compute_gradient_chunk, tasks)

    return aggregate_chunk_gradients(chunk_results)
