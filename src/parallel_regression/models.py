"""Custom linear regression training with serial and parallel gradient descent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parallel_regression.parallel import (
    add_intercept_column,
    aggregate_chunk_gradients,
    compute_gradient_chunk,
    split_arrays,
)


@dataclass(frozen=True)
class TrainingResult:
    """Result returned by custom gradient descent trainers."""

    weights: np.ndarray
    loss_history: list[float]
    n_epochs: int


def predict(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Predict target values for X using weights with intercept at index 0."""

    X_with_intercept = add_intercept_column(X)
    weight_vector = np.asarray(weights, dtype=float).reshape(-1)
    if X_with_intercept.shape[1] != weight_vector.shape[0]:
        raise ValueError(
            "weights uzunluğu, intercept dahil feature sayısıyla eşleşmelidir."
        )
    return X_with_intercept @ weight_vector


def compute_loss(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.0,
) -> float:
    """Compute mean squared error plus optional Ridge penalty."""

    _validate_alpha(alpha)
    y_vector = _as_vector(y)
    predictions = predict(X, weights)
    if predictions.shape[0] != y_vector.shape[0]:
        raise ValueError("X ve y aynı sayıda örnek içermelidir.")
    mse = float(np.mean((predictions - y_vector) ** 2))
    penalty = float(alpha * np.sum(np.asarray(weights, dtype=float).reshape(-1)[1:] ** 2))
    return mse + penalty


def compute_gradient_serial(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.0,
) -> np.ndarray:
    """Compute the full-batch gradient in a single NumPy operation."""

    _validate_alpha(alpha)
    gradient_sum, n_samples = compute_gradient_chunk(X, y, weights)
    gradient = gradient_sum / n_samples
    return _add_ridge_gradient(gradient, weights, alpha)


def train_serial_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1000,
    learning_rate: float = 0.01,
    alpha: float = 0.0,
    random_state: int | None = None,
    early_stopping: bool = False,
    tol: float = 1e-6,
    patience: int = 20,
) -> TrainingResult:
    """Train linear regression with a real serial NumPy gradient baseline."""

    X_array, y_vector = _validate_training_inputs(X, y, epochs, learning_rate, alpha)
    weights = _initial_weights(X_array.shape[1], random_state)
    return _run_gradient_descent(
        X=X_array,
        y=y_vector,
        weights=weights,
        epochs=epochs,
        learning_rate=learning_rate,
        alpha=alpha,
        gradient_fn=lambda current_weights: compute_gradient_serial(
            X_array, y_vector, current_weights, alpha
        ),
        early_stopping=early_stopping,
        tol=tol,
        patience=patience,
    )


def train_parallel_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 1000,
    learning_rate: float = 0.01,
    num_processes: int = 2,
    alpha: float = 0.0,
    random_state: int | None = None,
    early_stopping: bool = False,
    tol: float = 1e-6,
    patience: int = 20,
) -> TrainingResult:
    """Train linear regression by computing gradient chunks in worker processes."""

    import multiprocessing as mp

    if num_processes < 1:
        raise ValueError("num_processes en az 1 olmalıdır.")

    X_array, y_vector = _validate_training_inputs(X, y, epochs, learning_rate, alpha)
    weights = _initial_weights(X_array.shape[1], random_state)
    chunks = split_arrays(X_array, y_vector, num_processes)

    with mp.Pool(processes=num_processes) as pool:

        def gradient_fn(current_weights: np.ndarray) -> np.ndarray:
            tasks = [
                (X_chunk, y_chunk, current_weights)
                for X_chunk, y_chunk in chunks
            ]
            chunk_results = pool.starmap(compute_gradient_chunk, tasks)
            gradient = aggregate_chunk_gradients(chunk_results)
            return _add_ridge_gradient(gradient, current_weights, alpha)

        return _run_gradient_descent(
            X=X_array,
            y=y_vector,
            weights=weights,
            epochs=epochs,
            learning_rate=learning_rate,
            alpha=alpha,
            gradient_fn=gradient_fn,
            early_stopping=early_stopping,
            tol=tol,
            patience=patience,
        )


def _run_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    epochs: int,
    learning_rate: float,
    alpha: float,
    gradient_fn,
    early_stopping: bool,
    tol: float,
    patience: int,
) -> TrainingResult:
    loss_history: list[float] = []
    best_loss = float("inf")
    stale_epochs = 0

    for epoch in range(epochs):
        gradient = gradient_fn(weights)
        weights = weights - learning_rate * gradient
        loss = compute_loss(X, y, weights, alpha)
        loss_history.append(loss)

        if early_stopping:
            if best_loss - loss > tol:
                best_loss = loss
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    return TrainingResult(
                        weights=weights,
                        loss_history=loss_history,
                        n_epochs=epoch + 1,
                    )

    return TrainingResult(weights=weights, loss_history=loss_history, n_epochs=epochs)


def _initial_weights(n_features: int, random_state: int | None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=0.0, scale=0.01, size=n_features + 1)


def _add_ridge_gradient(
    gradient: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if alpha == 0:
        return gradient
    ridge_gradient = np.zeros_like(gradient, dtype=float)
    ridge_gradient[1:] = 2.0 * alpha * np.asarray(weights, dtype=float).reshape(-1)[1:]
    return gradient + ridge_gradient


def _validate_training_inputs(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    if epochs < 1:
        raise ValueError("epochs en az 1 olmalıdır.")
    if learning_rate <= 0:
        raise ValueError("learning_rate pozitif olmalıdır.")
    _validate_alpha(alpha)

    X_array = np.asarray(X, dtype=float)
    y_vector = _as_vector(y)
    if X_array.ndim != 2:
        raise ValueError(f"X iki boyutlu olmalıdır; gelen shape: {X_array.shape}")
    if X_array.shape[0] != y_vector.shape[0]:
        raise ValueError("X ve y aynı sayıda örnek içermelidir.")
    if X_array.shape[0] == 0:
        raise ValueError("Eğitim için en az bir örnek gereklidir.")
    return X_array, y_vector


def _as_vector(y: np.ndarray) -> np.ndarray:
    y_array = np.asarray(y, dtype=float)
    if y_array.ndim == 2 and y_array.shape[1] == 1:
        y_array = y_array.reshape(-1)
    if y_array.ndim != 1:
        raise ValueError(f"y tek boyutlu olmalıdır; gelen shape: {y_array.shape}")
    return y_array


def _validate_alpha(alpha: float) -> None:
    if alpha < 0:
        raise ValueError("alpha negatif olamaz.")
