"""Model evaluation metrics."""

from __future__ import annotations

import numpy as np


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MSE, RMSE, MAE and R2 for regression predictions."""

    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape[0] != predicted.shape[0]:
        raise ValueError("y_true ve y_pred aynı uzunlukta olmalıdır.")

    errors = predicted - actual
    mse = float(np.mean(errors**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(errors)))

    total_sum_squares = float(np.sum((actual - np.mean(actual)) ** 2))
    residual_sum_squares = float(np.sum(errors**2))
    r2 = 1.0 - residual_sum_squares / total_sum_squares if total_sum_squares else 0.0

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": float(r2),
    }
