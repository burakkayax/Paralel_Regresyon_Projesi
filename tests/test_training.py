from __future__ import annotations

import numpy as np

from parallel_regression.metrics import evaluate_model
from parallel_regression.models import (
    predict,
    train_parallel_gradient_descent,
    train_serial_gradient_descent,
)


def _linear_data(seed: int = 21):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 3))
    true_weights = np.array([1.2, 2.0, -1.5, 0.75])
    y = true_weights[0] + X @ true_weights[1:]
    return X, y, true_weights


def test_training_loss_decreases():
    X, y, _ = _linear_data()
    result = train_serial_gradient_descent(
        X,
        y,
        epochs=120,
        learning_rate=0.05,
        random_state=42,
    )

    assert result.loss_history[0] > result.loss_history[-1]


def test_predict_returns_one_dimensional_shape():
    X, y, _ = _linear_data()
    result = train_serial_gradient_descent(X, y, epochs=50, learning_rate=0.05)

    predictions = predict(X[:10], result.weights)

    assert predictions.shape == (10,)


def test_evaluate_model_returns_expected_metrics():
    metrics = evaluate_model(
        y_true=np.array([1.0, 2.0, 3.0]),
        y_pred=np.array([1.0, 2.5, 2.5]),
    )

    assert set(metrics) == {"MSE", "RMSE", "MAE", "R2"}
    assert metrics["MSE"] >= 0
    assert metrics["RMSE"] >= 0
    assert metrics["MAE"] >= 0


def test_model_converges_close_to_true_coefficients_on_synthetic_data():
    X, y, true_weights = _linear_data()
    result = train_serial_gradient_descent(
        X,
        y,
        epochs=800,
        learning_rate=0.04,
        random_state=0,
    )

    np.testing.assert_allclose(result.weights, true_weights, atol=0.08)


def test_parallel_training_matches_serial_training():
    X, y, _ = _linear_data()
    serial = train_serial_gradient_descent(
        X,
        y,
        epochs=80,
        learning_rate=0.04,
        random_state=123,
    )
    parallel = train_parallel_gradient_descent(
        X,
        y,
        epochs=80,
        learning_rate=0.04,
        num_processes=2,
        random_state=123,
    )

    np.testing.assert_allclose(parallel.weights, serial.weights, rtol=1e-10, atol=1e-10)
