"""Reusable tools for serial and parallel linear regression experiments."""

from parallel_regression.data import (
    TARGET_SCALE,
    PreparedData,
    build_preprocessor,
    load_raw_data,
    prepare_train_test_data,
    save_processed_splits,
)
from parallel_regression.metrics import evaluate_model
from parallel_regression.models import (
    TrainingResult,
    compute_gradient_serial,
    compute_loss,
    predict,
    train_parallel_gradient_descent,
    train_serial_gradient_descent,
)
from parallel_regression.parallel import compute_gradient_chunk, compute_parallel_gradient

__all__ = [
    "TARGET_SCALE",
    "PreparedData",
    "TrainingResult",
    "build_preprocessor",
    "compute_gradient_chunk",
    "compute_gradient_serial",
    "compute_loss",
    "compute_parallel_gradient",
    "evaluate_model",
    "load_raw_data",
    "predict",
    "prepare_train_test_data",
    "save_processed_splits",
    "train_parallel_gradient_descent",
    "train_serial_gradient_descent",
]
