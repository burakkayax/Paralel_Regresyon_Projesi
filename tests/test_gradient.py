from __future__ import annotations

import numpy as np

from parallel_regression.models import compute_gradient_serial
from parallel_regression.parallel import (
    aggregate_chunk_gradients,
    compute_gradient_chunk,
    compute_parallel_gradient,
    split_arrays,
)


def test_serial_and_chunked_gradient_match_for_uneven_chunks():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(11, 3))
    y = rng.normal(size=11)
    weights = rng.normal(size=4)

    serial_gradient = compute_gradient_serial(X, y, weights)
    chunk_results = [
        compute_gradient_chunk(X_chunk, y_chunk, weights)
        for X_chunk, y_chunk in split_arrays(X, y, num_chunks=4)
    ]
    chunked_gradient = aggregate_chunk_gradients(chunk_results)

    np.testing.assert_allclose(chunked_gradient, serial_gradient, rtol=1e-12, atol=1e-12)


def test_parallel_gradient_matches_serial_when_process_count_changes():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(17, 4))
    y = rng.normal(size=17)
    weights = rng.normal(size=5)
    serial_gradient = compute_gradient_serial(X, y, weights)

    for process_count in (1, 2, 4):
        parallel_gradient = compute_parallel_gradient(X, y, weights, process_count)
        np.testing.assert_allclose(
            parallel_gradient,
            serial_gradient,
            rtol=1e-12,
            atol=1e-12,
        )
