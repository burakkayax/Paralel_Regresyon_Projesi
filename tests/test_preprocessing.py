from __future__ import annotations

import numpy as np
import pandas as pd

from parallel_regression.data import build_preprocessor, prepare_train_test_data


def test_preprocessor_fits_imputer_on_train_only_and_handles_unknown_category():
    X_train = pd.DataFrame(
        {
            "numeric_feature": [1.0, np.nan, 3.0],
            "category": ["A", "A", "B"],
        }
    )
    X_test = pd.DataFrame(
        {
            "numeric_feature": [1000.0, np.nan],
            "category": ["C", "A"],
        }
    )

    preprocessor = build_preprocessor(X_train)
    transformed_train = preprocessor.fit_transform(X_train)
    transformed_test = preprocessor.transform(X_test)

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    imputer = numeric_pipeline.named_steps["imputer"]

    assert imputer.statistics_[0] == 2.0
    assert transformed_train.shape[1] == transformed_test.shape[1]
    assert transformed_test.shape == (2, 3)


def test_prepare_train_test_data_scales_target_and_keeps_shapes(tmp_path):
    csv_path = tmp_path / "housing.csv"
    pd.DataFrame(
        {
            "longitude": [-1.0, -2.0, -3.0, -4.0, -5.0],
            "latitude": [1.0, 2.0, 3.0, 4.0, 5.0],
            "total_bedrooms": [1.0, np.nan, 3.0, 4.0, 5.0],
            "ocean_proximity": ["NEAR BAY", "INLAND", "NEAR BAY", "ISLAND", "INLAND"],
            "median_house_value": [100000, 200000, 300000, 400000, 500000],
        }
    ).to_csv(csv_path, index=False)

    prepared = prepare_train_test_data(
        csv_path,
        test_size=0.4,
        random_state=0,
    )

    assert prepared.X_train.shape[0] == prepared.y_train.shape[0]
    assert prepared.X_test.shape[0] == prepared.y_test.shape[0]
    assert prepared.y_train.max() <= 5.0
    assert prepared.target_scale == 100_000.0
