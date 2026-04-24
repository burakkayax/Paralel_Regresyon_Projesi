"""Data loading and leakage-safe preprocessing utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_SCALE = 100_000.0


@dataclass(frozen=True)
class PreparedData:
    """Preprocessed train/test arrays and the fitted preprocessing pipeline."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: list[str]
    target_column: str
    target_scale: float


def load_raw_data(data_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV dataset with a clear error when the file is missing."""

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"CSV dosyası bulunamadı: {path}. "
            "Lütfen --data-path ile geçerli bir dosya yolu verin."
        )
    return pd.read_csv(path)


def _dense_one_hot_encoder() -> OneHotEncoder:
    """Create a dense OneHotEncoder across supported scikit-learn versions."""

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical feature columns."""

    if features.empty:
        raise ValueError("Preprocessing için en az bir feature kolonu gerekli.")

    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _dense_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_train_test_data(
    data_path: str | Path,
    target_column: str = "median_house_value",
    test_size: float = 0.2,
    random_state: int = 42,
    target_scale: float = TARGET_SCALE,
) -> PreparedData:
    """Split raw data first, then fit preprocessing only on the train split."""

    df = load_raw_data(data_path)
    if target_column not in df.columns:
        raise ValueError(
            f"Target kolonu bulunamadı: {target_column}. "
            f"Mevcut kolonlar: {', '.join(df.columns)}"
        )
    if target_scale <= 0:
        raise ValueError("target_scale pozitif olmalıdır.")

    features = df.drop(columns=[target_column])
    target = df[target_column].astype(float) / target_scale

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(X_train_raw)
    X_train = np.asarray(preprocessor.fit_transform(X_train_raw), dtype=float)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=float)

    feature_names = _get_feature_names(preprocessor)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.to_numpy(dtype=float),
        y_test=y_test.to_numpy(dtype=float),
        preprocessor=preprocessor,
        feature_names=feature_names,
        target_column=target_column,
        target_scale=target_scale,
    )


def save_processed_splits(
    prepared_data: PreparedData,
    output_dir: str | Path,
    train_filename: str = "housing_train_processed.csv",
    test_filename: str = "housing_test_processed.csv",
) -> tuple[Path, Path]:
    """Save leakage-safe processed train and test splits as separate CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df = _processed_frame(
        prepared_data.X_train,
        prepared_data.y_train,
        prepared_data.feature_names,
        prepared_data.target_column,
    )
    test_df = _processed_frame(
        prepared_data.X_test,
        prepared_data.y_test,
        prepared_data.feature_names,
        prepared_data.target_column,
    )

    train_path = output_path / train_filename
    test_path = output_path / test_filename
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def _processed_frame(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    target_column: str,
) -> pd.DataFrame:
    frame = pd.DataFrame(X, columns=feature_names)
    frame[target_column] = y
    return frame


def _get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        names: Any = preprocessor.get_feature_names_out()
        return [str(name) for name in names]
    except Exception:
        return [f"x_{idx}" for idx in range(preprocessor.transformers_[0][1].n_features_in_)]
