# src/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


def load_dataset(dataset_name: str):
    """
    Load model_dataset.csv or model_dataset_transformed.csv
    and sort observations chronologically so that a split with
    shuffle=False respects time order.
    """
    file_path = DATA_DIR / dataset_name
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    # Ensure time-series order: first by quarter, then by fund
    if "quarter_id" in df.columns:
        sort_cols = ["quarter_id"]
        if "filer_id" in df.columns:
            sort_cols.append("filer_id")
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def build_X_y(df: pd.DataFrame):
    """
    Create feature matrix X and target y.
    Excludes:
    - future returns
    - IDs
    - quarter index
    - (in transformed version) includes engineered features
    """

    drop_cols = [
        "filer_id",
        "quarter_id",
        "fund_return_next",
        "sp500_return_next",
        "excess_return_next",  # target
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["excess_return_next"]

    # Drop rows containing missing values in X
    mask = X.notnull().all(axis=1)
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    return X, y


def split_and_scale(X, y, test_size=0.25):
    """
    Chronological train/test split (NO SCALING).
    This matches the original experiment exactly.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=False,   # keep time order
    )

    return X_train, X_test, y_train, y_test