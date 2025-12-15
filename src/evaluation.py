# src/evaluation.py

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


def evaluate_model(model, X_test, y_test):
    """
    Compute out-of-sample R² and RMSE for a fitted model.
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return r2, rmse


def compare_results(results_dict):
    """
    Convert a dict of {model_name: [R2, RMSE]} into a clean DataFrame.
    """
    df = pd.DataFrame(results_dict).T
    df.columns = ["R2", "RMSE"]
    return df


def expanding_window_cv_by_quarter(
    X: pd.DataFrame,
    y: pd.Series,
    quarter_id: pd.Series,
    model_trainer,
    n_folds: int = 5,
    min_train_fraction: float = 0.5,
):
    """
    Expanding-window time-series cross-validation using QUARTER BLOCKS.

    - Observations are sorted by quarter_id
    - All rows belonging to the same quarter are assigned together
    - Training window expands by entire quarters
    """

    if not (0.0 < min_train_fraction < 1.0):
        raise ValueError("min_train_fraction must be in (0, 1).")

    if len(X) != len(y) or len(X) != len(quarter_id):
        raise ValueError("X, y, and quarter_id must have the same length.")

    # ------------------------------------------------------------------
    # Sort everything by quarter_id (stable sort)
    # ------------------------------------------------------------------
    order = np.argsort(quarter_id.values, kind="mergesort")
    X = X.iloc[order].reset_index(drop=True)
    y = y.iloc[order].reset_index(drop=True)
    q = pd.Series(quarter_id.values).iloc[order].reset_index(drop=True)

    quarters = pd.Index(q.unique())
    n_quarters = len(quarters)

    if n_quarters < 2 * n_folds:
        raise ValueError(
            f"Not enough quarters ({n_quarters}) for {n_folds} folds."
        )

    min_train_q = int(np.floor(n_quarters * min_train_fraction))
    if min_train_q >= n_quarters:
        raise ValueError("min_train_fraction too large for number of quarters.")

    remaining_q = n_quarters - min_train_q
    test_block_q = max(1, remaining_q // n_folds)

    r2_scores = []
    rmse_scores = []

    end_train_q = min_train_q

    # ------------------------------------------------------------------
    # Expanding-window evaluation
    # ------------------------------------------------------------------
    for fold in range(n_folds):
        start_test_q = end_train_q

        if fold == n_folds - 1:
            end_test_q = n_quarters
        else:
            end_test_q = min(n_quarters, start_test_q + test_block_q)

        if start_test_q >= n_quarters or end_test_q <= start_test_q:
            break

        train_quarters = set(quarters[:end_train_q])
        test_quarters = set(quarters[start_test_q:end_test_q])

        # Safety check: no leakage
        assert train_quarters.isdisjoint(test_quarters), \
            "Leakage detected: same quarter_id in train and test."

        train_mask = q.isin(train_quarters)
        test_mask = q.isin(test_quarters)

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]

        # Scale using TRAINING DATA ONLY
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = model_trainer(X_train_scaled, y_train)

        # Evaluate
        r2, rmse = evaluate_model(model, X_test_scaled, y_test)
        r2_scores.append(r2)
        rmse_scores.append(rmse)

        # Expand training window by full quarters
        end_train_q = end_test_q

    return {
        "r2_scores": r2_scores,
        "rmse_scores": rmse_scores,
        "mean_r2": float(np.mean(r2_scores)) if r2_scores else np.nan,
        "mean_rmse": float(np.mean(rmse_scores)) if rmse_scores else np.nan,
    }
