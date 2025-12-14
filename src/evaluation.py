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


def expanding_window_cv(
    X,
    y,
    model_trainer,
    n_folds: int = 5,
    min_train_fraction: float = 0.5,
):
    """
    Expanding-window time-series cross-validation.

    Assumes X and y are already sorted in chronological order.
    Procedure:
        - Use the first min_train_fraction of the data as the initial
          training window.
        - Split the remaining observations into n_folds consecutive
          test blocks.
        - For each fold:
            * Train on all data up to that point (expanding window).
            * Test on the next block.
            * Scale features using ONLY the training window.
            * Fit a new model via model_trainer(X_train_scaled, y_train).
            * Compute R² and RMSE on the test block.

    Returns:
        dict with:
            "r2_scores": list of R² per fold
            "rmse_scores": list of RMSE per fold
            "mean_r2": average R² across folds
            "mean_rmse": average RMSE across folds
    """

    if not (0.0 < min_train_fraction < 1.0):
        raise ValueError("min_train_fraction must be in (0, 1).")

    n = len(X)
    if n < 2 * n_folds:
        raise ValueError(
            f"Not enough observations ({n}) for {n_folds} folds "
            "of expanding-window CV."
        )

    min_train_size = int(n * min_train_fraction)
    if min_train_size >= n:
        raise ValueError(
            "min_train_fraction is too large: training window "
            "would cover all observations."
        )

    # Size of each test block after the initial training window
    remaining = n - min_train_size
    test_block_size = max(1, remaining // n_folds)

    r2_scores = []
    rmse_scores = []

    start_train = 0
    end_train = min_train_size

    for fold in range(n_folds):
        start_test = end_train

        # Last fold: test on everything remaining
        if fold == n_folds - 1:
            end_test = n
        else:
            end_test = min(n, start_test + test_block_size)

        # Safety check
        if start_test >= n or end_test <= start_test:
            break

        X_train = X.iloc[start_train:end_train]
        y_train = y.iloc[start_train:end_train]
        X_test = X.iloc[start_test:end_test]
        y_test = y.iloc[start_test:end_test]

        # Scale using ONLY training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train a fresh model on this window
        model = model_trainer(X_train_scaled, y_train)

        # Evaluate on the next block
        r2, rmse = evaluate_model(model, X_test_scaled, y_test)
        r2_scores.append(r2)
        rmse_scores.append(rmse)

        # Expand training window to include this test block
        end_train = end_test

    mean_r2 = float(np.mean(r2_scores)) if r2_scores else np.nan
    mean_rmse = float(np.mean(rmse_scores)) if rmse_scores else np.nan

    return {
        "r2_scores": r2_scores,
        "rmse_scores": rmse_scores,
        "mean_r2": mean_r2,
        "mean_rmse": mean_rmse,
    }
