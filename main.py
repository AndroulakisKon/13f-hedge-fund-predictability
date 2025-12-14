# main.py
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from src.data_loader import load_dataset, build_X_y
from src.models import (
    train_ols,
    train_ridge,
    train_lasso,
    train_random_forest,
)
from src.evaluation import evaluate_model


def expanding_window_cv(dataset_name: str, n_splits: int = 5):
    print(f"\n=== Expanding-window time-series CV: {dataset_name} ===")

    # ------------------------------------------------------------------
    # 1) Load dataset and ensure correct time ordering
    # ------------------------------------------------------------------
    df = load_dataset(dataset_name)

    if "quarter_id" not in df.columns:
        raise ValueError("quarter_id column not found. Cannot enforce time ordering.")

    df = df.sort_values("quarter_id").reset_index(drop=True)

    X, y = build_X_y(df)

    # ------------------------------------------------------------------
    # 2) Define expanding-window cross-validation
    # ------------------------------------------------------------------
    tscv = TimeSeriesSplit(n_splits=n_splits)

    model_trainers = {
        "OLS": train_ols,
        "Ridge": train_ridge,
        "Lasso": train_lasso,
        "RandomForest": train_random_forest,
    }

    results = {name: [] for name in model_trainers}

    # ------------------------------------------------------------------
    # 3) Walk-forward evaluation
    # ------------------------------------------------------------------
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        print(f"\n--- Fold {fold} ---")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, trainer in model_trainers.items():
            model = trainer(X_train, y_train)
            r2, rmse = evaluate_model(model, X_test, y_test)

            results[name].append((r2, rmse))
            print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}")

    # ------------------------------------------------------------------
    # 4) Aggregate fold results
    # ------------------------------------------------------------------
    summary = {}
    for name, scores in results.items():
        r2s = [s[0] for s in scores]
        rmses = [s[1] for s in scores]

        summary[name] = {
            "mean_R2": np.mean(r2s),
            "mean_RMSE": np.mean(rmses),
        }

    return pd.DataFrame(summary).T


def main():
    print("=" * 70)
    print("Hedge Fund Excess Return Prediction")
    print("Expanding-Window Time-Series Cross-Validation")
    print("=" * 70)

    baseline = expanding_window_cv("model_dataset.csv", n_splits=5)
    transformed = expanding_window_cv("model_dataset_transformed.csv", n_splits=5)

    final_results = (
        baseline.add_prefix("baseline_")
        .join(transformed.add_prefix("transformed_"))
    )

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "model_comparison_expanding_window.csv"
    final_results.to_csv(output_path)

    print("\n=== Final Expanding-Window Results ===")
    print(final_results)
    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
