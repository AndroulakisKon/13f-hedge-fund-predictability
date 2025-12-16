# main.py
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


from src.data_loader import load_dataset, build_X_y
from src.models import (
    train_ols,
    train_ridge,
    train_lasso,
    train_random_forest,
)
from src.evaluation import expanding_window_cv_by_quarter


def run_expanding_window(dataset_name: str, n_folds: int = 5):
    print(f"\n=== Expanding-window time-series CV: {dataset_name} ===")

    # ------------------------------------------------------------------
    # 1) Load data and enforce chronological ordering
    # ------------------------------------------------------------------
    df = load_dataset(dataset_name)

    if "quarter_id" not in df.columns:
        raise ValueError("quarter_id column not found in dataset.")

    df = df.sort_values("quarter_id").reset_index(drop=True)

    # Build X and y from the SAME dataframe
    X, y = build_X_y(df)

    # ------------------------------------------------------------------
    # 2) Define models
    # ------------------------------------------------------------------
    model_trainers = {
        "OLS": train_ols,
        "Ridge": train_ridge,
        "Lasso": train_lasso,
        "RandomForest": train_random_forest,
    }

    # ------------------------------------------------------------------
    # 3) Quarter-based expanding-window evaluation
    # ------------------------------------------------------------------
    summary = {}

    for name, trainer in model_trainers.items():
        cv_out = expanding_window_cv_by_quarter(
            X=X,
            y=y,
            quarter_id=df["quarter_id"],  # quarter blocks
            model_trainer=trainer,
            n_folds=n_folds,
            min_train_fraction=0.5,
        )

        summary[name] = {
            "mean_R2": cv_out["mean_r2"],
            "mean_RMSE": cv_out["mean_rmse"],
        }

    return pd.DataFrame(summary).T


def main():
    print("=" * 70)
    print("Hedge Fund Excess Return Prediction")
    print("Expanding-Window Time-Series Cross-Validation")
    print("=" * 70)

    baseline = run_expanding_window("model_dataset.csv", n_folds=5)
    transformed = run_expanding_window("model_dataset_transformed.csv", n_folds=5)

    final_results = (
        baseline.add_prefix("baseline_")
        .join(transformed.add_prefix("transformed_"))
    )

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "model_comparison_expanding_window.csv"
    final_results.to_csv(output_path)

    print("\n=== Final Expanding-Window Results ===")
    print(final_results)
    print("\nSaved to:", output_path)


if __name__ == "__main__":
    main()
