import numpy as np
import pandas as pd
from pathlib import Path

# Base directory = project root (hedgefund_project)
BASE_DIR = Path(__file__).resolve().parent.parent

IN_FILE  = BASE_DIR / "data" / "processed" / "fund_quarter_with_macro.csv"
OUT_FILE = BASE_DIR / "data" / "processed" / "model_dataset.csv"


def main():
    # 1) Load merged fund + macro data
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {IN_FILE}")

    print(f"Loading merged data from: {IN_FILE}")
    df = pd.read_csv(IN_FILE)

    # 2) Sanity checks: we need these columns to build Y
    required_cols = ["filer_id", "quarter_id", "fund_return", "sp500_return"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input: {missing}")

    # 3) Global clean-up: replace infinities with NaN
    num_inf = np.isinf(df.select_dtypes(include=[float, int])).sum().sum()
    if num_inf > 0:
        print(f"Found {num_inf} infinite values in the dataset. Converting to NaN.")
        df.replace([np.inf, -np.inf], pd.NA, inplace=True)

    # 4) Drop clearly broken fund-quarter rows (if MV columns exist)
    if {"total_mv_prev", "total_mv"}.issubset(df.columns):
        broken_mask = (
            (df["total_mv_prev"] > 0)
            & (df["total_mv"] == 0)
            & (df["fund_return"] <= -1 + 1e-9)
        )
        n_broken = int(broken_mask.sum())
        if n_broken > 0:
            print(
                f"Dropping {n_broken} broken fund-quarter rows "
                f"(total_mv_prev > 0, total_mv = 0, fund_return ≈ -1)."
            )
            df = df[~broken_mask].copy()
    else:
        print("Warning: total_mv_prev / total_mv not in input; "
              "cannot drop broken MV rows based on MV logic.")

    # 5) Sort by fund and time to ensure correct shifting
    df = df.sort_values(["filer_id", "quarter_id"]).reset_index(drop=True)

    # 6) Build next-quarter returns per fund
    df["fund_return_next"] = df.groupby("filer_id")["fund_return"].shift(-1)
    df["sp500_return_next"] = df.groupby("filer_id")["sp500_return"].shift(-1)

    # 7) Dependent variable: next-quarter excess return
    df["excess_return_next"] = df["fund_return_next"] - df["sp500_return_next"]

    # 8) Safety: convert any new infinities to NaN
    num_inf_after = np.isinf(df.select_dtypes(include=[float, int])).sum().sum()
    if num_inf_after > 0:
        print(f"Found {num_inf_after} infinite values after shifting. Converting to NaN.")
        df.replace([np.inf, -np.inf], pd.NA, inplace=True)

    # 9) Drop rows where we don't know next-quarter excess returns
    before = len(df)
    df_model = df.dropna(subset=["excess_return_next"]).copy()
    after = len(df_model)
    print(f"Dropped {before - after} rows with missing next-quarter excess return.")

    # 9b) ***Option A: hard drop of ALL remaining NaN rows (predictors, macros, etc.)***
    before_full = len(df_model)
    df_model = df_model.dropna().copy()
    after_full = len(df_model)
    print(f"Dropped {before_full - after_full} additional rows with NaN in any column.")

    # 10) Save final modeling dataset
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_model.to_csv(OUT_FILE, index=False)

    print(f"Model dataset saved to: {OUT_FILE}")
    print(
        df_model[
            [
                "filer_id",
                "quarter_id",
                "fund_return",
                "sp500_return",
                "fund_return_next",
                "sp500_return_next",
                "excess_return_next",
            ]
        ]
        .head(10)
    )


if __name__ == "__main__":
    main()
