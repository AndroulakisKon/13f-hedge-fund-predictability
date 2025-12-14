"""
build_model_dataset_transformed.py

As a senior quant, this script is designed to:
- Take the existing cleaned modeling dataset (model_dataset.csv),
- Engineer economically and academically motivated transformations,
- Output a new dataset (model_dataset_transformed.csv) for model comparison.

Transformations implemented:
1) log_total_mv       = ln(total_mv)
2) log_total_mv_prev  = ln(1 + total_mv_prev)
3) log_num_positions  = ln(num_positions)
4) log_VIX            = ln(VIX)
5) term_spread_sq     = term_spread**2
6) hhi_weight_sq      = hhi_weight**2
7) top10_weight_sq    = top10_weight**2
8) sector_hhi_sq      = sector_hhi**2

The goal is to keep the pipeline reproducible and transparent for your MSc report.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Base directory = project root (hedgefund_project)
BASE_DIR = Path(__file__).resolve().parent.parent

IN_FILE  = BASE_DIR / "data" / "processed" / "model_dataset.csv"
OUT_FILE = BASE_DIR / "data" / "processed" / "model_dataset_transformed.csv"


def main() -> None:
    # -------------------------------------------------------------------------
    # 1) Load base modeling dataset
    # -------------------------------------------------------------------------
    if not IN_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {IN_FILE}")

    print(f"Loading base model dataset from: {IN_FILE}")
    df = pd.read_csv(IN_FILE)

    # -------------------------------------------------------------------------
    # 2) Check that required columns exist
    # -------------------------------------------------------------------------
    required_cols = [
        "total_mv",
        "total_mv_prev",
        "num_positions",
        "VIX",
        "term_spread",
        "hhi_weight",
        "top10_weight",
        "sector_hhi",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following required columns are missing from model_dataset.csv: {missing}"
        )

    # -------------------------------------------------------------------------
    # 3) Sanity checks before transformations
    # -------------------------------------------------------------------------
    # We check conditions under which the logs make economic and mathematical sense.

    # total_mv should be strictly positive for log
    if not (df["total_mv"] > 0).all():
        problematic = df.loc[df["total_mv"] <= 0, ["filer_id", "quarter_id", "total_mv"]]
        raise ValueError(
            "Found non-positive total_mv values; cannot take log(total_mv). "
            f"Problematic rows:\n{problematic.head()}"
        )

    # total_mv_prev should be >= 0 for log1p; economically, negative is nonsensical
    if not (df["total_mv_prev"] >= 0).all():
        problematic = df.loc[df["total_mv_prev"] < 0, ["filer_id", "quarter_id", "total_mv_prev"]]
        raise ValueError(
            "Found negative total_mv_prev values; log1p(total_mv_prev) is not meaningful here. "
            f"Problematic rows:\n{problematic.head()}"
        )

    # num_positions must be strictly positive to take log
    if not (df["num_positions"] > 0).all():
        problematic = df.loc[df["num_positions"] <= 0, ["filer_id", "quarter_id", "num_positions"]]
        raise ValueError(
            "Found non-positive num_positions; cannot take log(num_positions). "
            f"Problematic rows:\n{problematic.head()}"
        )

    # VIX must be strictly positive for log
    if not (df["VIX"] > 0).all():
        problematic = df.loc[df["VIX"] <= 0, ["filer_id", "quarter_id", "VIX"]]
        raise ValueError(
            "Found non-positive VIX values; cannot take log(VIX). "
            f"Problematic rows:\n{problematic.head()}"
        )

    # -------------------------------------------------------------------------
    # 4) Engineer transformed features
    # -------------------------------------------------------------------------
    print("Engineering transformed features:")

    # 1) log_total_mv = ln(total_mv)
    print(" - Creating log_total_mv = ln(total_mv)")
    df["log_total_mv"] = np.log(df["total_mv"])

    # 2) log_total_mv_prev = ln(1 + total_mv_prev)
    print(" - Creating log_total_mv_prev = ln(1 + total_mv_prev)")
    df["log_total_mv_prev"] = np.log1p(df["total_mv_prev"])

    # 3) log_num_positions = ln(num_positions)
    print(" - Creating log_num_positions = ln(num_positions)")
    df["log_num_positions"] = np.log(df["num_positions"])

    # 4) log_VIX = ln(VIX)
    print(" - Creating log_VIX = ln(VIX)")
    df["log_VIX"] = np.log(df["VIX"])

    # 5) term_spread_sq = term_spread**2
    print(" - Creating term_spread_sq = term_spread**2")
    df["term_spread_sq"] = df["term_spread"] ** 2

    # 6) hhi_weight_sq = hhi_weight**2
    print(" - Creating hhi_weight_sq = hhi_weight**2")
    df["hhi_weight_sq"] = df["hhi_weight"] ** 2

    # 7) top10_weight_sq = top10_weight**2
    print(" - Creating top10_weight_sq = top10_weight**2")
    df["top10_weight_sq"] = df["top10_weight"] ** 2

    # 8) sector_hhi_sq = sector_hhi**2
    print(" - Creating sector_hhi_sq = sector_hhi**2")
    df["sector_hhi_sq"] = df["sector_hhi"] ** 2

    # -------------------------------------------------------------------------
    # 5) Save transformed dataset
    # -------------------------------------------------------------------------
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"Transformed model dataset saved to: {OUT_FILE}")
    print("Preview of new columns:")
    print(
        df[
            [
                "filer_id",
                "quarter_id",
                "log_total_mv",
                "log_total_mv_prev",
                "log_num_positions",
                "log_VIX",
                "term_spread_sq",
                "hhi_weight_sq",
                "top10_weight_sq",
                "sector_hhi_sq",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
