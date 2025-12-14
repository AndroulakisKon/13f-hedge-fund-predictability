import pandas as pd
from pathlib import Path

# Base directory = project root (hedgefund_project)
BASE_DIR = Path(__file__).resolve().parent.parent

MACRO_FILE = BASE_DIR / "data" / "raw" / "macro_sp500_quarterly.csv"
FUND_FILE  = BASE_DIR / "data" / "processed" / "fund_quarter_features.csv"
OUT_FILE   = BASE_DIR / "data" / "processed" / "fund_quarter_with_macro.csv"


def main():
    # 1) Load the two CSVs
    macro = pd.read_csv(MACRO_FILE)
    fund  = pd.read_csv(FUND_FILE)

    # 2) Basic sanity checks
    if "quarter_id" not in macro.columns:
        raise KeyError("Column 'quarter_id' not found in macro_sp500_quarterly.csv")

    if "quarter_id" not in fund.columns:
        raise KeyError("Column 'quarter_id' not found in fund_quarter_features.csv")

    # Make sure quarter_id is same dtype on both sides
    macro["quarter_id"] = macro["quarter_id"].astype(int)
    fund["quarter_id"]  = fund["quarter_id"].astype(int)

    # Optional: choose only macro columns you want to add
    # (so you avoid name conflicts like year/quarter if they also exist in fund)
    macro_cols_to_keep = [
        "quarter_id",
        "VIX",
        "y10",
        "y3m",
        "term_spread",
        "sp500_price",
        "sp500_return",
    ]
    # Filter to columns that actually exist (robustness)
    macro_cols_to_keep = [c for c in macro_cols_to_keep if c in macro.columns]
    macro_small = macro[macro_cols_to_keep].copy()

    # 3) Merge on quarter_id (SMART MERGE, NOT BY ROW ORDER)
    # Every fund row keeps its own quarter_id; macro data is matched by that key.
    merged = fund.merge(
        macro_small,
        on="quarter_id",
        how="left"   # keep all fund rows, even if some quarter_ids have no macro
    )

    # 4) Save result
    merged.to_csv(OUT_FILE, index=False)
    print(f"Saved merged dataset to: {OUT_FILE}")
    print(merged.head(10))


if __name__ == "__main__":
    main()
