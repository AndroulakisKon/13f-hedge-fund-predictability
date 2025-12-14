import numpy as np
import pandas as pd
from pathlib import Path

# Paths inside your new project
RAW_PATH = Path("data/raw/13f_raw_holdings.csv")
CLEAN_PATH = Path("data/processed/13f_clean_holdings.csv")
FUND_QUARTER_PATH = Path("data/processed/fund_quarter_features.csv")


def load_and_clean_holdings() -> pd.DataFrame:
    """
    Load raw WhaleWisdom holdings and clean them according to the project's methodology.
    - Keep only SH + 13F rows
    - Drop NA tickers / shares
    - Convert types
    - Keep only relevant columns (including sector & industry)
    - Save cleaned CSV
    """
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw holdings file not found: {RAW_PATH.resolve()}")

    print(f"Loading raw holdings from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)
    print(f"Raw rows: {len(df):,}")

    # 1. Keep only normal common stock 13F positions
    mask = (df.get("security_type") == "SH") & (df.get("source") == "13F")
    df = df.loc[mask].copy()
    print(f"After filtering SH & 13F → {len(df):,} rows")

    # 2. Drop rows without ticker or shares
    df = df.dropna(subset=["stock_ticker", "current_shares"])
    print(f"After dropping NA tickers/shares → {len(df):,} rows")

    # 3. Convert numeric types
    numeric_cols = [
        "filer_id",
        "quarter_id",
        "current_shares",
        "previous_shares",
        "current_mv",
        "previous_mv",
        "current_percent_of_portfolio",
        "previous_percent_of_portfolio",
        "shares_change",
        "percent_change",
        "quarter_end_price",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Keep only the useful columns (including sector & industry)
    keep_cols = [
        "filer_id",
        "filer_name",
        "quarter_id",
        "stock_id",
        "stock_ticker",
        "stock_name",
        "sector",          # sector from API
        "industry",        # industry from API
        "security_type",
        "source",
        "current_shares",
        "previous_shares",
        "current_mv",
        "previous_mv",
        "current_percent_of_portfolio",
        "previous_percent_of_portfolio",
        "position_change_type",
        "shares_change",
        "percent_change",
        "quarter_end_price",
    ]

    df = df[[c for c in keep_cols if c in df.columns]]
    print(f"After selecting relevant columns → {len(df):,} rows")

    # 5. Ensure processed folder exists
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 6. Save cleaned CSV
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned holdings → {CLEAN_PATH}")
    print(df.head())

    return df


def build_fund_quarter_features(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    From cleaned position-level holdings (one row per stock per fund per quarter),
    compute fund–quarter level portfolio metrics:
    - num_positions, total_mv, avg_weight, hhi_weight
    - top10_weight, top20_weight, top50_weight
    - sector weights (sector_*)
    - cyclical_share, defensive_share, sector_hhi
    - turnover
    """
    df = df_clean.copy()

    # 0) Normalize sector names (uppercase, underscores, AND)
    df["sector"] = df["sector"].astype(str).str.upper().str.strip()
    df["sector"] = df["sector"].str.replace(" ", "_")
    df["sector"] = df["sector"].str.replace("&", "AND")

    # 1) Compute portfolio weights per stock within each fund–quarter
    total_mv_per_fq = df.groupby(["filer_id", "quarter_id"])["current_mv"].transform("sum")
    df["weight"] = df["current_mv"] / total_mv_per_fq

    # 2) Basic aggregates per fund–quarter
    grouped = df.groupby(["filer_id", "quarter_id"])

    base = grouped.agg(
    num_positions=("stock_id", "nunique"),
    total_mv=("current_mv", "sum"),
    avg_weight=("weight", "mean"),
)

# Add previous market value and compute fund return
    base["total_mv_prev"] = grouped["previous_mv"].sum()
    base["fund_return"] = base["total_mv"] / base["total_mv_prev"] - 1


    # 3) HHI at position level (weight^2 sum)
    hhi_weight = grouped["weight"].apply(lambda w: (w ** 2).sum())
    base["hhi_weight"] = hhi_weight

    # 4) Top-N weights
    df_sorted = df.sort_values(["filer_id", "quarter_id", "weight"],
                               ascending=[True, True, False])

    top10 = df_sorted.groupby(["filer_id", "quarter_id"])["weight"].apply(
        lambda w: w.head(10).sum()
    )
    top20 = df_sorted.groupby(["filer_id", "quarter_id"])["weight"].apply(
        lambda w: w.head(20).sum()
    )
    top50 = df_sorted.groupby(["filer_id", "quarter_id"])["weight"].apply(
        lambda w: w.head(50).sum()
    )

    base["top10_weight"] = top10
    base["top20_weight"] = top20
    base["top50_weight"] = top50

    # 5) Sector weights: pivot on sector, sum of weights
    sector_table = df.pivot_table(
        index=["filer_id", "quarter_id"],
        columns="sector",
        values="weight",
        aggfunc="sum",
        fill_value=0.0,
    )

    # Add prefix "sector_"
    sector_table.columns = [f"sector_{c}" for c in sector_table.columns]

    # 6) Cyclical vs Defensive share
    cyclical_sectors = [
        "sector_COMMUNICATIONS",
        "sector_CONSUMER_DISCRETIONARY",
        "sector_INDUSTRIALS",
        "sector_INFORMATION_TECHNOLOGY",
        "sector_MATERIALS",
        "sector_TRANSPORTS",
        "sector_REAL_ESTATE",
    ]

    defensive_sectors = [
        "sector_CONSUMER_STAPLES",
        "sector_HEALTH_CARE",
        "sector_UTILITIES_AND_TELECOMMUNICATIONS",
        "sector_MUTUAL_FUND",
        "sector_FINANCE",
        "sector_ENERGY",
    ]

    # ensure all columns exist
    for col in cyclical_sectors + defensive_sectors:
        if col not in sector_table.columns:
            sector_table[col] = 0.0

    cyclical_share = sector_table[cyclical_sectors].sum(axis=1)
    defensive_share = sector_table[defensive_sectors].sum(axis=1)

    # 7) Sector HHI based on sector weights
    sector_hhi = (sector_table ** 2).sum(axis=1)

    # 8) Turnover: for each fund, compare weight vectors across consecutive quarters
    def compute_turnover_for_fund(df_fund: pd.DataFrame) -> pd.Series:
        """
        df_fund: rows for a single filer_id, all quarters.
        We pivot to (quarter_id x stock_id) matrix of weights, fill missing with 0,
        then compute 0.5 * sum |w_t - w_{t-1}| for each t.
        """
        mat = df_fund.pivot_table(
            index="quarter_id",
            columns="stock_id",
            values="weight",
            aggfunc="sum",
            fill_value=0.0,
        ).sort_index()

        diff = mat.diff().abs().sum(axis=1) * 0.5
        return diff  # Series indexed by quarter_id

    # groupby.apply returns a Series with MultiIndex (filer_id, quarter_id)
    turnover = df.groupby("filer_id").apply(compute_turnover_for_fund)
    turnover.name = "turnover"

    # 9) Combine everything into one fund–quarter DataFrame
    features = base.join(sector_table, how="left")

    # Drop spurious sector_ column (unknown or empty sector labels)
    if "sector_" in features.columns:
        features = features.drop(columns=["sector_"])
    # Drop sector_NAN column (holdings with missing sector classification)
    # Drop sector_NAN column (holdings with missing sector classification)
    if "sector_NAN" in features.columns:
        features = features.drop(columns=["sector_NAN"])


    # attach portfolio-level metrics
    features["cyclical_share"] = cyclical_share
    features["defensive_share"] = defensive_share
    features["sector_hhi"] = sector_hhi

    # join turnover using the same MultiIndex (filer_id, quarter_id)
    features = features.join(turnover, how="left")

    # Reset index so we have filer_id/quarter_id as columns
    features = features.reset_index()

    return features


def build_fund_quarter_panel() -> pd.DataFrame:
    """
    High-level function:
    - load & clean holdings
    - compute fund–quarter features
    - save to CSV
    """
    df_clean = load_and_clean_holdings()
    panel = build_fund_quarter_features(df_clean)

    FUND_QUARTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(FUND_QUARTER_PATH, index=False)
    print(f"Saved fund–quarter features → {FUND_QUARTER_PATH}")
    print(panel.head())

    return panel


if __name__ == "__main__":
    build_fund_quarter_panel()
