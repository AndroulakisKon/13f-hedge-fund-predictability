"""
Build quarterly macro + S&P500 dataset in ONE CSV file.

Outputs:
    data/raw/macro_sp500_quarterly.csv

Columns:
    quarter_id,
    year, quarter,
    VIX,
    y10, y3m, term_spread,
    sp500_price, sp500_return
"""

from datetime import date
import pandas as pd

from src.yfinance_provider import YFinanceProvider

OUT_FILE = "data/raw/macro_sp500_quarterly.csv"

START_YEAR = 2000
END_YEAR   = 2025


def build_quarter_end_dates(start_year: int, end_year: int) -> list[date]:
    """
    Build list of calendar quarter-end dates (Mar 31, Jun 30, Sep 30, Dec 31)
    between start_year and end_year (inclusive).
    """
    dates = []
    for y in range(start_year, end_year + 1):
        dates.append(date(y, 3, 31))   # Q1
        dates.append(date(y, 6, 30))   # Q2
        dates.append(date(y, 9, 30))   # Q3
        dates.append(date(y, 12, 31))  # Q4
    return dates


def main():
    # 1) Build quarter-end calendar
    quarter_ends = build_quarter_end_dates(START_YEAR, END_YEAR)

    # 2) Init provider
    provider = YFinanceProvider()

    # 3) Download data for all tickers at once
    tickers = ["^VIX", "^TNX", "^IRX", "^GSPC"]
    raw = provider.fetch_data(tickers=tickers, quarter_ends=quarter_ends)

    if raw.empty:
        print("No data returned from yfinance. Check your internet, tickers, or dates.")
        return

    df = raw.copy()
    df["price_date"] = pd.to_datetime(df["price_date"])
    df["year"] = df["price_date"].dt.year
    df["quarter"] = df["price_date"].dt.quarter

    # 4) For each (ticker, year, quarter), keep the last available price in that quarter
    df_q = (
        df.sort_values("price_date")
          .groupby(["ticker", "year", "quarter"], as_index=False)
          .agg(close_price=("close_price", "last"))
    )

    # 5) Pivot to wide format: columns per ticker
    wide = (
        df_q.pivot(index=["year", "quarter"], columns="ticker", values="close_price")
            .reset_index()
    )

    # Make sure rows are ordered chronologically
    wide = wide.sort_values(["year", "quarter"]).reset_index(drop=True)

    # 6) Rename columns to meaningful names
    # VIX level
    if "^VIX" in wide.columns:
        wide["VIX"] = wide["^VIX"]

    # Yields (Yahoo: TNX/IRX are "percent * 10")
    # Example: 38.50 -> 3.85% -> 0.0385
    if "^TNX" in wide.columns:
        wide["y10"] = (wide["^TNX"] / 10.0) / 100.0
    else:
        wide["y10"] = pd.NA

    if "^IRX" in wide.columns:
        wide["y3m"] = (wide["^IRX"] / 10.0) / 100.0
    else:
        wide["y3m"] = pd.NA

    wide["term_spread"] = wide["y10"] - wide["y3m"]

    # S&P500 price and quarterly returns
    if "^GSPC" in wide.columns:
        wide["sp500_price"] = wide["^GSPC"]
        wide["sp500_return"] = wide["sp500_price"].pct_change()
    else:
        wide["sp500_price"] = pd.NA
        wide["sp500_return"] = pd.NA

    # 7) Select final columns and drop the raw tickers
    final = wide[[
        "year", "quarter",
        "VIX",
        "y10", "y3m", "term_spread",
        "sp500_price", "sp500_return",
    ]].copy()

    # 7b) Add sequential quarter_id mapping
    # 2000Q1 -> 1, 2000Q2 -> 2, ..., 2001Q1 -> 5, ...
    final = final.sort_values(["year", "quarter"]).reset_index(drop=True)
    final["quarter_id"] = range(1, len(final) + 1)

    # Reorder columns so quarter_id is first (optional, but clean)
    final = final[[
        "quarter_id",
        "year", "quarter",
        "VIX",
        "y10", "y3m", "term_spread",
        "sp500_price", "sp500_return",
    ]]

    # 8) Save to CSV
    final.to_csv(OUT_FILE, index=False)
    print(f"Saved quarterly macro + S&P500 data to: {OUT_FILE}")
    print(final.head(10))


if __name__ == "__main__":
    main()
