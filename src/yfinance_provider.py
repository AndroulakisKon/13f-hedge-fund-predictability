"""
Minimal YFinanceProvider for your hedgefund_project.

Used by:
- src/build_quarterly_macro_sp500.py

It exposes a single method:
    fetch_data(tickers, quarter_ends)
and returns a DataFrame with columns:
    ticker, price_date, close_price, currency, vendor, created_at
"""

from __future__ import annotations

from datetime import timedelta
from typing import List

import pandas as pd
import yfinance as yf


class YFinanceProvider:
    def __init__(self) -> None:
        # no special state needed
        pass

    def __repr__(self) -> str:
        return "YFinanceProvider"

    def fetch_data(self, tickers: List[str], quarter_ends: List[pd.Timestamp]) -> pd.DataFrame:
        """
        Fetch daily prices from yfinance for the given tickers over a date range
        that covers all requested quarter_ends, then for each (ticker, quarter_end)
        pick the last available close <= that date.

        Parameters
        ----------
        tickers : list of str
            e.g. ["^GSPC"], ["^VIX"], ["^TNX","^IRX"], ...
        quarter_ends : list of date-like
            Python date objects or timestamps representing target dates.

        Returns
        -------
        df : pd.DataFrame
            Columns: ticker, price_date, close_price, currency, vendor, created_at
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        # Convert to pandas Timestamps
        dates = pd.to_datetime(quarter_ends)
        if dates.empty:
            return pd.DataFrame(
                columns=["ticker", "price_date", "close_price", "currency", "vendor", "created_at"]
            )

        start = dates.min() - timedelta(days=10)
        end = dates.max() + timedelta(days=2)

        # Download once for all tickers over the full range
        data = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

        rows = []

        # Helper to pull close price series for one ticker
        def _get_close_series(ticker: str) -> pd.Series:
            if isinstance(data.columns, pd.MultiIndex):
                # Multi-ticker: ('Close', ticker)
                return data["Close"][ticker].dropna()
            else:
                # Single ticker: 'Close'
                return data["Close"].dropna()

        for tkr in tickers:
            close = _get_close_series(tkr)
            if close.empty:
                continue

            # For each requested date, pick last available <= that date
            for target in dates:
                sub = close[close.index <= target]
                if sub.empty:
                    continue
                price_date = sub.index[-1]
                close_price = float(sub.iloc[-1])

                rows.append(
                    {
                        "ticker": tkr,
                        "price_date": price_date,
                        "close_price": close_price,
                        "currency": "USD",         # reasonable default
                        "vendor": "yfinance",
                        "created_at": pd.Timestamp.utcnow(),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=["ticker", "price_date", "close_price", "currency", "vendor", "created_at"]
            )

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["ticker", "price_date"]).reset_index(drop=True)
        return df
