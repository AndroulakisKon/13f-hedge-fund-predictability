# whales_provider.py
"""
Minimal WhaleWisdom provider for your capstone project.

Uses the same API idea as in Exechon but stripped down so it
doesn't depend on external project modules.
"""

from __future__ import annotations

import io
import os
import time
import json
import base64
import hmac
import hashlib
from typing import List, Tuple

import http.client
import urllib3
import requests
import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
BASE_URL = "https://whalewisdom.com/shell/command"
FILERS_CSV_URL = "https://whalewisdom.com/filers.csv"
STOCKS_CSV_URL = "https://whalewisdom.com/stocks.csv"

RATE_LIMIT_INTERVAL = 3.0      # seconds between API calls
CONNECTION_TIMEOUT = 300.0     # max seconds per HTTP request
HOLDINGS_BATCH_SIZE = 1        # filers per API call (keep it small!)


# ---------------------------------------------------------------------
# Small helper: GET with retry + backoff
# ---------------------------------------------------------------------
def _get_with_retry(
    url: str,
    *,
    timeout: float,
    max_retries: int = 5,
) -> requests.Response:
    """
    Robust wrapper around requests.get that retries on timeouts,
    connection drops and incomplete / chunked reads.
    """
    last_err: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            # stream=False → download whole body in one shot
            resp = requests.get(url, timeout=timeout, stream=False)
            resp.raise_for_status()

            # Force full read now so any IncompleteRead happens here
            _ = resp.content  # noqa: F841
            return resp

        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            urllib3.exceptions.ProtocolError,
            http.client.IncompleteRead,
        ) as e:
            last_err = e
            if attempt == max_retries:
                break

            # simple backoff: 2s, 4s, 6s, ...
            print(f"[WARN] Attempt {attempt}/{max_retries} failed: {e}")
            time.sleep(2 * attempt)

    # if we are here, all retries failed
    if last_err is not None:
        raise last_err
    raise RuntimeError("Unknown error in _get_with_retry")


def _download_csv(url: str) -> pd.DataFrame:
    resp = _get_with_retry(url, timeout=CONNECTION_TIMEOUT)
    return pd.read_csv(io.StringIO(resp.text))


# ---------------------------------------------------------------------
# Main provider
# ---------------------------------------------------------------------
class WhalesDataProvider:
    """
    Very small WhaleWisdom client, enough for:
    - quarters
    - filers
    - holdings
    """

    _OUTPUT = {
        "quarters":     "json",
        "filers":       "csv",
        "stocks":       "csv",
        "stock_lookup": "json",
        "filer_lookup": "json",
        "holdings":     "json",
        "holders":      "json",
    }

    def __init__(self) -> None:
        self.shared_key = os.getenv("WHALES_WISDOM_SHARED_KEY")
        self.secret_key = os.getenv("WHALES_WISDOM_SECRET_KEY")
        if not self.shared_key or not self.secret_key:
            raise EnvironmentError(
                "Set WHALES_WISDOM_SHARED_KEY & WHALES_WISDOM_SECRET_KEY env vars."
            )
        self._last_call = 0.0

    # ------------------------------------------------------------------ #
    # Public method
    # ------------------------------------------------------------------ #
    def fetch_data(self, command: str, **params) -> pd.DataFrame:
        if command not in self._OUTPUT:
            raise ValueError(f"Unsupported command: {command}")

        # CSV endpoints don't need signing
        if command == "filers":
            return _download_csv(FILERS_CSV_URL)
        if command == "stocks":
            return _download_csv(STOCKS_CSV_URL)

        # Holdings can be very heavy → batch by small filer groups
        if command == "holdings":
            filer_ids = params.get("filer_ids", [])
            if isinstance(filer_ids, int):
                filer_ids = [filer_ids]
            if isinstance(filer_ids, list) and len(filer_ids) > HOLDINGS_BATCH_SIZE:
                dfs: list[pd.DataFrame] = []
                for start in range(0, len(filer_ids), HOLDINGS_BATCH_SIZE):
                    batch = filer_ids[start:start + HOLDINGS_BATCH_SIZE]
                    batch_params = {**params, "filer_ids": batch}
                    dfs.append(self.fetch_data("holdings", **batch_params))
                return pd.concat(dfs, ignore_index=True)

        # normal flow: signed API call
        self._respect_rate_limit()
        url = self._build_url(command, **params)

        resp = _get_with_retry(url, timeout=CONNECTION_TIMEOUT)

        if self._OUTPUT[command] == "html":
            soup = BeautifulSoup(resp.content, "html.parser")
            return pd.read_html(str(soup))[0]

        if self._OUTPUT[command] == "json":
            payload = resp.json()
            if isinstance(payload, dict) and "errors" in payload:
                msg = "; ".join(payload["errors"])
                raise RuntimeError(f"WhaleWisdom API error: {msg}")
            return self._post_process_json(command, payload)

        raise RuntimeError("Unexpected output type mapping")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _respect_rate_limit(self) -> None:
        delta = time.time() - self._last_call
        if delta < RATE_LIMIT_INTERVAL:
            time.sleep(RATE_LIMIT_INTERVAL - delta)
        self._last_call = time.time()

    def _build_url(self, command: str, **params) -> str:
        args = {"command": command, **params}
        ts, sig = self._signature(args)
        qs = {
            "args": json.dumps(args, separators=(",", ":")),
            "api_shared_key": self.shared_key,
            "api_sig": sig,
            "timestamp": ts,
        }
        return f"{BASE_URL}.{self._OUTPUT[command]}?{requests.compat.urlencode(qs)}"

    def _signature(self, args: dict) -> Tuple[str, str]:
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        raw = json.dumps(args, separators=(",", ":")) + "\n" + ts
        digest = hmac.new(self.secret_key.encode(), raw.encode(), hashlib.sha1).digest()
        return ts, base64.b64encode(digest).decode().rstrip()

    def _post_process_json(self, command: str, data: dict) -> pd.DataFrame:
        if command == "quarters":
            df = pd.json_normalize(data.get("quarters", []))
            df = df.rename(
                columns={
                    "ID": "id",
                    "Filing Period": "filing_period",
                    "Status": "status",
                }
            )
            df["filing_period"] = pd.to_datetime(df["filing_period"], format="%m/%d/%Y")
            return df

        if command == "stock_lookup":
            return pd.json_normalize(data.get("stocks", []))

        if command == "filer_lookup":
            return pd.json_normalize(data.get("filers", []))

        if command == "holdings":
            rows: List[dict] = []
            for res in data.get("results", []):
                for rec in res.get("records", []):
                    for h in rec.get("holdings", []):
                        h.update(
                            quarter_id=rec["quarter_id"],
                            filer_id=res["filer_id"],
                            filer_name=res["filer_name"],
                        )
                        rows.append(h)
            return pd.DataFrame(rows)

        if command == "holders":
            return pd.json_normalize(data.get("results", []))

        raise ValueError(f"Unhandled JSON command: {command}")

    def __repr__(self) -> str:
        return "WhaleWisdom API"
