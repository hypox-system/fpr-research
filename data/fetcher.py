"""
Bybit V5 API data fetcher for BTCUSDT Perpetual 1m OHLCV data.
Caches data to parquet files organized by month.

Ported from fpr-backtest with minor adjustments for v2.1 spec.
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Constants
BYBIT_BASE_URL = "https://api.bybit.com"
KLINE_ENDPOINT = "/v5/market/kline"
MAX_LIMIT = 1000
RATE_LIMIT_DELAY = 0.1  # seconds between requests

CACHE_DIR = Path(__file__).parent / "cache"

# Default date ranges from spec
DEFAULT_IS_START = "2025-08-01"
DEFAULT_IS_END = "2025-12-31"
DEFAULT_HOLDOUT_START = "2026-01-01"
DEFAULT_HOLDOUT_END = "2026-02-15"


def timestamp_to_ms(dt: datetime) -> int:
    """Convert datetime to milliseconds timestamp."""
    return int(dt.timestamp() * 1000)


def ms_to_datetime(ms: int) -> datetime:
    """Convert milliseconds timestamp to datetime."""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def fetch_klines_batch(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = MAX_LIMIT
) -> list:
    """
    Fetch a single batch of klines from Bybit V5 API.

    Returns list of [timestamp, open, high, low, close, volume, turnover]
    Results are returned in DESCENDING order (newest first).
    """
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "start": start_ms,
        "end": end_ms,
        "limit": limit
    }

    response = requests.get(
        f"{BYBIT_BASE_URL}{KLINE_ENDPOINT}",
        params=params,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()
    if data.get("retCode") != 0:
        raise ValueError(f"API error: {data.get('retMsg')}")

    return data.get("result", {}).get("list", [])


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1",
    start_date: str = DEFAULT_IS_START,
    end_date: str = DEFAULT_HOLDOUT_END,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fetch all klines for the given date range, paginating as needed.

    Args:
        symbol: Trading pair symbol
        interval: Kline interval (1 = 1 minute)
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        verbose: Print progress

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(
        hour=23, minute=59, second=59, tzinfo=timezone.utc
    )

    start_ms = timestamp_to_ms(start_dt)
    end_ms = timestamp_to_ms(end_dt)

    all_data = []
    current_end = end_ms

    if verbose:
        print(f"Fetching {symbol} {interval}m data from {start_date} to {end_date}...")

    while current_end > start_ms:
        batch = fetch_klines_batch(symbol, interval, start_ms, current_end)

        if not batch:
            break

        # Results are descending (newest first), so oldest is last
        all_data.extend(batch)

        # Get oldest timestamp from batch for next pagination
        oldest_ts = int(batch[-1][0])

        if oldest_ts <= start_ms:
            break

        # Set end to just before oldest timestamp for next batch
        current_end = oldest_ts - 1

        if verbose:
            oldest_date = ms_to_datetime(oldest_ts).strftime("%Y-%m-%d %H:%M")
            print(f"  Fetched to {oldest_date}, total rows: {len(all_data)}")

        time.sleep(RATE_LIMIT_DELAY)

    if not all_data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Drop turnover column (not needed)
    df = df.drop(columns=["turnover"])

    # Results were descending, sort ascending
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Filter to exact date range
    df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]

    # Remove duplicates
    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    if verbose:
        print(f"Total: {len(df)} rows from {df['timestamp'].min()} to {df['timestamp'].max()}")

    return df


def save_to_cache(df: pd.DataFrame, symbol: str = "BTCUSDT") -> None:
    """Save DataFrame to parquet files, one per month."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Group by year-month
    df = df.copy()
    df["year_month"] = df["timestamp"].dt.to_period("M")

    for period, group in df.groupby("year_month"):
        filename = f"{symbol}_{period}.parquet"
        filepath = CACHE_DIR / filename
        group.drop(columns=["year_month"]).to_parquet(filepath, index=False)
        print(f"Saved {filepath} ({len(group)} rows)")


def load_cached_data(
    symbol: str = "BTCUSDT",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load cached parquet data for the given symbol and date range.

    Returns empty DataFrame if no cached data found.
    """
    if not CACHE_DIR.exists():
        return pd.DataFrame()

    # Find all parquet files for symbol
    files = sorted(CACHE_DIR.glob(f"{symbol}_*.parquet"))

    if not files:
        return pd.DataFrame()

    # Load and concatenate
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Ensure timestamp is datetime with UTC
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    elif df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # Filter by date range if specified
    if start_date:
        start_dt = pd.to_datetime(start_date, utc=True)
        df = df[df["timestamp"] >= start_dt]

    if end_date:
        end_dt = pd.to_datetime(end_date, utc=True).replace(
            hour=23, minute=59, second=59
        )
        df = df[df["timestamp"] <= end_dt]

    # Sort and remove duplicates
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    return df


def get_data(
    symbol: str = "BTCUSDT",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
    force_fetch: bool = False
) -> pd.DataFrame:
    """
    Get OHLCV data, using cache if available or fetching from API.

    Args:
        symbol: Trading pair symbol
        start_date: Start date (default: IS start)
        end_date: End date (default: holdout end)
        use_cache: Whether to use cached data
        force_fetch: Force re-fetch even if cache exists

    Returns:
        DataFrame with timestamp, open, high, low, close, volume
    """
    if start_date is None:
        start_date = DEFAULT_IS_START
    if end_date is None:
        end_date = DEFAULT_HOLDOUT_END

    # Try cache first
    if use_cache and not force_fetch:
        df = load_cached_data(symbol, start_date, end_date)
        if not df.empty:
            print(f"Loaded {len(df)} rows from cache")
            return df

    # Fetch from API
    df = fetch_klines(
        symbol=symbol,
        interval="1",
        start_date=start_date,
        end_date=end_date
    )

    # Save to cache
    if not df.empty:
        save_to_cache(df, symbol)

    return df


if __name__ == "__main__":
    # Test fetch
    df = get_data()
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nSample:\n{df.head()}")
