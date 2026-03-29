"""Data fetching layer with disk caching via requests_cache."""

import pathlib
import tempfile

import pandas as pd
import requests_cache
import yfinance as yf

# Cache yfinance HTTP calls to disk for 24 hours.
# Use /tmp on HF Spaces (ephemeral filesystem); fallback to system temp.
_cache_dir = pathlib.Path(tempfile.gettempdir()) / "market_regime_cache"
_cache_dir.mkdir(exist_ok=True)
requests_cache.install_cache(
    str(_cache_dir / "yfinance_cache"),
    backend="sqlite",
    expire_after=86400,  # 24 hours
)


def fetch_ohlcv(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Download daily OHLCV for ``ticker`` using yfinance.

    Returns DataFrame with columns: Open, High, Low, Close, Volume
    and a DatetimeIndex named 'Date'.

    Raises ValueError if ticker is invalid or has < 252 rows.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker cannot be empty")

    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)

    if data is None or data.empty:
        raise ValueError(f"Invalid ticker: {ticker}")

    # yfinance may return MultiIndex columns for single ticker; flatten
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Ensure we have the expected columns
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(data.columns)):
        raise ValueError(f"Invalid ticker: {ticker}")

    if len(data) < 252:
        raise ValueError(
            f"Insufficient data for {ticker}: got {len(data)} trading days, need at least 252"
        )

    data.index.name = "Date"
    return data[["Open", "High", "Low", "Close", "Volume"]]
