"""Data retrieval: ticker lists, historical prices, and optional S3 caching."""

import datetime
import io
import logging

import pandas as pd
import pytz as tz
from dateutil.relativedelta import relativedelta

from alpaca_trading.config import HISTORICAL_YEARS, S3_CACHE_BUCKET, STOCK_UNIVERSE

logger = logging.getLogger(__name__)

SP100_WIKI_URL = "https://en.wikipedia.org/wiki/S%26P_100"
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_sp100_tickers():
    """Scrape S&P 100 tickers from Wikipedia.

    Returns a deduplicated list with GOOGL removed (keeps GOOG).
    """
    tables = pd.read_html(SP100_WIKI_URL)
    tickers = tables[2]["Symbol"].tolist()
    if "GOOGL" in tickers:
        tickers.remove("GOOGL")
    logger.info("Retrieved %d S&P 100 tickers", len(tickers))
    return tickers


def get_sp500_tickers():
    """Scrape S&P 500 tickers from Wikipedia.

    Returns a deduplicated list with GOOGL removed (keeps GOOG).
    """
    tables = pd.read_html(SP500_WIKI_URL)
    tickers = tables[0]["Symbol"].tolist()
    # Clean tickers: Wikipedia sometimes has dots instead of dashes (e.g. BRK.B)
    tickers = [t.replace(".", "-") for t in tickers]
    if "GOOGL" in tickers:
        tickers.remove("GOOGL")
    logger.info("Retrieved %d S&P 500 tickers", len(tickers))
    return tickers


def get_tickers(universe=STOCK_UNIVERSE):
    """Get tickers based on the configured universe.

    Args:
        universe: "sp100", "sp500", or a comma-separated string of tickers.

    Returns:
        List of ticker symbols.
    """
    universe = universe.strip().lower()
    if universe == "sp100":
        return get_sp100_tickers()
    elif universe == "sp500":
        return get_sp500_tickers()
    else:
        # Assume comma-separated custom ticker list
        tickers = [t.strip().upper() for t in universe.split(",") if t.strip()]
        logger.info("Using custom universe: %d tickers", len(tickers))
        return tickers


def _s3_cache_key(tickers, start_date, end_date):
    """Build an S3 key for the cached price data."""
    # Hash the ticker list to keep the key short
    import hashlib

    ticker_hash = hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:12]
    return f"price_cache/{ticker_hash}_{start_date}_{end_date}.parquet"


def _load_from_s3(bucket, key):
    """Attempt to load a parquet DataFrame from S3. Returns None on failure."""
    try:
        import boto3

        s3 = boto3.client("s3")
        response = s3.get_object(Bucket=bucket, Key=key)
        data = response["Body"].read()
        df = pd.read_parquet(io.BytesIO(data))
        logger.info("Loaded cached data from s3://%s/%s", bucket, key)
        return df
    except Exception as e:
        logger.debug("S3 cache miss (%s): %s", key, e)
        return None


def _save_to_s3(df, bucket, key):
    """Save a DataFrame to S3 as parquet. Fails silently."""
    try:
        import boto3

        buf = io.BytesIO()
        df.to_parquet(buf, index=True)
        buf.seek(0)
        s3 = boto3.client("s3")
        s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
        logger.info("Saved data to s3://%s/%s", bucket, key)
    except Exception as e:
        logger.warning("Failed to save to S3 cache: %s", e)


def get_historical_returns(tickers, years=HISTORICAL_YEARS):
    """Download historical price data and compute daily returns.

    If S3_CACHE_BUCKET is configured, attempts to load from cache first
    and saves to cache after downloading.

    Returns a DataFrame of daily percentage returns with NAs cleaned.
    """
    est = tz.timezone("US/Eastern")
    end_date = datetime.datetime.now(est).date()
    start_date = end_date - relativedelta(years=years)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    # Try S3 cache (caches Adj Close prices as a flat DataFrame)
    adj_close = None
    cache_key = None
    if S3_CACHE_BUCKET:
        cache_key = _s3_cache_key(tickers, start_str, end_str)
        adj_close = _load_from_s3(S3_CACHE_BUCKET, cache_key)

    # Download from yfinance if no cache hit
    if adj_close is None:
        ticker_str = " ".join(tickers)
        logger.info(
            "Downloading %d years of data for %d tickers (%s to %s)",
            years,
            len(tickers),
            start_str,
            end_str,
        )
        import yfinance as yf

        data = yf.download(ticker_str, start=start_str, end=end_str)
        adj_close = data["Adj Close"]

        # yfinance returns a Series for a single ticker; ensure it's a DataFrame
        if isinstance(adj_close, pd.Series):
            adj_close = adj_close.to_frame(name=tickers[0])

        # Cache the flat Adj Close DataFrame (avoids MultiIndex parquet issues)
        if S3_CACHE_BUCKET and cache_key:
            _save_to_s3(adj_close, S3_CACHE_BUCKET, cache_key)

    rets = adj_close.pct_change()
    rets.dropna(axis=1, inplace=True, how="all")
    rets.dropna(axis=0, inplace=True, how="all")

    logger.info("Returns matrix shape: %s", rets.shape)
    return rets
