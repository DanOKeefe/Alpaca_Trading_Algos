import datetime
import logging

import pandas as pd
import pytz as tz
import yfinance as yf
from alpaca_trade_api.rest import APIError
from dateutil.relativedelta import relativedelta

from src.utils.config import (
    DUPLICATE_TICKERS,
    LOOKBACK_YEARS,
    TICKER_COLUMN_NAME,
    TICKER_SOURCE_URL,
    TICKER_TABLE_INDEX,
    TIMEZONE,
)

logger = logging.getLogger(__name__)


def fetch_sp100_tickers():
    """Fetch S&P 100 tickers from Wikipedia, removing duplicate share classes."""
    sp100_data = pd.read_html(TICKER_SOURCE_URL)
    tickers = sp100_data[TICKER_TABLE_INDEX][TICKER_COLUMN_NAME].tolist()
    for dup in DUPLICATE_TICKERS:
        if dup in tickers:
            tickers.remove(dup)
    return tickers


def download_returns(tickers):
    """Download historical OHLC data and compute daily returns."""
    est = tz.timezone(TIMEZONE)
    end_date = datetime.datetime.now(est).date()
    start_date = end_date - relativedelta(years=LOOKBACK_YEARS)

    tickers_str = ' '.join(tickers)
    data = yf.download(
        tickers_str,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
    )

    rets = data['Adj Close'].pct_change()
    rets.dropna(axis=1, inplace=True, how='all')
    rets.dropna(axis=0, inplace=True, how='all')
    return data, rets


def get_latest_prices(api, stocks):
    """Get latest price snapshots from Alpaca. Returns None on failure."""
    try:
        return api.get_snapshots(stocks)
    except APIError as e:
        logger.error("Failed to get snapshots: %s", e)
        return None
