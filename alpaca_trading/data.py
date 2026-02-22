"""Data retrieval: ticker lists and historical prices."""

import datetime
import logging

import pandas as pd
import pytz as tz
from dateutil.relativedelta import relativedelta

from alpaca_trading.config import HISTORICAL_YEARS

logger = logging.getLogger(__name__)

SP100_WIKI_URL = "https://en.wikipedia.org/wiki/S%26P_100"


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


def get_historical_returns(tickers, years=HISTORICAL_YEARS):
    """Download historical price data and compute daily returns.

    Returns a DataFrame of daily percentage returns with NAs cleaned.
    """
    est = tz.timezone("US/Eastern")
    end_date = datetime.datetime.now(est).date()
    start_date = end_date - relativedelta(years=years)

    ticker_str = " ".join(tickers)
    logger.info(
        "Downloading %d years of data for %d tickers (%s to %s)",
        years,
        len(tickers),
        start_date,
        end_date,
    )
    import yfinance as yf

    data = yf.download(
        ticker_str,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
    )

    rets = data["Adj Close"].pct_change()
    rets.dropna(axis=1, inplace=True, how="all")
    rets.dropna(axis=0, inplace=True, how="all")

    logger.info("Returns matrix shape: %s", rets.shape)
    return rets
