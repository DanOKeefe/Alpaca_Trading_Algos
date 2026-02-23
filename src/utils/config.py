import os

# Trading mode: "paper" or "live"
TRADING_MODE = os.environ.get('TRADING_MODE', 'paper')

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
LIVE_BASE_URL = "https://api.alpaca.markets"

# Strategy parameters
LOOKBACK_YEARS = 5
WEIGHT_ROUNDING_PRECISION = 5
TICKER_SOURCE_URL = 'https://en.wikipedia.org/wiki/S%26P_100'
TICKER_TABLE_INDEX = 2
TICKER_COLUMN_NAME = 'Symbol'
DUPLICATE_TICKERS = ['GOOGL']  # Tickers to remove (e.g., duplicate share classes)
TIMEZONE = 'US/Eastern'


def get_base_url():
    if TRADING_MODE == 'live':
        return LIVE_BASE_URL
    return PAPER_BASE_URL


def get_api_credentials():
    """Return (api_key, api_secret) from environment variables, or raise."""
    api_key = os.environ.get('API_KEY')
    api_secret = os.environ.get('API_SECRET')
    if not api_key or not api_secret:
        raise EnvironmentError(
            "API_KEY and API_SECRET environment variables must be set. "
            "Get your keys from https://app.alpaca.markets"
        )
    return api_key, api_secret
