"""Configuration and environment variable management."""

import logging
import os

logger = logging.getLogger(__name__)

PAPER_TRADING_URL = "https://paper-api.alpaca.markets"
LIVE_TRADING_URL = "https://api.alpaca.markets"

# Portfolio constraints
MAX_WEIGHT_PER_STOCK = 0.10  # 10% max allocation to any single stock
REBALANCE_THRESHOLD = 0.02  # 2% drift threshold before rebalancing a position
HISTORICAL_YEARS = 5  # years of price history to download


def get_api_credentials():
    """Retrieve Alpaca API credentials from environment variables."""
    api_key = os.environ.get("API_KEY")
    api_secret = os.environ.get("API_SECRET")
    if not api_key or not api_secret:
        raise EnvironmentError(
            "API_KEY and API_SECRET environment variables must be set."
        )
    return api_key, api_secret


def get_base_url():
    """Return the API base URL. Defaults to paper trading."""
    use_live = os.environ.get("LIVE_TRADING", "false").lower() == "true"
    url = LIVE_TRADING_URL if use_live else PAPER_TRADING_URL
    if use_live:
        logger.warning("LIVE TRADING is enabled.")
    return url
