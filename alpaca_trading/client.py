"""Alpaca API client factory."""

import logging

from alpaca_trading.config import get_api_credentials, get_base_url

logger = logging.getLogger(__name__)


def create_client():
    """Create and return an authenticated Alpaca REST client."""
    import alpaca_trade_api as tradeapi

    api_key, api_secret = get_api_credentials()
    base_url = get_base_url()
    logger.info("Connecting to Alpaca API at %s", base_url)
    return tradeapi.REST(api_key, api_secret, base_url, "v2")
