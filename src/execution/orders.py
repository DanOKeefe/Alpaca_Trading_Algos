import logging

from alpaca_trade_api.rest import APIError

logger = logging.getLogger(__name__)


def submit_order(api, qty, stock, side):
    """Submit a market day order via the Alpaca API."""
    if qty <= 0:
        logger.info("Quantity for %s is zero, skipping order", stock)
        return None
    try:
        api.submit_order(stock, qty, side, 'market', 'day')
        logger.info("Submitted order to %s %d share(s) of %s", side, qty, stock)
    except APIError as e:
        logger.error("Order failed for %s %d share(s) of %s: %s", side, qty, stock, e)
    return None
