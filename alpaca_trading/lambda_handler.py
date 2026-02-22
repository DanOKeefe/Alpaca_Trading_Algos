"""AWS Lambda entry point for portfolio rebalancing."""

import logging

import numpy as np
import pandas as pd

from alpaca_trading.client import create_client
from alpaca_trading.data import get_historical_returns, get_sp100_tickers
from alpaca_trading.execution import build_orders, execute_orders
from alpaca_trading.strategies import GMVStrategy

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def rebalance_portfolio(strategy=None):
    """Run the full rebalancing workflow.

    Args:
        strategy: a Strategy instance. Defaults to GMVStrategy.

    Returns:
        JSON string of executed orders.
    """
    if strategy is None:
        strategy = GMVStrategy()

    logger.info("Starting rebalance with strategy: %s", strategy.name)

    api = create_client()

    # Check market hours
    clock = api.get_clock()
    if not clock.is_open:
        logger.info("Market is closed. Skipping rebalance.")
        return "Stock market is closed today."

    # Get tickers and historical data
    tickers = get_sp100_tickers()
    returns = get_historical_returns(tickers)

    # Compute target weights
    weights = strategy.compute_weights(returns)
    portfolio_value = int(float(api.get_account().portfolio_value))
    dollar_amounts = weights * portfolio_value
    logger.info("Portfolio value: $%d", portfolio_value)

    # Cancel existing open orders
    orders = api.list_orders(status="open")
    for order in orders:
        api.cancel_order(order.id)
    if orders:
        logger.info("Cancelled %d open orders", len(orders))

    # Filter to tradable stocks
    stocks = list(returns.columns)
    assets = api.list_assets()
    tradable_symbols = [a.symbol for a in assets if a.tradable and a.status == "active"]
    stocks = [s for s in stocks if s in tradable_symbols]

    # Current positions
    positions = api.list_positions()
    positions_df = pd.DataFrame(
        {
            "Symbol": [p.symbol for p in positions],
            "Qty": [int(p.qty) for p in positions],
        }
    )

    if "GOOGL" in stocks:
        stocks.remove("GOOGL")

    price_df = api.get_barset(stocks, "minute", 1).df

    # Build and execute orders
    orders_df = build_orders(stocks, dollar_amounts, positions_df, price_df, tradable_symbols)
    execute_orders(api, orders_df)

    logger.info("Rebalance complete. %d orders submitted.", len(orders_df))
    return orders_df.to_json()


def lambda_handler(event, context):
    """AWS Lambda handler."""
    strategy_name = (event or {}).get("strategy", "gmv")

    strategies = {
        "gmv": GMVStrategy,
    }

    strategy_cls = strategies.get(strategy_name, GMVStrategy)
    return rebalance_portfolio(strategy=strategy_cls())
