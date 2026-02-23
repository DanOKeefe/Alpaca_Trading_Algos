"""AWS Lambda entry point for portfolio rebalancing."""

import logging

import pandas as pd

from alpaca_trading.client import create_client
from alpaca_trading.data import get_historical_returns, get_tickers
from alpaca_trading.execution import build_orders, execute_orders
from alpaca_trading.notifications import format_summary, send_rebalance_summary
from alpaca_trading.strategies import (
    EqualWeightStrategy,
    GMVStrategy,
    MSRStrategy,
    RiskParityStrategy,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def rebalance_portfolio(strategy=None, universe=None):
    """Run the full rebalancing workflow.

    Args:
        strategy: a Strategy instance. Defaults to GMVStrategy.
        universe: stock universe ("sp100", "sp500", or comma-separated tickers).
            Defaults to config.STOCK_UNIVERSE.

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
    tickers = get_tickers(universe) if universe else get_tickers()
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

    # Filter to tradable stocks (keeping dollar_amounts aligned)
    all_stocks = list(returns.columns)
    assets = api.list_assets()
    tradable_symbols = [a.symbol for a in assets if a.tradable and a.status == "active"]

    # Build aligned lists, filtering out non-tradable and GOOGL
    stocks = []
    aligned_amounts = []
    for sym, amt in zip(all_stocks, dollar_amounts):
        if sym == "GOOGL":
            continue
        if sym in tradable_symbols:
            stocks.append(sym)
            aligned_amounts.append(amt)

    # Current positions
    positions = api.list_positions()
    positions_df = pd.DataFrame(
        {
            "Symbol": [p.symbol for p in positions],
            "Qty": [int(p.qty) for p in positions],
        }
    )

    price_df = api.get_barset(stocks, "minute", 1).df

    # Build and execute orders
    orders_df = build_orders(
        stocks, aligned_amounts, positions_df, price_df, tradable_symbols,
        portfolio_value=portfolio_value,
    )
    execute_orders(api, orders_df)

    logger.info("Rebalance complete. %d orders submitted.", len(orders_df))
    logger.info("\n%s", format_summary(orders_df, portfolio_value, strategy.name))
    send_rebalance_summary(orders_df, portfolio_value, strategy.name)

    return orders_df.to_json()


def lambda_handler(event, context):
    """AWS Lambda handler."""
    strategy_name = (event or {}).get("strategy", "gmv")

    strategies = {
        "gmv": GMVStrategy,
        "msr": MSRStrategy,
        "equal_weight": EqualWeightStrategy,
        "risk_parity": RiskParityStrategy,
    }

    universe = (event or {}).get("universe")

    strategy_cls = strategies.get(strategy_name, GMVStrategy)
    return rebalance_portfolio(strategy=strategy_cls(), universe=universe)
