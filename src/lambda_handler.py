import logging

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd

from src.data.market_data import (
    download_returns,
    fetch_sp100_tickers,
    get_latest_prices,
)
from src.execution.orders import submit_order
from src.strategies.gmv import gmv
from src.utils.config import (
    DUPLICATE_TICKERS,
    WEIGHT_ROUNDING_PRECISION,
    get_api_credentials,
    get_base_url,
)
from src.utils.log_config import setup_logging

logger = logging.getLogger(__name__)


def rebalance_portfolio():
    api_key, api_secret = get_api_credentials()
    base_url = get_base_url()
    alpaca = tradeapi.REST(api_key, api_secret, base_url, 'v2')

    # Stop here if the market is closed
    clock = alpaca.get_clock()
    if not clock.is_open:
        logger.info("Stock market is closed today.")
        return 'Stock market is closed today.'

    tickers = fetch_sp100_tickers()
    data, rets = download_returns(tickers)

    weights = np.round(gmv(rets.cov()), WEIGHT_ROUNDING_PRECISION)
    portfolio_value = int(float(alpaca.get_account().portfolio_value))
    dollar_amounts = weights * portfolio_value

    # Cancel existing open orders
    open_orders = alpaca.list_orders(status='open')
    for order in open_orders:
        alpaca.cancel_order(order.id)

    stocks = list(data['Adj Close'].columns)
    assets = alpaca.list_assets()
    tradable_symbols = [a.symbol for a in assets if a.tradable and a.status == 'active']
    stocks = [s for s in stocks if s in tradable_symbols]

    positions = alpaca.list_positions()
    positions_df = pd.DataFrame({
        'Symbol': [p.symbol for p in positions],
        'Qty': [int(p.qty) for p in positions],
    })

    for dup in DUPLICATE_TICKERS:
        if dup in stocks:
            stocks.remove(dup)

    snapshots = get_latest_prices(alpaca, stocks)
    if snapshots is None:
        return '{"error": "Failed to get price snapshots"}'

    orders = {
        'Side': [],
        'Ticker': [],
        'Qty': [],
    }

    for stock, target_value in zip(stocks, dollar_amounts):
        try:
            if stock not in snapshots:
                logger.warning("No snapshot data for %s, skipping", stock)
                continue
            price = snapshots[stock].latest_trade.p
            target_qty = int(target_value // price)

            if stock in positions_df['Symbol'].tolist():
                mask = positions_df['Symbol'] == stock
                current_qty = positions_df[mask]['Qty'].iloc[0]
            else:
                current_qty = 0

            if target_qty > current_qty and stock in tradable_symbols:
                qty = target_qty - current_qty
                logger.info(
                    "Go from %d to %d shares of %s. Target: $%.2f",
                    current_qty, target_qty, stock, price * qty,
                )
                orders['Side'].append('Buy')
                orders['Ticker'].append(stock)
                orders['Qty'].append(qty)

            elif target_qty < current_qty and stock in tradable_symbols:
                qty = current_qty - target_qty
                logger.info(
                    "Go from %d to %d shares of %s. Target: $%.2f",
                    current_qty, target_qty, stock, price * qty,
                )
                orders['Side'].append('Sell')
                orders['Ticker'].append(stock)
                orders['Qty'].append(qty)

        except (KeyError, IndexError, ZeroDivisionError) as e:
            logger.warning("Could not process %s: %s", stock, e)

    orders_df = pd.DataFrame(orders)

    # Submit sells first to free up capital
    sell_df = orders_df[orders_df['Side'] == 'Sell']
    for _, row in sell_df.iterrows():
        submit_order(api=alpaca, qty=row['Qty'], stock=row['Ticker'], side='sell')

    # Then submit buys
    buy_df = orders_df[orders_df['Side'] == 'Buy']
    for _, row in buy_df.iterrows():
        submit_order(api=alpaca, qty=row['Qty'], stock=row['Ticker'], side='buy')

    return orders_df.to_json()


def lambda_handler(event, context):
    setup_logging()
    return rebalance_portfolio()
