import logging
import datetime

import pytz as tz
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import APIError
from dateutil.relativedelta import relativedelta

from config import (
    get_api_credentials,
    get_base_url,
    LOOKBACK_YEARS,
    WEIGHT_ROUNDING_PRECISION,
    TICKER_SOURCE_URL,
    TICKER_TABLE_INDEX,
    TICKER_COLUMN_NAME,
    DUPLICATE_TICKERS,
    TIMEZONE,
)

logger = logging.getLogger(__name__)


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights.
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix.
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights.
    weights are a numpy array or N x 1 matrix and covmat is an N x N matrix.
    """
    return (weights.T @ covmat @ weights) ** 0.5


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix.
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix.
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1,
    }

    def neg_sharpe(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    result = minimize(
        neg_sharpe,
        init_guess,
        args=(riskfree_rate, er, cov),
        method='SLSQP',
        options={'disp': False},
        constraints=(weights_sum_to_1,),
        bounds=bounds,
    )
    return result.x


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


def rebalance_portfolio():
    api_key, api_secret = get_api_credentials()
    base_url = get_base_url()
    alpaca = tradeapi.REST(api_key, api_secret, base_url, 'v2')

    # Stop here if the market is closed
    clock = alpaca.get_clock()
    if not clock.is_open:
        logger.info("Stock market is closed today.")
        return 'Stock market is closed today.'

    sp100_data = pd.read_html(TICKER_SOURCE_URL)
    tickers = sp100_data[TICKER_TABLE_INDEX][TICKER_COLUMN_NAME].tolist()

    est = tz.timezone(TIMEZONE)
    end_date = datetime.datetime.now(est).date()
    start_date = end_date - relativedelta(years=LOOKBACK_YEARS)
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date_str = start_date.strftime('%Y-%m-%d')

    for dup in DUPLICATE_TICKERS:
        if dup in tickers:
            tickers.remove(dup)
    tickers_str = ' '.join(tickers)

    # Download OHLC data for the lookback period
    data = yf.download(tickers_str, start=start_date_str, end=end_date_str)

    rets = data['Adj Close'].pct_change()
    rets.dropna(axis=1, inplace=True, how='all')
    rets.dropna(axis=0, inplace=True, how='all')

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

    # Get latest prices via snapshots (replaces deprecated get_barset)
    try:
        snapshots = alpaca.get_snapshots(stocks)
    except APIError as e:
        logger.error("Failed to get snapshots: %s", e)
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
                current_qty = positions_df[positions_df['Symbol'] == stock]['Qty'].iloc[0]
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
    logging.basicConfig(level=logging.INFO)
    return rebalance_portfolio()
