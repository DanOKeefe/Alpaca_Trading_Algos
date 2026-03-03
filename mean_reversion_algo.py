import os
import datetime
import pytz as tz
import numpy as np
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from dateutil.relativedelta import relativedelta


def calculate_z_scores(prices, lookback_days=60):
    """
    Calculate z-scores for each stock based on how far the current price
    deviates from its rolling mean over the lookback period.
    Negative z-scores indicate the stock is trading below its mean (buy signal).
    """
    rolling_mean = prices.rolling(window=lookback_days).mean()
    rolling_std = prices.rolling(window=lookback_days).std()
    z_scores = (prices.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
    return z_scores


def submitOrder(api, qty, stock, side):
    if qty > 0:
        try:
            api.submit_order(stock, qty, side, 'market', 'day')
            print('Submitted order to ' + side + ' ' + str(qty) + ' shares(s) of ' + stock)
        except:
            print('Order failed to submit: ' + side + ' of ' + str(qty) + ' share(s) of ' + stock)
    else:
        print('Quantity for ' + stock + ' is zero b/c dollar_amount < share_price')

    return None


def rebalance_portfolio():
    API_KEY = os.environ['API_KEY']
    API_SECRET = os.environ['API_SECRET']
    APCA_API_BASE_URL = "https://paper-api.alpaca.markets"
    alpaca = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')

    # Stop here if the market is closed
    clock = alpaca.get_clock()
    if clock.is_open == False:
        return 'Stock market is closed today.'

    # Number of oversold stocks to buy
    top_n = 20
    # Z-score threshold: buy stocks with z-score below this value
    z_threshold = -1.0

    sp100_data = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
    tickers = sp100_data[2]['Symbol'].tolist()

    EST = tz.timezone('US/Eastern')
    end_date = datetime.datetime.now(EST).date()
    start_date = end_date - relativedelta(months=6)

    end_date = end_date.strftime('%Y-%m-%d')
    start_date = start_date.strftime('%Y-%m-%d')

    if 'GOOGL' in tickers:
        tickers.remove('GOOGL')
    tickers_str = ' '.join(tickers)

    data = yf.download(tickers_str, start=start_date, end=end_date)
    prices = data['Adj Close'].dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Calculate z-scores
    z_scores = calculate_z_scores(prices)
    z_scores = z_scores.dropna().sort_values(ascending=True)

    # Select stocks with z-score below the threshold (oversold)
    oversold = z_scores[z_scores < z_threshold]
    selected_stocks = oversold.head(top_n).index.tolist()

    if len(selected_stocks) == 0:
        print('No stocks meet the mean reversion criteria. No trades to make.')
        return '{}'

    # Equal-weight the selected oversold stocks
    n = len(selected_stocks)
    weights = np.array([1.0 / n] * n)

    portfolio_value = int(float(alpaca.get_account().portfolio_value))
    dollar_amounts = weights * portfolio_value

    # Cancel existing open orders
    orders = alpaca.list_orders(status='open')
    for order in orders:
        alpaca.cancel_order(order.id)

    assets = alpaca.list_assets()
    tradable_symbols = [a.symbol for a in assets if a.tradable and a.status == 'active']
    selected_stocks = [s for s in selected_stocks if s in tradable_symbols]

    positions = alpaca.list_positions()
    positions_df = pd.DataFrame({
        'Symbol': [p.symbol for p in positions],
        'Qty': [int(p.qty) for p in positions]
    })

    # Close positions in stocks that have reverted to mean (z-score above 0)
    for _, row in positions_df.iterrows():
        symbol = row['Symbol']
        if symbol in z_scores.index and z_scores[symbol] > 0:
            try:
                alpaca.close_position(symbol)
                print(f"Closing position in {symbol} (z-score reverted to {z_scores[symbol]:.2f})")
            except:
                print(f"Failed to close position in {symbol}")

    price_df = alpaca.get_barset(selected_stocks, 'minute', 1).df

    orders = {
        'Side': [],
        'Ticker': [],
        'Qty': []
    }

    for stock, target_value in zip(selected_stocks, dollar_amounts):
        try:
            price = price_df[stock].dropna().close[0]
            target_qty = int(target_value // price)

            if stock in positions_df['Symbol'].tolist():
                current_qty = positions_df[positions_df['Symbol'] == stock]['Qty'].iloc[0]
            else:
                current_qty = 0

            if target_qty > current_qty:
                qty = target_qty - current_qty
                print(
                    f'Go from {current_qty} shares to {target_qty} shares of {stock}. '
                    f'Z-score: {z_scores[stock]:.2f}. '
                    f'Target dollar amount: ${round(price * qty, 2)}'
                )
                orders['Side'].append('Buy')
                orders['Ticker'].append(stock)
                orders['Qty'].append(qty)

            elif target_qty < current_qty:
                qty = current_qty - target_qty
                print(
                    f'Go from {current_qty} shares to {target_qty} shares of {stock}. '
                    f'Z-score: {z_scores[stock]:.2f}. '
                    f'Target dollar amount: ${round(price * qty, 2)}'
                )
                orders['Side'].append('Sell')
                orders['Ticker'].append(stock)
                orders['Qty'].append(qty)
        except:
            print(f'Could not pull stock data for {stock} from Alpaca')

    orders_df = pd.DataFrame(orders)

    # Submit sell orders first
    sell_df = orders_df[orders_df['Side'] == 'Sell']
    for i, row in sell_df.iterrows():
        submitOrder(api=alpaca, qty=row['Qty'], stock=row['Ticker'], side='sell')

    # Then submit buy orders
    buy_df = orders_df[orders_df['Side'] == 'Buy']
    for i, row in buy_df.iterrows():
        submitOrder(api=alpaca, qty=row['Qty'], stock=row['Ticker'], side='buy')

    return orders_df.to_json()


def lambda_handler(event, context):
    return rebalance_portfolio()
