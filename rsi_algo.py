import os
import datetime
import pytz as tz
import numpy as np
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from dateutil.relativedelta import relativedelta


def calculate_rsi(prices, period=14):
    """
    Calculate the Relative Strength Index (RSI) for each stock.
    RSI ranges from 0 to 100:
      - Below 30: oversold (buy signal)
      - Above 70: overbought (sell signal)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


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

    # RSI thresholds
    oversold_threshold = 30
    overbought_threshold = 70
    # Max number of positions to hold
    max_positions = 20

    sp100_data = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
    tickers = sp100_data[2]['Symbol'].tolist()

    EST = tz.timezone('US/Eastern')
    end_date = datetime.datetime.now(EST).date()
    start_date = end_date - relativedelta(months=3)

    end_date = end_date.strftime('%Y-%m-%d')
    start_date = start_date.strftime('%Y-%m-%d')

    if 'GOOGL' in tickers:
        tickers.remove('GOOGL')
    tickers_str = ' '.join(tickers)

    data = yf.download(tickers_str, start=start_date, end=end_date)
    prices = data['Adj Close'].dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Calculate RSI for each stock
    rsi_values = calculate_rsi(prices)
    rsi_values = rsi_values.dropna()

    # Identify oversold stocks to buy
    oversold_stocks = rsi_values[rsi_values < oversold_threshold].sort_values(ascending=True)
    stocks_to_buy = oversold_stocks.head(max_positions).index.tolist()

    # Identify overbought stocks to sell (if we hold them)
    overbought_stocks = rsi_values[rsi_values > overbought_threshold].index.tolist()

    portfolio_value = int(float(alpaca.get_account().portfolio_value))

    # Cancel existing open orders
    orders = alpaca.list_orders(status='open')
    for order in orders:
        alpaca.cancel_order(order.id)

    assets = alpaca.list_assets()
    tradable_symbols = [a.symbol for a in assets if a.tradable and a.status == 'active']
    stocks_to_buy = [s for s in stocks_to_buy if s in tradable_symbols]

    positions = alpaca.list_positions()
    positions_df = pd.DataFrame({
        'Symbol': [p.symbol for p in positions],
        'Qty': [int(p.qty) for p in positions]
    })

    # Close positions in overbought stocks
    for _, row in positions_df.iterrows():
        if row['Symbol'] in overbought_stocks:
            try:
                alpaca.close_position(row['Symbol'])
                print(f"Closing position in {row['Symbol']} (RSI: {rsi_values.get(row['Symbol'], 'N/A'):.1f}, overbought)")
            except:
                print(f"Failed to close position in {row['Symbol']}")

    if len(stocks_to_buy) == 0:
        print('No oversold stocks found. No new buy orders.')
        return '{}'

    # Equal-weight allocation across oversold stocks
    n = len(stocks_to_buy)
    weights = np.array([1.0 / n] * n)
    dollar_amounts = weights * portfolio_value

    price_df = alpaca.get_barset(stocks_to_buy, 'minute', 1).df

    orders = {
        'Side': [],
        'Ticker': [],
        'Qty': []
    }

    for stock, target_value in zip(stocks_to_buy, dollar_amounts):
        try:
            price = price_df[stock].dropna().close[0]
            target_qty = int(target_value // price)

            if stock in positions_df['Symbol'].tolist():
                current_qty = positions_df[positions_df['Symbol'] == stock]['Qty'].iloc[0]
            else:
                current_qty = 0

            if target_qty > current_qty:
                qty = target_qty - current_qty
                rsi_val = rsi_values.get(stock, 0)
                print(
                    f'Go from {current_qty} shares to {target_qty} shares of {stock}. '
                    f'RSI: {rsi_val:.1f}. '
                    f'Target dollar amount: ${round(price * qty, 2)}'
                )
                orders['Side'].append('Buy')
                orders['Ticker'].append(stock)
                orders['Qty'].append(qty)

            elif target_qty < current_qty:
                qty = current_qty - target_qty
                rsi_val = rsi_values.get(stock, 0)
                print(
                    f'Go from {current_qty} shares to {target_qty} shares of {stock}. '
                    f'RSI: {rsi_val:.1f}. '
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
