import os
import datetime
import pytz as tz
import numpy as np
import pandas as pd
import yfinance as yf
import alpaca_trade_api as tradeapi
from dateutil.relativedelta import relativedelta


def calculate_momentum(prices, lookback_months=12, skip_months=1):
    """
    Calculate momentum scores for each stock.
    Uses the standard 12-1 momentum factor: 12-month return skipping the most recent month.
    """
    end = len(prices) - 1
    skip_start = end - skip_months * 21  # approx 21 trading days per month
    lookback_start = end - lookback_months * 21

    if lookback_start < 0:
        lookback_start = 0

    recent_prices = prices.iloc[skip_start]
    past_prices = prices.iloc[lookback_start]

    momentum = (recent_prices - past_prices) / past_prices
    return momentum


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

    # Number of top momentum stocks to hold
    top_n = 20

    sp100_data = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
    tickers = sp100_data[2]['Symbol'].tolist()

    EST = tz.timezone('US/Eastern')
    end_date = datetime.datetime.now(EST).date()
    start_date = end_date - relativedelta(months=14)

    end_date = end_date.strftime('%Y-%m-%d')
    start_date = start_date.strftime('%Y-%m-%d')

    if 'GOOGL' in tickers:
        tickers.remove('GOOGL')
    tickers_str = ' '.join(tickers)

    data = yf.download(tickers_str, start=start_date, end=end_date)
    prices = data['Adj Close'].dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Calculate momentum scores and pick top N stocks
    momentum_scores = calculate_momentum(prices)
    momentum_scores = momentum_scores.dropna().sort_values(ascending=False)
    selected_stocks = momentum_scores.head(top_n).index.tolist()

    # Equal-weight the top momentum stocks
    weights = np.array([1.0 / top_n] * top_n)

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

    # Close positions in stocks no longer in the momentum portfolio
    for _, row in positions_df.iterrows():
        if row['Symbol'] not in selected_stocks:
            try:
                alpaca.close_position(row['Symbol'])
                print(f"Closing position in {row['Symbol']} (no longer in top momentum)")
            except:
                print(f"Failed to close position in {row['Symbol']}")

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
                    f'Target dollar amount: ${round(price * qty, 2)}'
                )
                orders['Side'].append('Buy')
                orders['Ticker'].append(stock)
                orders['Qty'].append(qty)

            elif target_qty < current_qty:
                qty = current_qty - target_qty
                print(
                    f'Go from {current_qty} shares to {target_qty} shares of {stock}. '
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
