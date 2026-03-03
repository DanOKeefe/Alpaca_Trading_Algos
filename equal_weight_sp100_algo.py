import os
import datetime
import pytz as tz
import numpy as np
import pandas as pd
import alpaca_trade_api as tradeapi


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

    sp100_data = pd.read_html('https://en.wikipedia.org/wiki/S%26P_100')
    tickers = sp100_data[2]['Symbol'].tolist()

    if 'GOOGL' in tickers:
        tickers.remove('GOOGL')

    # Filter to tradable symbols
    assets = alpaca.list_assets()
    tradable_symbols = [a.symbol for a in assets if a.tradable and a.status == 'active']
    tickers = [t for t in tickers if t in tradable_symbols]

    n = len(tickers)
    weights = np.array([1.0 / n] * n)

    portfolio_value = int(float(alpaca.get_account().portfolio_value))
    dollar_amounts = weights * portfolio_value

    # Cancel existing open orders
    orders = alpaca.list_orders(status='open')
    for order in orders:
        alpaca.cancel_order(order.id)

    positions = alpaca.list_positions()
    positions_df = pd.DataFrame({
        'Symbol': [p.symbol for p in positions],
        'Qty': [int(p.qty) for p in positions]
    })

    # Close positions not in the S&P 100
    for _, row in positions_df.iterrows():
        if row['Symbol'] not in tickers:
            try:
                alpaca.close_position(row['Symbol'])
                print(f"Closing position in {row['Symbol']} (no longer in S&P 100)")
            except:
                print(f"Failed to close position in {row['Symbol']}")

    price_df = alpaca.get_barset(tickers, 'minute', 1).df

    orders = {
        'Side': [],
        'Ticker': [],
        'Qty': []
    }

    for stock, target_value in zip(tickers, dollar_amounts):
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
