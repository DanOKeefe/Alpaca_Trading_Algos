import os
import time
import datetime
import pytz as tz
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import alpaca_trade_api as tradeapi
from dateutil.relativedelta import relativedelta

def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights
    weights are a numpy array or N x 1 maxtrix and covmat is an N x N matrix
    """
    vol = (weights.T @ covmat @ weights)**0.5
    return vol 

def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)

def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio
        of the given portfolio
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

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

    # Retrieve the current date in the EST timezone
    EST = tz.timezone('US/Eastern')
    end_date = datetime.datetime.now(EST).date()
    start_date = end_date - relativedelta(years=5)

    # Convert to string with format YYYY-MM-DD
    end_date = end_date.strftime('%Y-%m-%d')
    start_date = start_date.strftime('%Y-%m-%d')

    # The S&P 100 includes 101 tickers because of the A and C shares of Google.
    # Remove GOOGL and keep GOOG.
    if 'GOOGL' in tickers: tickers.remove('GOOGL')
    tickers = ' '.join(tickers)

    # Download OHLC data for the past year
    data = yf.download(tickers, start=start_date, end=end_date)

    rets = data['Adj Close'].pct_change()
    # first drop columns where all of the values are NA
    rets.dropna(axis=1, inplace=True, how='all')
    # then drop the first row (where all the values are NA)
    rets.dropna(axis=0, inplace=True, how='all')

    weights = np.round(gmv(rets.cov()), 5)
    portfolio_value = int(float(alpaca.get_account().portfolio_value))
    
    # Calculate the dollar amount I want to buy of each stock
    dollar_amounts = weights*portfolio_value

    # Cancel existing open orders
    orders = alpaca.list_orders(status='open')
    for order in orders:
        alpaca.cancel_order(order.id)

    stocks = list(data['Adj Close'].columns)
    assets = alpaca.list_assets()
    tradable_symbols = [a.symbol for a in assets if a.tradable and a.status == 'active']
    stocks = [s for s in stocks if s in tradable_symbols]

    positions = alpaca.list_positions()
    positions_df = pd.DataFrame({
        'Symbol': [p.symbol for p in positions],
        'Qty': [int(p.qty) for p in positions]
    })

    if 'GOOGL' in stocks: stocks.remove('GOOGL')
    price_df = alpaca.get_barset(stocks, 'minute', 1).df

    orders = {
        'Side': [],
        'Ticker': [],
        'Qty': []
    }

    for stock, target_value in zip(stocks, dollar_amounts):
        try:
            # Calculate my target quantity
            price = price_df[stock].dropna().close[0]
            # print(f'Price of {stock} is {price}')
            target_qty = int(target_value//price)
            # Compare how many shares I own to my target quantity
            # Retrieve my current position
            if stock in positions_df['Symbol'].tolist():
                current_qty = positions_df[positions_df['Symbol']==stock]['Qty'].iloc[0]
            else:
                current_qty = 0
            # print(f'Target quantity of {stock} is {target_qty}. Target value is {target_value}.')
            if target_qty > current_qty:
                # I want to increase my position. Calculate how many shares to buy and
                # round down to the nearest integer value
                if stock in tradable_symbols:
                    qty = target_qty - current_qty
                    print(
                        f'Go from {current_qty} shares to {target_qty} shares of {stock}. '
                        f'Target dollar amount: ${round(price*(qty), 2)}'
                    )
                    orders['Side'].append('Buy')
                    orders['Ticker'].append(stock)
                    orders['Qty'].append(qty)


            elif target_qty < current_qty:
                # I want to decrease my position. Calculate how many shares to sell and
                # round up to the nearest integer value
                if stock in tradable_symbols:
                    qty = current_qty - target_qty
                    print(
                        f'Go from {current_qty} shares to {target_qty} shares of {stock}. '
                        f'Target dollar amount: ${round(price*(qty), 2)}'
                    )
                    orders['Side'].append('Sell')
                    orders['Ticker'].append(stock)
                    orders['Qty'].append(qty)

            else:
                #print(f'Staying with {current_qty} shares of {stock}')
                pass
        except:
            print(f'Could not pull stock data for {stock} from Alpaca')

    orders_df = pd.DataFrame(orders)

    # Submit all the sell orders first.
    sell_df = orders_df[orders_df['Side']=='Sell']
    for i, row in sell_df.iterrows():
        submitOrder(api=alpaca, qty=row['Qty'], stock=row['Ticker'], side='sell')

    # Submit all the buy orders
    buy_df = orders_df[orders_df['Side']=='Buy']
    for i, row in buy_df.iterrows():
        submitOrder(api=alpaca, qty=row['Qty'], stock=row['Ticker'], side='buy')
        
    return orders_df.to_json()
    
def lambda_handler(event, context):
    return rebalance_portfolio()
