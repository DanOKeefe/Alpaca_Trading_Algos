# Alpaca Trading Algos

Trading algorithms that interface with the [Alpaca](https://alpaca.markets/) brokerage API and can be deployed as scheduled jobs using AWS Lambda.

## Overview

This project implements a **Global Minimum Variance (GMV)** portfolio strategy that automatically rebalances your portfolio across S&P 100 stocks. The algorithm:

1. Pulls the current S&P 100 constituents from Wikipedia
2. Downloads 5 years of historical price data via Yahoo Finance
3. Computes the covariance matrix of returns
4. Solves for the Global Minimum Variance portfolio weights using constrained optimization
5. Compares target allocations against your current Alpaca holdings
6. Submits sell orders first (to free up capital), then buy orders to rebalance

### What is Global Minimum Variance?

The GMV portfolio is the portfolio on the [efficient frontier](https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix-BEAMER.pdf) with the lowest possible volatility. Unlike mean-variance optimization, it only depends on the covariance matrix of returns (not expected returns), which makes it more robust since covariance estimates tend to be more stable than return estimates.

The optimization problem solved is:

```
minimize    w^T * Sigma * w
subject to  sum(w) = 1
            0 <= w_i <= 1  for all i
```

where `w` is the weight vector and `Sigma` is the covariance matrix.

## Project Structure

```
Alpaca_Trading_Algos/
├── gmv_algo.py        # GMV portfolio rebalancing algorithm
├── requirements.txt   # Python dependencies
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Prerequisites

- Python 3.8+
- An [Alpaca](https://alpaca.markets/) brokerage account (paper or live)
- Your Alpaca API key and secret

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/DanOKeefe/Alpaca_Trading_Algos.git
cd Alpaca_Trading_Algos
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

The algorithm requires the following environment variables:

| Variable      | Description                        |
|---------------|------------------------------------|
| `API_KEY`     | Your Alpaca API key                |
| `API_SECRET`  | Your Alpaca API secret key         |

```bash
export API_KEY="your-alpaca-api-key"
export API_SECRET="your-alpaca-api-secret"
```

> **Note:** The algorithm is configured to use the Alpaca **paper trading** endpoint (`https://paper-api.alpaca.markets`) by default. Modify `APCA_API_BASE_URL` in `gmv_algo.py` to switch to live trading.

## Usage

### Run locally

```bash
python -c "from gmv_algo import rebalance_portfolio; rebalance_portfolio()"
```

### Deploy to AWS Lambda

The script includes a `lambda_handler` function, making it ready for AWS Lambda deployment.

1. **Package the dependencies:**

```bash
mkdir package
pip install -r requirements.txt -t package/
cp gmv_algo.py package/
cd package && zip -r ../lambda_function.zip . && cd ..
```

2. **Create the Lambda function:**
   - Runtime: Python 3.8+
   - Handler: `gmv_algo.lambda_handler`
   - Timeout: 5 minutes (data download can be slow)
   - Memory: 512 MB minimum

3. **Set environment variables** in the Lambda console:
   - `API_KEY` — your Alpaca API key
   - `API_SECRET` — your Alpaca API secret

4. **Schedule with EventBridge (CloudWatch Events):**
   - Create a rule with a cron expression to run on market days, e.g.:
   - `cron(0 14 ? * MON-FRI *)` — runs at 9:00 AM ET (14:00 UTC) every weekday

## Algorithm Details

### gmv_algo.py

**Key functions:**

| Function | Description |
|---|---|
| `portfolio_return(weights, returns)` | Computes portfolio return as `w^T * r` |
| `portfolio_vol(weights, covmat)` | Computes portfolio volatility as `sqrt(w^T * Sigma * w)` |
| `gmv(cov)` | Returns GMV portfolio weights by calling `msr` with equal expected returns |
| `msr(riskfree_rate, er, cov)` | Solves for maximum Sharpe ratio weights via `scipy.optimize.minimize` (SLSQP) |
| `submitOrder(api, qty, stock, side)` | Submits a market order through the Alpaca API |
| `rebalance_portfolio()` | Main orchestration — fetches data, computes weights, and executes trades |
| `lambda_handler(event, context)` | AWS Lambda entry point |

**Execution flow:**

1. Connect to Alpaca API using environment credentials
2. Check if the market is open; exit early if closed
3. Scrape S&P 100 tickers from Wikipedia (`pd.read_html`)
4. Download 5 years of adjusted close prices from Yahoo Finance
5. Compute daily returns and the covariance matrix
6. Solve the GMV optimization for target weights
7. Convert weights to target share quantities based on portfolio value
8. Cancel any existing open orders
9. Compare current positions to target positions
10. Submit sell orders first, then buy orders

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Matrix operations for portfolio math |
| `pandas` | DataFrames for price/return data |
| `yfinance` | Historical stock price data from Yahoo Finance |
| `scipy` | Constrained optimization (SLSQP solver) |
| `alpaca-trade-api` | Alpaca brokerage API client |
| `pytz` | Timezone handling (EST for market hours) |
| `python-dateutil` | Relative date calculations |
| `lxml` | HTML parsing backend for `pd.read_html` |

## Disclaimer

This software is for **educational purposes only**. It is not financial advice. Use at your own risk. The default configuration uses Alpaca's paper trading API. Always test thoroughly before using real money.
