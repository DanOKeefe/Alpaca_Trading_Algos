# Alpaca Trading Algos

Automated portfolio rebalancing algorithms that trade through the [Alpaca](https://alpaca.markets/) brokerage API, designed to run as scheduled AWS Lambda functions.

## What's in here

### `gmv_algo.py` -- Global Minimum Variance Portfolio

This algorithm rebalances your portfolio daily to track the [Global Minimum Variance (GMV)](https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix-BEAMER.pdf) portfolio of the S&P 100.

**How it works:**

1. Scrapes the current S&P 100 ticker list from Wikipedia
2. Downloads 5 years of historical price data from Yahoo Finance
3. Computes a covariance matrix from daily returns
4. Solves for the portfolio weights that minimize overall volatility (long-only, fully invested)
5. Compares the target allocation against your current Alpaca holdings
6. Sells overweight positions first (to free up cash), then buys underweight positions

The optimization uses SciPy's SLSQP solver to minimize portfolio volatility subject to weights summing to 1 and no short selling.

## Prerequisites

- Python 3.7+
- An [Alpaca](https://alpaca.markets/) brokerage account (paper or live)
- AWS account (if deploying to Lambda)

## Dependencies

```
alpaca-trade-api
yfinance
pandas
numpy
scipy
pytz
python-dateutil
```

Install them with:

```bash
pip install alpaca-trade-api yfinance pandas numpy scipy pytz python-dateutil
```

## Configuration

The algorithm reads credentials from environment variables:

| Variable | Description |
|---|---|
| `API_KEY` | Your Alpaca API key |
| `API_SECRET` | Your Alpaca API secret key |

Set them locally:

```bash
export API_KEY="your-alpaca-api-key"
export API_SECRET="your-alpaca-secret-key"
```

Or configure them in your AWS Lambda function's environment variables.

> **Note:** The script currently points at Alpaca's **paper trading** endpoint (`https://paper-api.alpaca.markets`). Change the `APCA_API_BASE_URL` in `rebalance_portfolio()` to `https://api.alpaca.markets` for live trading.

## Usage

### Run locally

```bash
python -c "from gmv_algo import rebalance_portfolio; rebalance_portfolio()"
```

The function checks whether the market is open before doing anything. If the market is closed, it exits early.

### Deploy to AWS Lambda

The script exposes a `lambda_handler(event, context)` entry point. Package the code with its dependencies and deploy it as a Lambda function, then trigger it on a schedule with EventBridge (e.g., weekdays at market open).

## How the math works

The Global Minimum Variance portfolio finds the set of weights **w** that minimizes:

```
portfolio volatility = sqrt(w' * Cov * w)
```

subject to:
- All weights sum to 1
- Each weight is between 0 and 1 (no shorting)

This gives you the least volatile portfolio possible from the given universe of stocks. The algorithm uses equal expected returns so the optimization reduces to pure risk minimization rather than a return/risk tradeoff.

## Limitations

- Uses market orders, so fills may differ from the prices used to calculate target quantities
- Integer share quantities only (fractional shares not supported)
- The S&P 100 ticker list is scraped from Wikipedia, which may occasionally be out of date
- Bare `except` clauses silence errors for individual stocks -- check your logs
- The `get_barset` API call used for current prices is deprecated in newer versions of the Alpaca SDK
