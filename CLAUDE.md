# CLAUDE.md

## Project Overview

Alpaca Trading Algos — trading algorithms that interface with the Alpaca brokerage API, designed for deployment on AWS Lambda.

## Structure

- `gmv_algo.py` — Global Minimum Variance portfolio rebalancing algorithm
- Entry point: `lambda_handler(event, context)` for AWS Lambda

## Dependencies

Python 3 with these third-party packages (no requirements.txt exists):
- `alpaca_trade_api` — Alpaca brokerage API client
- `yfinance` — Yahoo Finance historical data
- `numpy` — numerical computation
- `pandas` — data manipulation
- `scipy` — portfolio optimization (scipy.optimize.minimize)
- `pytz` — timezone handling
- `python-dateutil` — date utilities

## Environment Variables

- `API_KEY` — Alpaca API key
- `API_SECRET` — Alpaca API secret

## How It Works

1. Connects to the Alpaca paper trading API
2. Scrapes S&P 100 tickers from Wikipedia
3. Downloads 5 years of historical price data via yfinance
4. Computes Global Minimum Variance portfolio weights using scipy optimization
5. Compares target positions against current holdings
6. Submits sell orders first, then buy orders to rebalance

## Testing

No automated tests or test framework configured. No CI/CD pipeline.

## Linting / Formatting

No linting or formatting tools configured.
