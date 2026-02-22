# CLAUDE.md

## Project Overview

Alpaca Trading Algos — trading algorithms that interface with the Alpaca brokerage API, designed for deployment on AWS Lambda.

## Structure

```
gmv_algo.py                          # Legacy single-file script (kept for reference)
alpaca_trading/                      # Main package
├── __init__.py
├── config.py                        # Environment variables, constants
├── client.py                        # Alpaca API client factory
├── data.py                          # Ticker scraping + historical data download
├── optimization.py                  # portfolio_return, portfolio_vol, gmv, msr, clip_weights
├── execution.py                     # Order building and submission
├── lambda_handler.py                # AWS Lambda entry point
└── strategies/
    ├── __init__.py
    ├── base.py                      # Abstract Strategy interface
    └── gmv.py                       # Global Minimum Variance strategy
tests/                               # Unit tests (pytest)
├── test_optimization.py
├── test_data.py
├── test_execution.py
└── test_strategies.py
```

- Entry point: `alpaca_trading.lambda_handler.lambda_handler(event, context)` for AWS Lambda
- Strategy is selectable via the Lambda event payload `{"strategy": "gmv"}`

## Dependencies

Python 3.9+ — see `requirements.txt` (runtime) and `requirements-dev.txt` (dev/test).

Key runtime packages:
- `alpaca_trade_api` — Alpaca brokerage API client
- `yfinance` — Yahoo Finance historical data
- `numpy`, `pandas`, `scipy` — numerical computation and optimization
- `pytz`, `python-dateutil` — timezone and date handling

Dev packages: `pytest`, `pytest-cov`, `ruff`, `black`

## Environment Variables

- `API_KEY` — Alpaca API key
- `API_SECRET` — Alpaca API secret
- `LIVE_TRADING` — set to `"true"` to use live trading API (default: paper trading)

## How It Works

1. Connects to the Alpaca paper trading API
2. Scrapes S&P 100 tickers from Wikipedia
3. Downloads 5 years of historical price data via yfinance
4. Computes portfolio weights using the selected strategy (default: Global Minimum Variance)
5. Applies position sizing limits (configurable max weight per stock)
6. Compares target positions against current holdings
7. Submits sell orders first, then buy orders to rebalance

## Testing

Run tests with: `pytest tests/ -v`

31 unit tests covering optimization math, data retrieval (mocked), order execution, and strategy behavior.

## Linting / Formatting

- Lint: `ruff check alpaca_trading/ tests/`
- Format: `black alpaca_trading/ tests/`
- Config in `pyproject.toml`

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint + tests on push/PR to main.
