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
├── data.py                          # Ticker scraping, historical data, S3 caching
├── optimization.py                  # portfolio_return, portfolio_vol, gmv, msr, clip_weights
├── execution.py                     # Order building, threshold filtering, submission
├── notifications.py                 # Post-rebalance SNS notifications
├── lambda_handler.py                # AWS Lambda entry point
└── strategies/
    ├── __init__.py
    ├── base.py                      # Abstract Strategy interface
    ├── gmv.py                       # Global Minimum Variance
    ├── msr.py                       # Maximum Sharpe Ratio
    ├── equal_weight.py              # Equal Weight (1/N)
    └── risk_parity.py               # Risk Parity (inverse vol)
tests/                               # Unit tests (pytest)
├── test_optimization.py
├── test_data.py
├── test_execution.py
├── test_notifications.py
└── test_strategies.py
```

- Entry point: `alpaca_trading.lambda_handler.lambda_handler(event, context)` for AWS Lambda
- Strategy and universe are selectable via the Lambda event payload:
  ```json
  {"strategy": "gmv", "universe": "sp100"}
  ```

## Strategies

| Key            | Class                | Description                              |
|----------------|----------------------|------------------------------------------|
| `gmv`          | `GMVStrategy`        | Global Minimum Variance (default)        |
| `msr`          | `MSRStrategy`        | Maximum Sharpe Ratio                     |
| `equal_weight` | `EqualWeightStrategy`| Equal Weight (1/N)                       |
| `risk_parity`  | `RiskParityStrategy` | Risk Parity (inverse volatility)         |

## Dependencies

Python 3.9+ — see `requirements.txt` (runtime) and `requirements-dev.txt` (dev/test).

Key runtime packages:
- `alpaca_trade_api` — Alpaca brokerage API client
- `yfinance` — Yahoo Finance historical data
- `numpy`, `pandas`, `scipy` — numerical computation and optimization
- `pytz`, `python-dateutil` — timezone and date handling

Optional: `boto3` (for SNS notifications and S3 data caching)

Dev packages: `pytest`, `pytest-cov`, `ruff`, `black`

## Environment Variables

- `API_KEY` — Alpaca API key
- `API_SECRET` — Alpaca API secret
- `LIVE_TRADING` — set to `"true"` to use live trading API (default: paper trading)
- `STOCK_UNIVERSE` — `"sp100"` (default), `"sp500"`, or comma-separated tickers (e.g. `"AAPL,MSFT,GOOG"`)
- `S3_CACHE_BUCKET` — optional; S3 bucket name for caching historical price data
- `SNS_TOPIC_ARN` — optional; SNS topic ARN for post-rebalance notifications

## How It Works

1. Connects to the Alpaca paper (or live) trading API
2. Gets tickers from configured universe (S&P 100, S&P 500, or custom list)
3. Downloads 5 years of historical price data via yfinance (with optional S3 caching)
4. Computes portfolio weights using the selected strategy
5. Applies position sizing limits (configurable max weight per stock)
6. Skips trades below the rebalance threshold or where transaction cost exceeds benefit
7. Compares target positions against current holdings
8. Submits sell orders first, then buy orders to rebalance
9. Sends a summary notification via SNS (if configured)

## Testing

Run tests with: `pytest tests/ -v`

77 unit tests covering optimization math, data retrieval (mocked), order execution, threshold logic, notifications, and all four strategy implementations.

## Linting / Formatting

- Lint: `ruff check alpaca_trading/ tests/`
- Format: `black alpaca_trading/ tests/`
- Config in `pyproject.toml`

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs lint + tests on push/PR to main.
