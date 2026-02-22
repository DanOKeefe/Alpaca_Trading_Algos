# Implementation Plan — Alpaca Trading Algos

## Phase 1: Foundation (Infrastructure & Code Quality)

### 1.1 Add `requirements.txt`
- Pin all dependencies with versions
- Add `requirements-dev.txt` for test/lint tooling

### 1.2 Refactor `gmv_algo.py` into a package
Reorganize the single-file script into a modular structure:
```
alpaca_trading/
├── __init__.py
├── config.py            # env vars, constants, API base URLs
├── client.py            # Alpaca API connection factory
├── data.py              # ticker scraping + historical data download
├── optimization.py      # portfolio_return, portfolio_vol, gmv, msr
├── execution.py         # submitOrder, order-building logic
├── strategies/
│   ├── __init__.py
│   ├── base.py          # abstract Strategy interface
│   └── gmv.py           # GMV strategy (extracted from rebalance_portfolio)
└── lambda_handler.py    # AWS Lambda entry point
```

### 1.3 Improve error handling & logging
- Replace `print()` with Python `logging` module
- Replace bare `except:` with specific exception types
- Add meaningful error messages

### 1.4 Add linting & formatting
- Add `pyproject.toml` with `ruff` config
- Format existing code with `black`

### 1.5 Add unit tests
- `tests/test_optimization.py` — test `portfolio_return`, `portfolio_vol`, `gmv`, `msr` with known covariance matrices and expected outputs
- `tests/test_data.py` — test ticker scraping with a mocked HTML response
- `tests/test_execution.py` — test order building logic with mocked Alpaca API

### 1.6 Add CI/CD
- GitHub Actions workflow: lint, test on push

---

## Phase 2: Risk & Robustness

### 2.1 Position sizing limits
- Add configurable max weight per stock (e.g., 10%)
- Clip weights and redistribute excess proportionally

### 2.2 Rebalancing threshold
- Only trade when a position drifts beyond a configurable tolerance band (e.g., 2%)
- Avoids unnecessary small trades and transaction costs

### 2.3 Transaction cost awareness
- Estimate trade cost (commission + spread) for each order
- Skip orders where cost exceeds rebalancing benefit

### 2.4 Notifications
- Post-rebalance summary via SNS or email
- Include: trades executed, new weights, portfolio value

---

## Phase 3: New Strategies

### 3.1 Maximum Sharpe Ratio strategy
- Wire up existing `msr()` function as a selectable strategy
- Accept expected returns as input (e.g., historical mean returns)

### 3.2 Equal Weight strategy
- Simple 1/N allocation as a baseline

### 3.3 Risk Parity strategy
- Allocate so each asset contributes equally to total portfolio risk

### 3.4 Strategy selection via config
- Choose strategy via Lambda event payload or environment variable
- All strategies implement the common `Strategy` interface from Phase 1

---

## Phase 4: Data & API Modernization

### 4.1 Migrate to `alpaca-py`
- Replace deprecated `alpaca_trade_api` with official `alpaca-py` SDK

### 4.2 Cache historical data
- Store downloaded price data in S3
- Only fetch incremental data on subsequent runs

### 4.3 Configurable stock universe
- Support S&P 500, custom watchlists, or ETFs
- Move Wikipedia scraping behind an interface so sources are swappable

---

## Phase 5: Analytics & Backtesting

### 5.1 Performance reporting
- Track daily returns, Sharpe ratio, max drawdown
- Store results in S3 or DynamoDB

### 5.2 Backtesting framework
- Simulate any strategy against historical data
- Compare strategy performance before going live

---

## Implementation Order

We'll work through these phases sequentially. Within Phase 1, the order is:
1. `requirements.txt` (quick win, unblocks everything)
2. Refactor into package (enables all future work)
3. Error handling & logging (improves debugging during development)
4. Linting & formatting (clean up refactored code)
5. Unit tests (lock in correctness of refactored code)
6. CI/CD (automate quality checks going forward)

Then Phases 2–5 build on the clean foundation.
