# Development Roadmap — Alpaca Trading Algos

## Current State

The project has a single trading algorithm (`gmv_algo.py`) implementing a **Global Minimum Variance** portfolio strategy over the S&P 100. It's deployed via AWS Lambda and uses the Alpaca paper-trading API. There are no tests, no dependency manifest, no CI/CD, no backtesting framework, and no structured logging.

---

## Phase 1: Stabilize the Foundation

Goal: Make the existing code production-grade before adding new features.

### 1.1 — Dependency Management
- [x] Add `requirements.txt` pinning all dependencies (`alpaca-trade-api`, `yfinance`, `numpy`, `pandas`, `scipy`, `pytz`, `python-dateutil`)
- [x] Add `requirements-dev.txt` for linting/testing tools (`pytest`, `pytest-cov`, `ruff`, `mypy`)

### 1.2 — Fix Known Issues in `gmv_algo.py`
- [x] Replace bare `except:` clauses with specific exception types (`APIError`, `KeyError`, `IndexError`, `ZeroDivisionError`)
- [x] Migrate from deprecated `get_barset()` to `get_snapshots()` (Alpaca v2 data API)
- [x] Validate that required environment variables (`API_KEY`, `API_SECRET`) are set on startup, with clear error messages
- [x] Replace `print()` calls with Python `logging` module at appropriate levels (INFO, WARNING, ERROR)

### 1.3 — Configuration
- [x] Extract hardcoded values into `config.py` (base URL, ticker source, lookback period)
- [x] Support switching between paper and live trading via `TRADING_MODE` environment variable
- [x] Move magic numbers (e.g., rounding precision `5`, `relativedelta(years=5)`) into named constants

### 1.4 — Testing
- [x] Add unit tests for pure math functions (`portfolio_return`, `portfolio_vol`, `gmv`, `msr`)
- [x] Add integration tests with mocked Alpaca API responses for `rebalance_portfolio`
- [x] Add tests for `submit_order` function with mocked API
- [x] Achieved 89% code coverage (target was ≥80%)

---

## Phase 2: Project Structure & DevOps

Goal: Set up a professional project layout and automated quality gates.

### 2.1 — Project Restructure
- [x] Split monolithic `gmv_algo.py` into modular `src/` package:
  - `src/strategies/gmv.py` — portfolio math (portfolio_return, portfolio_vol, gmv, msr)
  - `src/execution/orders.py` — order submission logic
  - `src/data/market_data.py` — ticker fetching, price downloads, snapshots
  - `src/utils/config.py` — configuration constants and credentials
  - `src/utils/log_config.py` — logging setup
  - `src/lambda_handler.py` — rebalance orchestrator and Lambda entry point
- [x] Add `pyproject.toml` replacing `requirements.txt` / `requirements-dev.txt`
- [x] Update all tests (35 total, 95% coverage) for new import paths
- [x] Add `test_market_data.py` with tests for data-fetching functions
- [x] Add `backtests/` directory placeholder

### 2.2 — CI/CD Pipeline
- [x] GitHub Actions workflow (`.github/workflows/ci.yml`): lint (`ruff`), type-check (`mypy`), test (`pytest`) on every PR and push to `main`
- [ ] Automated deployment to AWS Lambda on merge to `main` (future: add deploy step to CI)

### 2.3 — Infrastructure as Code
- [x] AWS SAM template (`template.yaml`) defining Lambda function, CloudWatch Events cron schedule (daily at 9:35 AM ET, weekdays)
- [x] API credentials passed via `NoEcho` parameters (deploy with `sam deploy --parameter-overrides`)

---

## Phase 3: Backtesting Framework

Goal: Evaluate strategies on historical data before deploying with real capital.

### 3.1 — Core Backtesting Engine
- [x] `backtests/engine.py`: `Backtester` class replays historical data through any weight function and records simulated trades
- [x] `backtests/metrics.py`: total return, annualized return (CAGR), Sharpe ratio, max drawdown, annualized volatility
- [x] `BacktestConfig` supports configurable date ranges, initial capital, rebalance frequency, lookback period, and risk-free rate
- [x] 25 new tests (60 total) covering engine, metrics, and edge cases

### 3.2 — Backtest the GMV Strategy
- [x] `backtests/run_gmv_backtest.py`: runs GMV over 1-year, 3-year, 5-year, and 10-year windows (`python -m backtests.run_gmv_backtest`)
- [x] Equal-weight benchmark runs side by side for comparison
- [x] `backtests/report.py`: generates both text and HTML performance reports with comparison tables

### 3.3 — Visualization
- [x] `backtests/visualize.py`: equity curves, drawdown charts, and weight allocation stacked area plots (matplotlib, Agg backend)
- [ ] Optional: Jupyter notebook for interactive exploration (future enhancement)

---

## Phase 4: New Strategies

Goal: Expand beyond a single strategy to a multi-strategy platform.

### 4.1 — Strategy Interface
- [ ] Define a base `Strategy` class/protocol with methods like `calculate_weights(data) -> np.ndarray` and `get_tickers() -> list[str]`
- [ ] Refactor `gmv_algo.py` to conform to this interface

### 4.2 — Additional Strategies
- [ ] **Maximum Sharpe Ratio** — already partially implemented in `msr()`, expose it as a standalone strategy
- [ ] **Risk Parity** — allocate such that each asset contributes equally to portfolio risk
- [ ] **Momentum** — rank assets by trailing returns and go long the top decile
- [ ] **Mean-Variance with Black-Litterman** — incorporate subjective views into the optimization

### 4.3 — Strategy Selection
- [ ] Allow the Lambda handler to accept a strategy name as input (event parameter)
- [ ] Support running multiple strategies on independent schedules

---

## Phase 5: Risk Management & Monitoring

Goal: Protect capital and gain visibility into algorithm behavior.

### 5.1 — Pre-Trade Risk Checks
- [ ] Maximum single-position size limit (e.g., no more than 10% in one stock)
- [ ] Maximum total order value per rebalance (circuit breaker)
- [ ] Sector concentration limits

### 5.2 — Monitoring & Alerts
- [ ] Structured JSON logging for CloudWatch ingestion
- [ ] CloudWatch alarms on Lambda errors and execution duration
- [ ] SNS or Slack notifications on: rebalance completion, order failures, risk limit breaches

### 5.3 — Portfolio Analytics Dashboard
- [ ] Daily portfolio snapshot stored in S3 or DynamoDB
- [ ] Simple dashboard (Streamlit, Grafana, or CloudWatch dashboard) showing current holdings, P&L, and recent trades

---

## Phase 6: Advanced Features

Goal: Longer-term enhancements once the core platform is solid.

### 6.1 — Live vs. Paper Toggle with Safeguards
- [ ] Require explicit confirmation (e.g., environment variable `LIVE_TRADING=true`) to trade real money
- [ ] Parallel paper-trading shadow mode: run live + paper simultaneously and compare

### 6.2 — Multi-Asset Support
- [ ] Extend beyond equities to crypto (Alpaca supports crypto trading)
- [ ] Support ETFs and fractional shares

### 6.3 — Transaction Cost Modeling
- [ ] Account for bid-ask spread, slippage, and commission in backtests
- [ ] Implement a minimum-trade-size threshold to avoid churning small positions

### 6.4 — ML-Enhanced Strategies
- [ ] Use ML models (e.g., covariance shrinkage estimators, return prediction) as inputs to optimization
- [ ] Walk-forward validation to avoid overfitting

---

## Priority Summary

| Phase | Focus | Priority |
|-------|-------|----------|
| 1 | Stabilize & harden existing code | **Done** |
| 2 | Project structure & CI/CD | **Done** |
| 3 | Backtesting framework | **Done** |
| 4 | New strategies | **Medium** |
| 5 | Risk management & monitoring | **Medium** |
| 6 | Advanced features | **Low — future** |
