# Plan: Address All Limitations in gmv_algo.py

This plan modernizes `gmv_algo.py` to fix every limitation listed in the README.

---

## Step 1: Migrate from `alpaca-trade-api` to `alpaca-py`

The `alpaca-trade-api` package is deprecated. The `get_barset()` call on line 136 no longer works in newer SDK versions.

**Changes:**

- Replace `import alpaca_trade_api as tradeapi` with imports from the `alpaca` namespace:
  ```python
  from alpaca.trading.client import TradingClient
  from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
  from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
  from alpaca.data.historical import StockHistoricalDataClient
  from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
  from alpaca.data.timeframe import TimeFrame
  ```
- Replace `tradeapi.REST(key, secret, url, 'v2')` with two clients:
  - `TradingClient(key, secret, paper=True)` for orders/positions/account
  - `StockHistoricalDataClient(key, secret)` for market data
- Replace `alpaca.get_barset(stocks, 'minute', 1).df` with `data_client.get_stock_bars(StockBarsRequest(...)).df`
- Update all API calls to new method names:
  - `list_orders()` -> `get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))`
  - `cancel_order(id)` -> `cancel_order_by_id(id)`
  - `list_assets()` -> `get_all_assets()`
  - `list_positions()` -> `get_all_positions()`
  - `submit_order(symbol, qty, side, type, tif)` -> `submit_order(MarketOrderRequest(...))`
- Update `requirements.txt` (new file): replace `alpaca-trade-api` with `alpaca-py`

---

## Step 2: Add fractional share support using notional orders

Currently line 149 does `int(target_value // price)`, truncating to whole shares and losing precision.

**Changes:**

- Use **notional orders** (dollar-amount based) instead of share-quantity orders. This eliminates rounding entirely -- Alpaca handles the fractional math.
- Replace `submitOrder(api, qty, stock, side)` with a function that submits `MarketOrderRequest(symbol=stock, notional=dollar_amount, side=..., time_in_force=TimeInForce.DAY)`
- Remove the per-stock price lookup and integer truncation logic (lines 147-149). Instead, compute the dollar difference between current position value and target value directly.
- For sells: calculate the excess dollar amount and submit a notional sell.
- For buys: calculate the shortfall dollar amount and submit a notional buy.
- Add a check for `asset.fractionable` before using notional orders; fall back to whole-share qty for non-fractionable assets.

---

## Step 3: Use limit orders to control slippage

Currently all orders are market orders (line 67), which can fill at unexpected prices.

**Changes:**

- Fetch the latest quote for each stock before ordering using `StockLatestQuoteRequest`
- For buys: set limit price = ask price * 1.005 (0.5% slippage tolerance)
- For sells: set limit price = bid price * 0.995
- Use `LimitOrderRequest` instead of `MarketOrderRequest`
- Add a configurable `SLIPPAGE_TOLERANCE` constant at the top of the file (default 0.005)
- Keep `time_in_force=TimeInForce.DAY` so unfilled limit orders expire at close

---

## Step 4: Make the S&P 100 ticker source more robust

Currently the ticker list is scraped from a Wikipedia HTML table (line 87-88), which is fragile.

**Changes:**

- Add a `TICKERS_FALLBACK` constant containing a hardcoded list of S&P 100 tickers as a fallback
- Wrap the Wikipedia scrape in a try/except
- Add validation after scraping: check that we got between 95-105 tickers and they look like valid symbols (1-5 uppercase letters)
- If scraping fails or validation fails, log a warning and fall back to the hardcoded list
- Add a comment noting the fallback list should be updated periodically

---

## Step 5: Replace bare `except` clauses with proper error handling

There are two bare `except:` clauses (lines 69 and 187) and no logging.

**Changes:**

- Add `import logging` and `logger = logging.getLogger(__name__)` at module level
- Add `import requests.exceptions` for network error handling

**`submitOrder` (line 69):**
- Catch `alpaca` API errors and `requests.exceptions.RequestException` separately
- Add retry with exponential backoff (3 attempts, 2s/4s/6s) for transient failures (rate limits, server errors)
- Return a success/failure indicator so the caller can track results
- Log with `logger.error(..., exc_info=True)` to capture tracebacks

**Rebalance loop (line 187):**
- Catch `KeyError`/`IndexError` for missing data (log at WARNING, skip stock)
- Catch `ZeroDivisionError`/`TypeError` for bad data (log at ERROR with traceback)
- Collect skipped stocks into a list and log a summary at the end

**`rebalance_portfolio` top-level:**
- Wrap env var access in try/except `KeyError`, log at CRITICAL, re-raise
- Wrap Wikipedia fetch in try/except, fall back to hardcoded tickers (see Step 4)
- Wrap order cancellation loop in per-order try/except (handle race condition where order fills between list and cancel)
- Collect and log all failed orders at the end of the function

**Replace all `print()` calls with `logger.info()` / `logger.warning()` / `logger.error()`.**

---

## Step 6: Create a `requirements.txt`

The repo currently has no dependency manifest.

**Create `requirements.txt`:**
```
alpaca-py
yfinance
pandas
numpy
scipy
pytz
python-dateutil
```

---

## Step 7: Update the README

- Remove the "Limitations" section entries that have been fixed
- Update the dependencies list to show `alpaca-py` instead of `alpaca-trade-api`
- Note the new features: fractional shares, limit orders, robust ticker sourcing
- Update any code references if needed

---

## Order of implementation

Steps 1-5 all modify `gmv_algo.py` and are interdependent (e.g., the new error handling in Step 5 catches exceptions from the new SDK in Step 1). They should be implemented together in a single pass through the file, roughly top-to-bottom:

1. **Imports and module-level setup** (Steps 1 + 5: new SDK imports, logging setup)
2. **Portfolio math functions** (unchanged -- `portfolio_return`, `portfolio_vol`, `gmv`, `msr`)
3. **`submitOrder`** (Steps 1 + 2 + 3 + 5: new SDK, notional/limit orders, error handling with retry)
4. **`rebalance_portfolio`** (Steps 1 + 2 + 3 + 4 + 5: new SDK calls, robust tickers, limit orders, proper error handling throughout)
5. **`lambda_handler`** (unchanged)
6. **New files** (Step 6: `requirements.txt`)
7. **README update** (Step 7)
