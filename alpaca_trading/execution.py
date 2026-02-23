"""Order building and submission logic."""

import logging

import pandas as pd

from alpaca_trading.config import ESTIMATED_COST_PER_DOLLAR, REBALANCE_THRESHOLD

logger = logging.getLogger(__name__)


def submit_order(api, qty, stock, side):
    """Submit a market order to Alpaca.

    Args:
        api: Alpaca REST client.
        qty: number of shares (integer).
        stock: ticker symbol.
        side: 'buy' or 'sell'.
    """
    if qty <= 0:
        logger.debug("Skipping %s for %s: quantity is zero", side, stock)
        return

    try:
        api.submit_order(stock, qty, side, "market", "day")
        logger.info("Submitted %s order: %d share(s) of %s", side, qty, stock)
    except Exception as e:
        logger.error("Order failed (%s %d %s): %s", side, qty, stock, e)


def _should_rebalance(current_value, target_value, portfolio_value, cost_per_dollar):
    """Decide whether a position change is worth executing.

    Skips the trade if:
    1. The drift is below the rebalance threshold, OR
    2. The estimated transaction cost exceeds the dollar benefit of rebalancing.

    Args:
        current_value: current dollar value of the position.
        target_value: target dollar value of the position.
        portfolio_value: total portfolio value.
        cost_per_dollar: estimated cost per dollar traded (e.g. 0.001).

    Returns:
        (should_trade, reason) tuple.
    """
    if portfolio_value <= 0:
        return False, "portfolio_value is zero"

    trade_value = abs(target_value - current_value)
    drift = trade_value / portfolio_value

    if drift < REBALANCE_THRESHOLD:
        return False, f"drift {drift:.4f} below threshold {REBALANCE_THRESHOLD}"

    trade_cost = trade_value * cost_per_dollar
    # Benefit is proportional to drift squared (variance reduction)
    # but as a simple heuristic, if cost > trade_value * threshold, skip
    if trade_cost > trade_value * REBALANCE_THRESHOLD:
        return False, f"trade cost ${trade_cost:.2f} exceeds benefit"

    return True, "ok"


def build_orders(
    stocks,
    target_values,
    positions_df,
    price_df,
    tradable_symbols,
    portfolio_value=0,
    cost_per_dollar=ESTIMATED_COST_PER_DOLLAR,
):
    """Compare target positions to current holdings and build an order list.

    Filters out trades that are below the rebalance threshold or where
    estimated transaction costs exceed the benefit.

    Args:
        stocks: list of ticker symbols.
        target_values: array of target dollar amounts per stock.
        positions_df: DataFrame with columns ['Symbol', 'Qty'].
        price_df: DataFrame of current prices from Alpaca.
        tradable_symbols: list of tradable ticker symbols.
        portfolio_value: total portfolio value (for threshold calculation).
        cost_per_dollar: estimated cost per dollar traded.

    Returns:
        DataFrame with columns ['Side', 'Ticker', 'Qty'].
    """
    orders = {"Side": [], "Ticker": [], "Qty": []}
    skipped = 0

    for stock, target_value in zip(stocks, target_values):
        if stock not in tradable_symbols:
            continue

        try:
            price = price_df[stock].dropna().close[0]
            target_qty = int(target_value // price)

            if stock in positions_df["Symbol"].tolist():
                current_qty = positions_df[positions_df["Symbol"] == stock][
                    "Qty"
                ].iloc[0]
            else:
                current_qty = 0

            diff = target_qty - current_qty
            if diff == 0:
                continue

            # Check rebalance threshold and transaction cost
            current_value = current_qty * price
            if portfolio_value > 0:
                should_trade, reason = _should_rebalance(
                    current_value, target_value, portfolio_value, cost_per_dollar
                )
                if not should_trade:
                    logger.debug("Skipping %s: %s", stock, reason)
                    skipped += 1
                    continue

            trade_qty = abs(diff)
            side = "Buy" if diff > 0 else "Sell"
            logger.info(
                "%s: %d -> %d shares (%s %d, ~$%.2f)",
                stock,
                current_qty,
                target_qty,
                side.lower(),
                trade_qty,
                price * trade_qty,
            )
            orders["Side"].append(side)
            orders["Ticker"].append(stock)
            orders["Qty"].append(trade_qty)

        except Exception as e:
            logger.warning("Could not process %s: %s", stock, e)

    if skipped:
        logger.info("Skipped %d positions below rebalance threshold", skipped)

    return pd.DataFrame(orders)


def execute_orders(api, orders_df):
    """Execute orders: sells first, then buys.

    Args:
        api: Alpaca REST client.
        orders_df: DataFrame with columns ['Side', 'Ticker', 'Qty'].
    """
    # Sell first to free up buying power
    sell_df = orders_df[orders_df["Side"] == "Sell"]
    for _, row in sell_df.iterrows():
        submit_order(api=api, qty=row["Qty"], stock=row["Ticker"], side="sell")

    buy_df = orders_df[orders_df["Side"] == "Buy"]
    for _, row in buy_df.iterrows():
        submit_order(api=api, qty=row["Qty"], stock=row["Ticker"], side="buy")
