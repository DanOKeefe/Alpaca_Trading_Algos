"""Order building and submission logic."""

import logging

import pandas as pd

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


def build_orders(stocks, target_values, positions_df, price_df, tradable_symbols):
    """Compare target positions to current holdings and build an order list.

    Args:
        stocks: list of ticker symbols.
        target_values: array of target dollar amounts per stock.
        positions_df: DataFrame with columns ['Symbol', 'Qty'].
        price_df: DataFrame of current prices from Alpaca.
        tradable_symbols: list of tradable ticker symbols.

    Returns:
        DataFrame with columns ['Side', 'Ticker', 'Qty'].
    """
    orders = {"Side": [], "Ticker": [], "Qty": []}

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

            if diff > 0:
                logger.info(
                    "%s: %d -> %d shares (buy %d, ~$%.2f)",
                    stock,
                    current_qty,
                    target_qty,
                    diff,
                    price * diff,
                )
                orders["Side"].append("Buy")
                orders["Ticker"].append(stock)
                orders["Qty"].append(diff)
            elif diff < 0:
                qty = abs(diff)
                logger.info(
                    "%s: %d -> %d shares (sell %d, ~$%.2f)",
                    stock,
                    current_qty,
                    target_qty,
                    qty,
                    price * qty,
                )
                orders["Side"].append("Sell")
                orders["Ticker"].append(stock)
                orders["Qty"].append(qty)
        except Exception as e:
            logger.warning("Could not process %s: %s", stock, e)

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
