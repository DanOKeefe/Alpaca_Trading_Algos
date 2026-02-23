"""Post-rebalance notifications via AWS SNS."""

import logging

from alpaca_trading.config import SNS_TOPIC_ARN

logger = logging.getLogger(__name__)


def send_rebalance_summary(orders_df, portfolio_value, strategy_name):
    """Send a rebalance summary notification via SNS.

    Does nothing if SNS_TOPIC_ARN is not configured.

    Args:
        orders_df: DataFrame with columns ['Side', 'Ticker', 'Qty'].
        portfolio_value: total portfolio value after rebalance.
        strategy_name: name of the strategy used.
    """
    if not SNS_TOPIC_ARN:
        logger.debug("SNS_TOPIC_ARN not set, skipping notification")
        return

    buys = orders_df[orders_df["Side"] == "Buy"]
    sells = orders_df[orders_df["Side"] == "Sell"]

    lines = [
        f"Rebalance Complete — {strategy_name}",
        f"Portfolio Value: ${portfolio_value:,}",
        f"Orders: {len(orders_df)} total ({len(buys)} buys, {len(sells)} sells)",
        "",
    ]

    if not orders_df.empty:
        lines.append("Trades:")
        for _, row in orders_df.iterrows():
            lines.append(f"  {row['Side']:4s} {row['Qty']:>5d} {row['Ticker']}")
    else:
        lines.append("No trades executed (positions within threshold).")

    message = "\n".join(lines)
    subject = f"Rebalance: {len(orders_df)} orders ({strategy_name})"

    try:
        import boto3

        sns = boto3.client("sns")
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject[:100],
            Message=message,
        )
        logger.info("Sent rebalance notification to %s", SNS_TOPIC_ARN)
    except Exception as e:
        logger.error("Failed to send SNS notification: %s", e)


def format_summary(orders_df, portfolio_value, strategy_name):
    """Format a rebalance summary as a string (for logging or other uses).

    Args:
        orders_df: DataFrame with columns ['Side', 'Ticker', 'Qty'].
        portfolio_value: total portfolio value.
        strategy_name: name of the strategy used.

    Returns:
        Formatted summary string.
    """
    buys = orders_df[orders_df["Side"] == "Buy"]
    sells = orders_df[orders_df["Side"] == "Sell"]

    lines = [
        f"Strategy: {strategy_name}",
        f"Portfolio Value: ${portfolio_value:,}",
        f"Orders: {len(orders_df)} ({len(buys)} buys, {len(sells)} sells)",
    ]

    if not orders_df.empty:
        for _, row in orders_df.iterrows():
            lines.append(f"  {row['Side']:4s} {row['Qty']:>5d} {row['Ticker']}")

    return "\n".join(lines)
