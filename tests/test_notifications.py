"""Tests for notification functions."""

import sys
from unittest.mock import MagicMock, patch

import pandas as pd

from alpaca_trading.notifications import format_summary, send_rebalance_summary


class TestFormatSummary:
    def test_includes_strategy_name(self):
        orders_df = pd.DataFrame({"Side": ["Buy"], "Ticker": ["AAPL"], "Qty": [10]})
        result = format_summary(orders_df, 100000, "Global Minimum Variance")
        assert "Global Minimum Variance" in result

    def test_includes_portfolio_value(self):
        orders_df = pd.DataFrame({"Side": ["Buy"], "Ticker": ["AAPL"], "Qty": [10]})
        result = format_summary(orders_df, 100000, "GMV")
        assert "$100,000" in result

    def test_includes_order_counts(self):
        orders_df = pd.DataFrame(
            {
                "Side": ["Buy", "Sell", "Buy"],
                "Ticker": ["AAPL", "MSFT", "GOOG"],
                "Qty": [10, 5, 3],
            }
        )
        result = format_summary(orders_df, 50000, "GMV")
        assert "2 buys" in result
        assert "1 sells" in result

    def test_empty_orders(self):
        orders_df = pd.DataFrame({"Side": [], "Ticker": [], "Qty": []})
        result = format_summary(orders_df, 50000, "GMV")
        assert "0" in result

    def test_includes_trade_details(self):
        orders_df = pd.DataFrame({"Side": ["Buy"], "Ticker": ["AAPL"], "Qty": [10]})
        result = format_summary(orders_df, 50000, "GMV")
        assert "AAPL" in result
        assert "10" in result


class TestSendRebalanceSummary:
    @patch("alpaca_trading.notifications.SNS_TOPIC_ARN", None)
    def test_does_nothing_without_topic(self):
        orders_df = pd.DataFrame({"Side": ["Buy"], "Ticker": ["AAPL"], "Qty": [10]})
        # Should not raise
        send_rebalance_summary(orders_df, 100000, "GMV")

    @patch("alpaca_trading.notifications.SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123:topic")
    def test_publishes_to_sns(self):
        mock_boto3 = MagicMock()
        mock_sns = MagicMock()
        mock_boto3.client.return_value = mock_sns

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            orders_df = pd.DataFrame({"Side": ["Buy"], "Ticker": ["AAPL"], "Qty": [10]})
            send_rebalance_summary(orders_df, 100000, "GMV")

        mock_boto3.client.assert_called_once_with("sns")
        mock_sns.publish.assert_called_once()
        call_kwargs = mock_sns.publish.call_args[1]
        assert call_kwargs["TopicArn"] == "arn:aws:sns:us-east-1:123:topic"
        assert "AAPL" in call_kwargs["Message"]

    @patch("alpaca_trading.notifications.SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:123:topic")
    def test_handles_sns_error(self):
        mock_boto3 = MagicMock()
        mock_sns = MagicMock()
        mock_sns.publish.side_effect = Exception("SNS error")
        mock_boto3.client.return_value = mock_sns

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            orders_df = pd.DataFrame({"Side": ["Buy"], "Ticker": ["AAPL"], "Qty": [10]})
            # Should not raise
            send_rebalance_summary(orders_df, 100000, "GMV")
