"""Tests for order execution logic."""

from unittest.mock import MagicMock, call

import pandas as pd

from alpaca_trading.execution import execute_orders, submit_order


class TestSubmitOrder:
    def test_submits_order_when_qty_positive(self):
        mock_api = MagicMock()
        submit_order(mock_api, 10, "AAPL", "buy")
        mock_api.submit_order.assert_called_once_with("AAPL", 10, "buy", "market", "day")

    def test_skips_when_qty_zero(self):
        mock_api = MagicMock()
        submit_order(mock_api, 0, "AAPL", "buy")
        mock_api.submit_order.assert_not_called()

    def test_skips_when_qty_negative(self):
        mock_api = MagicMock()
        submit_order(mock_api, -5, "AAPL", "buy")
        mock_api.submit_order.assert_not_called()

    def test_handles_api_error(self):
        mock_api = MagicMock()
        mock_api.submit_order.side_effect = Exception("API error")
        # Should not raise
        submit_order(mock_api, 10, "AAPL", "buy")


class TestExecuteOrders:
    def test_sells_before_buys(self):
        mock_api = MagicMock()
        orders_df = pd.DataFrame(
            {
                "Side": ["Buy", "Sell", "Buy", "Sell"],
                "Ticker": ["AAPL", "MSFT", "GOOG", "AMZN"],
                "Qty": [5, 3, 2, 7],
            }
        )

        execute_orders(mock_api, orders_df)

        calls = mock_api.submit_order.call_args_list
        # First two calls should be sells
        assert calls[0] == call("MSFT", 3, "sell", "market", "day")
        assert calls[1] == call("AMZN", 7, "sell", "market", "day")
        # Last two should be buys
        assert calls[2] == call("AAPL", 5, "buy", "market", "day")
        assert calls[3] == call("GOOG", 2, "buy", "market", "day")

    def test_empty_orders(self):
        mock_api = MagicMock()
        orders_df = pd.DataFrame({"Side": [], "Ticker": [], "Qty": []})
        execute_orders(mock_api, orders_df)
        mock_api.submit_order.assert_not_called()
