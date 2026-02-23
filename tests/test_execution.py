"""Tests for order execution logic."""

from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from alpaca_trading.execution import (
    _should_rebalance,
    build_orders,
    execute_orders,
    submit_order,
)


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


class TestShouldRebalance:
    def test_large_drift_returns_true(self):
        # 10% drift should be above 2% threshold
        should, reason = _should_rebalance(
            current_value=1000, target_value=2000, portfolio_value=10000, cost_per_dollar=0.001
        )
        assert should is True

    def test_small_drift_returns_false(self):
        # 0.1% drift should be below 2% threshold
        should, reason = _should_rebalance(
            current_value=1000, target_value=1010, portfolio_value=10000, cost_per_dollar=0.001
        )
        assert should is False
        assert "drift" in reason

    def test_zero_portfolio_value_returns_false(self):
        should, reason = _should_rebalance(
            current_value=100, target_value=200, portfolio_value=0, cost_per_dollar=0.001
        )
        assert should is False

    def test_drift_exactly_at_threshold(self):
        # Drift exactly at 2%: $200 change on $10000 portfolio
        should, reason = _should_rebalance(
            current_value=1000, target_value=1200, portfolio_value=10000, cost_per_dollar=0.001
        )
        assert should is True

    def test_high_cost_skips_trade(self):
        # With very high cost per dollar, the trade should be skipped
        should, reason = _should_rebalance(
            current_value=1000, target_value=1500, portfolio_value=10000, cost_per_dollar=0.5
        )
        assert should is False
        assert "cost" in reason


class TestBuildOrdersWithThreshold:
    def _make_price_df(self, prices):
        """Build a mock price DataFrame matching Alpaca's barset format."""
        data = {}
        for ticker, price in prices.items():
            data[ticker] = pd.DataFrame({"close": [price]})
        return pd.concat(data, axis=1)

    def test_skips_small_drift(self):
        """Positions with tiny drift should be filtered out."""
        prices = {"AAPL": 150.0, "MSFT": 300.0}
        price_df = self._make_price_df(prices)

        positions_df = pd.DataFrame({"Symbol": ["AAPL", "MSFT"], "Qty": [10, 5]})
        # Target values very close to current: AAPL=10*150=1500, MSFT=5*300=1500
        target_values = [1510.0, 1510.0]  # tiny differences

        orders = build_orders(
            stocks=["AAPL", "MSFT"],
            target_values=target_values,
            positions_df=positions_df,
            price_df=price_df,
            tradable_symbols=["AAPL", "MSFT"],
            portfolio_value=100000,
            cost_per_dollar=0.001,
        )
        # Drift is ~$10/$100000 = 0.01% — far below 2% threshold
        assert len(orders) == 0

    def test_includes_large_drift(self):
        """Positions with large drift should produce orders."""
        prices = {"AAPL": 150.0}
        price_df = self._make_price_df(prices)

        positions_df = pd.DataFrame({"Symbol": ["AAPL"], "Qty": [10]})
        # Current value: 10*150 = $1500, target: $5000 -> big drift
        target_values = [5000.0]

        orders = build_orders(
            stocks=["AAPL"],
            target_values=target_values,
            positions_df=positions_df,
            price_df=price_df,
            tradable_symbols=["AAPL"],
            portfolio_value=10000,
            cost_per_dollar=0.001,
        )
        assert len(orders) == 1
        assert orders.iloc[0]["Side"] == "Buy"
        assert orders.iloc[0]["Ticker"] == "AAPL"

    def test_backwards_compatible_without_portfolio_value(self):
        """When portfolio_value=0 (default), threshold check is skipped."""
        prices = {"AAPL": 150.0}
        price_df = self._make_price_df(prices)

        positions_df = pd.DataFrame({"Symbol": ["AAPL"], "Qty": [10]})
        target_values = [1800.0]  # need 12 shares, have 10 -> buy 2

        orders = build_orders(
            stocks=["AAPL"],
            target_values=target_values,
            positions_df=positions_df,
            price_df=price_df,
            tradable_symbols=["AAPL"],
        )
        assert len(orders) == 1
        assert orders.iloc[0]["Side"] == "Buy"
        assert orders.iloc[0]["Qty"] == 2
