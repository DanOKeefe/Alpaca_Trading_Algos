import json
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from config import get_api_credentials


class TestGetApiCredentials:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv('API_KEY', raising=False)
        monkeypatch.delenv('API_SECRET', raising=False)
        with pytest.raises(EnvironmentError, match="API_KEY and API_SECRET"):
            get_api_credentials()

    def test_missing_secret_raises(self, monkeypatch):
        monkeypatch.setenv('API_KEY', 'key')
        monkeypatch.delenv('API_SECRET', raising=False)
        with pytest.raises(EnvironmentError, match="API_KEY and API_SECRET"):
            get_api_credentials()

    def test_valid_credentials(self, monkeypatch):
        monkeypatch.setenv('API_KEY', 'my-key')
        monkeypatch.setenv('API_SECRET', 'my-secret')
        key, secret = get_api_credentials()
        assert key == 'my-key'
        assert secret == 'my-secret'


class TestRebalanceMarketClosed:
    @patch('gmv_algo.tradeapi.REST')
    def test_returns_early_when_market_closed(self, mock_rest_cls, monkeypatch):
        monkeypatch.setenv('API_KEY', 'test-key')
        monkeypatch.setenv('API_SECRET', 'test-secret')

        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_api.get_clock.return_value = mock_clock

        from gmv_algo import rebalance_portfolio
        result = rebalance_portfolio()

        assert result == 'Stock market is closed today.'
        mock_api.get_account.assert_not_called()


class TestRebalanceFullFlow:
    """Integration test that mocks all external deps and runs the full rebalance flow."""

    @patch('gmv_algo.yf')
    @patch('gmv_algo.pd.read_html')
    @patch('gmv_algo.tradeapi.REST')
    def test_generates_buy_and_sell_orders(self, mock_rest_cls, mock_read_html, mock_yf, monkeypatch):
        monkeypatch.setenv('API_KEY', 'test-key')
        monkeypatch.setenv('API_SECRET', 'test-secret')

        # --- Mock Alpaca API ---
        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api

        # Market is open
        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_api.get_clock.return_value = mock_clock

        # Account value
        mock_account = MagicMock()
        mock_account.portfolio_value = '100000'
        mock_api.get_account.return_value = mock_account

        # No open orders to cancel
        mock_api.list_orders.return_value = []

        # All tickers are tradable
        tickers = ['AAPL', 'MSFT', 'GOOG']
        mock_assets = []
        for t in tickers:
            a = MagicMock()
            a.symbol = t
            a.tradable = True
            a.status = 'active'
            mock_assets.append(a)
        mock_api.list_assets.return_value = mock_assets

        # Current positions: 10 shares of AAPL, 20 shares of MSFT, none of GOOG
        pos_aapl = MagicMock()
        pos_aapl.symbol = 'AAPL'
        pos_aapl.qty = '10'
        pos_msft = MagicMock()
        pos_msft.symbol = 'MSFT'
        pos_msft.qty = '20'
        mock_api.list_positions.return_value = [pos_aapl, pos_msft]

        # Snapshots with prices
        snap_aapl = MagicMock()
        snap_aapl.latest_trade.p = 150.0
        snap_msft = MagicMock()
        snap_msft.latest_trade.p = 300.0
        snap_goog = MagicMock()
        snap_goog.latest_trade.p = 100.0
        mock_api.get_snapshots.return_value = {
            'AAPL': snap_aapl,
            'MSFT': snap_msft,
            'GOOG': snap_goog,
        }

        # --- Mock Wikipedia ticker source ---
        mock_read_html.return_value = [
            None,  # table 0
            None,  # table 1
            pd.DataFrame({'Symbol': ['AAPL', 'MSFT', 'GOOG']}),  # table 2
        ]

        # --- Mock yfinance data ---
        # Create synthetic price data for 3 stocks over 10 days
        np.random.seed(42)
        dates = pd.date_range('2021-01-01', periods=10, freq='B')
        prices = pd.DataFrame({
            'AAPL': 150 + np.cumsum(np.random.randn(10)),
            'MSFT': 300 + np.cumsum(np.random.randn(10)),
            'GOOG': 100 + np.cumsum(np.random.randn(10)),
        }, index=dates)
        # yf.download returns a multi-level column DataFrame
        mock_data = pd.DataFrame(
            prices.values,
            index=dates,
            columns=pd.MultiIndex.from_product([['Adj Close'], ['AAPL', 'MSFT', 'GOOG']]),
        )
        mock_yf.download.return_value = mock_data

        # --- Run ---
        from gmv_algo import rebalance_portfolio
        result = rebalance_portfolio()

        # Verify result is valid JSON with expected keys
        result_data = json.loads(result)
        assert 'Side' in result_data
        assert 'Ticker' in result_data
        assert 'Qty' in result_data

        # Verify API was called correctly
        mock_api.get_clock.assert_called_once()
        mock_api.get_account.assert_called_once()
        mock_api.list_orders.assert_called_once_with(status='open')
        mock_api.list_assets.assert_called_once()
        mock_api.list_positions.assert_called_once()
        mock_api.get_snapshots.assert_called_once()

        # Verify orders were submitted (sells first, then buys)
        assert mock_api.submit_order.called

    @patch('gmv_algo.yf')
    @patch('gmv_algo.pd.read_html')
    @patch('gmv_algo.tradeapi.REST')
    def test_cancels_existing_open_orders(self, mock_rest_cls, mock_read_html, mock_yf, monkeypatch):
        monkeypatch.setenv('API_KEY', 'test-key')
        monkeypatch.setenv('API_SECRET', 'test-secret')

        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api

        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_api.get_clock.return_value = mock_clock

        mock_account = MagicMock()
        mock_account.portfolio_value = '50000'
        mock_api.get_account.return_value = mock_account

        # Two existing open orders to cancel
        order1 = MagicMock()
        order1.id = 'order-1'
        order2 = MagicMock()
        order2.id = 'order-2'
        mock_api.list_orders.return_value = [order1, order2]

        tickers = ['AAPL', 'MSFT']
        mock_assets = []
        for t in tickers:
            a = MagicMock()
            a.symbol = t
            a.tradable = True
            a.status = 'active'
            mock_assets.append(a)
        mock_api.list_assets.return_value = mock_assets
        mock_api.list_positions.return_value = []

        snap_aapl = MagicMock()
        snap_aapl.latest_trade.p = 150.0
        snap_msft = MagicMock()
        snap_msft.latest_trade.p = 300.0
        mock_api.get_snapshots.return_value = {
            'AAPL': snap_aapl,
            'MSFT': snap_msft,
        }

        mock_read_html.return_value = [
            None, None,
            pd.DataFrame({'Symbol': ['AAPL', 'MSFT']}),
        ]

        np.random.seed(42)
        dates = pd.date_range('2021-01-01', periods=10, freq='B')
        prices = pd.DataFrame({
            'AAPL': 150 + np.cumsum(np.random.randn(10)),
            'MSFT': 300 + np.cumsum(np.random.randn(10)),
        }, index=dates)
        mock_data = pd.DataFrame(
            prices.values, index=dates,
            columns=pd.MultiIndex.from_product([['Adj Close'], ['AAPL', 'MSFT']]),
        )
        mock_yf.download.return_value = mock_data

        from gmv_algo import rebalance_portfolio
        rebalance_portfolio()

        # Both open orders should be cancelled
        assert mock_api.cancel_order.call_count == 2
        mock_api.cancel_order.assert_any_call('order-1')
        mock_api.cancel_order.assert_any_call('order-2')

    @patch('gmv_algo.tradeapi.REST')
    def test_snapshot_api_error_returns_error(self, mock_rest_cls, monkeypatch):
        """If get_snapshots fails, the function should return an error JSON."""
        monkeypatch.setenv('API_KEY', 'test-key')
        monkeypatch.setenv('API_SECRET', 'test-secret')

        from gmv_algo import rebalance_portfolio
        from alpaca_trade_api.rest import APIError

        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api

        mock_clock = MagicMock()
        mock_clock.is_open = True
        mock_api.get_clock.return_value = mock_clock

        mock_account = MagicMock()
        mock_account.portfolio_value = '50000'
        mock_api.get_account.return_value = mock_account
        mock_api.list_orders.return_value = []
        mock_api.list_positions.return_value = []

        tickers = ['AAPL']
        mock_assets = []
        for t in tickers:
            a = MagicMock()
            a.symbol = t
            a.tradable = True
            a.status = 'active'
            mock_assets.append(a)
        mock_api.list_assets.return_value = mock_assets

        mock_api.get_snapshots.side_effect = APIError({'message': 'rate limit'})

        with patch('gmv_algo.pd.read_html') as mock_read_html, \
             patch('gmv_algo.yf') as mock_yf:
            mock_read_html.return_value = [
                None, None,
                pd.DataFrame({'Symbol': ['AAPL']}),
            ]
            np.random.seed(42)
            dates = pd.date_range('2021-01-01', periods=10, freq='B')
            prices = pd.DataFrame(
                {'AAPL': 150 + np.cumsum(np.random.randn(10))},
                index=dates,
            )
            mock_data = pd.DataFrame(
                prices.values, index=dates,
                columns=pd.MultiIndex.from_product([['Adj Close'], ['AAPL']]),
            )
            mock_yf.download.return_value = mock_data

            result = rebalance_portfolio()

        assert 'error' in result
