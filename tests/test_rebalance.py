import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.utils.config import get_api_credentials


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
    @patch('src.lambda_handler.tradeapi.REST')
    def test_returns_early_when_market_closed(self, mock_rest_cls, monkeypatch):
        monkeypatch.setenv('API_KEY', 'test-key')
        monkeypatch.setenv('API_SECRET', 'test-secret')

        mock_api = MagicMock()
        mock_rest_cls.return_value = mock_api
        mock_clock = MagicMock()
        mock_clock.is_open = False
        mock_api.get_clock.return_value = mock_clock

        from src.lambda_handler import rebalance_portfolio
        result = rebalance_portfolio()

        assert result == 'Stock market is closed today.'
        mock_api.get_account.assert_not_called()


def _make_mock_data(tickers, seed=42):
    """Helper to create mock price data and returns for testing."""
    np.random.seed(seed)
    dates = pd.date_range('2021-01-01', periods=10, freq='B')
    price_cols = {t: 100 + np.cumsum(np.random.randn(10)) for t in tickers}
    prices = pd.DataFrame(price_cols, index=dates)
    mock_data = pd.DataFrame(
        prices.values, index=dates,
        columns=pd.MultiIndex.from_product([['Adj Close'], tickers]),
    )
    mock_rets = mock_data['Adj Close'].pct_change().dropna()
    return mock_data, mock_rets


def _make_mock_api(tickers, portfolio_value='100000', positions=None, open_orders=None):
    """Helper to build a fully mocked Alpaca API."""
    mock_api = MagicMock()

    mock_clock = MagicMock()
    mock_clock.is_open = True
    mock_api.get_clock.return_value = mock_clock

    mock_account = MagicMock()
    mock_account.portfolio_value = portfolio_value
    mock_api.get_account.return_value = mock_account

    mock_api.list_orders.return_value = open_orders or []

    mock_assets = []
    for t in tickers:
        a = MagicMock()
        a.symbol = t
        a.tradable = True
        a.status = 'active'
        mock_assets.append(a)
    mock_api.list_assets.return_value = mock_assets

    pos_list = []
    for sym, qty in (positions or {}).items():
        p = MagicMock()
        p.symbol = sym
        p.qty = str(qty)
        pos_list.append(p)
    mock_api.list_positions.return_value = pos_list

    return mock_api


class TestRebalanceFullFlow:
    """Integration tests that mock external deps and run the full rebalance flow."""

    @patch('src.lambda_handler.get_latest_prices')
    @patch('src.lambda_handler.download_returns')
    @patch('src.lambda_handler.fetch_sp100_tickers')
    @patch('src.lambda_handler.tradeapi.REST')
    def test_generates_orders(
        self, mock_rest_cls, mock_fetch, mock_download, mock_prices, monkeypatch
    ):
        monkeypatch.setenv('API_KEY', 'test-key')
        monkeypatch.setenv('API_SECRET', 'test-secret')

        tickers = ['AAPL', 'MSFT', 'GOOG']
        mock_api = _make_mock_api(tickers, positions={'AAPL': 10, 'MSFT': 20})
        mock_rest_cls.return_value = mock_api

        mock_fetch.return_value = tickers

        mock_data, mock_rets = _make_mock_data(tickers)
        mock_download.return_value = (mock_data, mock_rets)

        snap_aapl = MagicMock()
        snap_aapl.latest_trade.p = 150.0
        snap_msft = MagicMock()
        snap_msft.latest_trade.p = 300.0
        snap_goog = MagicMock()
        snap_goog.latest_trade.p = 100.0
        mock_prices.return_value = {
            'AAPL': snap_aapl, 'MSFT': snap_msft, 'GOOG': snap_goog,
        }

        from src.lambda_handler import rebalance_portfolio
        result = rebalance_portfolio()

        result_data = json.loads(result)
        assert 'Side' in result_data
        assert 'Ticker' in result_data
        assert 'Qty' in result_data

        mock_api.get_clock.assert_called_once()
        mock_api.get_account.assert_called_once()
        mock_api.list_orders.assert_called_once_with(status='open')
        mock_api.list_assets.assert_called_once()
        mock_api.list_positions.assert_called_once()
        assert mock_api.submit_order.called

    @patch('src.lambda_handler.get_latest_prices')
    @patch('src.lambda_handler.download_returns')
    @patch('src.lambda_handler.fetch_sp100_tickers')
    @patch('src.lambda_handler.tradeapi.REST')
    def test_cancels_existing_open_orders(
        self, mock_rest_cls, mock_fetch, mock_download, mock_prices, monkeypatch
    ):
        monkeypatch.setenv('API_KEY', 'test-key')
        monkeypatch.setenv('API_SECRET', 'test-secret')

        tickers = ['AAPL', 'MSFT']

        order1 = MagicMock()
        order1.id = 'order-1'
        order2 = MagicMock()
        order2.id = 'order-2'

        mock_api = _make_mock_api(
            tickers, portfolio_value='50000', open_orders=[order1, order2],
        )
        mock_rest_cls.return_value = mock_api

        mock_fetch.return_value = tickers
        mock_data, mock_rets = _make_mock_data(tickers)
        mock_download.return_value = (mock_data, mock_rets)

        snap_aapl = MagicMock()
        snap_aapl.latest_trade.p = 150.0
        snap_msft = MagicMock()
        snap_msft.latest_trade.p = 300.0
        mock_prices.return_value = {'AAPL': snap_aapl, 'MSFT': snap_msft}

        from src.lambda_handler import rebalance_portfolio
        rebalance_portfolio()

        assert mock_api.cancel_order.call_count == 2
        mock_api.cancel_order.assert_any_call('order-1')
        mock_api.cancel_order.assert_any_call('order-2')

    @patch('src.lambda_handler.get_latest_prices')
    @patch('src.lambda_handler.download_returns')
    @patch('src.lambda_handler.fetch_sp100_tickers')
    @patch('src.lambda_handler.tradeapi.REST')
    def test_snapshot_failure_returns_error(
        self, mock_rest_cls, mock_fetch, mock_download, mock_prices, monkeypatch
    ):
        monkeypatch.setenv('API_KEY', 'test-key')
        monkeypatch.setenv('API_SECRET', 'test-secret')

        tickers = ['AAPL']
        mock_api = _make_mock_api(tickers)
        mock_rest_cls.return_value = mock_api

        mock_fetch.return_value = tickers
        mock_data, mock_rets = _make_mock_data(tickers)
        mock_download.return_value = (mock_data, mock_rets)

        mock_prices.return_value = None  # Simulates API failure

        from src.lambda_handler import rebalance_portfolio
        result = rebalance_portfolio()

        assert 'error' in result
