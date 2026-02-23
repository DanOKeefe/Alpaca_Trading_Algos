from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from alpaca_trade_api.rest import APIError


class TestFetchSP100Tickers:
    @patch('src.data.market_data.pd.read_html')
    def test_returns_tickers_without_duplicates(self, mock_read_html):
        mock_read_html.return_value = [
            None,  # table 0
            None,  # table 1
            pd.DataFrame({'Symbol': ['AAPL', 'GOOG', 'GOOGL', 'MSFT']}),
        ]
        from src.data.market_data import fetch_sp100_tickers
        tickers = fetch_sp100_tickers()
        assert 'GOOGL' not in tickers
        assert 'AAPL' in tickers
        assert 'GOOG' in tickers
        assert 'MSFT' in tickers

    @patch('src.data.market_data.pd.read_html')
    def test_handles_no_duplicates_present(self, mock_read_html):
        mock_read_html.return_value = [
            None, None,
            pd.DataFrame({'Symbol': ['AAPL', 'MSFT']}),
        ]
        from src.data.market_data import fetch_sp100_tickers
        tickers = fetch_sp100_tickers()
        assert tickers == ['AAPL', 'MSFT']


class TestDownloadReturns:
    @patch('src.data.market_data.yf')
    def test_returns_data_and_returns(self, mock_yf):
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

        from src.data.market_data import download_returns
        data, rets = download_returns(['AAPL', 'MSFT'])

        assert list(data['Adj Close'].columns) == ['AAPL', 'MSFT']
        assert len(rets) > 0
        mock_yf.download.assert_called_once()


class TestGetLatestPrices:
    def test_returns_snapshots_on_success(self):
        from src.data.market_data import get_latest_prices
        api = MagicMock()
        snap = MagicMock()
        api.get_snapshots.return_value = {'AAPL': snap}

        result = get_latest_prices(api, ['AAPL'])

        assert result == {'AAPL': snap}
        api.get_snapshots.assert_called_once_with(['AAPL'])

    def test_returns_none_on_api_error(self):
        from src.data.market_data import get_latest_prices
        api = MagicMock()
        api.get_snapshots.side_effect = APIError({'message': 'rate limit'})

        result = get_latest_prices(api, ['AAPL'])

        assert result is None
