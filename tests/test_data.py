"""Tests for data retrieval functions."""

import io
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alpaca_trading.data import (
    get_sp100_tickers,
    get_sp500_tickers,
    get_tickers,
    _s3_cache_key,
    _load_from_s3,
    _save_to_s3,
)


class TestGetSP100Tickers:
    @patch("alpaca_trading.data.pd.read_html")
    def test_returns_list_of_tickers(self, mock_read_html):
        mock_table = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "GOOG", "GOOGL", "AMZN"]})
        mock_read_html.return_value = [None, None, mock_table]

        tickers = get_sp100_tickers()

        assert isinstance(tickers, list)
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    @patch("alpaca_trading.data.pd.read_html")
    def test_removes_googl(self, mock_read_html):
        mock_table = pd.DataFrame({"Symbol": ["AAPL", "GOOG", "GOOGL"]})
        mock_read_html.return_value = [None, None, mock_table]

        tickers = get_sp100_tickers()

        assert "GOOGL" not in tickers
        assert "GOOG" in tickers

    @patch("alpaca_trading.data.pd.read_html")
    def test_no_googl_present(self, mock_read_html):
        mock_table = pd.DataFrame({"Symbol": ["AAPL", "MSFT"]})
        mock_read_html.return_value = [None, None, mock_table]

        tickers = get_sp100_tickers()

        assert tickers == ["AAPL", "MSFT"]


class TestGetSP500Tickers:
    @patch("alpaca_trading.data.pd.read_html")
    def test_returns_list_of_tickers(self, mock_read_html):
        mock_table = pd.DataFrame({"Symbol": ["AAPL", "MSFT", "BRK.B", "GOOGL"]})
        mock_read_html.return_value = [mock_table]

        tickers = get_sp500_tickers()

        assert "AAPL" in tickers
        assert "MSFT" in tickers

    @patch("alpaca_trading.data.pd.read_html")
    def test_converts_dots_to_dashes(self, mock_read_html):
        mock_table = pd.DataFrame({"Symbol": ["BRK.B", "BF.B"]})
        mock_read_html.return_value = [mock_table]

        tickers = get_sp500_tickers()

        assert "BRK-B" in tickers
        assert "BF-B" in tickers

    @patch("alpaca_trading.data.pd.read_html")
    def test_removes_googl(self, mock_read_html):
        mock_table = pd.DataFrame({"Symbol": ["AAPL", "GOOG", "GOOGL"]})
        mock_read_html.return_value = [mock_table]

        tickers = get_sp500_tickers()

        assert "GOOGL" not in tickers
        assert "GOOG" in tickers


class TestGetTickers:
    @patch("alpaca_trading.data.get_sp100_tickers")
    def test_sp100_universe(self, mock_sp100):
        mock_sp100.return_value = ["AAPL", "MSFT"]
        tickers = get_tickers("sp100")
        mock_sp100.assert_called_once()
        assert tickers == ["AAPL", "MSFT"]

    @patch("alpaca_trading.data.get_sp500_tickers")
    def test_sp500_universe(self, mock_sp500):
        mock_sp500.return_value = ["AAPL", "MSFT", "GOOG"]
        tickers = get_tickers("sp500")
        mock_sp500.assert_called_once()
        assert tickers == ["AAPL", "MSFT", "GOOG"]

    def test_custom_tickers(self):
        tickers = get_tickers("AAPL, MSFT, GOOG")
        assert tickers == ["AAPL", "MSFT", "GOOG"]

    def test_custom_tickers_uppercased(self):
        tickers = get_tickers("aapl,msft")
        assert tickers == ["AAPL", "MSFT"]

    def test_custom_tickers_strips_whitespace(self):
        tickers = get_tickers("  AAPL , MSFT  ,  GOOG  ")
        assert tickers == ["AAPL", "MSFT", "GOOG"]

    @patch("alpaca_trading.data.get_sp100_tickers")
    def test_case_insensitive_universe(self, mock_sp100):
        mock_sp100.return_value = ["AAPL"]
        get_tickers("SP100")
        mock_sp100.assert_called_once()


class TestS3CacheKey:
    def test_produces_consistent_key(self):
        key1 = _s3_cache_key(["AAPL", "MSFT"], "2020-01-01", "2025-01-01")
        key2 = _s3_cache_key(["AAPL", "MSFT"], "2020-01-01", "2025-01-01")
        assert key1 == key2

    def test_different_tickers_different_key(self):
        key1 = _s3_cache_key(["AAPL"], "2020-01-01", "2025-01-01")
        key2 = _s3_cache_key(["MSFT"], "2020-01-01", "2025-01-01")
        assert key1 != key2

    def test_different_dates_different_key(self):
        key1 = _s3_cache_key(["AAPL"], "2020-01-01", "2025-01-01")
        key2 = _s3_cache_key(["AAPL"], "2021-01-01", "2025-01-01")
        assert key1 != key2

    def test_key_format(self):
        key = _s3_cache_key(["AAPL"], "2020-01-01", "2025-01-01")
        assert key.startswith("price_cache/")
        assert key.endswith(".parquet")

    def test_order_independent(self):
        key1 = _s3_cache_key(["MSFT", "AAPL"], "2020-01-01", "2025-01-01")
        key2 = _s3_cache_key(["AAPL", "MSFT"], "2020-01-01", "2025-01-01")
        assert key1 == key2


class TestS3LoadSave:
    def test_load_returns_none_on_error(self):
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value.get_object.side_effect = Exception("not found")

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            result = _load_from_s3("my-bucket", "some-key")

        assert result is None

    def test_save_does_not_raise_on_error(self):
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value.put_object.side_effect = Exception("access denied")
        df = pd.DataFrame({"A": [1, 2, 3]})

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            # Should not raise
            _save_to_s3(df, "my-bucket", "some-key")
