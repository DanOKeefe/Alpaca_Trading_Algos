"""Tests for data retrieval functions."""

from unittest.mock import patch

import pandas as pd
import pytest

from alpaca_trading.data import get_sp100_tickers


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
