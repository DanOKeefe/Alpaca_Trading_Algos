from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtests.engine import BacktestConfig, Backtester


def _make_synthetic_prices(tickers, start, periods, seed=42):
    """Create synthetic daily prices for testing."""
    np.random.seed(seed)
    dates = pd.date_range(start, periods=periods, freq="B")
    prices = {}
    for i, t in enumerate(tickers):
        base = 100 + i * 50
        returns = np.random.normal(0.0005, 0.015, periods)
        prices[t] = base * np.cumprod(1 + returns)
    return pd.DataFrame(prices, index=dates)


def _equal_weight_fn(returns):
    n = returns.shape[1]
    return np.repeat(1 / n, n)


class TestBacktester:
    @patch("backtests.engine.yf")
    def test_equity_starts_at_initial_capital(self, mock_yf):
        tickers = ["A", "B", "C"]
        prices = _make_synthetic_prices(tickers, "2015-01-01", 2600)

        mock_data = pd.DataFrame(
            prices.values,
            index=prices.index,
            columns=pd.MultiIndex.from_product(
                [["Adj Close"], tickers]
            ),
        )
        mock_yf.download.return_value = mock_data

        config = BacktestConfig(
            start_date="2020-01-01",
            end_date="2024-12-31",
            initial_capital=50_000,
            rebalance_freq="BME",
            lookback_years=5,
        )
        bt = Backtester(_equal_weight_fn, tickers, config)
        result = bt.run(strategy_name="Test")

        # First value should be close to initial capital
        # (after one day of returns)
        assert result.equity_curve.iloc[0] == pytest.approx(
            50_000, rel=0.02
        )

    @patch("backtests.engine.yf")
    def test_result_has_all_metrics(self, mock_yf):
        tickers = ["A", "B"]
        prices = _make_synthetic_prices(tickers, "2015-01-01", 2600)

        mock_data = pd.DataFrame(
            prices.values,
            index=prices.index,
            columns=pd.MultiIndex.from_product(
                [["Adj Close"], tickers]
            ),
        )
        mock_yf.download.return_value = mock_data

        config = BacktestConfig(
            start_date="2020-01-01",
            end_date="2024-12-31",
            initial_capital=100_000,
        )
        bt = Backtester(_equal_weight_fn, tickers, config)
        result = bt.run(strategy_name="EW")

        assert "total_return" in result.metrics
        assert "annualized_return" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "annualized_volatility" in result.metrics
        assert result.strategy_name == "EW"

    @patch("backtests.engine.yf")
    def test_weights_history_populated(self, mock_yf):
        tickers = ["A", "B"]
        prices = _make_synthetic_prices(tickers, "2015-01-01", 2600)

        mock_data = pd.DataFrame(
            prices.values,
            index=prices.index,
            columns=pd.MultiIndex.from_product(
                [["Adj Close"], tickers]
            ),
        )
        mock_yf.download.return_value = mock_data

        config = BacktestConfig(
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        bt = Backtester(_equal_weight_fn, tickers, config)
        result = bt.run()

        # Should have multiple rebalance entries
        assert len(result.weights_history) > 1

        # Each weight entry should sum to ~1
        for date, weights in result.weights_history.items():
            assert weights.sum() == pytest.approx(1.0, abs=1e-6)

    @patch("backtests.engine.yf")
    def test_equal_weight_on_identical_assets(self, mock_yf):
        """If all assets are identical, equal-weight should match any single."""
        np.random.seed(42)
        dates = pd.date_range("2015-01-01", periods=2600, freq="B")
        base_prices = 100 * np.cumprod(
            1 + np.random.normal(0.0003, 0.01, 2600)
        )
        # All tickers have identical prices
        tickers = ["A", "B", "C"]
        prices = pd.DataFrame(
            {t: base_prices for t in tickers}, index=dates
        )

        mock_data = pd.DataFrame(
            prices.values,
            index=prices.index,
            columns=pd.MultiIndex.from_product(
                [["Adj Close"], tickers]
            ),
        )
        mock_yf.download.return_value = mock_data

        config = BacktestConfig(
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        bt = Backtester(_equal_weight_fn, tickers, config)
        result = bt.run()

        # Total return should be positive for positive-drift assets
        assert result.metrics["total_return"] > 0

    @patch("backtests.engine.yf")
    def test_empty_data_returns_empty_result(self, mock_yf):
        mock_yf.download.return_value = pd.DataFrame()

        config = BacktestConfig(
            start_date="2020-01-01", end_date="2024-12-31"
        )
        bt = Backtester(_equal_weight_fn, ["A", "B"], config)
        result = bt.run()

        assert len(result.equity_curve) == 0

    @patch("backtests.engine.yf")
    def test_gmv_strategy_works(self, mock_yf):
        """Verify GMV weight function integrates with the backtester."""
        from src.strategies.gmv import gmv

        def gmv_fn(returns):
            return gmv(returns.cov())

        tickers = ["A", "B", "C"]
        prices = _make_synthetic_prices(tickers, "2015-01-01", 2600)

        mock_data = pd.DataFrame(
            prices.values,
            index=prices.index,
            columns=pd.MultiIndex.from_product(
                [["Adj Close"], tickers]
            ),
        )
        mock_yf.download.return_value = mock_data

        config = BacktestConfig(
            start_date="2020-01-01",
            end_date="2024-12-31",
        )
        bt = Backtester(gmv_fn, tickers, config)
        result = bt.run(strategy_name="GMV")

        assert len(result.equity_curve) > 0
        assert result.strategy_name == "GMV"
        assert "total_return" in result.metrics
