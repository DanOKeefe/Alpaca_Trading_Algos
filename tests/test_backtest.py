"""Tests for backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from alpaca_trading.backtest import backtest, compare_strategies
from alpaca_trading.strategies.equal_weight import EqualWeightStrategy
from alpaca_trading.strategies.gmv import GMVStrategy
from alpaca_trading.strategies.risk_parity import RiskParityStrategy


def _make_returns(n_days=504, n_assets=5):
    """Create synthetic returns DataFrame with 2 years of data."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    vols = np.linspace(0.005, 0.02, n_assets)
    drifts = np.linspace(0.0001, 0.0005, n_assets)
    data = np.random.randn(n_days, n_assets) * vols + drifts
    cols = [f"ASSET_{i}" for i in range(n_assets)]
    return pd.DataFrame(data, index=dates, columns=cols)


class TestBacktest:
    def test_returns_expected_keys(self):
        returns = _make_returns()
        result = backtest(EqualWeightStrategy(), returns, rebalance_frequency=21)
        assert "daily_returns" in result
        assert "cumulative" in result
        assert "weights_history" in result
        assert "metrics" in result

    def test_daily_returns_length(self):
        returns = _make_returns(n_days=300)
        result = backtest(EqualWeightStrategy(), returns, rebalance_frequency=21)
        # Should have n_days - min_lookback entries
        assert len(result["daily_returns"]) == 300 - 63

    def test_cumulative_starts_near_initial(self):
        returns = _make_returns()
        result = backtest(EqualWeightStrategy(), returns, initial_capital=100000)
        # First value should be close to initial_capital * (1 + first_return)
        first_cum = result["cumulative"].iloc[0]
        assert 90000 < first_cum < 110000

    def test_weights_history_populated(self):
        returns = _make_returns()
        result = backtest(EqualWeightStrategy(), returns, rebalance_frequency=21)
        assert len(result["weights_history"]) > 0
        date, weights = result["weights_history"][0]
        assert len(weights) == 5
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-4)

    def test_metrics_has_sharpe(self):
        returns = _make_returns()
        result = backtest(EqualWeightStrategy(), returns)
        assert "sharpe_ratio" in result["metrics"]
        assert "max_drawdown" in result["metrics"]

    def test_gmv_strategy_runs(self):
        returns = _make_returns()
        result = backtest(GMVStrategy(max_weight=1.0), returns, rebalance_frequency=63)
        assert len(result["daily_returns"]) > 0

    def test_rebalance_frequency_affects_history(self):
        returns = _make_returns(n_days=504)
        fast = backtest(EqualWeightStrategy(), returns, rebalance_frequency=5)
        slow = backtest(EqualWeightStrategy(), returns, rebalance_frequency=63)
        # More frequent rebalancing -> more weight snapshots
        assert len(fast["weights_history"]) > len(slow["weights_history"])


class TestCompareStrategies:
    def test_returns_dataframe(self):
        returns = _make_returns()
        strategies = [
            EqualWeightStrategy(),
            RiskParityStrategy(max_weight=1.0),
        ]
        result = compare_strategies(strategies, returns, rebalance_frequency=63)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_index_is_strategy_names(self):
        returns = _make_returns()
        strategies = [
            EqualWeightStrategy(),
            RiskParityStrategy(max_weight=1.0),
        ]
        result = compare_strategies(strategies, returns, rebalance_frequency=63)
        assert "Equal Weight" in result.index
        assert "Risk Parity" in result.index

    def test_has_metric_columns(self):
        returns = _make_returns()
        strategies = [EqualWeightStrategy()]
        result = compare_strategies(strategies, returns, rebalance_frequency=63)
        assert "annualized_return" in result.columns
        assert "sharpe_ratio" in result.columns
        assert "max_drawdown" in result.columns
