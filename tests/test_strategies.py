"""Tests for strategy implementations."""

import numpy as np
import pandas as pd
import pytest

from alpaca_trading.strategies.gmv import GMVStrategy


class TestGMVStrategy:
    def _make_returns(self, n_assets=3):
        """Create a small synthetic returns DataFrame."""
        np.random.seed(42)
        n_days = 252
        cols = [f"ASSET_{i}" for i in range(n_assets)]
        # Give each asset different volatility so GMV produces non-equal weights
        vols = np.linspace(0.005, 0.03, n_assets)
        data = np.random.randn(n_days, n_assets) * vols
        return pd.DataFrame(data, columns=cols)

    def test_weights_sum_to_one(self):
        returns = self._make_returns()
        strategy = GMVStrategy(max_weight=1.0)
        weights = strategy.compute_weights(returns)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-4)

    def test_all_weights_non_negative(self):
        returns = self._make_returns()
        strategy = GMVStrategy(max_weight=1.0)
        weights = strategy.compute_weights(returns)
        assert np.all(weights >= -1e-10)

    def test_max_weight_constraint(self):
        # Use 20 assets so 10% cap is feasible (20 * 0.10 >= 1.0)
        returns = self._make_returns(n_assets=20)
        strategy = GMVStrategy(max_weight=0.10)
        weights = strategy.compute_weights(returns)
        assert np.all(weights <= 0.10 + 1e-6)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-4)

    def test_name(self):
        strategy = GMVStrategy()
        assert strategy.name == "Global Minimum Variance"
