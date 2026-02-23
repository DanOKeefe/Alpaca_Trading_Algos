"""Tests for strategy implementations."""

import numpy as np
import pandas as pd
import pytest

from alpaca_trading.strategies.equal_weight import EqualWeightStrategy
from alpaca_trading.strategies.gmv import GMVStrategy
from alpaca_trading.strategies.msr import MSRStrategy
from alpaca_trading.strategies.risk_parity import RiskParityStrategy


def _make_returns(n_assets=3):
    """Create a small synthetic returns DataFrame with different volatilities."""
    np.random.seed(42)
    n_days = 252
    cols = [f"ASSET_{i}" for i in range(n_assets)]
    vols = np.linspace(0.005, 0.03, n_assets)
    data = np.random.randn(n_days, n_assets) * vols
    return pd.DataFrame(data, columns=cols)


class TestGMVStrategy:
    def test_weights_sum_to_one(self):
        returns = _make_returns()
        strategy = GMVStrategy(max_weight=1.0)
        weights = strategy.compute_weights(returns)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-4)

    def test_all_weights_non_negative(self):
        returns = _make_returns()
        strategy = GMVStrategy(max_weight=1.0)
        weights = strategy.compute_weights(returns)
        assert np.all(weights >= -1e-10)

    def test_max_weight_constraint(self):
        returns = _make_returns(n_assets=20)
        strategy = GMVStrategy(max_weight=0.10)
        weights = strategy.compute_weights(returns)
        assert np.all(weights <= 0.10 + 1e-6)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-4)

    def test_name(self):
        strategy = GMVStrategy()
        assert strategy.name == "Global Minimum Variance"


class TestMSRStrategy:
    def test_weights_sum_to_one(self):
        returns = _make_returns(n_assets=5)
        strategy = MSRStrategy(riskfree_rate=0.02, max_weight=1.0)
        weights = strategy.compute_weights(returns)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-4)

    def test_all_weights_non_negative(self):
        returns = _make_returns(n_assets=5)
        strategy = MSRStrategy(riskfree_rate=0.02, max_weight=1.0)
        weights = strategy.compute_weights(returns)
        assert np.all(weights >= -1e-10)

    def test_max_weight_constraint(self):
        returns = _make_returns(n_assets=20)
        strategy = MSRStrategy(riskfree_rate=0.02, max_weight=0.10)
        weights = strategy.compute_weights(returns)
        assert np.all(weights <= 0.10 + 1e-6)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-4)

    def test_name(self):
        strategy = MSRStrategy()
        assert strategy.name == "Maximum Sharpe Ratio"

    def test_favors_higher_return_asset(self):
        """With divergent returns, MSR should overweight the higher-return asset."""
        np.random.seed(42)
        n_days = 252
        # Asset 0: positive drift, Asset 1: near-zero drift
        data = np.column_stack([
            np.random.randn(n_days) * 0.01 + 0.001,
            np.random.randn(n_days) * 0.01 - 0.001,
        ])
        returns = pd.DataFrame(data, columns=["WINNER", "LOSER"])
        strategy = MSRStrategy(riskfree_rate=0.0, max_weight=1.0)
        weights = strategy.compute_weights(returns)
        assert weights[0] > weights[1]


class TestEqualWeightStrategy:
    def test_weights_sum_to_one(self):
        returns = _make_returns(n_assets=10)
        strategy = EqualWeightStrategy()
        weights = strategy.compute_weights(returns)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-10)

    def test_all_weights_equal(self):
        returns = _make_returns(n_assets=5)
        strategy = EqualWeightStrategy()
        weights = strategy.compute_weights(returns)
        assert weights == pytest.approx(np.full(5, 0.2), abs=1e-10)

    def test_name(self):
        strategy = EqualWeightStrategy()
        assert strategy.name == "Equal Weight"

    def test_single_asset(self):
        returns = _make_returns(n_assets=1)
        strategy = EqualWeightStrategy()
        weights = strategy.compute_weights(returns)
        assert weights == pytest.approx([1.0])


class TestRiskParityStrategy:
    def test_weights_sum_to_one(self):
        returns = _make_returns(n_assets=5)
        strategy = RiskParityStrategy(max_weight=1.0)
        weights = strategy.compute_weights(returns)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-4)

    def test_all_weights_non_negative(self):
        returns = _make_returns(n_assets=5)
        strategy = RiskParityStrategy(max_weight=1.0)
        weights = strategy.compute_weights(returns)
        assert np.all(weights >= -1e-10)

    def test_max_weight_constraint(self):
        returns = _make_returns(n_assets=20)
        strategy = RiskParityStrategy(max_weight=0.10)
        weights = strategy.compute_weights(returns)
        assert np.all(weights <= 0.10 + 1e-6)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-4)

    def test_name(self):
        strategy = RiskParityStrategy()
        assert strategy.name == "Risk Parity"

    def test_overweights_low_vol_asset(self):
        """Risk parity should allocate more to lower-volatility assets."""
        returns = _make_returns(n_assets=3)
        strategy = RiskParityStrategy(max_weight=1.0)
        weights = strategy.compute_weights(returns)
        # ASSET_0 has lowest vol (0.005), ASSET_2 has highest (0.03)
        assert weights[0] > weights[2]
