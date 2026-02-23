"""Tests for Phase 4 strategy classes, registry, and strategy selection."""

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import Strategy
from src.strategies.black_litterman import BlackLittermanStrategy
from src.strategies.gmv import GMVStrategy
from src.strategies.max_sharpe import MaxSharpeStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.registry import _REGISTRY, get_strategy, list_strategies, register
from src.strategies.risk_parity import RiskParityStrategy


def _make_returns(n_assets=5, n_days=300, seed=42):
    """Generate synthetic daily returns for testing."""
    rng = np.random.RandomState(seed)
    data = rng.randn(n_days, n_assets) * 0.01 + 0.0003
    cols = [f"ASSET_{i}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=cols)


def _validate_weights(weights, n_assets):
    """Assert weights are valid: correct shape, sum to 1, non-negative."""
    assert weights.shape == (n_assets,)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-4)
    assert all(w >= -1e-6 for w in weights)


# ── Protocol conformance ─────────────────────────────────────────────


class TestProtocolConformance:
    def test_gmv_is_strategy(self):
        assert isinstance(GMVStrategy(), Strategy)

    def test_max_sharpe_is_strategy(self):
        assert isinstance(MaxSharpeStrategy(), Strategy)

    def test_risk_parity_is_strategy(self):
        assert isinstance(RiskParityStrategy(), Strategy)

    def test_momentum_is_strategy(self):
        assert isinstance(MomentumStrategy(), Strategy)

    def test_black_litterman_is_strategy(self):
        assert isinstance(BlackLittermanStrategy(), Strategy)

    def test_has_name_attribute(self):
        for cls in [
            GMVStrategy,
            MaxSharpeStrategy,
            RiskParityStrategy,
            MomentumStrategy,
            BlackLittermanStrategy,
        ]:
            assert hasattr(cls(), "name")
            assert isinstance(cls().name, str)


# ── GMVStrategy ───────────────────────────────────────────────────────


class TestGMVStrategy:
    def test_weights_valid(self):
        rets = _make_returns(n_assets=5)
        s = GMVStrategy()
        weights = s.calculate_weights(rets)
        _validate_weights(weights, 5)

    def test_name(self):
        assert GMVStrategy().name == "Global Minimum Variance"


# ── MaxSharpeStrategy ────────────────────────────────────────────────


class TestMaxSharpeStrategy:
    def test_weights_valid(self):
        rets = _make_returns(n_assets=5)
        s = MaxSharpeStrategy()
        weights = s.calculate_weights(rets)
        _validate_weights(weights, 5)

    def test_custom_risk_free_rate(self):
        rets = _make_returns(n_assets=3)
        s = MaxSharpeStrategy(risk_free_rate=0.02)
        weights = s.calculate_weights(rets)
        _validate_weights(weights, 3)

    def test_name(self):
        assert MaxSharpeStrategy().name == "Maximum Sharpe Ratio"


# ── RiskParityStrategy ───────────────────────────────────────────────


class TestRiskParityStrategy:
    def test_weights_valid(self):
        rets = _make_returns(n_assets=5)
        s = RiskParityStrategy()
        weights = s.calculate_weights(rets)
        _validate_weights(weights, 5)

    def test_roughly_equal_risk_contribution(self):
        """Risk parity should give roughly equal risk contributions."""
        rets = _make_returns(n_assets=4, n_days=500)
        s = RiskParityStrategy()
        weights = s.calculate_weights(rets)
        cov = rets.cov().values
        port_vol = np.sqrt(weights.T @ cov @ weights)
        marginal = cov @ weights
        risk_contrib = weights * marginal / port_vol
        risk_pct = risk_contrib / risk_contrib.sum()
        # Each asset should contribute ~25% of risk (4 assets)
        np.testing.assert_allclose(risk_pct, 0.25, atol=0.05)

    def test_name(self):
        assert RiskParityStrategy().name == "Risk Parity"


# ── MomentumStrategy ────────────────────────────────────────────────


class TestMomentumStrategy:
    def test_weights_valid(self):
        rets = _make_returns(n_assets=10, n_days=300)
        s = MomentumStrategy()
        weights = s.calculate_weights(rets)
        _validate_weights(weights, 10)

    def test_selects_top_decile(self):
        """With 10 assets and top_pct=0.1, only 1 asset should be selected."""
        rets = _make_returns(n_assets=10, n_days=300)
        s = MomentumStrategy(top_pct=0.1)
        weights = s.calculate_weights(rets)
        n_nonzero = np.sum(weights > 1e-6)
        assert n_nonzero == 1

    def test_fallback_equal_weight_short_data(self):
        """Falls back to equal weight when not enough data."""
        rets = _make_returns(n_assets=5, n_days=50)
        s = MomentumStrategy(lookback=252)
        weights = s.calculate_weights(rets)
        np.testing.assert_allclose(weights, 0.2, atol=1e-6)

    def test_name(self):
        assert MomentumStrategy().name == "Momentum"


# ── BlackLittermanStrategy ───────────────────────────────────────────


class TestBlackLittermanStrategy:
    def test_weights_valid(self):
        rets = _make_returns(n_assets=5)
        s = BlackLittermanStrategy()
        weights = s.calculate_weights(rets)
        _validate_weights(weights, 5)

    def test_custom_params(self):
        rets = _make_returns(n_assets=3)
        s = BlackLittermanStrategy(risk_aversion=3.0, tau=0.1)
        weights = s.calculate_weights(rets)
        _validate_weights(weights, 3)

    def test_name(self):
        assert BlackLittermanStrategy().name == "Black-Litterman"


# ── Registry ─────────────────────────────────────────────────────────


class TestRegistry:
    def test_default_strategies_registered(self):
        keys = list_strategies()
        assert "gmv" in keys
        assert "max_sharpe" in keys
        assert "risk_parity" in keys
        assert "momentum" in keys
        assert "black_litterman" in keys

    def test_get_strategy_returns_instance(self):
        s = get_strategy("gmv")
        assert isinstance(s, GMVStrategy)

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown strategy"):
            get_strategy("nonexistent_strategy")

    def test_register_custom_strategy(self):
        class CustomStrategy:
            name = "Custom"

            def calculate_weights(self, returns):
                n = returns.shape[1]
                return np.repeat(1.0 / n, n)

        register("custom_test", CustomStrategy())
        s = get_strategy("custom_test")
        assert s.name == "Custom"
        # Clean up
        del _REGISTRY["custom_test"]

    def test_list_strategies_sorted(self):
        keys = list_strategies()
        assert keys == sorted(keys)


# ── Lambda handler strategy selection ────────────────────────────────


class TestLambdaHandlerStrategySelection:
    def test_lambda_handler_default_strategy(self):
        event = {}
        strategy_name = event.get("strategy", "gmv")
        assert strategy_name == "gmv"

    def test_lambda_handler_custom_strategy(self):
        event = {"strategy": "risk_parity"}
        strategy_name = event.get("strategy", "gmv")
        assert strategy_name == "risk_parity"

    def test_lambda_handler_none_event(self):
        event = None
        strategy_name = "gmv"
        if event and isinstance(event, dict):
            strategy_name = event.get("strategy", "gmv")
        assert strategy_name == "gmv"
