"""Tests for performance metrics."""

import numpy as np
import pandas as pd
import pytest

from alpaca_trading.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    performance_summary,
    sharpe_ratio,
)


class TestAnnualizedReturn:
    def test_zero_returns(self):
        rets = pd.Series([0.0] * 252)
        assert annualized_return(rets) == pytest.approx(0.0, abs=1e-6)

    def test_positive_returns(self):
        # 0.1% daily for 252 days
        rets = pd.Series([0.001] * 252)
        result = annualized_return(rets)
        assert result > 0.2  # should be ~28.6%

    def test_empty_returns(self):
        rets = pd.Series([], dtype=float)
        assert annualized_return(rets) == 0.0


class TestAnnualizedVolatility:
    def test_zero_volatility(self):
        rets = pd.Series([0.01] * 100)
        assert annualized_volatility(rets) == pytest.approx(0.0, abs=1e-10)

    def test_known_daily_vol(self):
        # Daily vol of 1% -> annualized ~ 15.87%
        np.random.seed(42)
        rets = pd.Series(np.random.randn(10000) * 0.01)
        ann_vol = annualized_volatility(rets)
        assert ann_vol == pytest.approx(0.01 * np.sqrt(252), abs=0.01)


class TestSharpeRatio:
    def test_positive_sharpe(self):
        # Consistent positive returns -> positive Sharpe
        rets = pd.Series([0.001] * 252)
        sr = sharpe_ratio(rets, riskfree_rate=0.02)
        assert sr > 0

    def test_zero_vol_returns_zero(self):
        rets = pd.Series([0.0] * 252)
        assert sharpe_ratio(rets) == 0.0


class TestMaxDrawdown:
    def test_no_drawdown(self):
        # Monotonically increasing
        rets = pd.Series([0.01] * 100)
        assert max_drawdown(rets) == pytest.approx(0.0, abs=1e-10)

    def test_known_drawdown(self):
        # Go up 10%, then down 20%, then up 5%
        rets = pd.Series([0.10, -0.20, 0.05])
        dd = max_drawdown(rets)
        assert dd < 0
        # Peak is 1.1, trough is 1.1*0.8 = 0.88, drawdown = 0.88/1.1 - 1 = -0.2
        assert dd == pytest.approx(-0.2, abs=1e-10)

    def test_full_loss(self):
        rets = pd.Series([0.5, -1.0])
        dd = max_drawdown(rets)
        assert dd == pytest.approx(-1.0, abs=1e-10)


class TestPerformanceSummary:
    def test_returns_all_keys(self):
        np.random.seed(42)
        rets = pd.Series(np.random.randn(252) * 0.01)
        summary = performance_summary(rets)
        assert "annualized_return" in summary
        assert "annualized_volatility" in summary
        assert "sharpe_ratio" in summary
        assert "max_drawdown" in summary
        assert "total_return" in summary
        assert "n_days" in summary

    def test_n_days_correct(self):
        rets = pd.Series([0.01] * 100)
        summary = performance_summary(rets)
        assert summary["n_days"] == 100
