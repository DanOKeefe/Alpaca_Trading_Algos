import numpy as np
import pandas as pd
import pytest

from backtests.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    total_return,
)


class TestTotalReturn:
    def test_positive_return(self):
        curve = pd.Series([100, 110, 120])
        assert total_return(curve) == pytest.approx(0.20)

    def test_negative_return(self):
        curve = pd.Series([100, 90, 80])
        assert total_return(curve) == pytest.approx(-0.20)

    def test_zero_return(self):
        curve = pd.Series([100, 100, 100])
        assert total_return(curve) == pytest.approx(0.0)

    def test_single_point_returns_zero(self):
        curve = pd.Series([100])
        assert total_return(curve) == 0.0

    def test_empty_returns_zero(self):
        curve = pd.Series([], dtype=float)
        assert total_return(curve) == 0.0


class TestAnnualizedReturn:
    def test_one_year_doubling(self):
        # Double in exactly 365 days
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        curve = pd.Series(
            np.linspace(100, 200, 252), index=dates
        )
        result = annualized_return(curve)
        # Approximate — not exactly 100% due to number of days
        assert result > 0.5

    def test_flat_returns_zero(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        curve = pd.Series([100] * 100, index=dates)
        assert annualized_return(curve) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        curve = pd.Series([], dtype=float)
        assert annualized_return(curve) == 0.0


class TestSharpeRatio:
    def test_positive_sharpe(self):
        # Consistent small positive returns => positive Sharpe
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0.001, 0.01, 252))
        s = sharpe_ratio(rets)
        assert s > 0

    def test_zero_vol_returns_zero(self):
        rets = pd.Series([0.0, 0.0, 0.0])
        assert sharpe_ratio(rets) == 0.0

    def test_negative_returns_negative_sharpe(self):
        np.random.seed(42)
        rets = pd.Series(np.random.normal(-0.002, 0.01, 252))
        s = sharpe_ratio(rets)
        assert s < 0

    def test_empty_returns_zero(self):
        rets = pd.Series([], dtype=float)
        assert sharpe_ratio(rets) == 0.0


class TestMaxDrawdown:
    def test_simple_drawdown(self):
        # Goes up 50%, then drops by 1/3 (from 150 to 100)
        curve = pd.Series([100, 150, 100])
        dd = max_drawdown(curve)
        assert dd == pytest.approx(-1 / 3, abs=1e-6)

    def test_no_drawdown(self):
        curve = pd.Series([100, 110, 120, 130])
        dd = max_drawdown(curve)
        assert dd == pytest.approx(0.0)

    def test_total_loss(self):
        curve = pd.Series([100, 50, 10])
        dd = max_drawdown(curve)
        assert dd == pytest.approx(-0.90)

    def test_empty_returns_zero(self):
        curve = pd.Series([], dtype=float)
        assert max_drawdown(curve) == 0.0


class TestAnnualizedVolatility:
    def test_known_vol(self):
        np.random.seed(42)
        daily_vol = 0.01
        rets = pd.Series(np.random.normal(0, daily_vol, 10000))
        ann_vol = annualized_volatility(rets)
        expected = daily_vol * np.sqrt(252)
        assert ann_vol == pytest.approx(expected, rel=0.05)

    def test_zero_vol(self):
        rets = pd.Series([0.01, 0.01, 0.01])
        assert annualized_volatility(rets) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        rets = pd.Series([], dtype=float)
        assert annualized_volatility(rets) == 0.0
