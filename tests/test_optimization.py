"""Tests for portfolio optimization functions."""

import numpy as np
import pytest

from alpaca_trading.optimization import (
    clip_weights,
    gmv,
    msr,
    portfolio_return,
    portfolio_vol,
)


class TestPortfolioReturn:
    def test_equal_weights_equal_returns(self):
        weights = np.array([0.5, 0.5])
        returns = np.array([0.10, 0.10])
        assert portfolio_return(weights, returns) == pytest.approx(0.10)

    def test_concentrated_portfolio(self):
        weights = np.array([1.0, 0.0])
        returns = np.array([0.20, 0.05])
        assert portfolio_return(weights, returns) == pytest.approx(0.20)

    def test_weighted_average(self):
        weights = np.array([0.6, 0.4])
        returns = np.array([0.10, 0.20])
        expected = 0.6 * 0.10 + 0.4 * 0.20
        assert portfolio_return(weights, returns) == pytest.approx(expected)


class TestPortfolioVol:
    def test_single_asset(self):
        weights = np.array([1.0])
        covmat = np.array([[0.04]])  # vol = 0.2
        assert portfolio_vol(weights, covmat) == pytest.approx(0.2)

    def test_two_uncorrelated_assets(self):
        weights = np.array([0.5, 0.5])
        covmat = np.array([[0.04, 0.0], [0.0, 0.04]])
        # vol = sqrt(0.25*0.04 + 0.25*0.04) = sqrt(0.02)
        expected = np.sqrt(0.02)
        assert portfolio_vol(weights, covmat) == pytest.approx(expected)

    def test_perfectly_correlated_assets(self):
        # Two assets with vol=0.2, correlation=1.0
        covmat = np.array([[0.04, 0.04], [0.04, 0.04]])
        weights = np.array([0.5, 0.5])
        # vol = 0.2 (same as individual)
        assert portfolio_vol(weights, covmat) == pytest.approx(0.2)


class TestGMV:
    def test_equal_variance_uncorrelated(self):
        # Two uncorrelated assets with equal variance -> equal weights
        covmat = np.array([[0.04, 0.0], [0.0, 0.04]])
        weights = gmv(covmat)
        assert weights == pytest.approx([0.5, 0.5], abs=1e-4)

    def test_weights_sum_to_one(self):
        covmat = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = gmv(covmat)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_non_negative(self):
        covmat = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = gmv(covmat)
        assert np.all(weights >= -1e-10)

    def test_favors_lower_variance(self):
        # Asset 1 has lower variance, so GMV should allocate more to it
        covmat = np.array([[0.01, 0.005], [0.005, 0.09]])
        weights = gmv(covmat)
        assert weights[0] > weights[1]

    def test_three_assets(self):
        covmat = np.array([
            [0.04, 0.006, 0.002],
            [0.006, 0.09, 0.009],
            [0.002, 0.009, 0.01],
        ])
        weights = gmv(covmat)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-6)
        assert np.all(weights >= -1e-10)


class TestMSR:
    def test_weights_sum_to_one(self):
        er = np.array([0.10, 0.20])
        covmat = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = msr(0.02, er, covmat)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-6)

    def test_all_weights_non_negative(self):
        er = np.array([0.10, 0.20])
        covmat = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = msr(0.02, er, covmat)
        assert np.all(weights >= -1e-10)

    def test_favors_higher_sharpe(self):
        # Asset 2 has much higher return with same vol -> should get more weight
        er = np.array([0.05, 0.30])
        covmat = np.array([[0.04, 0.0], [0.0, 0.04]])
        weights = msr(0.02, er, covmat)
        assert weights[1] > weights[0]


class TestClipWeights:
    def test_no_clipping_needed(self):
        weights = np.array([0.05, 0.05, 0.90])
        clipped = clip_weights(weights, 1.0)
        assert clipped == pytest.approx(weights, abs=1e-6)

    def test_clips_and_redistributes(self):
        weights = np.array([0.5, 0.3, 0.2])
        clipped = clip_weights(weights, 0.4)
        assert np.all(clipped <= 0.4 + 1e-6)
        assert np.sum(clipped) == pytest.approx(1.0, abs=1e-6)

    def test_preserves_sum_to_one(self):
        weights = np.array([0.7, 0.2, 0.1])
        clipped = clip_weights(weights, 0.3)
        assert np.sum(clipped) == pytest.approx(1.0, abs=1e-6)

    def test_all_equal_above_cap(self):
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        clipped = clip_weights(weights, 0.10)
        assert clipped == pytest.approx([0.25, 0.25, 0.25, 0.25], abs=1e-6)
