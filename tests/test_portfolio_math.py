import numpy as np
import pytest

from gmv_algo import portfolio_return, portfolio_vol, gmv, msr


class TestPortfolioReturn:
    def test_equal_weights_equal_returns(self):
        weights = np.array([0.5, 0.5])
        returns = np.array([0.1, 0.1])
        assert portfolio_return(weights, returns) == pytest.approx(0.1)

    def test_single_asset(self):
        weights = np.array([1.0])
        returns = np.array([0.05])
        assert portfolio_return(weights, returns) == pytest.approx(0.05)

    def test_unequal_weights(self):
        weights = np.array([0.7, 0.3])
        returns = np.array([0.10, 0.20])
        expected = 0.7 * 0.10 + 0.3 * 0.20
        assert portfolio_return(weights, returns) == pytest.approx(expected)

    def test_zero_returns(self):
        weights = np.array([0.5, 0.5])
        returns = np.array([0.0, 0.0])
        assert portfolio_return(weights, returns) == pytest.approx(0.0)

    def test_negative_returns(self):
        weights = np.array([0.6, 0.4])
        returns = np.array([-0.05, 0.10])
        expected = 0.6 * (-0.05) + 0.4 * 0.10
        assert portfolio_return(weights, returns) == pytest.approx(expected)


class TestPortfolioVol:
    def test_single_asset(self):
        weights = np.array([1.0])
        covmat = np.array([[0.04]])
        assert portfolio_vol(weights, covmat) == pytest.approx(0.2)

    def test_two_uncorrelated_assets(self):
        weights = np.array([0.5, 0.5])
        covmat = np.array([[0.04, 0.0], [0.0, 0.04]])
        expected = (0.5**2 * 0.04 + 0.5**2 * 0.04) ** 0.5
        assert portfolio_vol(weights, covmat) == pytest.approx(expected)

    def test_two_correlated_assets(self):
        weights = np.array([0.6, 0.4])
        covmat = np.array([[0.04, 0.01], [0.01, 0.09]])
        expected = (weights.T @ covmat @ weights) ** 0.5
        assert portfolio_vol(weights, covmat) == pytest.approx(expected)

    def test_all_in_one_asset(self):
        weights = np.array([1.0, 0.0])
        covmat = np.array([[0.04, 0.01], [0.01, 0.09]])
        assert portfolio_vol(weights, covmat) == pytest.approx(0.2)


class TestGMV:
    def test_equal_vol_uncorrelated_gives_equal_weights(self):
        cov = np.array([[0.04, 0.0], [0.0, 0.04]])
        weights = gmv(cov)
        np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-4)

    def test_weights_sum_to_one(self):
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = gmv(cov)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self):
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16],
        ])
        weights = gmv(cov)
        assert all(w >= -1e-10 for w in weights)

    def test_prefers_lower_vol_asset(self):
        cov = np.array([[0.01, 0.0], [0.0, 0.16]])
        weights = gmv(cov)
        assert weights[0] > weights[1]

    def test_three_assets_sums_to_one(self):
        cov = np.array([
            [0.04, 0.006, 0.002],
            [0.006, 0.09, 0.009],
            [0.002, 0.009, 0.01],
        ])
        weights = gmv(cov)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-6)
        assert all(w >= -1e-10 for w in weights)


class TestMSR:
    def test_weights_sum_to_one(self):
        er = np.array([0.10, 0.15])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        weights = msr(0.02, er, cov)
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-6)

    def test_weights_non_negative(self):
        er = np.array([0.10, 0.15, 0.12])
        cov = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.09, 0.02],
            [0.005, 0.02, 0.16],
        ])
        weights = msr(0.02, er, cov)
        assert all(w >= -1e-10 for w in weights)

    def test_favors_high_sharpe_asset(self):
        er = np.array([0.20, 0.05])
        cov = np.array([[0.01, 0.0], [0.0, 0.16]])
        weights = msr(0.02, er, cov)
        assert weights[0] > weights[1]
