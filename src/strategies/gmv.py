import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_return(weights, returns):
    """
    Computes the return on a portfolio from constituent returns and weights.
    weights are a numpy array or Nx1 matrix and returns are a numpy array or Nx1 matrix.
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """
    Computes the vol of a portfolio from a covariance matrix and constituent weights.
    weights are a numpy array or N x 1 matrix and covmat is an N x N matrix.
    """
    return (weights.T @ covmat @ weights) ** 0.5


def gmv(cov):
    """
    Returns the weights of the Global Minimum Volatility portfolio
    given a covariance matrix.
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def msr(riskfree_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum sharpe ratio
    given the riskfree rate and expected returns and a covariance matrix.
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1,
    }

    def neg_sharpe(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    result = minimize(
        neg_sharpe,
        init_guess,
        args=(riskfree_rate, er, cov),
        method='SLSQP',
        options={'disp': False},
        constraints=(weights_sum_to_1,),
        bounds=bounds,
    )
    return result.x


class GMVStrategy:
    """Global Minimum Variance — minimizes total portfolio volatility."""

    name = "Global Minimum Variance"

    def calculate_weights(self, returns: pd.DataFrame) -> np.ndarray:
        return gmv(returns.cov())
