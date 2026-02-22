"""Portfolio optimization functions."""

import numpy as np
from scipy.optimize import minimize


def portfolio_return(weights, returns):
    """Compute portfolio return from constituent returns and weights.

    Args:
        weights: numpy array of portfolio weights (Nx1).
        returns: numpy array of expected returns (Nx1).

    Returns:
        Scalar portfolio return.
    """
    return weights.T @ returns


def portfolio_vol(weights, covmat):
    """Compute portfolio volatility from a covariance matrix and weights.

    Args:
        weights: numpy array of portfolio weights (Nx1).
        covmat: NxN covariance matrix.

    Returns:
        Scalar portfolio volatility.
    """
    return (weights.T @ covmat @ weights) ** 0.5


def msr(riskfree_rate, er, cov):
    """Compute Maximum Sharpe Ratio portfolio weights.

    Args:
        riskfree_rate: risk-free rate of return.
        er: numpy array of expected returns.
        cov: NxN covariance matrix.

    Returns:
        numpy array of optimal weights.
    """
    n = er.shape[0]
    init_guess = np.repeat(1 / n, n)
    bounds = ((0.0, 1.0),) * n

    weights_sum_to_1 = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    def neg_sharpe(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate) / vol

    result = minimize(
        neg_sharpe,
        init_guess,
        args=(riskfree_rate, er, cov),
        method="SLSQP",
        options={"disp": False},
        constraints=(weights_sum_to_1,),
        bounds=bounds,
    )
    return result.x


def gmv(cov):
    """Compute Global Minimum Variance portfolio weights.

    Args:
        cov: NxN covariance matrix.

    Returns:
        numpy array of optimal weights.
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def clip_weights(weights, max_weight):
    """Clip weights to a maximum value and redistribute excess proportionally.

    Iteratively clips and redistributes until no weight exceeds the cap.

    Args:
        weights: numpy array of portfolio weights.
        max_weight: maximum allowed weight per asset (e.g. 0.10 for 10%).

    Returns:
        numpy array of clipped weights that sum to 1.
    """
    clipped = weights.copy()
    for _ in range(100):  # safeguard against infinite loops
        over_mask = clipped > max_weight + 1e-10
        if not over_mask.any():
            break
        excess = (clipped[over_mask] - max_weight).sum()
        clipped[over_mask] = max_weight
        under_mask = clipped < max_weight - 1e-10
        if not under_mask.any():
            break
        under_total = clipped[under_mask].sum()
        if under_total > 0:
            clipped[under_mask] += excess * (clipped[under_mask] / under_total)
    # Final normalization: only scale uncapped weights to avoid pushing capped ones over
    total = clipped.sum()
    if abs(total - 1.0) > 1e-10:
        capped = clipped >= max_weight - 1e-10
        capped_sum = clipped[capped].sum()
        remaining = 1.0 - capped_sum
        uncapped = ~capped
        if uncapped.any() and clipped[uncapped].sum() > 0:
            clipped[uncapped] = clipped[uncapped] * (remaining / clipped[uncapped].sum())
        else:
            clipped = clipped / total
    return clipped
