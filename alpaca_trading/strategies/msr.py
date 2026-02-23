"""Maximum Sharpe Ratio strategy."""

import numpy as np
import pandas as pd

from alpaca_trading.config import MAX_WEIGHT_PER_STOCK
from alpaca_trading.optimization import clip_weights, msr
from alpaca_trading.strategies.base import Strategy


class MSRStrategy(Strategy):
    """Maximum Sharpe Ratio portfolio strategy.

    Maximizes the Sharpe ratio using historical mean returns as expected
    return estimates.
    """

    def __init__(self, riskfree_rate=0.02, max_weight=MAX_WEIGHT_PER_STOCK):
        self.riskfree_rate = riskfree_rate
        self.max_weight = max_weight

    @property
    def name(self):
        return "Maximum Sharpe Ratio"

    def compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        # Annualize daily returns for expected return estimates
        er = returns.mean() * 252
        cov = returns.cov() * 252
        weights = msr(self.riskfree_rate, er.values, cov.values)
        weights = np.round(weights, 5)
        if self.max_weight < 1.0:
            weights = clip_weights(weights, self.max_weight)
        return weights
