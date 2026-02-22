"""Global Minimum Variance strategy."""

import numpy as np
import pandas as pd

from alpaca_trading.config import MAX_WEIGHT_PER_STOCK
from alpaca_trading.optimization import clip_weights, gmv
from alpaca_trading.strategies.base import Strategy


class GMVStrategy(Strategy):
    """Global Minimum Variance portfolio strategy.

    Minimizes portfolio variance without requiring expected return estimates.
    """

    def __init__(self, max_weight=MAX_WEIGHT_PER_STOCK):
        self.max_weight = max_weight

    @property
    def name(self):
        return "Global Minimum Variance"

    def compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        cov = returns.cov()
        weights = gmv(cov)
        weights = np.round(weights, 5)
        if self.max_weight < 1.0:
            weights = clip_weights(weights, self.max_weight)
        return weights
