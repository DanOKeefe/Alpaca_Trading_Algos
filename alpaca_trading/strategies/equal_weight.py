"""Equal Weight strategy."""

import numpy as np
import pandas as pd

from alpaca_trading.strategies.base import Strategy


class EqualWeightStrategy(Strategy):
    """Equal Weight (1/N) portfolio strategy.

    Allocates equally across all assets. Serves as a simple baseline.
    """

    @property
    def name(self):
        return "Equal Weight"

    def compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        n = returns.shape[1]
        return np.full(n, 1.0 / n)
