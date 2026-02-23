"""Risk Parity strategy."""

import numpy as np
import pandas as pd

from alpaca_trading.config import MAX_WEIGHT_PER_STOCK
from alpaca_trading.optimization import clip_weights
from alpaca_trading.strategies.base import Strategy


class RiskParityStrategy(Strategy):
    """Risk Parity portfolio strategy.

    Allocates so each asset contributes equally to total portfolio risk.
    Uses the inverse-volatility heuristic: weight each asset proportional
    to 1/sigma, then normalize.
    """

    def __init__(self, max_weight=MAX_WEIGHT_PER_STOCK):
        self.max_weight = max_weight

    @property
    def name(self):
        return "Risk Parity"

    def compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        vols = returns.std()
        # Inverse volatility weights
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        weights = np.round(weights.values, 5)
        if self.max_weight < 1.0:
            weights = clip_weights(weights, self.max_weight)
        return weights
