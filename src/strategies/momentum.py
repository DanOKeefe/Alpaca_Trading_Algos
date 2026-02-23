import numpy as np
import pandas as pd

MOMENTUM_LOOKBACK = 252  # ~12 months of trading days
MOMENTUM_SKIP = 21  # skip most recent month (reversal effect)
TOP_DECILE = 0.1


class MomentumStrategy:
    """Momentum — go long the top decile of assets ranked by trailing returns.

    Uses 12-month trailing returns, skipping the most recent month
    to avoid short-term reversal. Selected assets are equal-weighted.
    """

    name = "Momentum"

    def __init__(
        self,
        lookback: int = MOMENTUM_LOOKBACK,
        skip: int = MOMENTUM_SKIP,
        top_pct: float = TOP_DECILE,
    ):
        self.lookback = lookback
        self.skip = skip
        self.top_pct = top_pct

    def calculate_weights(self, returns: pd.DataFrame) -> np.ndarray:
        n = returns.shape[1]

        if len(returns) < self.lookback:
            # Not enough data — fall back to equal weight
            return np.repeat(1.0 / n, n)

        # Trailing returns excluding the most recent `skip` days
        if self.skip > 0:
            trailing = returns.iloc[-self.lookback : -self.skip]
        else:
            trailing = returns.iloc[-self.lookback :]

        # Cumulative return per asset
        cum_returns = (1 + trailing).prod() - 1

        # Select top decile
        n_select = max(1, int(n * self.top_pct))
        top_assets = cum_returns.nlargest(n_select).index

        weights = np.zeros(n)
        for i, col in enumerate(returns.columns):
            if col in top_assets:
                weights[i] = 1.0 / n_select

        return weights
