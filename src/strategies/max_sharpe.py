import numpy as np
import pandas as pd

from src.strategies.gmv import msr

TRADING_DAYS_PER_YEAR = 252


class MaxSharpeStrategy:
    """Maximum Sharpe Ratio — maximizes risk-adjusted return.

    Uses trailing mean returns (annualized) as expected returns.
    """

    name = "Maximum Sharpe Ratio"

    def __init__(self, risk_free_rate: float = 0.0):
        self.risk_free_rate = risk_free_rate

    def calculate_weights(self, returns: pd.DataFrame) -> np.ndarray:
        er = returns.mean().values * TRADING_DAYS_PER_YEAR
        cov = returns.cov().values
        return msr(self.risk_free_rate, er, cov)
