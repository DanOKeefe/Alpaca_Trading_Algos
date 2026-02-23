import numpy as np
import pandas as pd

from src.strategies.gmv import msr

TRADING_DAYS_PER_YEAR = 252


class BlackLittermanStrategy:
    """Mean-Variance with Black-Litterman implied equilibrium returns.

    Uses equal-weight as the market portfolio to derive implied equilibrium
    returns, then optimizes for maximum Sharpe ratio with those returns.
    Views can be incorporated by subclassing and overriding `_get_views`.
    """

    name = "Black-Litterman"

    def __init__(
        self, risk_aversion: float = 2.5, tau: float = 0.05
    ):
        self.risk_aversion = risk_aversion
        self.tau = tau

    def calculate_weights(self, returns: pd.DataFrame) -> np.ndarray:
        cov = returns.cov().values * TRADING_DAYS_PER_YEAR
        n = cov.shape[0]

        # Market portfolio (equal-weight as proxy)
        w_mkt = np.repeat(1.0 / n, n)

        # Implied equilibrium excess returns
        pi = self.risk_aversion * cov @ w_mkt

        # Get views (default: no views)
        P, Q, omega = self._get_views(n)

        if P is not None and Q is not None and omega is not None:
            # Posterior expected returns with views
            tau_cov_inv = np.linalg.inv(self.tau * cov)
            omega_inv = np.linalg.inv(omega)
            posterior_cov = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P)
            posterior_mean = posterior_cov @ (
                tau_cov_inv @ pi + P.T @ omega_inv @ Q
            )
        else:
            # No views: posterior = equilibrium
            posterior_mean = pi

        return msr(0, posterior_mean, cov)

    def _get_views(self, n):
        """Return (P, Q, omega) or (None, None, None) for no views.

        Override this method to incorporate investor views.
        P: k x n pick matrix (k views on n assets)
        Q: k-vector of view returns
        omega: k x k uncertainty matrix for views
        """
        return None, None, None
