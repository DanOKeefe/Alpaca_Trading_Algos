import numpy as np
import pandas as pd
from scipy.optimize import minimize


class RiskParityStrategy:
    """Risk Parity — each asset contributes equally to portfolio risk."""

    name = "Risk Parity"

    def calculate_weights(self, returns: pd.DataFrame) -> np.ndarray:
        cov = returns.cov().values
        n = cov.shape[0]
        target_contrib = np.repeat(1.0 / n, n)

        def objective(weights):
            port_vol = np.sqrt(weights.T @ cov @ weights)
            if port_vol < 1e-12:
                return 0.0
            marginal = cov @ weights
            risk_contrib = weights * marginal / port_vol
            total_risk = risk_contrib.sum()
            if total_risk < 1e-12:
                return 0.0
            risk_pct = risk_contrib / total_risk
            return np.sum((risk_pct - target_contrib) ** 2)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = ((1e-4, 1.0),) * n
        init = np.repeat(1.0 / n, n)

        result = minimize(
            objective,
            init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"disp": False, "maxiter": 1000},
        )
        return result.x
