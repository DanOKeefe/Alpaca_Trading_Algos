from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Strategy(Protocol):
    """Protocol that all trading strategies must implement."""

    name: str

    def calculate_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute portfolio weights from a DataFrame of daily returns.

        Args:
            returns: DataFrame where each column is an asset's daily returns.

        Returns:
            Array of portfolio weights that sum to 1.0.
        """
        ...
