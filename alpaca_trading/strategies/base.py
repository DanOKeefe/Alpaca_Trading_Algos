"""Abstract base class for trading strategies."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Strategy(ABC):
    """Interface that all trading strategies must implement."""

    @abstractmethod
    def compute_weights(self, returns: pd.DataFrame) -> np.ndarray:
        """Compute portfolio weights from historical returns.

        Args:
            returns: DataFrame of daily returns (rows=dates, cols=tickers).

        Returns:
            numpy array of portfolio weights (one per ticker).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""
        ...
