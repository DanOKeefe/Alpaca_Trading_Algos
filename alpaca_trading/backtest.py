"""Backtesting engine for portfolio strategies."""

import logging

import numpy as np
import pandas as pd

from alpaca_trading.metrics import performance_summary
from alpaca_trading.strategies.base import Strategy

logger = logging.getLogger(__name__)


def backtest(
    strategy: Strategy,
    returns: pd.DataFrame,
    rebalance_frequency: int = 21,
    initial_capital: float = 100000.0,
):
    """Backtest a strategy against historical returns.

    Simulates periodic rebalancing and tracks daily portfolio returns.

    Args:
        strategy: a Strategy instance.
        returns: DataFrame of daily returns (rows=dates, cols=tickers).
        rebalance_frequency: number of trading days between rebalances (default 21 ~ monthly).
        initial_capital: starting portfolio value (for reporting only).

    Returns:
        dict with keys:
            - "daily_returns": pd.Series of daily portfolio returns
            - "cumulative": pd.Series of cumulative portfolio value
            - "weights_history": list of (date, weights_array) tuples
            - "metrics": dict of performance metrics
    """
    n_days, n_assets = returns.shape
    dates = returns.index

    daily_portfolio_returns = []
    weights_history = []
    current_weights = np.full(n_assets, 1.0 / n_assets)  # start equal weight

    # Minimum lookback for covariance estimation
    min_lookback = 63  # ~3 months

    for i in range(min_lookback, n_days):
        # Rebalance on schedule
        if (i - min_lookback) % rebalance_frequency == 0:
            lookback = returns.iloc[:i]
            try:
                current_weights = strategy.compute_weights(lookback)
                weights_history.append((dates[i], current_weights.copy()))
            except Exception as e:
                logger.warning("Rebalance failed at %s: %s", dates[i], e)

        # Daily return = weighted sum of individual asset returns
        day_returns = returns.iloc[i].values
        portfolio_ret = np.dot(current_weights, day_returns)
        daily_portfolio_returns.append(portfolio_ret)

        # Update weights for drift (buy-and-hold between rebalances)
        drifted = current_weights * (1 + day_returns)
        total = drifted.sum()
        if total > 0:
            current_weights = drifted / total

    daily_returns = pd.Series(
        daily_portfolio_returns,
        index=dates[min_lookback:],
        name="portfolio_return",
    )

    cumulative = initial_capital * (1 + daily_returns).cumprod()

    metrics = performance_summary(daily_returns)

    return {
        "daily_returns": daily_returns,
        "cumulative": cumulative,
        "weights_history": weights_history,
        "metrics": metrics,
    }


def compare_strategies(
    strategies: list[Strategy],
    returns: pd.DataFrame,
    rebalance_frequency: int = 21,
):
    """Run backtest for multiple strategies and return comparative results.

    Args:
        strategies: list of Strategy instances.
        returns: DataFrame of daily returns.
        rebalance_frequency: days between rebalances.

    Returns:
        pd.DataFrame with one row per strategy and columns for each metric.
    """
    results = []
    for strategy in strategies:
        logger.info("Backtesting: %s", strategy.name)
        bt = backtest(strategy, returns, rebalance_frequency)
        row = {"strategy": strategy.name, **bt["metrics"]}
        results.append(row)

    return pd.DataFrame(results).set_index("strategy")
