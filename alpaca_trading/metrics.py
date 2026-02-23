"""Portfolio performance metrics."""

import numpy as np


def annualized_return(daily_returns):
    """Compute annualized return from a Series of daily returns.

    Args:
        daily_returns: pd.Series of daily portfolio returns.

    Returns:
        Annualized return as a float.
    """
    total = (1 + daily_returns).prod()
    n_days = len(daily_returns)
    if n_days == 0:
        return 0.0
    return total ** (252 / n_days) - 1


def annualized_volatility(daily_returns):
    """Compute annualized volatility from a Series of daily returns.

    Args:
        daily_returns: pd.Series of daily portfolio returns.

    Returns:
        Annualized volatility as a float.
    """
    return daily_returns.std() * np.sqrt(252)


def sharpe_ratio(daily_returns, riskfree_rate=0.02):
    """Compute annualized Sharpe ratio.

    Args:
        daily_returns: pd.Series of daily portfolio returns.
        riskfree_rate: annualized risk-free rate (default 2%).

    Returns:
        Sharpe ratio as a float.
    """
    ann_ret = annualized_return(daily_returns)
    ann_vol = annualized_volatility(daily_returns)
    if ann_vol == 0:
        return 0.0
    return (ann_ret - riskfree_rate) / ann_vol


def max_drawdown(daily_returns):
    """Compute maximum drawdown from a Series of daily returns.

    Args:
        daily_returns: pd.Series of daily portfolio returns.

    Returns:
        Maximum drawdown as a negative float (e.g. -0.15 for 15% drawdown).
    """
    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    return drawdowns.min()


def performance_summary(daily_returns, riskfree_rate=0.02):
    """Compute a full performance summary.

    Args:
        daily_returns: pd.Series of daily portfolio returns.
        riskfree_rate: annualized risk-free rate.

    Returns:
        Dict with performance metrics.
    """
    return {
        "annualized_return": annualized_return(daily_returns),
        "annualized_volatility": annualized_volatility(daily_returns),
        "sharpe_ratio": sharpe_ratio(daily_returns, riskfree_rate),
        "max_drawdown": max_drawdown(daily_returns),
        "total_return": (1 + daily_returns).prod() - 1,
        "n_days": len(daily_returns),
    }
