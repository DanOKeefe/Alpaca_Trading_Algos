import numpy as np

TRADING_DAYS_PER_YEAR = 252


def total_return(equity_curve):
    """Total return over the entire period."""
    if len(equity_curve) < 2:
        return 0.0
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1


def annualized_return(equity_curve):
    """Annualized (CAGR) return over the period."""
    if len(equity_curve) < 2:
        return 0.0
    total = total_return(equity_curve)
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days <= 0:
        return 0.0
    years = days / 365.25
    if total <= -1:
        return -1.0
    return (1 + total) ** (1 / years) - 1


def sharpe_ratio(daily_returns, risk_free_rate=0.0):
    """Annualized Sharpe ratio from daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    std = excess.std()
    if std == 0:
        return 0.0
    return (excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(equity_curve):
    """Maximum drawdown (most negative value, e.g., -0.15 for 15% drawdown)."""
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()


def annualized_volatility(daily_returns):
    """Annualized volatility from daily returns."""
    if len(daily_returns) < 2:
        return 0.0
    return daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
