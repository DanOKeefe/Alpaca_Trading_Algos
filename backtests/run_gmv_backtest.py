"""
Run GMV and equal-weight backtests over multiple time windows.

Usage:
    python -m backtests.run_gmv_backtest
"""

import logging
import os

import numpy as np

from backtests.engine import BacktestConfig, Backtester
from backtests.report import comparison_table, html_report, text_report
from backtests.visualize import plot_drawdowns, plot_equity_curves
from src.strategies.gmv import gmv

logger = logging.getLogger(__name__)


def gmv_weight_fn(returns):
    """Compute Global Minimum Variance weights from a returns DataFrame."""
    return gmv(returns.cov())


def equal_weight_fn(returns):
    """Equal-weight allocation across all assets."""
    n = returns.shape[1]
    return np.repeat(1 / n, n)


# S&P 100 tickers (subset for faster backtesting)
SP100_SAMPLE = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BLK", "BMY", "BRK-B", "C", "CAT",
    "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX",
    "GD", "GE", "GILD", "GM", "GOOG", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT",
    "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM",
    "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT",
    "TMO", "TMUS", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ",
    "WBA", "WFC", "WMT", "XOM",
]

# Backtest windows: (label, start_date, end_date)
WINDOWS = [
    ("1 Year", "2024-01-01", "2024-12-31"),
    ("3 Years", "2022-01-01", "2024-12-31"),
    ("5 Years", "2020-01-01", "2024-12-31"),
    ("10 Years", "2015-01-01", "2024-12-31"),
]


def run_all():
    """Run GMV and equal-weight backtests over all windows."""
    all_results = []

    for label, start, end in WINDOWS:
        config = BacktestConfig(
            start_date=start,
            end_date=end,
            initial_capital=100_000,
            rebalance_freq="BME",
            lookback_years=5,
        )

        logger.info("Running GMV backtest: %s", label)
        gmv_bt = Backtester(gmv_weight_fn, SP100_SAMPLE, config)
        gmv_result = gmv_bt.run(strategy_name=f"GMV ({label})")
        all_results.append(gmv_result)

        logger.info("Running Equal-Weight backtest: %s", label)
        ew_bt = Backtester(equal_weight_fn, SP100_SAMPLE, config)
        ew_result = ew_bt.run(strategy_name=f"Equal-Weight ({label})")
        all_results.append(ew_result)

    # Print individual reports
    for r in all_results:
        print(text_report(r))
        print()

    # Print comparison table
    print(comparison_table(all_results))

    # Save HTML report
    os.makedirs("backtests/output", exist_ok=True)
    html = html_report(all_results)
    with open("backtests/output/report.html", "w") as f:
        f.write(html)
    logger.info("HTML report saved to backtests/output/report.html")

    # Save charts for the longest window (10-year)
    ten_yr_results = [r for r in all_results if "10 Years" in r.strategy_name]
    if ten_yr_results:
        plot_equity_curves(
            ten_yr_results,
            save_path="backtests/output/equity_curves.png",
        )
        plot_drawdowns(
            ten_yr_results,
            save_path="backtests/output/drawdowns.png",
        )
        logger.info("Charts saved to backtests/output/")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all()
