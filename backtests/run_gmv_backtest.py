"""
Run all strategies against equal-weight benchmark over multiple time windows.

Usage:
    python -m backtests.run_gmv_backtest
"""

import logging
import os

import numpy as np

from backtests.engine import BacktestConfig, Backtester
from backtests.report import comparison_table, html_report, text_report
from backtests.visualize import plot_drawdowns, plot_equity_curves
from src.strategies import (
    BlackLittermanStrategy,
    GMVStrategy,
    MaxSharpeStrategy,
    MomentumStrategy,
    RiskParityStrategy,
)

logger = logging.getLogger(__name__)


def equal_weight_fn(returns):
    """Equal-weight allocation across all assets."""
    n = returns.shape[1]
    return np.repeat(1 / n, n)


# All strategies to benchmark
STRATEGIES = [
    GMVStrategy(),
    MaxSharpeStrategy(),
    RiskParityStrategy(),
    MomentumStrategy(),
    BlackLittermanStrategy(),
]

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
    """Run all strategies and equal-weight benchmark over all windows."""
    all_results = []

    for label, start, end in WINDOWS:
        config = BacktestConfig(
            start_date=start,
            end_date=end,
            initial_capital=100_000,
            rebalance_freq="BME",
            lookback_years=5,
        )

        for strategy in STRATEGIES:
            logger.info("Running %s backtest: %s", strategy.name, label)
            bt = Backtester(
                strategy.calculate_weights, SP100_SAMPLE, config
            )
            result = bt.run(
                strategy_name=f"{strategy.name} ({label})"
            )
            all_results.append(result)

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
    ten_yr = [r for r in all_results if "10 Years" in r.strategy_name]
    if ten_yr:
        plot_equity_curves(
            ten_yr, save_path="backtests/output/equity_curves.png"
        )
        plot_drawdowns(
            ten_yr, save_path="backtests/output/drawdowns.png"
        )
        logger.info("Charts saved to backtests/output/")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_all()
