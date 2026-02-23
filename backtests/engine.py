import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta

from backtests.metrics import (
    annualized_return,
    annualized_volatility,
    max_drawdown,
    sharpe_ratio,
    total_return,
)

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252
MIN_OBSERVATIONS = 60


@dataclass
class BacktestConfig:
    start_date: str  # YYYY-MM-DD
    end_date: str  # YYYY-MM-DD
    initial_capital: float = 100_000.0
    rebalance_freq: str = "BME"  # pandas offset: BME=business month end
    lookback_years: int = 5
    risk_free_rate: float = 0.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series = field(default_factory=pd.Series)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    weights_history: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    config: BacktestConfig = field(default_factory=BacktestConfig)
    strategy_name: str = ""


class Backtester:
    """Replay historical data through a strategy and record simulated trades."""

    def __init__(self, weight_fn, tickers, config):
        """
        Args:
            weight_fn: callable(pd.DataFrame) -> np.ndarray
                Takes a DataFrame of daily returns and returns portfolio weights.
            tickers: list of ticker symbols.
            config: BacktestConfig with date range and parameters.
        """
        self.weight_fn = weight_fn
        self.tickers = tickers
        self.config = config

    def run(self, strategy_name="Strategy"):
        """Execute the backtest and return a BacktestResult."""
        prices = self._download_prices()
        if prices.empty:
            logger.error("No price data downloaded")
            return BacktestResult(
                config=self.config, strategy_name=strategy_name
            )

        daily_rets = prices.pct_change().iloc[1:]

        backtest_start = pd.Timestamp(self.config.start_date)
        backtest_end = pd.Timestamp(self.config.end_date)

        # Trading days within the backtest window
        mask = (daily_rets.index >= backtest_start) & (
            daily_rets.index <= backtest_end
        )
        trading_days = daily_rets.index[mask]
        if len(trading_days) == 0:
            logger.error("No trading days in the specified range")
            return BacktestResult(
                config=self.config, strategy_name=strategy_name
            )

        # Generate rebalance dates and snap to actual trading days
        rebal_schedule = pd.date_range(
            start=backtest_start, end=backtest_end, freq=self.config.rebalance_freq,
        )
        rebalance_dates = []
        for d in rebal_schedule:
            candidates = trading_days[trading_days >= d]
            if len(candidates) > 0:
                rebalance_dates.append(candidates[0])
        # Ensure the first trading day is a rebalance point
        if len(rebalance_dates) == 0 or rebalance_dates[0] != trading_days[0]:
            rebalance_dates.insert(0, trading_days[0])

        # Run the backtest
        equity = self.config.initial_capital
        equity_values = {}
        weights_history = {}
        current_weights = None
        current_stocks = None

        rebal_set = set(rebalance_dates)

        for date in trading_days:
            if date in rebal_set:
                new_weights, new_stocks = self._compute_weights(
                    daily_rets, date
                )
                if new_weights is not None:
                    current_weights = new_weights
                    current_stocks = new_stocks
                    weights_history[date] = pd.Series(
                        new_weights, index=new_stocks
                    )

            if current_weights is not None and current_stocks is not None:
                day_rets = daily_rets.loc[date, current_stocks]
                if day_rets.isna().any():
                    day_rets = day_rets.fillna(0.0)
                port_ret = np.dot(current_weights, day_rets.values)
                equity *= 1 + port_ret

            equity_values[date] = equity

        equity_curve = pd.Series(equity_values, name="Portfolio Value")
        equity_curve.index.name = "Date"

        port_daily_returns = equity_curve.pct_change().dropna()

        metrics = {
            "total_return": total_return(equity_curve),
            "annualized_return": annualized_return(equity_curve),
            "sharpe_ratio": sharpe_ratio(
                port_daily_returns, self.config.risk_free_rate
            ),
            "max_drawdown": max_drawdown(equity_curve),
            "annualized_volatility": annualized_volatility(port_daily_returns),
        }

        return BacktestResult(
            equity_curve=equity_curve,
            daily_returns=port_daily_returns,
            weights_history=weights_history,
            metrics=metrics,
            config=self.config,
            strategy_name=strategy_name,
        )

    def _download_prices(self):
        """Download adjusted close prices for the full required period."""
        data_start = (
            pd.Timestamp(self.config.start_date)
            - relativedelta(years=self.config.lookback_years)
            - pd.Timedelta(days=30)  # buffer
        )
        tickers_str = " ".join(self.tickers)
        logger.info(
            "Downloading data for %d tickers from %s to %s",
            len(self.tickers),
            data_start.strftime("%Y-%m-%d"),
            self.config.end_date,
        )
        data = yf.download(
            tickers_str,
            start=data_start.strftime("%Y-%m-%d"),
            end=self.config.end_date,
        )
        if data.empty:
            return pd.DataFrame()
        prices = data["Adj Close"]
        prices = prices.dropna(axis=1, how="all")
        return prices

    def _compute_weights(self, daily_rets, rebal_date):
        """Compute strategy weights using trailing data up to rebal_date."""
        lookback_start = rebal_date - relativedelta(
            years=self.config.lookback_years
        )
        trailing = daily_rets.loc[lookback_start:rebal_date]

        # Keep stocks with sufficient data
        valid_counts = trailing.count()
        valid_stocks = valid_counts[
            valid_counts >= MIN_OBSERVATIONS
        ].index.tolist()
        if len(valid_stocks) < 2:
            logger.warning(
                "Fewer than 2 stocks with sufficient data at %s",
                rebal_date,
            )
            return None, None

        trailing_clean = trailing[valid_stocks].dropna()
        if len(trailing_clean) < MIN_OBSERVATIONS:
            return None, None

        try:
            weights = self.weight_fn(trailing_clean)
            return weights, valid_stocks
        except Exception as e:
            logger.warning(
                "Weight computation failed at %s: %s", rebal_date, e
            )
            return None, None
