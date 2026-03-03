# Alpaca_Trading_Algos

Trading algorithms that interface with the Alpaca trading brokerage API and can be deployed using AWS Lambda.

### gmv_algo.py
- Rebalance your portfolio to utilize the [Global Minimum Variance](https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix-BEAMER.pdf) portfolio strategy.
- Uses 5 years of historical data from the S&P 100 to compute the covariance matrix and optimize for minimum portfolio volatility.

### momentum_algo.py
- Implements a **cross-sectional momentum** strategy on the S&P 100.
- Uses the standard 12-1 momentum factor: ranks stocks by their 12-month return (skipping the most recent month to avoid short-term reversal effects).
- Equal-weights the top 20 momentum stocks and rebalances daily.
- Automatically closes positions in stocks that fall out of the top momentum ranking.

### mean_reversion_algo.py
- Implements a **mean reversion** strategy using z-scores on the S&P 100.
- Calculates how far each stock's price has deviated from its 60-day rolling mean.
- Buys stocks with a z-score below -1.0 (oversold) and closes positions once the z-score reverts above 0.
- Equal-weights selected stocks and rebalances daily.

### equal_weight_sp100_algo.py
- Implements a simple **equal-weight S&P 100** portfolio strategy.
- Allocates an equal dollar amount to each of the ~100 stocks in the S&P 100 index.
- Rebalances daily to maintain equal weighting as prices drift.
- A straightforward benchmark strategy with broad diversification.

### rsi_algo.py
- Implements a **Relative Strength Index (RSI)** trading strategy on the S&P 100.
- Calculates the 14-day RSI for each stock to identify overbought and oversold conditions.
- Buys stocks with RSI below 30 (oversold) and sells positions in stocks with RSI above 70 (overbought).
- Equal-weights up to 20 oversold positions and rebalances daily.
