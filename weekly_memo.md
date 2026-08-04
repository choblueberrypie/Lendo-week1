# Weekly Memo — Week 1 Market Data Foundations

**Prepared for:** Lendo Quant Research Internship  
**Dataset:** Yahoo Finance daily OHLCV, 100 current S&P 500 constituents  
**Period:** 2016-08-01 to 2026-07-29  
**Rows:** 247,816

## Data issues found

The raw schema is complete for `date`, `ticker`, OHLC, `volume`, and `adj_close`, with no duplicate date/ticker rows, no missing adjusted closes, and no nonpositive adjusted prices. Four tickers do not cover the full global trading calendar: APP (1184 days), ABNB (1099 days), CARR (914 days), CVNA (187 days). These gaps are consistent with later listing dates rather than bad rows. I flagged 2 daily return observations above ±50% (APA, CVNA); the largest was CVNA on 2023-06-08 (56.02%). I also flagged 198 volume spikes above 10× the prior 60-day median across 54 tickers (ABBV, ABNB, ACGL, ADBE, ADM, ADP, ADSK, AIZ, AJG, AKAM, ALGN, AMCR, ...); the largest was AMCR on 2019-06-26 (187,122.0× prior median). These events are retained for review, not silently deleted.

## Cleaning decisions made

I used Yahoo Finance `adj_close` for split/dividend-adjusted returns. Cleaning standardizes schema, removes unusable date/ticker rows, keeps the last duplicate date/ticker record, removes nonpositive adjusted prices, converts negative volume to missing, and forward-fills only isolated missing adjusted closes for at most three observations within a ticker. It never fabricates pre-listing history and does not interpolate OHLC values. Return outliers and volume spikes remain visible in the quality report.

## Key observations

Daily return distributions for AAPL, AMZN, AMD, BAC, T are not normal: skewness ranges from -0.25 to 0.35, and excess kurtosis ranges from 4.57 to 11.16. The median monthly/daily volatility ratio is 4.20, close to the square-root-of-time approximation sqrt(21) = 4.58, but not exact because real returns show volatility clustering and fat tails.

For the 50-stock portfolios, selection and value weights use only the first 252 trading days of average dollar volume; evaluation starts after that formation window. From 2017-08-01 to 2026-07-29, the equal-weight portfolio produced 16.11% annualized return, 19.46% annualized volatility, and a 0.828 Sharpe ratio; $1 grew to $3.82. The dollar-volume value-weight proxy produced 20.49% annualized return, 21.20% annualized volatility, and a 0.967 Sharpe ratio; $1 grew to $5.32. Annualization uses 252 trading days and a 0% risk-free rate.

## Limitation

The universe is a current S&P 500 snapshot, so delisted firms and historical index changes are missing. Results are useful for validating the research pipeline, but they are not a point-in-time investable S&P 500 backtest. The value-weight portfolio is a liquidity-value proxy, not true market-cap weighting.
