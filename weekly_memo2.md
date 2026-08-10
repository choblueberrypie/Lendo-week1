# Week 2 Factor Memo — Value & Momentum on a 100-Stock S&P 500 Universe

**Author:** Intern — Quant Research
**Period analyzed:** 2016-08 → 2026-08 (121 month-end rebalance dates)
**Deliverables:** `src/factor_utils.py`, `notebooks/week2/factor_value_momentum.ipynb`, this memo

---

## 1. Objective

Construct two classic equity factors — **Value** (B/M and E/P) and **Momentum** (12-1) —
on a 100-stock S&P 500 universe and evaluate them with quintile sorts: at each month-end,
stocks are ranked by signal into Q1 (lowest) – Q5 (highest), held equal-weighted for one
month, and scored on the next-period return.

## 2. Data & Construction (point-in-time, no look-ahead)

| Layer | Source | Key detail |
|---|---|---|
| Prices | Yahoo Finance OHLCV | split-adjusted OHLC; `adj_close` for returns/momentum |
| Fundamentals | **SEC EDGAR companyfacts API** | equity, net income, shares — each with its true `filing_date` |
| Risk-free | `^IRX` 13-week T-bill | Sharpe denominators |

- **PIT rule:** a fundamental figure is visible at rebalance date *t* only if
  `filing_date ≤ t`; restatements resolve to the version known at *t*. No assumed lags.
- **TTM earnings:** last 4 consecutive quarters known at *t*; missing Q4 derived as
  `FY − (Q1+Q2+Q3)` from 10-K records.
- **Market cap:** Yahoo prices are split-adjusted, so SEC share counts are multiplied by
  the cumulative post-*t* split factor (28 events, 24 tickers) — verified against known
  market caps (e.g., AAPL Jan-2019 = $785B ✓).
- **Data hygiene:** share-count placeholders/unit errors filtered (valid range 1M–100B);
  staleness caps (shares 400d, equity 550d, TTM 200d); entity changes handled by merging
  predecessor CIKs (XOM, BLK, BG, APO, APA); multi-class share gaps filled with
  weighted-average share counts.
- **Coverage after cleaning:** B/M 94.0%, E/P 97.6%, momentum 90.0% of panel cells;
  ~94/100 tickers have ≥36 quarters of fundamentals. Negative-equity firms (AZO, BA in
  2020-23, MO, BKNG) are excluded from B/M ranking only — an economic reality of heavy
  buyback programs, not missing data.

## 3. Results (equal-weighted quintiles, 2016-08 → 2026-08)

| Factor | Q5−Q1 (ann.) | t-stat | Mean IC | ICIR | Q5 ann. | Q1 ann. |
|---|---|---|---|---|---|---|
| **Momentum 12-1** | **+8.8%** | 1.43 | +0.021 | 0.108 | 24.4% | 13.3% |
| B/M (value) | −6.0% | −1.24 | −0.018 | −0.092 | 17.4% | 24.4% |
| E/P (value) | −5.0% | −0.98 | +0.008 | 0.046 | 18.0% | 22.8% |

**Long–short by year (B/M):** 2017 −26.3% · 2019 −7.4% · 2020 −16.1% · **2021 +23.4%** ·
**2022 +19.2%** · 2023 −18.3% · 2024 −29.2%
**Momentum:** 2019 +14.7% · 2021 −5.1% · 2023 −11.3% · **2024 +53.9%**

## 4. Interpretation

1. **Momentum is the only positively priced signal in this universe.** Winners beat losers
   by 0.73%/month; the premium concentrates in trending markets and reverses in sharp
   rotations (2021, 2023) — the classic momentum-crash pattern.
2. **Value's premium was negative over 2016–2026**, the well-documented "lost decade."
   Both definitions agree (B/M and E/P move together, including the 2021–22 rate-hike
   comeback), so this is a regime result, not a construction artifact. In a mega-cap,
   tech-tilted large-cap universe the effect is amplified.
3. **Statistical caveat:** ~120 monthly observations give t-stats of ±1–1.4 — directionally
   meaningful, not decisive. The year-by-year pattern (value works in hiking regimes,
   momentum in trends) is the more actionable finding.

## 5. Limitations

Survivorship bias (current constituents only); 100-name universe ⇒ ~20 names/quintile;
weighted-average shares proxy for multi-class firms; equal weighting per assignment; no
transaction costs/turnover; 6 tickers (ABNB, APP, CARR, CVNA, AMCR, BKR) have genuinely
short histories (post-2017 listings/spin-offs) and enter the sample when data exists.

## 6. Next Steps

Scale to 300–500 names; sector-neutral sorts; composite value+momentum signal; conditional
analysis (value in rising-rate regimes); turnover and cost modeling.

---

### Appendix — Reproducibility

| Step | Script | Output |
|---|---|---|
| Price download | `week2_data_fetch.py` (Yahoo, 2s batch delays) | `week2_sp500_100_ohlcv_10y.csv` |
| Fundamentals | `sec_edgar_batch_fetch.py` (+ pilot v1/v2) | `sec_edgar_100_raw_facts.csv` (146k rows) |
| Patches | `sec_edgar_patch_aep_brkb.py` (AEP local JSON, BRK-B Yahoo shares) | — |
| Splits | `fetch_splits.py` | `sp500_100_splits.csv` (28 events) |
| Factor panel | `build_factor_panel.py` → `src/factor_utils.py` | `factor_panel_monthly.csv` (11,941 rows) |
| Backtest | `run_factor_backtest.py` / notebook | `factors/backtest/*` |
