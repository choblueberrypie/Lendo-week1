# Week 3 Memo — Risk Metrics & Portfolio Construction

**Author:** Quant Research Intern · **Date:** August 2026
**Deliverables:** `src/risk.py`, `notebooks/week3/risk_portfolio_construction.ipynb`, chart pack (6 charts)

---

## What was built

A reusable risk module (`src/risk.py`) implementing `compute_risk_metrics()` — annualized
return (geometric), annualized vol, Sharpe (monthly excess over `^IRX`), max drawdown, and
drawdown duration — plus correlation/diversification tools, three allocation schemes
(EW, inverse-vol risk parity, constrained MVO via `scipy.optimize.minimize`), and a rolling
backtest engine. **One frequency (monthly), one rf treatment, one annualization rule across
every comparison.** All weights sum to 1 at each rebalance and use only prior information.

## Key findings

**1. Quintile risk profiles (Week-2 portfolios).** Momentum Q5 earns 24.4%/yr with Sharpe
1.15 and the *shallowest* max drawdown (−24.2%); momentum Q1 (losers) is the worst
risk-adjusted bucket (Sharpe 0.64). Value Q1 (growth) matches momentum Q5 on return (24.4%)
with the best Sharpe in the study (1.22) but the longest underwater stretches (27 months).

**2. Diversification is real but bounded.** Mean pairwise correlation of the 20 largest
stocks is ≈ 0.30; going from 1 to 50 random equal-weighted names cuts portfolio vol from
33.2% to 19.2% — a 42% reduction, mostly achieved by 20–30 names. Quintile betas vs SPY
(0.86–1.37) show factor sorts also sort *market exposure*: E/P Q1 (negative earners) is the
highest-beta bucket (1.37).

**3. EW vs risk parity vs MVO (full-sample, monthly rebalanced):**

| Method | Ann. return | Vol | Sharpe | Max DD | Turnover (ann.) |
|---|---|---|---|---|---|
| Equal weight (100) | 16.9% | 17.5% | 0.98 | −22.6% | 69% |
| Risk parity, inv-vol (100) | 14.7% | 16.0% | 0.94 | −21.9% | 106% |
| MVO max-Sharpe, rolling 24m (50) | 20.8% | 17.7% | 1.17 | −20.3% | **409%** |

Risk parity buys a modest vol/drawdown reduction for ~1.5× turnover — roughly a wash on
Sharpe. The constrained rolling MVO posts the best raw numbers here, **but**: (a) it trades
~4–6× more than EW, so any realistic cost assumption erodes the edge; (b) its weights are
driven by trailing-24-month μ̂ — the noisiest object in the pipeline; (c) the result is
carried by constraints (long-only, 10% position cap, 25% sector cap), which shrink the
solution toward diversification. This is exactly the DeMiguel–Garlappi–Uppal point:
**in-sample μ̂ describes the past; it is not a forecast.** The in-sample efficient frontier
is presented as a description of the sample, never as an expectation.

## Why simple methods are hard to beat

EW has zero estimation error in expected returns; inverse-vol RP estimates only
volatilities, which are far more stable than means. MVO amplifies μ̂ noise into large,
unstable, turnover-heavy weights — every trade pays costs without reliable expected gain.
Optimization still earns its keep for *enforcing constraints* and *targeting risk*, not in
naive plug-in form.

## Limitations

Survivorship bias (current constituents only); 100-name universe; gross of transaction
costs (decisive given MVO turnover); RP is the inverse-vol approximation, not full ERC;
monthly frequency throughout.

## Next steps

Add transaction-cost modeling (would reorder the three methods); shrinkage estimators for
μ̂/Σ̂ (Ledoit–Wolf) before trusting MVO further; full ERC risk parity; sector-neutral
variants.
