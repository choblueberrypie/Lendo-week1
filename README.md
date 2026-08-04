# Lendo Week 1 — Market Data Foundations

Quant research internship Week 1 project: reproducible environment, OHLCV data validation, cleaning, return computation, rolling diagnostics, and 50-stock portfolio analysis.

## Final Project Structure

```text
Lendo-week1/
├── .gitignore
├── README.md
├── requirements.txt
├── weekly_memo.md
├── src/
│   └── data_utils.py
└── notebooks/
    └── week1/
        ├── hello_world.ipynb
        └── data_quality_checks.ipynb
```

## Environment Setup

Use Python 3.11 with Conda:

```bash
conda create -n lendo-week1 python=3.11 -y
conda activate lendo-week1
pip install -r requirements.txt
```

## Data Source

The analysis uses a Yahoo Finance daily OHLCV CSV file for 100 current S&P 500 constituents:

- **100** tickers
- **247,816** daily OHLCV rows
- **2016-08-01 to 2026-07-29** date coverage
- Required schema: `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`, `adj_close`
- Source: Yahoo Finance via `yfinance`

The CSV file is stored locally outside the repository and is loaded through `DATA_PATH` in `notebooks/week1/data_quality_checks.ipynb`:

```python
DATA_PATH = Path("C:/Users/dwt04/Desktop/北美求职/LENDO/dataset/sp500_100_10y_ohlcv.csv")
raw = load_ohlcv(DATA_PATH)
```

Update this path if the CSV is moved or the project is run on another machine.

## Verification

Run all checks from the repository root:

```bash
ruff format --check src
ruff check src
jupyter nbconvert --to notebook --execute --inplace notebooks/week1/hello_world.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/week1/data_quality_checks.ipynb
```

Expected result: Ruff reports no issues, and both notebooks execute from top to bottom without errors.

## Week 1 Deliverables

- `notebooks/week1/hello_world.ipynb` — Task 1 environment verification notebook.
- `notebooks/week1/data_quality_checks.ipynb` — fully executed Week 1 notebook covering Tasks 2–5.
- `src/data_utils.py` — reusable functions for loading, validating, cleaning, returns, rolling statistics, and portfolio calculations.
- `weekly_memo.md` — one-page summary of data issues, cleaning decisions, and key observations.

## Methodology Notes

- Returns use Yahoo Finance `adj_close`, which is split/dividend adjusted.
- Weekly returns use Friday-ending intervals; monthly returns use month-end intervals.
- Simple returns are used for cross-sectional portfolio aggregation; log returns are included for time-aggregation intuition.
- Annualization uses 252 trading days.
- Sharpe ratios use a 0% annual risk-free rate.
- The 50-stock value-weight portfolio uses the first 252 trading days of average dollar volume (`adj_close × volume`) as a liquidity-value proxy, because true historical market-cap weights are not available.
- Portfolio evaluation starts after the formation window to avoid look-ahead bias in stock selection and weights.

## Limitations

The S&P 500 list is a **current constituent snapshot**, not point-in-time index membership. Delisted stocks and historical index changes are absent, so the analysis has survivorship-bias limitations. Results validate the research pipeline but are not a point-in-time investable S&P 500 backtest.

## Git Workflow

Work was organized on two feature branches:

- `feature/data-utils` — implements `src/data_utils.py`, including OHLCV loading, validation, cleaning, return computation, rolling statistics, and portfolio helpers.
- `feature/week1-quality-check` — contains the final Week 1 quality-check notebook and supporting deliverable updates.

Recommended workflow:

1. Keep `main` stable.
2. Develop utility-function changes on `feature/data-utils`.
3. Develop notebook and final Week 1 validation changes on `feature/week1-quality-check`.
4. Run Ruff and execute both notebooks before opening a pull request.
5. Merge each feature branch into `main` only after the relevant checks pass.
