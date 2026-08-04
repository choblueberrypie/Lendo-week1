"""Utilities for loading, cleaning, and analyzing OHLCV market data.

The functions in this module implement the Week 1 research pipeline:
load raw Yahoo Finance OHLCV data, document quality issues, clean adjusted
prices conservatively, compute explicit simple/log returns, and summarize
rolling portfolio risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
)

NUMERIC_OHLCV_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
)

PRICE_OHLCV_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adj_close",
)

TRADING_DAYS_PER_YEAR = 252


def load_ohlcv(
    path: str | Path,
    tickers: list[str] | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load a CSV or parquet OHLCV file with the Week 1 schema.

    Args:
        path: Path to a ``.csv`` or ``.parquet`` file.
        tickers: Optional list of ticker symbols to keep.
        start_date: Optional inclusive start date.
        end_date: Optional inclusive end date.

    Returns:
        DataFrame sorted by ticker and date with columns ``date``, ``ticker``,
        ``open``, ``high``, ``low``, ``close``, ``volume``, and ``adj_close``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension or required schema is unsupported.
    """

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"OHLCV file not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(data_path)
    elif suffix == ".parquet":
        raw = pd.read_parquet(data_path)
    else:
        raise ValueError("OHLCV files must use .csv or .parquet format")

    df = standardize_ohlcv_schema(raw)
    missing_columns = [
        column for column in REQUIRED_OHLCV_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

    df = df.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.tz_convert(
        None
    )
    df["ticker"] = df["ticker"].astype("string").str.upper().str.strip()

    for column in NUMERIC_OHLCV_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if tickers is not None:
        normalized_tickers = [ticker.upper().strip() for ticker in tickers]
        df = df.loc[df["ticker"].isin(normalized_tickers)]

    if start_date is not None:
        start = pd.Timestamp(start_date)
        df = df.loc[df["date"] >= start]

    if end_date is not None:
        end = pd.Timestamp(end_date)
        df = df.loc[df["date"] <= end]

    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def standardize_ohlcv_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with snake-case OHLCV column names."""

    result = df.copy()
    result.columns = [_standardize_column_name(column) for column in result.columns]
    aliases = {
        "adjclose": "adj_close",
        "adjusted_close": "adj_close",
        "symbol": "ticker",
    }
    return result.rename(columns=aliases)


def validate_ohlcv(df: pd.DataFrame, long_gap_days: int = 7) -> dict[str, Any]:
    """Build a schema, coverage, and trading-calendar quality report.

    The report flags issues instead of silently removing observations. Long gaps
    are calendar gaps longer than ``long_gap_days`` and should be reviewed rather
    than interpreted automatically as bad data.
    """

    missing_columns = [
        column for column in REQUIRED_OHLCV_COLUMNS if column not in df.columns
    ]
    report: dict[str, Any] = {
        "schema_valid": not missing_columns,
        "missing_columns": missing_columns,
        "row_count": int(len(df)),
    }
    if missing_columns:
        return report

    dates = df["date"].dropna().drop_duplicates().sort_values()
    report.update(
        {
            "ticker_count": int(df["ticker"].nunique()),
            "date_range": {
                "start": _format_date(dates.iloc[0]) if not dates.empty else None,
                "end": _format_date(dates.iloc[-1]) if not dates.empty else None,
            },
            "missing_by_column": {
                column: int(df[column].isna().sum())
                for column in REQUIRED_OHLCV_COLUMNS
            },
            "duplicate_date_ticker_rows": int(
                df.duplicated(subset=["date", "ticker"]).sum()
            ),
            "nonpositive_price_rows": {
                column: int((df[column] <= 0).sum()) for column in PRICE_OHLCV_COLUMNS
            },
            "negative_volume_rows": int((df["volume"] < 0).sum()),
        }
    )

    report["trading_calendar"] = _calendar_report(dates, long_gap_days)
    report["coverage"] = _coverage_report(df, dates)
    return report


def summarize_ohlcv(df: pd.DataFrame) -> dict[str, Any]:
    """Return Task 2 summary statistics and average-volume rankings."""

    summary_stats = (
        df.loc[:, list(NUMERIC_OHLCV_COLUMNS)]
        .describe()
        .loc[["count", "mean", "std", "min", "max"]]
        .round(6)
        .to_dict()
    )
    average_volume = df.groupby("ticker")["volume"].mean().sort_values(ascending=False)

    return {
        "summary_statistics": summary_stats,
        "top_10_by_average_daily_volume": average_volume.head(10).round(2).to_dict(),
        "bottom_10_by_average_daily_volume": average_volume.tail(10)
        .sort_values()
        .round(2)
        .to_dict(),
    }


def average_daily_volume(df: pd.DataFrame) -> pd.Series:
    """Return average daily volume by ticker, sorted descending."""

    return df.groupby("ticker")["volume"].mean().sort_values(ascending=False)


def pivot_ohlcv(df: pd.DataFrame, value: str = "adj_close") -> pd.DataFrame:
    """Pivot long OHLCV data to a date-indexed ticker-column matrix.

    Duplicate ``date``/``ticker`` pairs are not allowed here; call
    :func:`clean_ohlcv` first when the source may contain duplicates.
    """

    if value not in df.columns:
        raise ValueError(f"Unknown OHLCV value column: {value}")
    if df.duplicated(subset=["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker rows must be cleaned before pivoting")

    return (
        df.pivot(index="date", columns="ticker", values=value)
        .sort_index()
        .sort_index(axis=1)
    )


def clean_ohlcv(
    df: pd.DataFrame,
    max_ffill_periods: int = 3,
    return_outlier_threshold: float = 0.5,
    volume_spike_multiplier: float = 10.0,
    volume_window: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean OHLCV data conservatively and return a documented quality report.

    Cleaning rules:
    1. Standardize schema and drop rows without a usable date or ticker.
    2. Sort by ticker/date and keep the last duplicate observation for a
       duplicate ``date``/``ticker`` pair.
    3. Remove rows with nonpositive ``adj_close`` because they cannot produce
       valid returns.
    4. Convert negative volumes to missing values; do not invent volume.
    5. Forward-fill isolated missing ``adj_close`` values within each ticker for
       at most ``max_ffill_periods`` observations, but never before a ticker's
       first observed price.
    6. Drop rows whose ``adj_close`` remains missing after limited forward fill.
    7. Do not interpolate OHLC values and do not remove return outliers or
       volume spikes; flag them for review instead.

    Yahoo Finance ``adj_close`` is already split/dividend adjusted, so no extra
    split adjustment is applied here.
    """

    standardized = standardize_ohlcv_schema(df)
    missing_columns = [
        column
        for column in REQUIRED_OHLCV_COLUMNS
        if column not in standardized.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required OHLCV columns: {missing_columns}")

    working = standardized.loc[:, list(REQUIRED_OHLCV_COLUMNS)].copy()
    report: dict[str, Any] = {
        "input_rows": int(len(working)),
        "cleaning_rules": [
            "standardize schema and require date/ticker/open/high/low/close/volume/adj_close",
            "drop rows without a valid date or ticker",
            "keep the last duplicate date/ticker observation",
            "drop rows with nonpositive adjusted close",
            "convert negative volume to missing rather than inventing volume",
            f"forward-fill adjusted close within ticker for at most {max_ffill_periods} rows",
            "do not forward-fill before a ticker's first observed price",
            "drop adjusted-close rows still missing after limited forward fill",
            "do not interpolate OHLC fields",
            "flag return outliers and volume spikes instead of silently deleting them",
            "use Yahoo adjusted close for split/dividend adjustment",
        ],
    }

    working["date"] = pd.to_datetime(working["date"], errors="coerce", utc=True)
    working["date"] = working["date"].dt.tz_convert(None)
    working["ticker"] = working["ticker"].astype("string").str.upper().str.strip()
    for column in NUMERIC_OHLCV_COLUMNS:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    invalid_date_rows = int(working["date"].isna().sum())
    blank_ticker_rows = int(
        working["ticker"].isna().sum() + (working["ticker"] == "").sum()
    )
    working = working.loc[
        working["date"].notna() & working["ticker"].notna() & (working["ticker"] != "")
    ].copy()
    report["invalid_date_rows_removed"] = invalid_date_rows
    report["blank_ticker_rows_removed"] = blank_ticker_rows

    working = working.sort_values(["ticker", "date"], kind="mergesort")
    duplicate_rows = int(working.duplicated(subset=["ticker", "date"]).sum())
    working = working.drop_duplicates(subset=["ticker", "date"], keep="last")
    report["duplicate_rows_removed"] = duplicate_rows

    nonpositive_price_rows = {
        column: int((working[column] <= 0).sum()) for column in PRICE_OHLCV_COLUMNS
    }
    nonpositive_adj_close_rows = int((working["adj_close"] <= 0).sum())
    working = working.loc[
        (working["adj_close"] > 0) | working["adj_close"].isna()
    ].copy()
    report["nonpositive_price_rows"] = nonpositive_price_rows
    report["nonpositive_adj_close_rows_removed"] = nonpositive_adj_close_rows

    negative_volume_rows = int((working["volume"] < 0).sum())
    working.loc[working["volume"] < 0, "volume"] = np.nan
    report["negative_volume_rows_set_to_missing"] = negative_volume_rows

    report["adj_close_missing_before_fill"] = int(working["adj_close"].isna().sum())
    filled_adj_close = working.groupby("ticker", sort=False)["adj_close"].ffill(
        limit=max_ffill_periods
    )
    adj_close_filled = int(
        working["adj_close"].isna().sum() - filled_adj_close.isna().sum()
    )
    working["adj_close"] = filled_adj_close
    report["adj_close_values_forward_filled"] = adj_close_filled
    report["adj_close_missing_after_fill"] = int(working["adj_close"].isna().sum())

    remaining_missing_adj_close = int(working["adj_close"].isna().sum())
    working = working.loc[working["adj_close"].notna()].copy()
    report["rows_removed_with_unfilled_adj_close"] = remaining_missing_adj_close

    working = working.sort_values(["ticker", "date"]).reset_index(drop=True)
    report["output_rows"] = int(len(working))
    report["ticker_count"] = int(working["ticker"].nunique())

    report["missing_by_year"] = missing_by_year(working).to_dict(orient="records")

    prices = pivot_ohlcv(working, "adj_close")
    return_outliers = detect_return_outliers(prices, return_outlier_threshold)
    volume_spikes = detect_volume_spikes(
        working, window=volume_window, multiplier=volume_spike_multiplier
    )
    report["suspicious_return_outliers"] = _events_report(return_outliers)
    report["suspicious_volume_spikes"] = _events_report(volume_spikes)
    report["survivorship_bias_note"] = (
        "The universe is a current S&P 500 constituent snapshot; delisted stocks "
        "and historical index changes are not represented."
    )

    return working, report


def missing_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-cell counts and percentages by calendar year."""

    if df.empty:
        return pd.DataFrame(
            columns=[
                "year",
                "rows",
                "adj_close_missing",
                "adj_close_missing_pct",
                "volume_missing",
                "volume_missing_pct",
                "all_ohlcv_missing_pct",
            ]
        )

    working = df.copy()
    working["year"] = working["date"].dt.year
    grouped = working.groupby("year", sort=True)
    result = pd.DataFrame(
        {
            "rows": grouped.size(),
            "adj_close_missing": grouped["adj_close"].apply(lambda s: s.isna().sum()),
            "volume_missing": grouped["volume"].apply(lambda s: s.isna().sum()),
            "missing_cells": grouped.apply(
                lambda frame: frame.loc[:, list(NUMERIC_OHLCV_COLUMNS)]
                .isna()
                .sum()
                .sum(),
                include_groups=False,
            ),
        }
    ).reset_index()
    result["adj_close_missing_pct"] = (
        result["adj_close_missing"] / result["rows"] * 100
    ).round(4)
    result["volume_missing_pct"] = (
        result["volume_missing"] / result["rows"] * 100
    ).round(4)
    result["all_ohlcv_missing_pct"] = (
        result["missing_cells"] / (result["rows"] * len(NUMERIC_OHLCV_COLUMNS)) * 100
    ).round(4)
    return result.loc[
        :,
        [
            "year",
            "rows",
            "adj_close_missing",
            "adj_close_missing_pct",
            "volume_missing",
            "volume_missing_pct",
            "all_ohlcv_missing_pct",
        ],
    ]


def detect_return_outliers(
    prices: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Flag daily simple returns whose absolute value exceeds ``threshold``."""

    if threshold <= 0:
        raise ValueError("threshold must be positive")
    returns = compute_returns(prices, method="simple", frequency="D")
    events = returns.stack(future_stack=True).rename("simple_return").reset_index()
    events.columns = ["date", "ticker", "simple_return"]
    events = events.loc[
        events["simple_return"].abs() > threshold,
        ["date", "ticker", "simple_return"],
    ]
    return events.sort_values("simple_return", key=lambda s: s.abs(), ascending=False)


def detect_volume_spikes(
    df: pd.DataFrame,
    window: int = 60,
    multiplier: float = 10.0,
) -> pd.DataFrame:
    """Flag volume above ``multiplier`` times the prior rolling median volume."""

    if window < 2:
        raise ValueError("window must be at least 2")
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")

    volume = pivot_ohlcv(df, "volume")
    min_periods = max(5, window // 3)
    baseline = volume.rolling(window=window, min_periods=min_periods).median().shift(1)
    ratio = (volume / baseline).replace([np.inf, -np.inf], np.nan)
    events = (
        ratio.stack(future_stack=True)
        .rename("volume_ratio")
        .reset_index()
        .rename(columns={"level_1": "ticker"})
    )
    events = events.loc[events["volume_ratio"] >= multiplier].copy()
    if events.empty:
        return pd.DataFrame(
            columns=["date", "ticker", "volume", "median_volume", "volume_ratio"]
        )

    long_lookup = df.loc[:, ["date", "ticker", "volume"]]
    baseline_lookup = (
        baseline.stack(future_stack=True)
        .rename("median_volume")
        .reset_index()
        .rename(columns={"level_1": "ticker"})
    )
    events = events.merge(long_lookup, on=["date", "ticker"], how="left").merge(
        baseline_lookup, on=["date", "ticker"], how="left"
    )
    return events.sort_values("volume_ratio", ascending=False).loc[
        :, ["date", "ticker", "volume", "median_volume", "volume_ratio"]
    ]


def compute_returns(
    prices: pd.DataFrame,
    method: str = "simple",
    frequency: str = "D",
) -> pd.DataFrame:
    """Compute simple or log returns from adjusted-price matrix.

    Args:
        prices: Date-indexed DataFrame with one adjusted-price column per ticker.
        method: ``"simple"`` for arithmetic returns or ``"log"`` for log returns.
        frequency: ``"D"`` for daily, ``"W"`` for Friday-ending weekly, or ``"M"``
            for month-end returns. Resampled returns use the last available price
            in each interval, so simple returns compound correctly.

    Returns:
        Return matrix with the same ticker columns. Return type and frequency are
        explicit because log returns are appropriate for time aggregation, while
        simple returns are appropriate for cross-sectional portfolio aggregation.
    """

    if prices.empty:
        raise ValueError("prices must not be empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex")

    method_key = method.lower()
    if method_key not in {"simple", "log"}:
        raise ValueError("method must be 'simple' or 'log'")

    frequency_key = frequency.upper()
    if frequency_key == "D":
        sampled = prices.sort_index()
    elif frequency_key == "W":
        sampled = prices.sort_index().resample("W-FRI").last()
    elif frequency_key == "M":
        sampled = prices.sort_index().resample("ME").last()
    else:
        raise ValueError("frequency must be 'D', 'W', or 'M'")

    sampled = sampled.dropna(axis=0, how="all")
    if (sampled <= 0).any().any():
        raise ValueError("prices must be positive to compute returns")

    if method_key == "simple":
        returns = sampled.pct_change(fill_method=None)
    else:
        returns = np.log(sampled / sampled.shift(1))

    return returns.replace([np.inf, -np.inf], np.nan)


def compute_rolling_stats(
    returns: pd.DataFrame,
    windows: tuple[int, ...] = (20, 60, 252),
) -> dict[int, pd.DataFrame]:
    """Return rolling mean and rolling standard deviation for each window.

    Each result has MultiIndex columns ``(stat, ticker)`` where ``stat`` is
    ``"mean"`` or ``"std"``.
    """

    result: dict[int, pd.DataFrame] = {}
    for window in windows:
        if window < 2:
            raise ValueError("rolling windows must be at least 2")
        stats = pd.concat(
            {
                "mean": returns.rolling(window=window, min_periods=window).mean(),
                "std": returns.rolling(window=window, min_periods=window).std(),
            },
            axis=1,
        )
        result[window] = stats
    return result


def annualized_volatility(
    returns: pd.DataFrame | pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series | float:
    """Annualize sample volatility by the square-root-of-time rule."""

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    volatility = returns.std(ddof=1) * np.sqrt(periods_per_year)
    if isinstance(volatility, pd.Series):
        return volatility
    return float(volatility)


def portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Compute daily portfolio returns from simple security returns.

    With no weights, the portfolio is equal-weighted across all securities with
    available returns each day. With fixed weights, weights are normalized once,
    then renormalized daily across securities with observed returns. Use simple
    returns for this cross-sectional aggregation.
    """

    if returns.empty:
        raise ValueError("returns must not be empty")

    if weights is None:
        result = returns.mean(axis=1, skipna=True)
        result.name = "portfolio_return"
        return result

    aligned_weights = weights.reindex(returns.columns).fillna(0.0).astype(float)
    if (aligned_weights < 0).any():
        raise ValueError("portfolio weights must be nonnegative")
    weight_total = aligned_weights.sum()
    if weight_total <= 0:
        raise ValueError("portfolio weights must have a positive sum")
    aligned_weights = aligned_weights / weight_total

    observed = returns.notna()
    weighted_return_sum = returns.mul(aligned_weights, axis=1).sum(axis=1)
    observed_weight_sum = observed.mul(aligned_weights, axis=1).sum(axis=1)
    result = weighted_return_sum.div(observed_weight_sum.where(observed_weight_sum > 0))
    result.name = "portfolio_return"
    return result


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Return the growth of one dollar from a simple-return series."""

    if (returns <= -1).any():
        raise ValueError("simple returns cannot be less than or equal to -100%")
    result = (1 + returns.fillna(0.0)).cumprod()
    result.name = "cumulative_return"
    return result


def portfolio_performance(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Summarize annualized return, volatility, and Sharpe ratio.

    Annualized return compounds the observed simple-return series. Annualized
    volatility uses ``sqrt(periods_per_year)``. The risk-free rate is annual and
    defaults to zero.
    """

    clean_returns = returns.dropna()
    if clean_returns.empty:
        raise ValueError("returns must contain at least one non-missing value")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    total_growth = float((1 + clean_returns).prod())
    years = len(clean_returns) / periods_per_year
    annualized_return = total_growth ** (1 / years) - 1
    annualized_vol = float(annualized_volatility(clean_returns, periods_per_year))
    sharpe = (
        (annualized_return - risk_free_rate) / annualized_vol
        if annualized_vol > 0
        else np.nan
    )
    return pd.Series(
        {
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe_ratio": sharpe,
            "periods": float(len(clean_returns)),
        }
    )


def _standardize_column_name(column: object) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _calendar_report(dates: pd.Series, long_gap_days: int) -> dict[str, Any]:
    if dates.empty:
        return {
            "inferred_frequency": None,
            "median_spacing_days": None,
            "long_gap_count": 0,
            "long_gaps": [],
        }

    inferred_frequency = None
    if len(dates) >= 3:
        try:
            inferred_frequency = pd.infer_freq(dates)
        except ValueError:
            inferred_frequency = None

    spacing = dates.reset_index(drop=True).diff()
    median_spacing = spacing.median()
    gap_days = spacing.dt.days
    long_gaps = gap_days.loc[gap_days > long_gap_days]

    return {
        "inferred_frequency": inferred_frequency,
        "median_spacing_days": (
            int(median_spacing.days) if pd.notna(median_spacing) else None
        ),
        "long_gap_count": int(len(long_gaps)),
        "long_gaps": [
            {"after": _format_date(date), "gap_days": int(days)}
            for date, days in long_gaps.head(10).items()
        ],
    }


def _coverage_report(df: pd.DataFrame, all_dates: pd.Series) -> dict[str, Any]:
    if all_dates.empty:
        return {
            "global_trading_days": 0,
            "tickers_with_missing_global_dates": 0,
            "largest_missing_date_counts": {},
        }

    global_dates = pd.DatetimeIndex(all_dates)
    missing_counts: dict[str, int] = {}
    for ticker, group in df.groupby("ticker"):
        ticker_dates = pd.DatetimeIndex(group["date"].dropna().unique())
        missing_count = len(global_dates.difference(ticker_dates))
        if missing_count > 0:
            missing_counts[str(ticker)] = int(missing_count)

    largest_missing = dict(
        sorted(missing_counts.items(), key=lambda item: item[1], reverse=True)[:10]
    )
    return {
        "global_trading_days": int(len(global_dates)),
        "tickers_with_missing_global_dates": int(len(missing_counts)),
        "largest_missing_date_counts": largest_missing,
    }


def _events_report(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {"count": 0, "tickers": [], "top_events": []}

    ticker_column = "ticker"
    return {
        "count": int(len(events)),
        "tickers": sorted(events[ticker_column].astype(str).unique().tolist()),
        "top_events": events.head(20)
        .assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records"),
    }


def _format_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")
