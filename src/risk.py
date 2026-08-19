# -*- coding: utf-8 -*-
"""
risk.py — Week 3: risk metrics, diversification, allocation schemes, mean-variance optimization

设计约定 (全周一致性, 任务书 Task 1 quality check 要求):
  - 组合层面指标默认用【月度收益】, periods_per_year=12
  - 年化收益 = 几何年化 ((1+r).prod()**(ppy/n) - 1)
  - 年化波动 = std * sqrt(ppy)
  - Sharpe = 月均超额收益 / 月波动 * sqrt(12), 超额收益按月减去 ^IRX 月度无风险利率
  - 回撤基于累计净值; 回撤时长以"期"(月)计
  - 所有配置权重在每个调仓日 sum(w) == 1, 且只用调仓日之前的信息 (无 look-ahead)
"""

import numpy as np
import pandas as pd


# ================================================================ 数据准备辅助

def month_end_dates(ohlcv: pd.DataFrame) -> pd.DatetimeIndex:
    """每月最后一个交易日 (与 Week 2 调仓日历一致)"""
    d = ohlcv[["date"]].drop_duplicates().sort_values("date")
    return pd.DatetimeIndex(d.groupby(d["date"].dt.to_period("M"))["date"].max().values)


def monthly_returns_from_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """从日线 OHLCV 生成月末 × 股票的月度复权收益矩阵 (首月为 NaN)"""
    o = ohlcv.copy()
    o["date"] = pd.to_datetime(o["date"])
    me = month_end_dates(o)
    adj = o.pivot(index="date", columns="ticker", values="adj_close").reindex(me)
    return adj.pct_change()


def daily_returns_from_ohlcv(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """日收益矩阵 (date × ticker, 复权)"""
    o = ohlcv.copy()
    o["date"] = pd.to_datetime(o["date"])
    return o.pivot(index="date", columns="ticker", values="adj_close").pct_change()


def load_rf_monthly(rf_path, dates: pd.DatetimeIndex) -> pd.Series:
    """^IRX 年化% → 调仓日月度小数 (与 Week 2 factor_utils 口径一致)"""
    rf = pd.read_csv(rf_path)
    rf.columns = [c.strip().lower() for c in rf.columns]
    date_col = "date" if "date" in rf.columns else rf.columns[0]
    val_col = [c for c in rf.columns if c != date_col][-1]
    rf[date_col] = pd.to_datetime(rf[date_col])
    rf = rf.set_index(date_col)[val_col].sort_index()
    return rf.reindex(rf.index.union(dates)).ffill().reindex(dates) / 100.0 / 12.0


# ================================================================ Task 1: 风险指标

def drawdown_series(returns: pd.Series) -> pd.Series:
    """累计净值的回撤序列 (≤ 0)"""
    cum = (1 + returns).cumprod()
    return cum / cum.cummax() - 1.0


def max_drawdown_duration(returns: pd.Series) -> int:
    """最长水下时间 (期数): 净值处于历史高点之下的最长连续段"""
    dd = drawdown_series(returns)
    underwater = (dd < -1e-12).astype(int)
    # 连续 1 的最大长度
    grp = (underwater != underwater.shift()).cumsum()
    lengths = underwater.groupby(grp).sum()
    return int(lengths.max()) if len(lengths) else 0


def compute_risk_metrics(returns, rf=None, periods_per_year: int = 12) -> dict:
    """
    核心风险指标 (任务书 Task 1)。
    returns: 单期收益序列 (默认月度)
    rf:      同频率无风险利率序列 (月度小数) 或标量; None 视为 0
    """
    r = pd.Series(returns).dropna().astype(float)
    n = len(r)
    ppy = periods_per_year

    ann_ret = (1 + r).prod() ** (ppy / n) - 1.0
    ann_vol = r.std(ddof=1) * np.sqrt(ppy)

    if rf is None:
        excess = r
    elif np.isscalar(rf):
        excess = r - rf
    else:
        excess = r - pd.Series(rf).reindex(r.index).ffill().fillna(0.0)
    sharpe = excess.mean() / r.std(ddof=1) * np.sqrt(ppy) if r.std(ddof=1) > 0 else np.nan

    dd = drawdown_series(r)

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": dd.min(),
        "max_dd_duration_periods": max_drawdown_duration(r),
        "n_periods": n,
    }


def portfolio_beta(port_ret: pd.Series, mkt_ret: pd.Series) -> float:
    """beta = cov(R_p, R_m) / var(R_m), 只用两者都有值的日期 (aligned dates)"""
    df = pd.concat([port_ret, mkt_ret], axis=1, keys=["p", "m"]).dropna()
    return float(df["p"].cov(df["m"]) / df["m"].var(ddof=1))


def realized_quintile_returns(q_ret: pd.DataFrame, factor: str) -> pd.DataFrame:
    """
    Week2 五分位收益的口径修正:
    quintile_monthly_returns.csv 中 date=t 的 ret 是【t → 下一调仓日】实现的收益,
    与市场收益对齐做 beta/相关分析前, 必须把标签平移到【实现月】。
    返回 index=实现月月末, columns=Q1..Q5 的收益矩阵 (最后一期无实现月标签, 丢弃)。
    """
    p = q_ret[q_ret["factor"] == factor].pivot_table(
        index="date", columns="quintile", values="ret").sort_index()
    idx = p.index
    mapped = pd.Series(idx[1:].append(pd.DatetimeIndex([pd.NaT])), index=idx)
    p.index = pd.DatetimeIndex(mapped.values)
    return p[p.index.notna()]


# ================================================================ Task 2: 相关性与分散化

def correlation_matrix(daily_returns: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """个股日收益相关矩阵; 输入需已按日期对齐 (相同 index)"""
    return daily_returns[tickers].dropna().corr()


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """DR = 加权平均波动 / 组合波动。DR=1 表示完全集中, 越大越分散"""
    vols = np.sqrt(np.diag(cov))
    port_vol = float(np.sqrt(weights @ cov @ weights))
    return float((weights * vols).sum() / port_vol)


def diversification_curve(daily_returns: pd.DataFrame, n_grid: list,
                          n_samples: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    分散化曲线: 随机抽 N 只股票等权持有, 看组合年化波动随 N 的变化。
    采样规则 (quality check 要求明确说明):
      每次从全池【不放回均匀随机】抽 N 只, 等权, 用全样本日收益,
      每个 N 重复 n_samples 次, 记录均值与 10-90% 分位。
    """
    rng = np.random.default_rng(seed)
    tickers = np.array(daily_returns.columns)
    rows = []
    for n in n_grid:
        vols = []
        for _ in range(n_samples):
            pick = rng.choice(tickers, size=min(n, len(tickers)), replace=False)
            port = daily_returns[pick].mean(axis=1).dropna()
            vols.append(port.std(ddof=1) * np.sqrt(252))
        rows.append({"n_stocks": n,
                     "vol_mean": np.mean(vols),
                     "vol_p10": np.percentile(vols, 10),
                     "vol_p90": np.percentile(vols, 90)})
    return pd.DataFrame(rows)


# ================================================================ Task 3: 配置方案

def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def inverse_vol_weights(hist_returns: pd.DataFrame) -> np.ndarray:
    """
    简单风险平价近似 (任务书允许): w_i ∝ 1/σ_i,
    σ_i 用 hist_returns (调仓日之前的历史) 估计 —— 无 look-ahead。
    """
    vols = hist_returns.std(ddof=1).to_numpy()
    inv = np.where(vols > 0, 1.0 / vols, 0.0)
    return inv / inv.sum()


# ================================================================ Task 4: 均值方差优化

def _sector_constraints(sectors: pd.Series, sector_cap: float):
    """每个板块权重 ≤ sector_cap 的 scipy 约束列表"""
    cons = []
    for sec in sectors.unique():
        idx = np.where(sectors.to_numpy() == sec)[0]
        cons.append({"type": "ineq", "fun": lambda w, idx=idx: sector_cap - w[idx].sum()})
    return cons


def min_variance_weights(hist_returns: pd.DataFrame, sectors: pd.Series = None,
                         max_weight: float = 0.10, sector_cap: float = 0.25,
                         target_return: float = None) -> np.ndarray:
    """
    约束均值方差优化 (scipy.optimize.minimize, SLSQP):
      min  w'Σw
      s.t. sum(w)=1, 0 ≤ w_i ≤ max_weight (long-only + 单票上限),
           板块权重 ≤ sector_cap,
           μ'w = target_return (给定时; 有效前沿用)
    """
    from scipy.optimize import minimize

    mu = hist_returns.mean().to_numpy()
    cov = hist_returns.cov().to_numpy()
    n = len(mu)

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if sectors is not None:
        cons += _sector_constraints(sectors, sector_cap)
    if target_return is not None:
        cons.append({"type": "eq", "fun": lambda w: w @ mu - target_return})

    res = minimize(lambda w: w @ cov @ w, equal_weight(n),
                   method="SLSQP", bounds=[(0.0, max_weight)] * n,
                   constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"MVO 优化失败: {res.message}")
    return res.x


def max_sharpe_weights(hist_returns: pd.DataFrame, rf_period: float = 0.0,
                       sectors: pd.Series = None, max_weight: float = 0.10,
                       sector_cap: float = 0.25) -> np.ndarray:
    """同上约束下的最大 Sharpe 组合"""
    from scipy.optimize import minimize

    mu = hist_returns.mean().to_numpy()
    cov = hist_returns.cov().to_numpy()
    n = len(mu)

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu - rf_period) / vol if vol > 0 else 1e6

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if sectors is not None:
        cons += _sector_constraints(sectors, sector_cap)

    res = minimize(neg_sharpe, equal_weight(n), method="SLSQP",
                   bounds=[(0.0, max_weight)] * n, constraints=cons,
                   options={"maxiter": 1000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"max-Sharpe 优化失败: {res.message}")
    return res.x


def efficient_frontier(hist_returns: pd.DataFrame, sectors: pd.Series = None,
                       max_weight: float = 0.10, sector_cap: float = 0.25,
                       n_points: int = 25) -> pd.DataFrame:
    """
    有效前沿: 在 [GMV 收益, 收益上限] 之间取 n_points 个目标收益, 各解一个最小方差。
    收益上限取截面预期收益的 90% 分位, 保证约束下数值可行 (quality check)。
    """
    mu = hist_returns.mean()
    w_gmv = min_variance_weights(hist_returns, sectors, max_weight, sector_cap)
    r_lo = float(w_gmv @ mu.to_numpy())
    r_hi = float(mu.quantile(0.90))

    rows = []
    for target in np.linspace(r_lo, r_hi, n_points):
        try:
            w = min_variance_weights(hist_returns, sectors, max_weight, sector_cap,
                                     target_return=target)
            rows.append({"target_return": target,
                         "port_return": float(w @ mu.to_numpy()),
                         "port_vol": float(np.sqrt(w @ hist_returns.cov().to_numpy() @ w))})
        except RuntimeError:
            continue  # 不可行的目标点跳过
    return pd.DataFrame(rows)


# ================================================================ 回测引擎 (Task 3/4 共用)

def backtest_allocation(monthly_returns: pd.DataFrame, method: str,
                        lookback: int = 12, min_history: int = 12,
                        sectors: pd.Series = None, rf_monthly: pd.Series = None,
                        max_weight: float = 0.10, sector_cap: float = 0.25) -> dict:
    """
    月度调仓回测, 只用调仓日之前的信息 (rolling lookback 窗口)。

    method: "ew" 等权 | "rp" 风险平价(逆波动) | "mvo" 最大Sharpe
    资格规则: 调仓日往前 lookback 个月收益完整 (无 NaN) 的股票才入池。
    返回: port_ret (组合月收益), weights (逐期权重), turnover (逐期换手)
    """
    dates = monthly_returns.index
    tickers = list(monthly_returns.columns)

    weights_hist, port_ret, turnover_hist = {}, {}, {}
    prev_w = None  # (index tickers, weights)

    for i, t in enumerate(dates):
        hist = monthly_returns.iloc[max(0, i - lookback):i]
        if len(hist) < min_history:
            continue
        eligible = [c for c in tickers if hist[c].notna().all()]
        if len(eligible) < 10:
            continue
        h = hist[eligible]

        if method == "ew":
            w = equal_weight(len(eligible))
        elif method == "rp":
            w = inverse_vol_weights(h)
        elif method == "mvo":
            rf_p = float(rf_monthly.get(t, 0.0)) if rf_monthly is not None else 0.0
            sec = sectors.reindex(eligible) if sectors is not None else None
            try:
                w = max_sharpe_weights(h, rf_p, sec, max_weight, sector_cap)
            except RuntimeError:
                w = equal_weight(len(eligible))  # 优化失败回退等权
        else:
            raise ValueError(method)

        assert abs(w.sum() - 1.0) < 1e-6, "权重必须归一"

        # 下期收益
        if i + 1 >= len(dates):
            break
        r_next = monthly_returns.iloc[i + 1][eligible].fillna(0.0).to_numpy()
        port_ret[dates[i + 1]] = float(w @ r_next)
        weights_hist[t] = pd.Series(w, index=eligible)

        # 换手: |新权重 - 漂移后旧权重|; 新进/调出股票按 0→w 或 w→0 计
        if prev_w is not None:
            old_idx, old_w = prev_w
            drifted = pd.Series(old_w, index=old_idx) * (1 + monthly_returns.iloc[i][old_idx].fillna(0.0))
            drifted = drifted / drifted.sum()
            new_w = pd.Series(w, index=eligible)
            all_idx = drifted.index.union(new_w.index)
            turnover_hist[t] = float((new_w.reindex(all_idx, fill_value=0.0)
                                      - drifted.reindex(all_idx, fill_value=0.0)).abs().sum())
        prev_w = (eligible, w)

    return {"port_ret": pd.Series(port_ret).sort_index(),
            "weights": pd.DataFrame(weights_hist),
            "turnover": pd.Series(turnover_hist).sort_index()}
