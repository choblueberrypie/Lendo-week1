# -*- coding: utf-8 -*-
"""
factor_utils.py — Week 2: point-in-time factor construction (Value & Momentum)

数据源:
  - OHLCV (Yahoo Finance): 未复权 close 用于市值, 复权 adj_close 用于收益率/动量
  - SEC EDGAR companyfacts: 股东权益 / 净利润 / 股本, 全部带真实 filing_date

因子定义 (每月末调仓日 t):
  - B/M      = 最新已披露股东权益(t 或之前 filed) / 市值(t)
  - E/P      = TTM 净利润(t 或之前 filed 的最近 4 个季度之和) / 市值(t)
  - MOM 12-1 = adj_close(t-1月) / adj_close(t-12月) - 1   (跳过最近 1 个月)

Point-in-time 规则:
  - 任何财务数据必须满足 filing_date <= t 才可用 (无前视偏差)
  - 同一 period_end 有多次披露(重述)时, 取 t 时点已知的最新版本
  - 市值 = 未复权 close × 股本; 收益率 = 复权 adj_close
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 常量

# 每个字段的 tag 优先级 (数字越小越优先; 用于同 period_end 多 tag 冲突时选值)
TAG_PRIORITY = {
    "us-gaap:StockholdersEquity": 0,
    "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": 1,
    "derived:AssetsMinusLiabilities": 2,
    "us-gaap:NetIncomeLoss": 0,
    "us-gaap:ProfitLoss": 1,
    "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic": 2,
    "dei:EntityCommonStockSharesOutstanding": 0,
    "yfinance:SharesOutstanding(B-equivalent)": 0,
    "us-gaap:CommonStockSharesOutstanding": 1,
    "us-gaap:CommonStockSharesIssued": 2,
    "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic": 3,
    "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": 4,
}

QUARTER_DAYS = (60, 130)     # 季度 duration 合理范围
ANNUAL_DAYS = (300, 400)     # 年度 duration 合理范围
MAX_GAP_DAYS = 170           # TTM 相邻季度允许的最大间隔
SHARES_STALE_DAYS = 400      # 股本数据允许的最大陈旧度
EQUITY_STALE_DAYS = 550      # 权益数据允许的最大陈旧度 (~1.5 年)
NI_STALE_DAYS = 200          # TTM 最近一个季度允许的最大陈旧度


# ---------------------------------------------------------------- 数据加载

def load_ohlcv(path) -> pd.DataFrame:
    """读取 OHLCV, 统一列名为 date/ticker/open/high/low/close/volume/adj_close"""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_sec_facts(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
    df["duration_days"] = (df["period_end"] - df["start"]).dt.days
    df["priority"] = df["tag"].map(TAG_PRIORITY).fillna(9).astype(int)
    # filing_date 缺失的行无法做 point-in-time, 直接丢弃 (数据中实际为 0)
    df = df.dropna(subset=["period_end", "value", "filing_date"])
    # 股本合理性过滤: 标普500级别公司股本不可能 < 100 万股或 > 1000 亿股
    # (SEC 原始数据混有 100 股之类的占位值, 也有 1e14 之类的单位错误值)
    bad_shares = (df["field"] == "shares_outstanding") & \
                 ((df["value"] < 1e6) | (df["value"] > 1e11))
    return df[~bad_shares]


def load_splits(path) -> pd.DataFrame:
    """读取拆股事件: ticker, split_date, ratio (ratio=新/旧, 如 1拆4 → 4.0)"""
    df = pd.read_csv(path)
    df["split_date"] = pd.to_datetime(df["split_date"])
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def split_factor_series(splits: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    每个调仓日 t 的累计未来拆股因子: Π(t 之后发生的拆股比例)。
    用于把当时实际股本换算成拆股后等价股本, 与 Yahoo 拆股调整后价格对齐。
    返回 DataFrame(index=dates, columns=tickers), 无拆股的股票恒为 1。
    """
    tickers = splits["ticker"].unique() if len(splits) else []
    factor = pd.DataFrame(1.0, index=dates, columns=tickers)
    for tk, g in splits.groupby("ticker"):
        g = g.sort_values("split_date")
        cum = 1.0
        # 从最晚的拆股往前累计
        for dt, ratio in zip(g["split_date"][::-1], g["ratio"][::-1]):
            cum *= ratio
            factor.loc[factor.index < dt, tk] = cum
    return factor


def get_rebalance_dates(ohlcv: pd.DataFrame) -> pd.DatetimeIndex:
    """每月最后一个交易日 = 调仓日"""
    d = ohlcv[["date"]].drop_duplicates().sort_values("date")
    return pd.DatetimeIndex(d.groupby(d["date"].dt.to_period("M"))["date"].max().values)


# ------------------------------------------------- point-in-time 取值核心

class PITSeries:
    """单 (ticker, field) 的 facts 序列, 按 filing_date 排序, 支持快速切片"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("filing_date").reset_index(drop=True)
        self._filing = self.df["filing_date"].values

    def upto(self, t: pd.Timestamp) -> pd.DataFrame:
        """filing_date <= t 的全部记录"""
        idx = np.searchsorted(self._filing, np.datetime64(t), side="right")
        return self.df.iloc[:idx]


def _dedup_best(df: pd.DataFrame) -> pd.DataFrame:
    """每个 period_end 保留 PIT 最优记录: tag 优先级最高, 其次 filing 最新"""
    d = df.sort_values(["period_end", "priority", "filing_date"],
                       ascending=[True, True, False])
    return d.drop_duplicates("period_end", keep="first")


def _pit_latest(pit: PITSeries, t: pd.Timestamp, max_stale_days: int):
    """t 时点某瞬时字段的最新已知值; 陈旧超限视为缺失。返回 (value, period_end)"""
    if pit is None:
        return np.nan, None
    g = pit.upto(t)
    if g.empty:
        return np.nan, None
    row = _dedup_best(g).iloc[-1]  # period_end 升序, 最后一行即最新
    pe = row["period_end"]
    if (t - pe).days > max_stale_days:
        return np.nan, None
    return row["value"], pe


def pit_equity(pit: PITSeries, t: pd.Timestamp):
    return _pit_latest(pit, t, EQUITY_STALE_DAYS)


def pit_shares(pit: PITSeries, t: pd.Timestamp):
    return _pit_latest(pit, t, SHARES_STALE_DAYS)


def pit_ni_ttm(pit: PITSeries, t: pd.Timestamp):
    """
    t 时点可知的 TTM 净利润:
      1. 季度记录 (duration 60-130 天) 按 PIT 规则取每季度最优值
      2. 年度记录 (duration 300-400 天) 用于推导缺失的 Q4:  Q4 = FY - (Q1+Q2+Q3)
      3. 最近 4 个连续季度求和; 最近季度陈旧度 > NI_STALE_DAYS 则缺失
    """
    if pit is None:
        return np.nan, None
    g = pit.upto(t)
    if g.empty:
        return np.nan, None

    q = g[g["duration_days"].between(*QUARTER_DAYS)]
    quarterly = {}
    if not q.empty:
        qb = _dedup_best(q)
        quarterly = dict(zip(qb["period_end"], qb["value"]))

    a = g[g["duration_days"].between(*ANNUAL_DAYS)]
    if not a.empty:
        ab = _dedup_best(a)
        for _, row in ab.iterrows():
            fy_start, fy_end, fy_val = row["start"], row["period_end"], row["value"]
            if pd.isna(fy_start):
                continue
            inner_q = [v for k, v in quarterly.items() if fy_start < k < fy_end]
            # 若该财年最后一个季度(Q4)没有独立季度记录, 用 FY - 前3季度 推导
            if fy_end not in quarterly and len(inner_q) == 3:
                quarterly[fy_end] = fy_val - sum(inner_q)

    if len(quarterly) < 4:
        return np.nan, None

    pes = sorted(quarterly.keys())
    last4 = pes[-4:]
    gaps = [(last4[i + 1] - last4[i]).days for i in range(3)]
    if not all(QUARTER_DAYS[0] <= x <= MAX_GAP_DAYS for x in gaps):
        return np.nan, None
    if (t - last4[-1]).days > NI_STALE_DAYS:
        return np.nan, None
    return sum(quarterly[pe] for pe in last4), last4[-1]


# ---------------------------------------------------------------- 因子面板

def build_factor_panel(ohlcv: pd.DataFrame, facts: pd.DataFrame,
                       rebalance_dates: pd.DatetimeIndex,
                       splits: pd.DataFrame = None) -> pd.DataFrame:
    """
    构建月度因子面板。
    返回列: date, ticker, close, shares, equity, ni_ttm, mktcap,
            bm, ep, mom_12_1, fwd_ret_1m

    splits: 拆股事件 (load_splits 输出)。Yahoo 价格为拆股调整口径,
            股本需乘以 split factor 对齐后才能算正确市值。
    """
    close_p = ohlcv.pivot(index="date", columns="ticker", values="close").reindex(rebalance_dates)
    adj_p = ohlcv.pivot(index="date", columns="ticker", values="adj_close").reindex(rebalance_dates)

    # 动量 12-1 和下期收益 (用复权价), 转 numpy 加速
    mom_m = (adj_p.shift(1) / adj_p.shift(12) - 1.0).to_numpy()
    fwd_m = (adj_p.shift(-1) / adj_p - 1.0).to_numpy()
    close_m = close_p.to_numpy()
    tickers = list(close_p.columns)

    # 拆股因子矩阵 (与 close_m 同形)
    sf_m = np.ones_like(close_m)
    if splits is not None and len(splits):
        sf = split_factor_series(splits, rebalance_dates)
        sf = sf.reindex(columns=tickers).fillna(1.0)
        sf_m = sf.to_numpy()

    pit_map = {}
    for (tk, f), g in facts.groupby(["ticker", "field"]):
        pit_map[(tk, f)] = PITSeries(g)

    records = []
    for i, t in enumerate(rebalance_dates):
        for j, tk in enumerate(tickers):
            px = close_m[i, j]
            if not np.isfinite(px):
                continue  # 该日未上市/无交易
            equity, _ = pit_equity(pit_map.get((tk, "equity")), t)
            shares_raw, _ = pit_shares(pit_map.get((tk, "shares_outstanding")), t)
            ni_ttm, _ = pit_ni_ttm(pit_map.get((tk, "net_income")), t)

            # 股本换算成拆股后等价口径, 与 Yahoo 拆股调整后价格对齐
            shares = shares_raw * sf_m[i, j] if pd.notna(shares_raw) else np.nan

            mktcap = px * shares if pd.notna(shares) and shares > 0 else np.nan
            bm = equity / mktcap if pd.notna(equity) and pd.notna(mktcap) and equity > 0 else np.nan
            ep = ni_ttm / mktcap if pd.notna(ni_ttm) and pd.notna(mktcap) else np.nan

            records.append({
                "date": t, "ticker": tk, "close": px, "shares": shares,
                "equity": equity, "ni_ttm": ni_ttm, "mktcap": mktcap,
                "bm": bm, "ep": ep,
                "mom_12_1": mom_m[i, j], "fwd_ret_1m": fwd_m[i, j],
            })
    return pd.DataFrame(records)


def panel_coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    """每个调仓日的因子覆盖率"""
    rows = []
    for t, g in panel.groupby("date"):
        rows.append({
            "date": t, "n_stocks": len(g),
            "bm_valid": g["bm"].notna().sum(),
            "ep_valid": g["ep"].notna().sum(),
            "mom_valid": g["mom_12_1"].notna().sum(),
            "fwd_valid": g["fwd_ret_1m"].notna().sum(),
        })
    return pd.DataFrame(rows)


# ================================================================ 回测
# 五分位组合回测: 与 Week 2 任务书一致的方法
#   - 每个调仓日按 signal 横截面排序, Q1(最低) ~ Q5(最高)
#   - 等权持有至下一调仓日, 用 fwd_ret_1m (复权收益) 评价
#   - 指标: 各组月均/年化/Sharpe/t值, 多空 Q5-Q1, Spearman IC/ICIR

FACTOR_LABELS = {
    "bm": "B/M (Book-to-Market)",
    "ep": "E/P (Earnings-to-Price)",
    "mom_12_1": "Momentum 12-1",
}

DEFAULT_MIN_STOCKS = 25
DEFAULT_N_Q = 5


def load_rf_monthly(rf_path, dates: pd.DatetimeIndex) -> pd.Series:
    """^IRX 年化收益率 -> 调仓日对应的月度无风险利率 (小数)"""
    rf = pd.read_csv(rf_path)
    rf.columns = [c.strip().lower() for c in rf.columns]
    date_col = "date" if "date" in rf.columns else rf.columns[0]
    val_col = [c for c in rf.columns if c != date_col][-1]
    rf[date_col] = pd.to_datetime(rf[date_col])
    rf = rf.set_index(date_col)[val_col].sort_index()
    # ^IRX 单位是年化 %; 对齐到调仓日 (取当日或之前最近值), 转月度小数
    return rf.reindex(rf.index.union(dates)).ffill().reindex(dates) / 100.0 / 12.0


def spearman_ic(x: pd.Series, y: pd.Series) -> float:
    """Spearman 秩相关 = 各自排名后的 Pearson 相关 (避免 scipy 依赖)"""
    rx, ry = x.rank(), y.rank()
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return rx.corr(ry)


def assign_quintiles(g: pd.DataFrame, signal: str, n_q: int = DEFAULT_N_Q) -> pd.Series:
    """Q1 = signal 最低 ... Q5 = 最高; rank(method='first') 处理并列"""
    return pd.qcut(g[signal].rank(method="first"), n_q,
                   labels=[f"Q{i}" for i in range(1, n_q + 1)])


def backtest_factor(panel: pd.DataFrame, signal: str,
                    min_stocks: int = DEFAULT_MIN_STOCKS,
                    n_q: int = DEFAULT_N_Q) -> dict:
    """
    单因子五分位回测。
    返回 dict:
      q_ret  — 长表: date/factor/quintile/n/ret (等权) /ret_vw (市值加权, 参考)
      ic     — 每月 Spearman IC
      ls     — 每月多空收益 (Q5 - Q1)
      pivot  — date × quintile 的等权月收益矩阵
    """
    rows_q, ic_rows = [], []
    for t, g in panel.groupby("date"):
        g = g.dropna(subset=[signal, "fwd_ret_1m"])
        if len(g) < min_stocks:
            continue
        g = g.copy()
        g["q"] = assign_quintiles(g, signal, n_q)
        for q, qg in g.groupby("q", observed=True):
            vw = np.average(qg["fwd_ret_1m"], weights=qg["mktcap"]) \
                if qg["mktcap"].notna().all() else np.nan
            rows_q.append({"date": t, "factor": signal, "quintile": q,
                           "n": len(qg), "ret": qg["fwd_ret_1m"].mean(), "ret_vw": vw})
        ic_rows.append({"date": t, "factor": signal,
                        "ic": spearman_ic(g[signal], g["fwd_ret_1m"])})

    q_ret = pd.DataFrame(rows_q)
    ic = pd.DataFrame(ic_rows)
    pivot = q_ret.pivot_table(index="date", columns="quintile", values="ret")
    ls = pd.DataFrame({"date": pivot.index, "factor": signal,
                       "ls_ret": pivot[f"Q{n_q}"] - pivot["Q1"]})
    return {"q_ret": q_ret, "ic": ic, "ls": ls, "pivot": pivot}


def quintile_stats(pivot: pd.DataFrame, rf_m: pd.Series = None) -> pd.DataFrame:
    """各组统计: 月均/年化收益/年化波动/Sharpe/t值/月数, 最后一行多空 Q5-Q1"""
    rows = []
    n_q = len(pivot.columns)
    for q in pivot.columns:
        r = pivot[q].dropna()
        rf = rf_m.reindex(r.index).fillna(0) if rf_m is not None else 0
        ex = r - rf
        ann = (1 + r).prod() ** (12 / len(r)) - 1
        vol = r.std() * np.sqrt(12)
        sharpe = ex.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
        t_stat = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else np.nan
        rows.append({"portfolio": q, "mean_monthly": r.mean(), "ann_return": ann,
                     "ann_vol": vol, "sharpe": sharpe, "t_stat": t_stat, "months": len(r)})
    ls = pivot[f"Q{n_q}"] - pivot["Q1"]
    ls = ls.dropna()
    rows.append({"portfolio": f"Q{n_q}-Q1", "mean_monthly": ls.mean(),
                 "ann_return": ls.mean() * 12, "ann_vol": ls.std() * np.sqrt(12),
                 "sharpe": np.nan,
                 "t_stat": ls.mean() / (ls.std() / np.sqrt(len(ls))), "months": len(ls)})
    return pd.DataFrame(rows)


def ic_stats(ic: pd.DataFrame) -> dict:
    v = ic["ic"].dropna()
    return {"mean_ic": v.mean(),
            "icir": v.mean() / v.std() if v.std() > 0 else np.nan,
            "t_stat": v.mean() / (v.std() / np.sqrt(len(v))) if v.std() > 0 else np.nan,
            "pct_positive": (v > 0).mean(), "months": len(v)}


def yearly_ls_returns(ls: pd.DataFrame) -> pd.Series:
    """多空收益按年度复合"""
    s = ls.copy()
    s["year"] = pd.to_datetime(s["date"]).dt.year
    return s.groupby("year")["ls_ret"].apply(lambda x: (1 + x).prod() - 1)
