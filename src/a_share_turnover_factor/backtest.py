from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    group_returns: pd.DataFrame
    average_group_returns: pd.Series
    ic_series: pd.Series
    sample_counts: pd.Series


def run_turnover_factor_backtest(
    stock_basic: pd.DataFrame,
    namechange: pd.DataFrame,
    daily: pd.DataFrame,
    daily_basic: pd.DataFrame,
    adj_factor: pd.DataFrame,
    trade_dates: list[str],
    start_date: str,
    end_date: str,
    group_count: int = 5,
    rolling_window: int = 20,
    min_listed_days: int = 180,
    rebalance_frequency: str = "monthly",
) -> BacktestResult:
    dates = pd.Index(pd.to_datetime(trade_dates), name="trade_date")
    if dates.empty:
        raise ValueError("No trading dates were provided.")

    stock_info = _prepare_stock_basic(stock_basic)
    stock_info = stock_info.loc[_is_a_share(stock_info)].copy()
    codes = sorted(stock_info.index.tolist())
    if not codes:
        raise ValueError("No A-share stocks found in stock_basic.")

    turnover = _pivot_panel(daily_basic, "turnover_rate", dates, codes)
    close = _pivot_panel(daily, "close", dates, codes)
    factor_df = _pivot_panel(adj_factor, "adj_factor", dates, codes)
    adj_close = close * factor_df

    factor = turnover.rolling(rolling_window, min_periods=rolling_window).mean()
    rebalance_dates = _rebalance_trade_dates(dates, start_date, end_date, rebalance_frequency)
    if len(rebalance_dates) < 2:
        raise ValueError("Need at least two rebalance dates to compute forward returns.")

    factor_rebalance = factor.reindex(rebalance_dates)
    rebalance_close = adj_close.reindex(rebalance_dates)
    forward_returns = rebalance_close.shift(-1) / rebalance_close - 1.0

    active_mask = _active_on_dates(stock_info, rebalance_dates)
    listed_mask = _listed_days_mask(stock_info, rebalance_dates, min_listed_days)
    st_mask = _st_mask(stock_info, namechange, rebalance_dates)
    eligible = active_mask & listed_mask & ~st_mask

    group_returns: dict[pd.Timestamp, pd.Series] = {}
    ic_values: dict[pd.Timestamp, float] = {}
    sample_counts: dict[pd.Timestamp, int] = {}
    labels = [f"G{i}" for i in range(1, group_count + 1)]

    for date in rebalance_dates[:-1]:
        factor_s = factor_rebalance.loc[date]
        return_s = forward_returns.loc[date]
        valid = eligible.loc[date] & factor_s.notna() & return_s.notna()
        sample_counts[date] = int(valid.sum())
        if valid.sum() < group_count:
            group_returns[date] = pd.Series(index=labels, dtype=float)
            ic_values[date] = np.nan
            continue

        factor_valid = factor_s.loc[valid]
        return_valid = return_s.loc[valid]
        groups = _assign_quantile_groups(factor_valid, group_count)
        grouped_ret = return_valid.groupby(groups).mean().reindex(labels)
        group_returns[date] = grouped_ret
        ic_values[date] = _spearman_ic(factor_valid, return_valid)

    group_returns_df = pd.DataFrame.from_dict(group_returns, orient="index").sort_index()
    group_returns_df.index.name = "rebalance_date"
    ic_series = pd.Series(ic_values, name="IC").sort_index()
    ic_series.index.name = "rebalance_date"
    sample_counts_series = pd.Series(sample_counts, name="sample_count").sort_index()
    sample_counts_series.index.name = "rebalance_date"
    average_group_returns = group_returns_df.mean(axis=0)
    average_group_returns.name = "average_period_return"

    return BacktestResult(
        group_returns=group_returns_df,
        average_group_returns=average_group_returns,
        ic_series=ic_series,
        sample_counts=sample_counts_series,
    )


def _prepare_stock_basic(stock_basic: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "name", "list_date"}
    missing = required - set(stock_basic.columns)
    if missing:
        raise ValueError(f"stock_basic missing columns: {sorted(missing)}")

    info = stock_basic.copy()
    info["ts_code"] = info["ts_code"].astype(str)
    info["name"] = info["name"].fillna("").astype(str)
    info["list_date"] = pd.to_datetime(info["list_date"].astype("string"), format="%Y%m%d", errors="coerce")
    if "delist_date" in info.columns:
        info["delist_date"] = pd.to_datetime(info["delist_date"].astype("string"), format="%Y%m%d", errors="coerce")
    else:
        info["delist_date"] = pd.NaT
    return info.drop_duplicates("ts_code").set_index("ts_code")


def _is_a_share(stock_info: pd.DataFrame) -> pd.Series:
    suffix_ok = stock_info.index.to_series().str.endswith((".SH", ".SZ", ".BJ"))
    if "exchange" not in stock_info.columns:
        return suffix_ok
    exchange_ok = stock_info["exchange"].isin(["SSE", "SZSE", "BSE"]) | stock_info["exchange"].isna()
    return suffix_ok & exchange_ok


def _pivot_panel(frame: pd.DataFrame, value_col: str, dates: pd.Index, codes: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=dates, columns=codes, dtype=float)
    required = {"ts_code", "trade_date", value_col}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"daily frame missing columns for {value_col}: {sorted(missing)}")
    prepared = frame.loc[:, ["ts_code", "trade_date", value_col]].copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    prepared["ts_code"] = prepared["ts_code"].astype(str)
    prepared[value_col] = pd.to_numeric(prepared[value_col], errors="coerce")
    panel = prepared.pivot_table(index="trade_date", columns="ts_code", values=value_col, aggfunc="last")
    return panel.reindex(index=dates, columns=codes).astype(float)


def _month_end_trade_dates(dates: pd.Index, start_date: str, end_date: str) -> pd.DatetimeIndex:
    return _period_end_trade_dates(dates, start_date, end_date, "M", "month_end")


def _week_end_trade_dates(dates: pd.Index, start_date: str, end_date: str) -> pd.DatetimeIndex:
    return _period_end_trade_dates(dates, start_date, end_date, "W-FRI", "week_end")


def _rebalance_trade_dates(
    dates: pd.Index,
    start_date: str,
    end_date: str,
    rebalance_frequency: str,
) -> pd.DatetimeIndex:
    if rebalance_frequency == "monthly":
        return _month_end_trade_dates(dates, start_date, end_date)
    if rebalance_frequency == "weekly":
        return _week_end_trade_dates(dates, start_date, end_date)
    raise ValueError("rebalance_frequency must be 'monthly' or 'weekly'.")


def _period_end_trade_dates(
    dates: pd.Index,
    start_date: str,
    end_date: str,
    period_freq: str,
    index_name: str,
) -> pd.DatetimeIndex:
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    scoped = pd.Series(dates[(dates >= start) & (dates <= end)])
    period_ends = scoped.groupby(scoped.dt.to_period(period_freq)).max().sort_values()
    return pd.DatetimeIndex(period_ends.tolist(), name=index_name)


def _active_on_dates(stock_info: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    output = pd.DataFrame(False, index=dates, columns=stock_info.index)
    for code, row in stock_info.iterrows():
        listed = dates >= row["list_date"]
        delist_date = row.get("delist_date", pd.NaT)
        if pd.isna(delist_date):
            active = listed
        else:
            active = listed & (dates <= delist_date)
        output[code] = active
    return output


def _listed_days_mask(stock_info: pd.DataFrame, dates: pd.DatetimeIndex, min_listed_days: int) -> pd.DataFrame:
    output = pd.DataFrame(False, index=dates, columns=stock_info.index)
    for code, list_date in stock_info["list_date"].items():
        output[code] = (dates - list_date).days >= min_listed_days
    return output


def _st_mask(stock_info: pd.DataFrame, namechange: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    output = pd.DataFrame(False, index=dates, columns=stock_info.index)
    if namechange.empty or not {"ts_code", "name", "start_date"}.issubset(namechange.columns):
        current_st = stock_info["name"].str.contains("ST", case=False, regex=False, na=False)
        for code in current_st[current_st].index:
            output[code] = True
        return output

    changes = namechange.copy()
    changes["ts_code"] = changes["ts_code"].astype(str)
    changes = changes.loc[changes["ts_code"].isin(stock_info.index)]
    changes["name"] = changes["name"].fillna("").astype(str)
    changes = changes.loc[changes["name"].str.contains("ST", case=False, regex=False)]
    changes["start_date"] = pd.to_datetime(changes["start_date"].astype("string"), format="%Y%m%d", errors="coerce")
    if "end_date" in changes.columns:
        changes["end_date"] = pd.to_datetime(changes["end_date"].astype("string"), format="%Y%m%d", errors="coerce")
    else:
        changes["end_date"] = pd.NaT

    for _, row in changes.iterrows():
        if pd.isna(row["start_date"]):
            continue
        end_date = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp.max
        mask = (dates >= row["start_date"]) & (dates <= end_date)
        output.loc[mask, row["ts_code"]] = True

    return output


def _assign_quantile_groups(factor: pd.Series, group_count: int) -> pd.Series:
    labels = [f"G{i}" for i in range(1, group_count + 1)]
    ranked = factor.rank(method="first")
    groups = pd.qcut(ranked, q=group_count, labels=labels)
    return pd.Series(groups.astype(str), index=factor.index, name="group")


def _spearman_ic(factor: pd.Series, forward_return: pd.Series) -> float:
    factor_rank = factor.rank(method="average")
    return_rank = forward_return.rank(method="average")
    if len(factor_rank) < 2 or factor_rank.std(ddof=0) == 0 or return_rank.std(ddof=0) == 0:
        return np.nan
    return float(factor_rank.corr(return_rank))
