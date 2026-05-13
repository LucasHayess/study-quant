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


@dataclass(frozen=True)
class PortfolioBacktestResult:
    period_returns: pd.DataFrame
    nav: pd.DataFrame
    annual_returns: pd.DataFrame
    metrics: pd.Series
    max_drawdown_start: pd.Timestamp
    max_drawdown_end: pd.Timestamp
    holdings: pd.DataFrame


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
    daily_amount: pd.DataFrame | None = None,
    liquidity_window: int = 20,
    liquidity_min_avg_amount_yuan: float | None = None,
    factor_type: str = "turnover",
    momentum_window: int = 20,
    momentum_skip_days: int = 1,
    financial_indicator: pd.DataFrame | None = None,
    income_statement: pd.DataFrame | None = None,
    analyst_report: pd.DataFrame | None = None,
) -> BacktestResult:
    dates = pd.Index(pd.to_datetime(trade_dates), name="trade_date")
    if dates.empty:
        raise ValueError("No trading dates were provided.")

    stock_info = _prepare_stock_basic(stock_basic)
    stock_info = stock_info.loc[_is_a_share(stock_info)].copy()
    codes = sorted(stock_info.index.tolist())
    if not codes:
        raise ValueError("No A-share stocks found in stock_basic.")

    close = _pivot_panel(daily, "close", dates, codes)
    factor_df = _pivot_panel(adj_factor, "adj_factor", dates, codes)
    adj_close = close * factor_df
    rebalance_dates = _rebalance_trade_dates(dates, start_date, end_date, rebalance_frequency)
    if len(rebalance_dates) < 2:
        raise ValueError("Need at least two rebalance dates to compute forward returns.")

    if factor_type == "turnover":
        turnover = _pivot_panel(daily_basic, "turnover_rate", dates, codes)
        factor = turnover.rolling(rolling_window, min_periods=rolling_window).mean()
    elif factor_type == "turnover_inverse":
        turnover = _pivot_panel(daily_basic, "turnover_rate", dates, codes)
        factor = -turnover.rolling(rolling_window, min_periods=rolling_window).mean()
    elif factor_type == "momentum":
        factor = adj_close.shift(momentum_skip_days) / adj_close.shift(momentum_skip_days + momentum_window) - 1.0
    elif factor_type == "momentum_inverse":
        momentum = adj_close.shift(momentum_skip_days) / adj_close.shift(momentum_skip_days + momentum_window) - 1.0
        factor = -momentum
    elif factor_type == "composite":
        turnover = _pivot_panel(daily_basic, "turnover_rate", dates, codes)
        turnover_inverse = -turnover.rolling(rolling_window, min_periods=rolling_window).mean()
        momentum = adj_close.shift(momentum_skip_days) / adj_close.shift(momentum_skip_days + momentum_window) - 1.0
        momentum_inverse = -momentum
        turnover_rank = turnover_inverse.rank(axis=1, method="average", pct=True)
        momentum_rank = momentum_inverse.rank(axis=1, method="average", pct=True)
        factor = 0.5 * turnover_rank + 0.5 * momentum_rank
    elif factor_type == "quality_roe":
        if financial_indicator is None or financial_indicator.empty:
            raise ValueError("financial_indicator is required when factor_type='quality_roe'.")
        factor = _latest_annual_quality_panel(financial_indicator, rebalance_dates, codes, "roe")
    elif factor_type == "quality_grossprofit_margin_yoy":
        if financial_indicator is None or financial_indicator.empty:
            raise ValueError(
                "financial_indicator is required when factor_type='quality_grossprofit_margin_yoy'."
            )
        factor = _latest_annual_quality_panel(
            financial_indicator,
            rebalance_dates,
            codes,
            "grossprofit_margin_yoy",
        )
    elif factor_type == "quality_ocf_to_np":
        if financial_indicator is None or financial_indicator.empty:
            raise ValueError("financial_indicator is required when factor_type='quality_ocf_to_np'.")
        factor = _latest_annual_quality_panel(financial_indicator, rebalance_dates, codes, "ocf_to_profit")
    elif factor_type == "quality_net_profit_yoy":
        if income_statement is None or income_statement.empty:
            raise ValueError("income_statement is required when factor_type='quality_net_profit_yoy'.")
        factor = _latest_annual_net_profit_yoy_panel(income_statement, rebalance_dates, codes)
    elif factor_type == "analyst_eps_revision_count":
        if analyst_report is None or analyst_report.empty:
            raise ValueError("analyst_report is required when factor_type='analyst_eps_revision_count'.")
        factor = _analyst_eps_revision_panel(analyst_report, rebalance_dates, codes, "count")
    elif factor_type == "analyst_eps_revision_magnitude":
        if analyst_report is None or analyst_report.empty:
            raise ValueError("analyst_report is required when factor_type='analyst_eps_revision_magnitude'.")
        factor = _analyst_eps_revision_panel(analyst_report, rebalance_dates, codes, "magnitude")
    else:
        raise ValueError(
            "factor_type must be 'turnover', 'turnover_inverse', 'momentum', "
            "'momentum_inverse', 'composite', 'quality_roe', "
            "'quality_grossprofit_margin_yoy', 'quality_ocf_to_np', "
            "'quality_net_profit_yoy', 'analyst_eps_revision_count', "
            "or 'analyst_eps_revision_magnitude'."
        )

    factor_rebalance = factor.reindex(rebalance_dates)
    rebalance_close = adj_close.reindex(rebalance_dates)
    forward_returns = rebalance_close.shift(-1) / rebalance_close - 1.0

    active_mask = _active_on_dates(stock_info, rebalance_dates)
    listed_mask = _listed_days_mask(stock_info, rebalance_dates, min_listed_days)
    st_mask = _st_mask(stock_info, namechange, rebalance_dates)
    eligible = active_mask & listed_mask & ~st_mask
    if liquidity_min_avg_amount_yuan is not None:
        if daily_amount is None or daily_amount.empty:
            raise ValueError("daily_amount is required when liquidity_min_avg_amount_yuan is set.")
        amount = _pivot_panel(daily_amount, "amount", dates, codes)
        amount_threshold = liquidity_min_avg_amount_yuan / 1000.0
        avg_amount = amount.rolling(liquidity_window, min_periods=liquidity_window).mean()
        liquidity_mask = avg_amount.reindex(rebalance_dates) >= amount_threshold
        eligible = eligible & liquidity_mask

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


def run_top_quantile_portfolio_backtest(
    stock_basic: pd.DataFrame,
    namechange: pd.DataFrame,
    daily: pd.DataFrame,
    daily_basic: pd.DataFrame,
    adj_factor: pd.DataFrame,
    daily_amount: pd.DataFrame,
    benchmark_daily: pd.DataFrame,
    trade_dates: list[str],
    start_date: str,
    end_date: str,
    benchmark_500_daily: pd.DataFrame | None = None,
    top_quantile: float = 0.2,
    rolling_window: int = 5,
    min_listed_days: int = 180,
    rebalance_frequency: str = "monthly",
    liquidity_window: int = 20,
    liquidity_min_avg_amount_yuan: float = 20_000_000,
    momentum_window: int = 20,
    momentum_skip_days: int = 1,
    buy_commission_rate: float = 0.0003,
    sell_commission_rate: float = 0.0003,
    stamp_tax_rate: float = 0.001,
    market_risk_control: bool = False,
    market_risk_lookback_days: int = 20,
    market_risk_threshold: float = -0.05,
    defensive_exposure: float = 0.5,
    stock_stop_loss: bool = False,
    stop_loss_threshold: float = -0.10,
) -> PortfolioBacktestResult:
    if not 0 < top_quantile <= 1:
        raise ValueError("top_quantile must be in (0, 1].")
    if rebalance_frequency != "monthly":
        raise ValueError("Portfolio metrics assume monthly returns; use rebalance_frequency='monthly'.")
    if not 0 <= defensive_exposure <= 1:
        raise ValueError("defensive_exposure must be in [0, 1].")

    dates = pd.Index(pd.to_datetime(trade_dates), name="trade_date")
    if dates.empty:
        raise ValueError("No trading dates were provided.")

    stock_info = _prepare_stock_basic(stock_basic)
    stock_info = stock_info.loc[_is_a_share(stock_info)].copy()
    codes = sorted(stock_info.index.tolist())
    if not codes:
        raise ValueError("No A-share stocks found in stock_basic.")

    close = _pivot_panel(daily, "close", dates, codes)
    factor_df = _pivot_panel(adj_factor, "adj_factor", dates, codes)
    adj_close = close * factor_df
    turnover = _pivot_panel(daily_basic, "turnover_rate", dates, codes)
    amount = _pivot_panel(daily_amount, "amount", dates, codes)

    turnover_inverse = -turnover.rolling(rolling_window, min_periods=rolling_window).mean()
    momentum = adj_close.shift(momentum_skip_days) / adj_close.shift(momentum_skip_days + momentum_window) - 1.0
    momentum_inverse = -momentum
    turnover_rank = turnover_inverse.rank(axis=1, method="average", pct=True)
    momentum_rank = momentum_inverse.rank(axis=1, method="average", pct=True)
    composite_score = 0.5 * turnover_rank + 0.5 * momentum_rank

    rebalance_dates = _rebalance_trade_dates(
        dates,
        dates.min().strftime("%Y%m%d"),
        end_date,
        rebalance_frequency,
    )
    if len(rebalance_dates) < 2:
        raise ValueError("Need at least two rebalance dates to compute portfolio returns.")

    active_mask = _active_on_dates(stock_info, rebalance_dates)
    listed_mask = _listed_days_mask(stock_info, rebalance_dates, min_listed_days)
    st_mask = _st_mask(stock_info, namechange, rebalance_dates)
    amount_threshold = liquidity_min_avg_amount_yuan / 1000.0
    avg_amount = amount.rolling(liquidity_window, min_periods=liquidity_window).mean()
    liquidity_mask = avg_amount.reindex(rebalance_dates) >= amount_threshold
    eligible = active_mask & listed_mask & ~st_mask & liquidity_mask

    benchmark_close = _prepare_benchmark_close(benchmark_daily, dates)
    benchmark_trailing_return = (
        benchmark_close / benchmark_close.shift(market_risk_lookback_days) - 1.0
        if market_risk_control
        else pd.Series(index=benchmark_close.index, dtype=float)
    )
    benchmark_500_close = (
        _prepare_benchmark_close(benchmark_500_daily, dates)
        if benchmark_500_daily is not None and not benchmark_500_daily.empty
        else None
    )
    current_weights = pd.Series(dtype=float)
    next_stop_loss_exclusions: set[str] = set()
    rows: list[dict[str, object]] = []
    holding_rows: list[dict[str, object]] = []
    start_ts = pd.to_datetime(start_date, format="%Y%m%d")
    end_ts = pd.to_datetime(end_date, format="%Y%m%d")

    for rebalance_date, next_date in zip(rebalance_dates[:-1], rebalance_dates[1:]):
        if next_date < start_ts or next_date > end_ts:
            continue

        factor_s = composite_score.loc[rebalance_date]
        period_return_s = adj_close.loc[next_date] / adj_close.loc[rebalance_date] - 1.0
        valid = eligible.loc[rebalance_date] & factor_s.notna() & period_return_s.notna()
        stop_loss_exclusions = next_stop_loss_exclusions if stock_stop_loss else set()
        if stop_loss_exclusions:
            valid = valid & ~valid.index.isin(stop_loss_exclusions)
        sample_count = int(valid.sum())
        market_signal = (
            float(benchmark_trailing_return.loc[rebalance_date])
            if market_risk_control and rebalance_date in benchmark_trailing_return.index
            else np.nan
        )
        target_exposure = (
            defensive_exposure
            if market_risk_control and pd.notna(market_signal) and market_signal < market_risk_threshold
            else 1.0
        )
        if sample_count == 0:
            target_weights = pd.Series(dtype=float)
        else:
            selected_count = max(1, int(np.ceil(sample_count * top_quantile)))
            selected_codes = (
                factor_s.loc[valid]
                .sort_values(ascending=False, kind="mergesort")
                .head(selected_count)
                .index
            )
            target_weights = pd.Series(target_exposure / selected_count, index=selected_codes, dtype=float)
        if target_weights.empty and current_weights.empty:
            next_stop_loss_exclusions = set()
            continue

        aligned_index = current_weights.index.union(target_weights.index)
        previous_weights = current_weights.reindex(aligned_index, fill_value=0.0)
        target_aligned = target_weights.reindex(aligned_index, fill_value=0.0)
        trades = target_aligned - previous_weights
        buy_turnover = float(trades.clip(lower=0.0).sum())
        sell_turnover = float((-trades.clip(upper=0.0)).sum())
        buy_cost = buy_turnover * buy_commission_rate
        sell_cost = sell_turnover * (sell_commission_rate + stamp_tax_rate)
        total_cost = buy_cost + sell_cost

        if target_weights.empty:
            gross_return = 0.0
            net_return = -total_cost
            current_weights = pd.Series(dtype=float)
            stopped_codes: list[str] = []
        else:
            selected_returns = period_return_s.reindex(target_weights.index)
            gross_return = float((target_weights * selected_returns).sum())
            net_return = gross_return - total_cost
            growth = target_weights * (1.0 + selected_returns)
            ending_value_before_cost = 1.0 + gross_return
            current_weights = (
                growth / ending_value_before_cost
                if ending_value_before_cost != 0
                else target_weights
            )
            stopped_codes = (
                selected_returns.loc[selected_returns < stop_loss_threshold]
                .index.astype(str)
                .tolist()
                if stock_stop_loss
                else []
            )
        next_stop_loss_exclusions = set(stopped_codes) if stock_stop_loss else set()

        benchmark_return = _period_benchmark_return(benchmark_close, rebalance_date, next_date)
        row = {
            "rebalance_date": rebalance_date,
            "return_end_date": next_date,
            "strategy_gross_return": gross_return,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "buy_cost": buy_cost,
            "sell_cost": sell_cost,
            "total_cost": total_cost,
            "strategy_return": net_return,
            "benchmark_return": benchmark_return,
            "sample_count": sample_count,
            "holding_count": int(len(target_weights)),
            "target_exposure": target_exposure if not target_weights.empty else 0.0,
            "market_trailing_return": market_signal,
            "stop_loss_exclusion_count": int(len(stop_loss_exclusions)),
            "new_stop_loss_count": int(len(stopped_codes)),
        }
        if benchmark_500_close is not None:
            row["benchmark_500_return"] = _period_benchmark_return(
                benchmark_500_close,
                rebalance_date,
                next_date,
            )
        rows.append(row)
        holding_rows.append(
            {
                "rebalance_date": rebalance_date,
                "return_end_date": next_date,
                "ts_codes": ",".join(target_weights.index.astype(str).tolist()),
            }
        )

    period_returns = pd.DataFrame(rows)
    if period_returns.empty:
        raise ValueError("No portfolio return periods were produced.")
    period_returns = period_returns.set_index("return_end_date").sort_index()
    period_returns.index.name = "return_end_date"

    strategy_nav = (1.0 + period_returns["strategy_return"]).cumprod()
    benchmark_nav = (1.0 + period_returns["benchmark_return"]).cumprod()
    nav = pd.DataFrame({"strategy_nav": strategy_nav, "benchmark_nav": benchmark_nav})
    if "benchmark_500_return" in period_returns:
        nav["benchmark_500_nav"] = (1.0 + period_returns["benchmark_500_return"]).cumprod()
    nav["excess_nav"] = nav["strategy_nav"] / nav["benchmark_nav"]
    if "benchmark_500_nav" in nav:
        nav["excess_500_nav"] = nav["strategy_nav"] / nav["benchmark_500_nav"]
    nav.index.name = "date"

    annual_returns = _annual_returns(period_returns)
    metrics, max_drawdown_start, max_drawdown_end = _portfolio_metrics(period_returns, nav)
    holdings = pd.DataFrame(holding_rows).set_index("return_end_date").sort_index()

    return PortfolioBacktestResult(
        period_returns=period_returns,
        nav=nav,
        annual_returns=annual_returns,
        metrics=metrics,
        max_drawdown_start=max_drawdown_start,
        max_drawdown_end=max_drawdown_end,
        holdings=holdings,
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


def _latest_annual_roe_panel(
    financial_indicator: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    codes: list[str],
) -> pd.DataFrame:
    return _latest_annual_quality_panel(financial_indicator, rebalance_dates, codes, "roe")


def _latest_annual_quality_panel(
    financial_indicator: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    codes: list[str],
    factor_name: str,
) -> pd.DataFrame:
    value_columns = {
        "roe": ["roe"],
        "grossprofit_margin_yoy": ["grossprofit_margin"],
        "ocf_to_profit": ["ocf_to_profit"],
    }
    if factor_name not in value_columns:
        raise ValueError(
            "factor_name must be 'roe', 'grossprofit_margin_yoy', or 'ocf_to_profit'."
        )

    required = {"ts_code", "ann_date", "end_date", *value_columns[factor_name]}
    missing = required - set(financial_indicator.columns)
    if missing:
        raise ValueError(f"financial_indicator missing columns: {sorted(missing)}")

    prepared = financial_indicator.loc[:, ["ts_code", "ann_date", "end_date", *value_columns[factor_name]]].copy()
    prepared["ts_code"] = prepared["ts_code"].astype(str)
    prepared = prepared.loc[prepared["ts_code"].isin(codes)]
    prepared["ann_date"] = pd.to_datetime(prepared["ann_date"].astype("string"), format="%Y%m%d", errors="coerce")
    prepared["end_date"] = pd.to_datetime(prepared["end_date"].astype("string"), format="%Y%m%d", errors="coerce")
    for column in value_columns[factor_name]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    annual = prepared.loc[
        prepared["ann_date"].notna()
        & prepared["end_date"].notna()
        & (prepared["end_date"].dt.month == 12)
        & (prepared["end_date"].dt.day == 31)
    ].copy()
    for column in value_columns[factor_name]:
        annual = annual.loc[annual[column].notna()]
    annual = annual.sort_values(["ann_date", "end_date", "ts_code"])

    output = pd.DataFrame(index=rebalance_dates, columns=codes, dtype=float)
    for date in rebalance_dates:
        disclosed = annual.loc[annual["ann_date"] <= date]
        if disclosed.empty:
            continue
        latest_by_report = disclosed.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(
            ["ts_code", "end_date"],
            keep="last",
        )
        if factor_name == "grossprofit_margin_yoy":
            gpm = latest_by_report.pivot_table(
                index="end_date",
                columns="ts_code",
                values="grossprofit_margin",
                aggfunc="last",
            ).sort_index()
            values: dict[str, float] = {}
            for code, series in gpm.items():
                valid_series = series.dropna()
                if valid_series.empty:
                    continue
                current_end_date = valid_series.index.max()
                previous_end_date = current_end_date - pd.DateOffset(years=1)
                if previous_end_date not in valid_series.index:
                    continue
                values[code] = float(valid_series.loc[current_end_date] - valid_series.loc[previous_end_date])
            if values:
                output.loc[date, list(values.keys())] = list(values.values())
            continue

        latest = latest_by_report.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(
            "ts_code",
            keep="last",
        )
        value_column = value_columns[factor_name][0]
        output.loc[date, latest["ts_code"].values] = latest[value_column].astype(float).values
    return output


def _analyst_eps_revision_panel(
    analyst_report: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    codes: list[str],
    metric: str,
) -> pd.DataFrame:
    if metric not in {"count", "magnitude"}:
        raise ValueError("metric must be 'count' or 'magnitude'.")
    required = {"ts_code", "report_date", "quarter", "eps", "org_name", "author_name"}
    missing = required - set(analyst_report.columns)
    if missing:
        raise ValueError(f"analyst_report missing columns: {sorted(missing)}")

    prepared = analyst_report.loc[
        :,
        ["ts_code", "report_date", "quarter", "eps", "org_name", "author_name"],
    ].copy()
    prepared["ts_code"] = prepared["ts_code"].astype(str)
    prepared = prepared.loc[prepared["ts_code"].isin(codes)]
    prepared["report_date"] = pd.to_datetime(
        prepared["report_date"].astype("string"),
        format="%Y%m%d",
        errors="coerce",
    )
    prepared["quarter"] = prepared["quarter"].fillna("").astype(str)
    prepared["eps"] = pd.to_numeric(prepared["eps"], errors="coerce")
    prepared["org_name"] = prepared["org_name"].fillna("").astype(str)
    prepared["author_name"] = prepared["author_name"].fillna("").astype(str)
    prepared["analyst_id"] = prepared["org_name"] + "|" + prepared["author_name"]
    prepared = prepared.loc[
        prepared["report_date"].notna()
        & prepared["eps"].notna()
        & prepared["quarter"].str.endswith("Q4")
        & (prepared["analyst_id"] != "|")
    ].copy()
    if prepared.empty:
        return pd.DataFrame(0.0, index=rebalance_dates, columns=codes, dtype=float)

    prepared = prepared.sort_values(["ts_code", "quarter", "analyst_id", "report_date"])
    prepared = prepared.drop_duplicates(
        ["ts_code", "quarter", "analyst_id", "report_date"],
        keep="last",
    )
    group_cols = ["ts_code", "quarter", "analyst_id"]
    prepared["previous_eps"] = prepared.groupby(group_cols)["eps"].shift(1)
    prepared["eps_revision"] = prepared["eps"] - prepared["previous_eps"]
    upward = prepared.loc[prepared["eps_revision"] > 0, ["ts_code", "report_date", "eps_revision"]].copy()

    output = pd.DataFrame(0.0, index=rebalance_dates, columns=codes, dtype=float)
    for date in rebalance_dates:
        window_start = date - pd.DateOffset(months=3)
        scoped = upward.loc[(upward["report_date"] > window_start) & (upward["report_date"] <= date)]
        if scoped.empty:
            continue
        if metric == "count":
            values = scoped.groupby("ts_code")["eps_revision"].size().astype(float)
        else:
            values = scoped.groupby("ts_code")["eps_revision"].sum().astype(float)
        aligned = values.reindex(codes).dropna()
        if not aligned.empty:
            output.loc[date, aligned.index] = aligned.values
    return output


def _latest_annual_net_profit_yoy_panel(
    income_statement: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    codes: list[str],
) -> pd.DataFrame:
    required = {"ts_code", "ann_date", "end_date", "n_income_attr_p"}
    missing = required - set(income_statement.columns)
    if missing:
        raise ValueError(f"income_statement missing columns: {sorted(missing)}")

    prepared = income_statement.loc[:, ["ts_code", "ann_date", "end_date", "n_income_attr_p"]].copy()
    prepared["ts_code"] = prepared["ts_code"].astype(str)
    prepared = prepared.loc[prepared["ts_code"].isin(codes)]
    prepared["ann_date"] = pd.to_datetime(prepared["ann_date"].astype("string"), format="%Y%m%d", errors="coerce")
    prepared["end_date"] = pd.to_datetime(prepared["end_date"].astype("string"), format="%Y%m%d", errors="coerce")
    prepared["n_income_attr_p"] = pd.to_numeric(prepared["n_income_attr_p"], errors="coerce")
    annual = prepared.loc[
        prepared["ann_date"].notna()
        & prepared["end_date"].notna()
        & prepared["n_income_attr_p"].notna()
        & (prepared["end_date"].dt.month == 12)
        & (prepared["end_date"].dt.day == 31)
    ].copy()
    annual = annual.sort_values(["ts_code", "end_date", "ann_date"])
    annual = annual.drop_duplicates(["ts_code", "end_date", "ann_date"], keep="last")

    output = pd.DataFrame(index=rebalance_dates, columns=codes, dtype=float)
    for date in rebalance_dates:
        disclosed = annual.loc[annual["ann_date"] <= date]
        if disclosed.empty:
            continue
        latest_by_report = disclosed.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(
            ["ts_code", "end_date"],
            keep="last",
        )
        profit_panel = latest_by_report.pivot_table(
            index="end_date",
            columns="ts_code",
            values="n_income_attr_p",
            aggfunc="last",
        ).sort_index()
        values: dict[str, float] = {}
        for code, series in profit_panel.items():
            valid_series = series.dropna()
            if valid_series.empty:
                continue
            current_end_date = valid_series.index.max()
            previous_end_date = current_end_date - pd.DateOffset(years=1)
            if previous_end_date not in valid_series.index:
                continue
            previous_profit = valid_series.loc[previous_end_date]
            if previous_profit <= 0:
                continue
            current_profit = valid_series.loc[current_end_date]
            values[code] = float((current_profit - previous_profit) / abs(previous_profit))
        if values:
            output.loc[date, list(values.keys())] = list(values.values())
    return output


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


def _prepare_benchmark_close(benchmark_daily: pd.DataFrame, dates: pd.Index) -> pd.Series:
    required = {"trade_date", "close"}
    missing = required - set(benchmark_daily.columns)
    if missing:
        raise ValueError(f"benchmark_daily missing columns: {sorted(missing)}")
    prepared = benchmark_daily.loc[:, ["trade_date", "close"]].copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    close = prepared.dropna(subset=["trade_date"]).drop_duplicates("trade_date", keep="last").set_index("trade_date")["close"]
    close = close.sort_index().reindex(dates)
    return close.astype(float)


def _period_benchmark_return(
    benchmark_close: pd.Series,
    rebalance_date: pd.Timestamp,
    next_date: pd.Timestamp,
) -> float:
    start_close = benchmark_close.loc[rebalance_date]
    end_close = benchmark_close.loc[next_date]
    if pd.isna(start_close) or pd.isna(end_close):
        return np.nan
    return float(end_close / start_close - 1.0)


def _annual_returns(period_returns: pd.DataFrame) -> pd.DataFrame:
    return_columns = ["strategy_return", "benchmark_return"]
    if "benchmark_500_return" in period_returns:
        return_columns.append("benchmark_500_return")
    annual = period_returns.groupby(period_returns.index.year)[return_columns].apply(
        lambda frame: (1.0 + frame).prod() - 1.0
    )
    annual.index.name = "year"
    annual["excess_return"] = annual["strategy_return"] - annual["benchmark_return"]
    if "benchmark_500_return" in annual:
        annual["excess_500_return"] = annual["strategy_return"] - annual["benchmark_500_return"]
    return annual


def _portfolio_metrics(
    period_returns: pd.DataFrame,
    nav: pd.DataFrame,
) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    strategy_returns = period_returns["strategy_return"].dropna()
    benchmark_returns = period_returns["benchmark_return"].dropna()
    months = len(strategy_returns)
    strategy_annual_return = _annualized_return(strategy_returns, 12)
    benchmark_annual_return = _annualized_return(benchmark_returns, 12)
    strategy_std = strategy_returns.std(ddof=1)
    sharpe = (
        float(strategy_returns.mean() / strategy_std * np.sqrt(12))
        if months > 1 and pd.notna(strategy_std) and strategy_std != 0
        else np.nan
    )

    drawdown = (nav["strategy_nav"] / nav["strategy_nav"].cummax() - 1.0).dropna()
    if drawdown.empty:
        max_drawdown = np.nan
        max_drawdown_start = nav.index[0]
        max_drawdown_end = nav.index[0]
    else:
        max_drawdown = float(drawdown.min())
        max_drawdown_end = drawdown.idxmin()
        max_drawdown_start = nav.loc[:max_drawdown_end, "strategy_nav"].idxmax()

    strategy_total_return = float(nav["strategy_nav"].iloc[-1] - 1.0)
    benchmark_total_return = float(nav["benchmark_nav"].iloc[-1] - 1.0)
    calmar = (
        float(strategy_annual_return / abs(max_drawdown))
        if pd.notna(strategy_annual_return) and pd.notna(max_drawdown) and max_drawdown < 0
        else np.nan
    )
    values = {
        "strategy_total_return": strategy_total_return,
        "benchmark_total_return": benchmark_total_return,
        "total_excess_return": strategy_total_return - benchmark_total_return,
        "strategy_annual_return": strategy_annual_return,
        "benchmark_annual_return": benchmark_annual_return,
        "annual_excess_return": strategy_annual_return - benchmark_annual_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "calmar_ratio": calmar,
        "average_monthly_turnover": float(
            (period_returns["buy_turnover"] + period_returns["sell_turnover"]).mean()
        ),
        "average_monthly_cost": float(period_returns["total_cost"].mean()),
        "return_months": int(months),
    }
    if "benchmark_500_return" in period_returns and "benchmark_500_nav" in nav:
        benchmark_500_returns = period_returns["benchmark_500_return"].dropna()
        benchmark_500_annual_return = _annualized_return(benchmark_500_returns, 12)
        benchmark_500_total_return = float(nav["benchmark_500_nav"].iloc[-1] - 1.0)
        values.update(
            {
                "benchmark_500_total_return": benchmark_500_total_return,
                "total_excess_500_return": strategy_total_return - benchmark_500_total_return,
                "benchmark_500_annual_return": benchmark_500_annual_return,
                "annual_excess_500_return": strategy_annual_return - benchmark_500_annual_return,
            }
        )
    metrics = pd.Series(values, name="value")
    return metrics, max_drawdown_start, max_drawdown_end


def _annualized_return(returns: pd.Series, periods_per_year: int) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total_growth = float((1.0 + returns).prod())
    if total_growth <= 0:
        return np.nan
    return total_growth ** (periods_per_year / len(returns)) - 1.0
