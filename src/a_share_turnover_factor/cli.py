from __future__ import annotations

import argparse
import os
from pathlib import Path

from .backtest import run_top_quantile_portfolio_backtest, run_turnover_factor_backtest
from .config import BacktestConfig
from .plotting import save_monthly_return_heatmap, save_nav_plot, save_plots, save_risk_control_nav_plot
from .tushare_client import (
    TushareDataClient,
    fetch_analyst_report_rc,
    fetch_annual_fina_indicator,
    fetch_daily_panels,
    fetch_income_statements,
)


def load_local_env(env_path: Path = Path(".env"), override: bool = False) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    load_local_env()
    load_local_env(Path(".env.local"), override=True)
    parser = argparse.ArgumentParser(
        description="Fetch A-share daily data from Tushare and backtest a 20-day turnover factor."
    )
    parser.add_argument("--token", default=os.getenv("TUSHARE_TOKEN"), help="Tushare Pro token. Defaults to TUSHARE_TOKEN.")
    parser.add_argument("--start-date", default=BacktestConfig.start_date, help="Factor test start date, YYYYMMDD.")
    parser.add_argument("--end-date", default=BacktestConfig.end_date, help="Factor test end date, YYYYMMDD.")
    parser.add_argument("--warmup-start-date", default=BacktestConfig.warmup_start_date, help="Fetch start date for rolling warmup.")
    parser.add_argument(
        "--fetch-end-date",
        default=None,
        help="Last date to fetch. Defaults to --end-date. Use 20240131 if you want a Dec-2023 forward return.",
    )
    parser.add_argument("--cache-dir", default=str(BacktestConfig.cache_dir), help="Local API cache directory.")
    parser.add_argument("--output-dir", default=str(BacktestConfig.output_dir), help="Output directory.")
    parser.add_argument("--groups", type=int, default=BacktestConfig.group_count, help="Number of equal quantile groups.")
    parser.add_argument(
        "--factor-type",
        choices=[
            "turnover",
            "turnover_inverse",
            "momentum",
            "momentum_inverse",
            "composite",
            "quality_roe",
            "quality_grossprofit_margin_yoy",
            "quality_ocf_to_np",
            "quality_net_profit_yoy",
            "analyst_eps_revision_count",
            "analyst_eps_revision_magnitude",
        ],
        default="turnover",
        help="Factor to test: turnover, momentum, composite score, or quality factors.",
    )
    parser.add_argument("--rolling-window", type=int, default=BacktestConfig.rolling_window, help="Rolling turnover window.")
    parser.add_argument("--momentum-window", type=int, default=20, help="Momentum cumulative return window.")
    parser.add_argument("--momentum-skip-days", type=int, default=1, help="Most recent trading days to skip in momentum factor.")
    parser.add_argument(
        "--rebalance-frequency",
        choices=["monthly", "weekly"],
        default="monthly",
        help="Rebalance on month-end or week-end trading dates.",
    )
    parser.add_argument("--run-experiments", action="store_true", help="Run the three comparison experiments requested.")
    parser.add_argument(
        "--run-top-portfolio",
        action="store_true",
        help="Run a top-quantile equal-weight portfolio backtest instead of only factor grouping/IC.",
    )
    parser.add_argument(
        "--run-risk-controls",
        action="store_true",
        help="Run baseline plus three independent risk-control portfolio backtests.",
    )
    parser.add_argument("--top-quantile", type=float, default=0.2, help="Portfolio keeps the highest-scoring quantile.")
    parser.add_argument("--benchmark-index", default="000300.SH", help="Tushare index code for benchmark.")
    parser.add_argument("--benchmark-500-index", default="000905.SH", help="Tushare index code for CSI 500 benchmark.")
    parser.add_argument("--buy-commission-rate", type=float, default=0.0003, help="Buy-side commission rate.")
    parser.add_argument("--sell-commission-rate", type=float, default=0.0003, help="Sell-side commission rate.")
    parser.add_argument("--stamp-tax-rate", type=float, default=0.001, help="Sell-side stamp tax rate.")
    parser.add_argument("--min-listed-days", type=int, default=BacktestConfig.min_listed_days, help="Minimum listed calendar days.")
    parser.add_argument(
        "--liquidity-min-avg-amount-yuan",
        type=float,
        default=None,
        help="Exclude stocks whose rolling average daily amount is below this RMB threshold. Example: 20000000.",
    )
    parser.add_argument("--liquidity-window", type=int, default=20, help="Rolling window for average daily amount filter.")
    parser.add_argument("--pause", type=float, default=BacktestConfig.pause_seconds, help="Pause seconds between uncached API calls.")
    parser.add_argument("--retry-count", type=int, default=BacktestConfig.retry_count, help="Retry count for Tushare calls.")
    return parser.parse_args()


def period_end_trade_dates(trade_dates: list[str], start_date: str, end_date: str, frequency: str) -> list[str]:
    import pandas as pd

    dates = pd.Series(pd.to_datetime(trade_dates, format="%Y%m%d"))
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    scoped = dates[(dates >= start) & (dates <= end)]
    period_freq = "M" if frequency == "monthly" else "W-FRI"
    period_ends = scoped.groupby(scoped.dt.to_period(period_freq)).max().sort_values()
    return [date.strftime("%Y%m%d") for date in period_ends]


def experiment_specs(default_window: int) -> list[dict[str, str | int]]:
    return [
        {
            "name": "experiment1_rolling5_monthly",
            "label": "实验1：rolling(5)，月末换仓",
            "rolling_window": 5,
            "rebalance_frequency": "monthly",
            "period_label": "月",
        },
        {
            "name": "experiment2_rolling60_monthly",
            "label": "实验2：rolling(60)，月末换仓",
            "rolling_window": 60,
            "rebalance_frequency": "monthly",
            "period_label": "月",
        },
        {
            "name": "experiment3_rolling20_weekly",
            "label": f"实验3：rolling({default_window})，周末换仓",
            "rolling_window": default_window,
            "rebalance_frequency": "weekly",
            "period_label": "周",
        },
    ]


def write_portfolio_outputs(output_dir: Path, result) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.period_returns.to_csv(output_dir / "monthly_returns.csv", float_format="%.8f")
    result.nav.to_csv(output_dir / "nav.csv", float_format="%.8f")
    result.annual_returns.to_csv(output_dir / "annual_returns.csv", float_format="%.8f")
    result.metrics.to_csv(output_dir / "metrics.csv", float_format="%.8f")
    result.holdings.to_csv(output_dir / "holdings.csv")
    (output_dir / "max_drawdown_period.txt").write_text(
        (
            f"start={result.max_drawdown_start.strftime('%Y-%m-%d')}\n"
            f"end={result.max_drawdown_end.strftime('%Y-%m-%d')}\n"
            f"max_drawdown={result.metrics['max_drawdown']:.8f}\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    quality_factor_types = {
        "quality_roe",
        "quality_grossprofit_margin_yoy",
        "quality_ocf_to_np",
        "quality_net_profit_yoy",
    }
    analyst_factor_types = {"analyst_eps_revision_count", "analyst_eps_revision_magnitude"}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fetch_end_date = args.fetch_end_date or args.end_date
    client = TushareDataClient(
        token=args.token,
        cache_dir=Path(args.cache_dir),
        pause_seconds=args.pause,
        retry_count=args.retry_count,
    )

    print("[step] Fetch stock basics and ST name-change history")
    stock_basic = client.stock_basic()
    namechange = client.namechange()

    print(f"[step] Fetch trading calendar {args.warmup_start_date} -> {fetch_end_date}")
    trade_dates = client.trade_dates(args.warmup_start_date, fetch_end_date)

    if args.run_top_portfolio or args.run_risk_controls:
        price_dates = trade_dates
    elif args.factor_type in {"momentum", "momentum_inverse", "composite", *quality_factor_types}:
        price_dates = trade_dates
    elif args.run_experiments:
        specs = experiment_specs(args.rolling_window)
        price_date_set = set()
        for spec in specs:
            price_date_set.update(
                period_end_trade_dates(
                    trade_dates,
                    args.start_date,
                    fetch_end_date,
                    str(spec["rebalance_frequency"]),
                )
            )
        price_dates = sorted(price_date_set)
    else:
        price_dates = period_end_trade_dates(trade_dates, args.start_date, fetch_end_date, args.rebalance_frequency)

    amount_dates = (
        trade_dates
        if args.run_top_portfolio or args.run_risk_controls or args.liquidity_min_avg_amount_yuan is not None
        else None
    )

    print("[step] Fetch daily turnover, optional liquidity data, and rebalance-date prices")
    daily, daily_basic, adj_factor, daily_amount = fetch_daily_panels(client, trade_dates, price_dates, amount_dates)
    financial_indicator = None
    income_statement = None
    analyst_report = None
    if args.factor_type in quality_factor_types - {"quality_net_profit_yoy"}:
        financial_start_year = int(args.start_date[:4]) - 3
        financial_end_year = int(args.end_date[:4])
        financial_codes = sorted(
            stock_basic.loc[
                stock_basic["ts_code"].astype(str).str.endswith((".SH", ".SZ", ".BJ")),
                "ts_code",
            ]
            .astype(str)
            .unique()
            .tolist()
        )
        print(f"[step] Fetch annual quality disclosures {financial_start_year} -> {financial_end_year}")
        financial_indicator = fetch_annual_fina_indicator(
            client,
            financial_start_year,
            financial_end_year,
            financial_codes,
        )
    if args.factor_type == "quality_net_profit_yoy":
        import pandas as pd

        income_start_date = (
            pd.to_datetime(args.start_date, format="%Y%m%d") - pd.DateOffset(years=3)
        ).strftime("%Y%m%d")
        stock_for_income = stock_basic.copy()
        stock_for_income["ts_code"] = stock_for_income["ts_code"].astype(str)
        stock_for_income["list_date"] = pd.to_datetime(
            stock_for_income["list_date"].astype("string"),
            format="%Y%m%d",
            errors="coerce",
        )
        stock_for_income["delist_date"] = pd.to_datetime(
            stock_for_income.get("delist_date", pd.Series(index=stock_for_income.index, dtype=object)).astype("string"),
            format="%Y%m%d",
            errors="coerce",
        )
        start_ts = pd.to_datetime(args.start_date, format="%Y%m%d")
        end_ts = pd.to_datetime(args.end_date, format="%Y%m%d")
        code_mask = (
            stock_for_income["ts_code"].str.endswith((".SH", ".SZ", ".BJ"))
            & stock_for_income["list_date"].notna()
            & (stock_for_income["list_date"] <= end_ts)
            & (stock_for_income["delist_date"].isna() | (stock_for_income["delist_date"] >= start_ts))
        )
        if "exchange" in stock_for_income.columns:
            code_mask = code_mask & stock_for_income["exchange"].isin(["SSE", "SZSE", "BSE"])
        income_codes = set(stock_for_income.loc[code_mask, "ts_code"].astype(str).unique().tolist())
        if args.liquidity_min_avg_amount_yuan is not None and not daily_amount.empty:
            amount_frame = daily_amount.loc[:, ["ts_code", "trade_date", "amount"]].copy()
            amount_frame["trade_date"] = pd.to_datetime(
                amount_frame["trade_date"].astype(str),
                format="%Y%m%d",
                errors="coerce",
            )
            amount_frame["amount"] = pd.to_numeric(amount_frame["amount"], errors="coerce")
            amount_panel = amount_frame.pivot_table(
                index="trade_date",
                columns="ts_code",
                values="amount",
                aggfunc="last",
            )
            threshold = args.liquidity_min_avg_amount_yuan / 1000.0
            liquid_codes = set(
                amount_panel.rolling(args.liquidity_window, min_periods=args.liquidity_window)
                .mean()
                .columns[
                    (
                        amount_panel.rolling(args.liquidity_window, min_periods=args.liquidity_window)
                        .mean()
                        >= threshold
                    ).any(axis=0)
                ]
                .astype(str)
                .tolist()
            )
            income_codes = income_codes & liquid_codes
        income_codes = sorted(income_codes)
        print(f"[step] Fetch income statements {income_start_date} -> {args.end_date}")
        income_statement = fetch_income_statements(client, income_start_date, args.end_date, income_codes)
    if args.factor_type in analyst_factor_types:
        import pandas as pd

        analyst_start_date = (
            pd.to_datetime(args.start_date, format="%Y%m%d") - pd.DateOffset(months=4)
        ).strftime("%Y%m%d")
        print(f"[step] Fetch analyst EPS forecast reports {analyst_start_date} -> {args.end_date}")
        analyst_report = fetch_analyst_report_rc(client, analyst_start_date, args.end_date)

    if args.run_top_portfolio or args.run_risk_controls:
        print(f"[step] Fetch benchmark index {args.benchmark_index}")
        benchmark_daily = client.index_daily(args.benchmark_index, args.warmup_start_date, fetch_end_date)
        print(f"[step] Fetch CSI 500 benchmark index {args.benchmark_500_index}")
        benchmark_500_daily = client.index_daily(args.benchmark_500_index, args.warmup_start_date, fetch_end_date)

        def run_portfolio(**risk_kwargs):
            return run_top_quantile_portfolio_backtest(
                stock_basic=stock_basic,
                namechange=namechange,
                daily=daily,
                daily_basic=daily_basic,
                adj_factor=adj_factor,
                daily_amount=daily_amount,
                benchmark_daily=benchmark_daily,
                benchmark_500_daily=benchmark_500_daily,
                trade_dates=trade_dates,
                start_date=args.start_date,
                end_date=args.end_date,
                top_quantile=args.top_quantile,
                rolling_window=args.rolling_window,
                min_listed_days=args.min_listed_days,
                rebalance_frequency=args.rebalance_frequency,
                liquidity_window=args.liquidity_window,
                liquidity_min_avg_amount_yuan=(
                    args.liquidity_min_avg_amount_yuan
                    if args.liquidity_min_avg_amount_yuan is not None
                    else 20_000_000
                ),
                momentum_window=args.momentum_window,
                momentum_skip_days=args.momentum_skip_days,
                buy_commission_rate=args.buy_commission_rate,
                sell_commission_rate=args.sell_commission_rate,
                stamp_tax_rate=args.stamp_tax_rate,
                **risk_kwargs,
            )

        if args.run_risk_controls:
            import pandas as pd

            print("[step] Run baseline portfolio backtest")
            baseline = run_portfolio()
            baseline_dir = output_dir / "baseline"
            write_portfolio_outputs(baseline_dir, baseline)

            schemes = [
                (
                    "scheme1_market_exposure",
                    "方案1：沪深300 20日跌幅<-5%降至50%仓位",
                    {"market_risk_control": True},
                ),
                (
                    "scheme2_stock_stop_loss",
                    "方案2：单只股票亏损<-10%下月剔除",
                    {"stock_stop_loss": True},
                ),
                (
                    "scheme3_combined",
                    "方案3：仓位控制+个股止损",
                    {"market_risk_control": True, "stock_stop_loss": True},
                ),
            ]
            summary_rows = []
            metric_keys = ["strategy_annual_return", "max_drawdown", "sharpe_ratio", "calmar_ratio"]
            baseline_metrics = baseline.metrics.reindex(metric_keys)
            summary_rows.append(
                {
                    "scheme": "baseline",
                    "label": "原始策略",
                    **baseline_metrics.to_dict(),
                    **{f"delta_{key}": 0.0 for key in metric_keys},
                }
            )
            for dirname, label, kwargs in schemes:
                print(f"[step] Run {label}")
                result = run_portfolio(**kwargs)
                scheme_dir = output_dir / dirname
                write_portfolio_outputs(scheme_dir, result)
                save_risk_control_nav_plot(
                    baseline.nav,
                    result.nav,
                    scheme_dir,
                    title=f"{label}：累计净值 vs 原始策略 vs 沪深300",
                    filename="risk_control_nav_compare.png",
                    risk_label=label.split("：", 1)[0],
                )
                metrics = result.metrics.reindex(metric_keys)
                row = {"scheme": dirname, "label": label, **metrics.to_dict()}
                row.update({f"delta_{key}": metrics[key] - baseline_metrics[key] for key in metric_keys})
                summary_rows.append(row)
            pd.DataFrame(summary_rows).set_index("scheme").to_csv(
                output_dir / "risk_control_summary.csv",
                float_format="%.8f",
            )
        else:
            print("[step] Run top 20% equal-weight portfolio backtest")
            result = run_portfolio()
            write_portfolio_outputs(output_dir, result)

            print("[step] Save portfolio NAV chart")
            save_nav_plot(
                result.nav,
                output_dir,
                title=(
                    "综合评分Top20%等权组合累计净值 vs 沪深300 vs 中证500"
                    f"（{args.start_date[:4]}-{args.end_date[:4]}，月末换仓，含交易成本）"
                ),
            )
            print("[step] Save monthly return heatmap")
            save_monthly_return_heatmap(
                result.period_returns["strategy_return"],
                output_dir,
                title=(
                    "综合评分Top20%等权组合月度收益热力图"
                    f"（{args.start_date[:4]}-{args.end_date[:4]}，含交易成本）"
                ),
            )
    elif args.run_experiments:
        print("[step] Run comparison experiments")
        for spec in specs:
            result = run_turnover_factor_backtest(
                stock_basic=stock_basic,
                namechange=namechange,
                daily=daily,
                daily_basic=daily_basic,
                adj_factor=adj_factor,
                trade_dates=trade_dates,
                start_date=args.start_date,
                end_date=args.end_date,
                group_count=args.groups,
                rolling_window=int(spec["rolling_window"]),
                min_listed_days=args.min_listed_days,
                rebalance_frequency=str(spec["rebalance_frequency"]),
                daily_amount=daily_amount,
                liquidity_window=args.liquidity_window,
                liquidity_min_avg_amount_yuan=args.liquidity_min_avg_amount_yuan,
                factor_type="turnover",
            )
            prefix = str(spec["name"])
            result.group_returns.to_csv(output_dir / f"{prefix}_group_returns.csv", float_format="%.8f")
            result.average_group_returns.to_csv(output_dir / f"{prefix}_average_group_returns.csv", float_format="%.8f")
            result.ic_series.to_csv(output_dir / f"{prefix}_ic_series.csv", float_format="%.8f")
            result.sample_counts.to_csv(output_dir / f"{prefix}_sample_counts.csv")
            save_plots(
                result.average_group_returns,
                result.ic_series,
                output_dir,
                group_title=f"{spec['label']}：五分组平均{spec['period_label']}收益",
                ic_title=f"{spec['label']}：IC 时序",
                period_label=str(spec["period_label"]),
                group_xlabel="分组（G1低换手率，G5高换手率）",
                group_filename=f"{prefix}_group_return.png",
                ic_filename=f"{prefix}_ic_series.png",
            )
    else:
        print("[step] Run factor backtest")
        result = run_turnover_factor_backtest(
            stock_basic=stock_basic,
            namechange=namechange,
            daily=daily,
            daily_basic=daily_basic,
            adj_factor=adj_factor,
            trade_dates=trade_dates,
            start_date=args.start_date,
            end_date=args.end_date,
            group_count=args.groups,
            rolling_window=args.rolling_window,
            min_listed_days=args.min_listed_days,
            rebalance_frequency=args.rebalance_frequency,
            daily_amount=daily_amount,
            liquidity_window=args.liquidity_window,
            liquidity_min_avg_amount_yuan=args.liquidity_min_avg_amount_yuan,
            factor_type=args.factor_type,
            momentum_window=args.momentum_window,
            momentum_skip_days=args.momentum_skip_days,
            financial_indicator=financial_indicator,
            income_statement=income_statement,
            analyst_report=analyst_report,
        )

        result.group_returns.to_csv(output_dir / "group_returns.csv", float_format="%.8f")
        result.average_group_returns.to_csv(output_dir / "group_average_returns.csv", float_format="%.8f")
        result.ic_series.to_csv(output_dir / "ic_series.csv", float_format="%.8f")
        result.sample_counts.to_csv(output_dir / "sample_counts.csv")
        ic_mean = result.ic_series.mean()
        ic_std = result.ic_series.std(ddof=1)
        ic_summary = {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": ic_mean / ic_std if ic_std and ic_std == ic_std else float("nan"),
            "n": int(result.ic_series.notna().sum()),
        }
        import pandas as pd

        pd.Series(ic_summary, name="value").to_csv(output_dir / "ic_summary.csv", float_format="%.8f")
        if args.factor_type in quality_factor_types or args.factor_type in analyst_factor_types:
            print("[step] Compare factor IC with turnover IC")
            turnover_result = run_turnover_factor_backtest(
                stock_basic=stock_basic,
                namechange=namechange,
                daily=daily,
                daily_basic=daily_basic,
                adj_factor=adj_factor,
                trade_dates=trade_dates,
                start_date=args.start_date,
                end_date=args.end_date,
                group_count=args.groups,
                rolling_window=args.rolling_window,
                min_listed_days=args.min_listed_days,
                rebalance_frequency=args.rebalance_frequency,
                daily_amount=daily_amount,
                liquidity_window=args.liquidity_window,
                liquidity_min_avg_amount_yuan=args.liquidity_min_avg_amount_yuan,
                factor_type="turnover_inverse",
                momentum_window=args.momentum_window,
                momentum_skip_days=args.momentum_skip_days,
            )
            ic_compare = pd.DataFrame(
                {
                    f"{args.factor_type}_ic": result.ic_series,
                    f"turnover_inverse_rolling{args.rolling_window}_ic": turnover_result.ic_series,
                }
            )
            ic_compare.to_csv(output_dir / "ic_series_comparison.csv", float_format="%.8f")
            correlations = pd.Series(
                {
                    f"corr_{args.factor_type}_vs_turnover_inverse_rolling{args.rolling_window}": ic_compare[
                        f"{args.factor_type}_ic"
                    ].corr(ic_compare[f"turnover_inverse_rolling{args.rolling_window}_ic"]),
                    "n": int(ic_compare.dropna().shape[0]),
                },
                name="value",
            )
            correlations.to_csv(output_dir / "ic_correlation.csv", float_format="%.8f")

        print("[step] Save charts")
        period_label = "月" if args.rebalance_frequency == "monthly" else "周"
        liquidity_note = ""
        if args.liquidity_min_avg_amount_yuan is not None:
            amount_wan = args.liquidity_min_avg_amount_yuan / 10000
            liquidity_note = f"，{args.liquidity_window}日均成交额>={amount_wan:.0f}万"
        if args.factor_type == "momentum":
            factor_label = f"过去{args.momentum_window}日累计收益率（跳过最近{args.momentum_skip_days}日）"
            group_xlabel = "分组（G1低动量，G5高动量）"
        elif args.factor_type == "momentum_inverse":
            factor_label = f"过去{args.momentum_window}日累计收益率取反（跳过最近{args.momentum_skip_days}日）"
            group_xlabel = "分组（G1高动量，G5低动量）"
        elif args.factor_type == "turnover_inverse":
            factor_label = f"{args.rolling_window}日平均换手率取反因子"
            group_xlabel = "分组（G1高换手率，G5低换手率）"
        elif args.factor_type == "composite":
            factor_label = (
                f"综合评分：0.5*{args.rolling_window}日平均换手率取反排名"
                f"+0.5*过去{args.momentum_window}日收益率取反排名"
            )
            group_xlabel = "分组（G1低综合评分，G5高综合评分）"
        elif args.factor_type == "quality_roe":
            factor_label = "质量因子：最近一期已披露年报ROE"
            group_xlabel = "分组（G1低ROE，G5高ROE）"
        elif args.factor_type == "quality_grossprofit_margin_yoy":
            factor_label = "质量因子：最近一期已披露年报毛利率同比变化"
            group_xlabel = "分组（G1低毛利率同比变化，G5高毛利率同比变化）"
        elif args.factor_type == "quality_ocf_to_np":
            factor_label = "质量因子：最近一期已披露年报经营现金流/净利润"
            group_xlabel = "分组（G1低经营现金流/净利润，G5高经营现金流/净利润）"
        elif args.factor_type == "quality_net_profit_yoy":
            factor_label = "质量因子：最近一期已披露年报净利润同比增速"
            group_xlabel = "分组（G1低净利润增速，G5高净利润增速）"
        elif args.factor_type == "analyst_eps_revision_count":
            factor_label = "分析师因子：过去3个月EPS预测上调次数"
            group_xlabel = "分组（G1低上调次数，G5高上调次数）"
        elif args.factor_type == "analyst_eps_revision_magnitude":
            factor_label = "分析师因子：过去3个月EPS预测上调幅度"
            group_xlabel = "分组（G1低上调幅度，G5高上调幅度）"
        else:
            factor_label = f"{args.rolling_window}日平均换手率因子"
            group_xlabel = "分组（G1低换手率，G5高换手率）"
        save_plots(
            result.average_group_returns,
            result.ic_series,
            output_dir,
            group_title=f"{factor_label}五分组平均{period_label}收益{liquidity_note}",
            ic_title=f"{factor_label} IC 时序（{period_label}末换仓{liquidity_note}）",
            period_label=period_label,
            group_xlabel=group_xlabel,
        )

    print(f"[done] Outputs written to {output_dir.resolve()}")
