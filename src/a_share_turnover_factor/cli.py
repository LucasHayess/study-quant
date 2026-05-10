from __future__ import annotations

import argparse
import os
from pathlib import Path

from .backtest import run_turnover_factor_backtest
from .config import BacktestConfig
from .plotting import save_plots
from .tushare_client import TushareDataClient, fetch_daily_panels


def load_local_env(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    load_local_env()
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
    parser.add_argument("--rolling-window", type=int, default=BacktestConfig.rolling_window, help="Rolling turnover window.")
    parser.add_argument(
        "--rebalance-frequency",
        choices=["monthly", "weekly"],
        default="monthly",
        help="Rebalance on month-end or week-end trading dates.",
    )
    parser.add_argument("--run-experiments", action="store_true", help="Run the three comparison experiments requested.")
    parser.add_argument("--min-listed-days", type=int, default=BacktestConfig.min_listed_days, help="Minimum listed calendar days.")
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


def main() -> None:
    args = parse_args()
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

    if args.run_experiments:
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

    print("[step] Fetch daily turnover and rebalance-date prices")
    daily, daily_basic, adj_factor = fetch_daily_panels(client, trade_dates, price_dates)

    if args.run_experiments:
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
        )

        result.group_returns.to_csv(output_dir / "group_returns.csv", float_format="%.8f")
        result.average_group_returns.to_csv(output_dir / "group_average_returns.csv", float_format="%.8f")
        result.ic_series.to_csv(output_dir / "ic_series.csv", float_format="%.8f")
        result.sample_counts.to_csv(output_dir / "sample_counts.csv")

        print("[step] Save charts")
        period_label = "月" if args.rebalance_frequency == "monthly" else "周"
        save_plots(
            result.average_group_returns,
            result.ic_series,
            output_dir,
            group_title=f"{args.rolling_window}日平均换手率因子五分组平均{period_label}收益",
            ic_title=f"{args.rolling_window}日平均换手率因子 IC 时序（{period_label}末换仓）",
            period_label=period_label,
        )

    print(f"[done] Outputs written to {output_dir.resolve()}")
