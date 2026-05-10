from __future__ import annotations

import argparse
import os
from pathlib import Path

from .backtest import run_turnover_factor_backtest
from .config import BacktestConfig
from .plotting import save_plots
from .tushare_client import TushareDataClient, fetch_daily_panels


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--min-listed-days", type=int, default=BacktestConfig.min_listed_days, help="Minimum listed calendar days.")
    parser.add_argument("--pause", type=float, default=BacktestConfig.pause_seconds, help="Pause seconds between uncached API calls.")
    parser.add_argument("--retry-count", type=int, default=BacktestConfig.retry_count, help="Retry count for Tushare calls.")
    return parser.parse_args()


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

    print("[step] Fetch daily close, adjustment factor, and daily turnover")
    daily, daily_basic, adj_factor = fetch_daily_panels(client, trade_dates)

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
    )

    result.group_returns.to_csv(output_dir / "group_monthly_returns.csv", float_format="%.8f")
    result.average_group_returns.to_csv(output_dir / "group_average_monthly_returns.csv", float_format="%.8f")
    result.ic_series.to_csv(output_dir / "ic_series.csv", float_format="%.8f")
    result.monthly_sample_counts.to_csv(output_dir / "monthly_sample_counts.csv")

    print("[step] Save charts")
    save_plots(result.average_group_returns, result.ic_series, output_dir)

    print(f"[done] Outputs written to {output_dir.resolve()}")
