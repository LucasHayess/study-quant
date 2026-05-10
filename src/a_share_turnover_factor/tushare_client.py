from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import pandas as pd


class TushareDataClient:
    """Thin cached wrapper around the Tushare Pro SDK."""

    def __init__(
        self,
        token: str,
        cache_dir: Path,
        pause_seconds: float = 0.35,
        retry_count: int = 3,
    ) -> None:
        if not token:
            raise ValueError("Tushare token is required. Pass --token or set TUSHARE_TOKEN.")

        import tushare as ts

        self.pro = ts.pro_api(token)
        self.cache_dir = cache_dir
        self.pause_seconds = pause_seconds
        self.retry_count = retry_count

    def stock_basic(self) -> pd.DataFrame:
        fields = ",".join(
            [
                "ts_code",
                "symbol",
                "name",
                "area",
                "industry",
                "market",
                "exchange",
                "list_status",
                "list_date",
                "delist_date",
            ]
        )
        frames = []
        for status in ("L", "D", "P"):
            frame = self._cached_call(
                "stock_basic",
                f"status_{status}",
                self.pro.stock_basic,
                exchange="",
                list_status=status,
                fields=fields,
            )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")

    def namechange(self) -> pd.DataFrame:
        fields = "ts_code,name,start_date,end_date,change_reason"
        try:
            return self._cached_call("namechange", "all", self.pro.namechange, fields=fields)
        except Exception as exc:
            print(f"[warn] namechange fetch failed, fallback to current names only: {exc}")
            return pd.DataFrame(columns=["ts_code", "name", "start_date", "end_date", "change_reason"])

    def trade_dates(self, start_date: str, end_date: str) -> list[str]:
        fields = "exchange,cal_date,is_open,pretrade_date"
        cal = self._cached_call(
            "trade_cal",
            f"{start_date}_{end_date}_open",
            self.pro.trade_cal,
            exchange="",
            start_date=start_date,
            end_date=end_date,
            is_open="1",
            fields=fields,
        )
        if cal.empty:
            return []
        return sorted(cal["cal_date"].astype(str).tolist())

    def daily(self, trade_date: str) -> pd.DataFrame:
        fields = "ts_code,trade_date,close"
        return self._cached_call(
            "daily",
            trade_date,
            self.pro.daily,
            trade_date=trade_date,
            fields=fields,
        )

    def daily_basic(self, trade_date: str) -> pd.DataFrame:
        fields = "ts_code,trade_date,turnover_rate"
        return self._cached_call(
            "daily_basic",
            trade_date,
            self.pro.daily_basic,
            trade_date=trade_date,
            fields=fields,
        )

    def adj_factor(self, trade_date: str) -> pd.DataFrame:
        fields = "ts_code,trade_date,adj_factor"
        return self._cached_call(
            "adj_factor",
            trade_date,
            self.pro.adj_factor,
            trade_date=trade_date,
            fields=fields,
        )

    def _cached_call(
        self,
        api_name: str,
        cache_key: str,
        caller: Callable[..., pd.DataFrame],
        **kwargs: object,
    ) -> pd.DataFrame:
        api_dir = self.cache_dir / api_name
        api_dir.mkdir(parents=True, exist_ok=True)
        cache_file = api_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            return pd.read_pickle(cache_file)

        last_error: Exception | None = None
        for attempt in range(1, self.retry_count + 1):
            try:
                frame = caller(**kwargs)
                if frame is None:
                    frame = pd.DataFrame()
                frame.to_pickle(cache_file)
                time.sleep(self.pause_seconds)
                return frame
            except Exception as exc:
                last_error = exc
                sleep_for = self.pause_seconds * attempt * 3
                print(f"[warn] {api_name} {cache_key} failed on attempt {attempt}: {exc}")
                time.sleep(sleep_for)

        raise RuntimeError(f"Tushare call failed: {api_name} {cache_key}") from last_error


def fetch_daily_panels(client: TushareDataClient, trade_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily_frames = []
    basic_frames = []
    adj_frames = []
    total = len(trade_dates)

    for idx, trade_date in enumerate(trade_dates, start=1):
        print(f"[fetch] {trade_date} ({idx}/{total})")
        daily_frames.append(client.daily(trade_date))
        basic_frames.append(client.daily_basic(trade_date))
        adj_frames.append(client.adj_factor(trade_date))

    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    daily_basic = pd.concat(basic_frames, ignore_index=True) if basic_frames else pd.DataFrame()
    adj_factor = pd.concat(adj_frames, ignore_index=True) if adj_frames else pd.DataFrame()
    return daily, daily_basic, adj_factor
