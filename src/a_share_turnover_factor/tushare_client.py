from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from hashlib import md5
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
        self._request_lock = Lock()
        self._last_request_at = 0.0

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

    def daily_amount(self, trade_date: str) -> pd.DataFrame:
        fields = "ts_code,trade_date,amount"
        return self._cached_call(
            "daily_amount",
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

    def index_daily(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        fields = "ts_code,trade_date,close"
        cache_key = f"{ts_code}_{start_date}_{end_date}".replace(".", "_")
        return self._cached_call(
            "index_daily",
            cache_key,
            self.pro.index_daily,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def annual_fina_indicator(self, period: str) -> pd.DataFrame:
        fields = "ts_code,ann_date,end_date,roe,grossprofit_margin,ocf_to_profit"
        return self._cached_call(
            "fina_indicator",
            f"period_{period}",
            self.pro.fina_indicator,
            period=period,
            fields=fields,
        )

    def fina_indicator_by_ts_code(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        fields = "ts_code,ann_date,end_date,roe,grossprofit_margin,ocf_to_profit"
        cache_key = f"{ts_code}_{start_date}_{end_date}".replace(".", "_")
        return self._cached_call(
            "fina_indicator_by_code",
            cache_key,
            self.pro.fina_indicator,
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def fina_indicator_by_ts_codes(self, ts_codes: list[str], period: str) -> pd.DataFrame:
        fields = "ts_code,ann_date,end_date,roe,grossprofit_margin,ocf_to_profit"
        joined_codes = ",".join(ts_codes)
        digest = md5(joined_codes.encode("utf-8")).hexdigest()[:12]
        cache_key = f"batch_{digest}_{period}"
        return self._cached_call(
            "fina_indicator_quality_v2_by_code_period_batch",
            cache_key,
            self.pro.fina_indicator,
            ts_code=joined_codes,
            period=period,
            fields=fields,
        )

    def analyst_report_rc(self, start_date: str, end_date: str, limit: int = 5000, offset: int = 0) -> pd.DataFrame:
        fields = "ts_code,report_date,quarter,eps,org_name,author_name"
        cache_key = f"{start_date}_{end_date}_{limit}_{offset}"
        return self._cached_call(
            "report_rc",
            cache_key,
            self.pro.report_rc,
            throttle_seconds=max(self.pause_seconds, 31.0),
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
            fields=fields,
        )

    def income_by_ts_code(self, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        fields = "ts_code,ann_date,end_date,n_income_attr_p"
        cache_key = f"{ts_code}_{start_date}_{end_date}".replace(".", "_")
        return self._cached_call(
            "income_by_code",
            cache_key,
            self.pro.income,
            throttle_seconds=max(self.pause_seconds, 0.32),
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
        )

    def _cached_call(
        self,
        api_name: str,
        cache_key: str,
        caller: Callable[..., pd.DataFrame],
        throttle_seconds: float | None = None,
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
                if throttle_seconds is not None:
                    with self._request_lock:
                        wait_for = throttle_seconds - (time.monotonic() - self._last_request_at)
                        if wait_for > 0:
                            time.sleep(wait_for)
                        self._last_request_at = time.monotonic()
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


def fetch_daily_panels(
    client: TushareDataClient,
    turnover_dates: list[str],
    price_dates: list[str] | None = None,
    amount_dates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if price_dates is None:
        price_dates = turnover_dates

    daily_frames = []
    basic_frames = []
    adj_frames = []
    amount_frames = []
    turnover_total = len(turnover_dates)
    price_total = len(price_dates)
    amount_total = len(amount_dates) if amount_dates is not None else 0

    for idx, trade_date in enumerate(turnover_dates, start=1):
        print(f"[fetch] turnover {trade_date} ({idx}/{turnover_total})")
        basic_frames.append(client.daily_basic(trade_date))

    if amount_dates is not None:
        for idx, trade_date in enumerate(amount_dates, start=1):
            print(f"[fetch] amount {trade_date} ({idx}/{amount_total})")
            amount_frames.append(client.daily_amount(trade_date))

    for idx, trade_date in enumerate(price_dates, start=1):
        print(f"[fetch] price {trade_date} ({idx}/{price_total})")
        daily_frames.append(client.daily(trade_date))
        adj_frames.append(client.adj_factor(trade_date))

    daily = pd.concat(daily_frames, ignore_index=True) if daily_frames else pd.DataFrame()
    daily_basic = pd.concat(basic_frames, ignore_index=True) if basic_frames else pd.DataFrame()
    adj_factor = pd.concat(adj_frames, ignore_index=True) if adj_frames else pd.DataFrame()
    daily_amount = pd.concat(amount_frames, ignore_index=True) if amount_frames else pd.DataFrame()
    return daily, daily_basic, adj_factor, daily_amount


def fetch_annual_fina_indicator(
    client: TushareDataClient,
    start_year: int,
    end_year: int,
    ts_codes: list[str],
) -> pd.DataFrame:
    frames = []
    periods = [f"{year}1231" for year in range(start_year, end_year + 1)]
    chunk_size = 200
    chunks = [ts_codes[idx : idx + chunk_size] for idx in range(0, len(ts_codes), chunk_size)]
    total = len(periods) * len(chunks)
    current = 0
    for period in periods:
        for chunk in chunks:
            current += 1
            print(f"[fetch] fina_indicator {period} batch {current}/{total} ({chunk[0]}..{chunk[-1]})")
            frame = client.fina_indicator_by_ts_codes(chunk, period)
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_analyst_report_rc(
    client: TushareDataClient,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frames = []
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    chunks = []
    chunk_start = start
    while chunk_start <= end:
        month_end = chunk_start + pd.offsets.MonthEnd(0)
        chunk_end = min(month_end, end)
        chunks.append((chunk_start.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        chunk_start = chunk_end + pd.Timedelta(days=1)

    total = len(chunks)
    page_size = 5000
    for idx, (chunk_start_s, chunk_end_s) in enumerate(chunks, start=1):
        offset = 0
        page = 1
        while True:
            print(
                f"[fetch] report_rc {chunk_start_s}->{chunk_end_s} "
                f"page {page} ({idx}/{total})"
            )
            frame = client.analyst_report_rc(chunk_start_s, chunk_end_s, limit=page_size, offset=offset)
            if not frame.empty:
                frames.append(frame)
            if len(frame) < page_size:
                break
            offset += page_size
            page += 1

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(ignore_index=True)


def fetch_income_statements(
    client: TushareDataClient,
    start_date: str,
    end_date: str,
    ts_codes: list[str],
    max_workers: int = 6,
) -> pd.DataFrame:
    frames: list[tuple[int, pd.DataFrame]] = []
    total = len(ts_codes)

    def fetch_one(idx_code: tuple[int, str]) -> tuple[int, str, pd.DataFrame]:
        idx, ts_code = idx_code
        frame = client.income_by_ts_code(ts_code, start_date, end_date)
        return idx, ts_code, frame

    workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, item) for item in enumerate(ts_codes, start=1)]
        for future in as_completed(futures):
            idx, ts_code, frame = future.result()
            print(f"[fetch] income {ts_code} ({idx}/{total})")
            if not frame.empty:
                frames.append((idx, frame))

    if not frames:
        return pd.DataFrame()

    ordered_frames = [frame for _, frame in sorted(frames, key=lambda item: item[0])]
    return pd.concat(ordered_frames, ignore_index=True)
