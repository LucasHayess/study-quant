from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str = "20220101"
    end_date: str = "20231231"
    warmup_start_date: str = "20211101"
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path("data/cache")
    group_count: int = 5
    rolling_window: int = 20
    min_listed_days: int = 180
    pause_seconds: float = 0.35
    retry_count: int = 3
