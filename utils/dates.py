from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Iterable

import pandas as pd


DATE_FMT = "%Y-%m-%d"


def to_date_str(value: str | datetime | pd.Timestamp) -> str:
    if isinstance(value, str):
        return pd.Timestamp(value).strftime(DATE_FMT)
    return pd.Timestamp(value).strftime(DATE_FMT)


def ensure_timestamp(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value)


@dataclass(slots=True)
class TradingCalendar:
    trading_days: list[pd.Timestamp]
    _day_set: set[pd.Timestamp] = field(init=False, repr=False)
    _series: pd.Series = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._day_set = {pd.Timestamp(day).normalize() for day in self.trading_days}
        self._series = pd.Series(range(len(self.trading_days)), index=pd.DatetimeIndex(self.trading_days))

    def is_trade_day(self, day: str | datetime | pd.Timestamp) -> bool:
        return ensure_timestamp(day).normalize() in self._day_set

    def next_trade_day(self, day: str | datetime | pd.Timestamp, offset: int = 1) -> pd.Timestamp:
        ts = ensure_timestamp(day).normalize()
        idx = self._series.index.searchsorted(ts, side="right")
        target = idx + offset - 1
        if target >= len(self.trading_days):
            raise IndexError("next trade day out of range")
        return self.trading_days[target]

    def prev_trade_day(self, day: str | datetime | pd.Timestamp, offset: int = 1) -> pd.Timestamp:
        ts = ensure_timestamp(day).normalize()
        idx = self._series.index.searchsorted(ts, side="left") - offset
        if idx < 0:
            raise IndexError("previous trade day out of range")
        return self.trading_days[idx]

    def range(self, start: str | datetime | pd.Timestamp, end: str | datetime | pd.Timestamp) -> list[pd.Timestamp]:
        left = self._series.index.searchsorted(ensure_timestamp(start).normalize(), side="left")
        right = self._series.index.searchsorted(ensure_timestamp(end).normalize(), side="right")
        return self.trading_days[left:right]


def build_calendar_from_series(days: Iterable[str | datetime | pd.Timestamp]) -> TradingCalendar:
    trading_days = sorted(pd.Timestamp(day).normalize() for day in days)
    return TradingCalendar(trading_days=trading_days)


@lru_cache(maxsize=1)
def default_business_calendar() -> TradingCalendar:
    days = pd.bdate_range("2023-01-01", "2025-12-31").to_list()
    return TradingCalendar(trading_days=days)
