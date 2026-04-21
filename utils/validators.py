from __future__ import annotations

from typing import Any

import pandas as pd


def validate_bar_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    open_ = float(record.get("open", 0) or 0)
    high = float(record.get("high", 0) or 0)
    low = float(record.get("low", 0) or 0)
    close = float(record.get("close", 0) or 0)
    volume = float(record.get("volume", 0) or 0)
    amount = float(record.get("amount", 0) or 0)
    if high < max(open_, close):
        errors.append("high_lt_open_or_close")
    if low > min(open_, close):
        errors.append("low_gt_open_or_close")
    if high < low:
        errors.append("high_lt_low")
    if min(open_, high, low, close, volume, amount) < 0:
        errors.append("negative_value")
    return errors


def validate_required(record: dict[str, Any], required_fields: list[str]) -> list[str]:
    return [field for field in required_fields if pd.isna(record.get(field)) or record.get(field) == ""]


def clip_extreme(series: pd.Series, lower: float, upper: float) -> pd.Series:
    return series.clip(lower=lower, upper=upper)

