from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from utils.dates import TradingCalendar
from utils.symbols import normalize_symbol
from utils.validators import validate_bar_record, validate_required


def normalize_securities(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        symbol = normalize_symbol(record["symbol"])
        if symbol in seen:
            continue
        seen.add(symbol)
        output.append(
            {
                **record,
                "symbol": symbol,
                "exchange": record.get("exchange") or symbol.split(".")[1],
                "is_st": bool(record.get("is_st", False)),
                "status": record.get("status") or "active",
            }
        )
    return output


def normalize_daily_bars(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid_rows: list[dict[str, Any]] = []
    bad_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, Any]] = set()
    for record in records:
        row = dict(record)
        row["symbol"] = normalize_symbol(row["symbol"])
        key = (row["symbol"], row["trade_date"])
        missing = validate_required(row, ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"])
        errors = validate_bar_record(row)
        if key in seen or missing or errors:
            bad_rows.append({"record": row, "missing": missing, "errors": errors})
            continue
        seen.add(key)
        valid_rows.append(row)
    return valid_rows, bad_rows


def clip_market_extremes(df: pd.DataFrame) -> pd.DataFrame:
    if "pct_chg" in df.columns:
        df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce").clip(-25, 25)
    if "turnover_rate" in df.columns:
        df["turnover_rate"] = pd.to_numeric(df["turnover_rate"], errors="coerce").clip(0, 100)
    if "main_net_inflow" in df.columns:
        q = df["main_net_inflow"].quantile([0.01, 0.99]).fillna(0)
        df["main_net_inflow"] = df["main_net_inflow"].clip(q.iloc[0], q.iloc[1])
    return df


def publication_to_trade_date(
    publish_time: pd.Timestamp,
    calendar: TradingCalendar,
    market_close_hour: int = 15,
) -> pd.Timestamp:
    normalized = publish_time.normalize()
    if calendar.is_trade_day(normalized) and publish_time.hour < market_close_hour:
        return normalized
    try:
        return calendar.next_trade_day(normalized, offset=1)
    except IndexError:
        if not calendar.trading_days:
            raise
        # When a publication lands after the last available trade day in our
        # local calendar snapshot, clamp it to the last known trade day so the
        # normalization step remains rerunnable instead of crashing the job.
        return pd.Timestamp(calendar.trading_days[-1]).normalize()


def dedupe_news_like(records: list[dict[str, Any]], id_key: str, title_key: str = "title") -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    seen_compounds: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in records:
        unique_id = row[id_key]
        compound = hashlib.md5(
            f"{row.get(title_key)}|{row.get('source')}|{row.get('publish_time')}".encode("utf-8")
        ).hexdigest()
        if unique_id in seen_ids or compound in seen_compounds:
            continue
        seen_ids.add(unique_id)
        seen_compounds.add(compound)
        row["mentioned_symbols"] = [normalize_symbol(s) for s in (row.get("mentioned_symbols") or [])]
        output.append(row)
    return output
