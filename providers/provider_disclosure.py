from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from providers.base import BaseProvider
from utils.symbols import normalize_symbol


class DisclosureProvider(BaseProvider):
    provider_name = "provider_disclosure"

    def fetch_announcements(self, symbol: str, start: str, end: str) -> list[dict[str, Any]]:
        ak = self.require_akshare()
        code = normalize_symbol(symbol).split(".")[0]

        def _fetch() -> list[dict[str, Any]]:
            try:
                df = ak.stock_notice_report(symbol=code)
            except Exception:
                return []
            if df is None or df.empty:
                return []
            df = df.rename(
                columns={
                    "公告时间": "publish_time",
                    "公告标题": "title",
                    "公告链接": "url",
                    "公告类型": "category",
                }
            )
            df["publish_time"] = pd.to_datetime(df["publish_time"])
            df = df[(df["publish_time"] >= pd.Timestamp(start)) & (df["publish_time"] <= pd.Timestamp(end) + pd.Timedelta(days=1))]
            rows: list[dict[str, Any]] = []
            for row in df.to_dict("records"):
                title = str(row.get("title") or "")
                publish_time = pd.Timestamp(row["publish_time"]).to_pydatetime()
                announcement_id = hashlib.md5(f"{code}|{title}|{publish_time.isoformat()}".encode("utf-8")).hexdigest()
                rows.append(
                    {
                        "announcement_id": announcement_id,
                        "publish_time": publish_time,
                        "source": "cninfo",
                        "title": title,
                        "category": row.get("category"),
                        "summary": None,
                        "content": row.get("url"),
                        "mentioned_symbols": [normalize_symbol(symbol)],
                        "data_version": self.config.project.get("data_version", "v1"),
                    }
                )
            return rows

        return self.wrap_fetch("announcement_raw", {"symbol": symbol, "start": start, "end": end}, _fetch)

