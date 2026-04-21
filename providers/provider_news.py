from __future__ import annotations

import hashlib
from datetime import datetime, time
from typing import Any

import pandas as pd

from providers.base import BaseProvider


class NewsProvider(BaseProvider):
    provider_name = "provider_news"

    def fetch_trade_date_news(self, trade_date: str) -> list[dict[str, Any]]:
        ak = self.require_akshare()
        target_ts = pd.Timestamp(trade_date).normalize()

        def _fetch() -> list[dict[str, Any]]:
            financial_candidates = self._build_financial_candidates(ak, target_ts)
            domestic_candidates = self._build_domestic_candidates(ak, target_ts)
            international_candidates = self._build_international_candidates(ak, target_ts)

            selected: list[dict[str, Any]] = []
            selected.extend(self._take_with_padding(financial_candidates, 5, "financial", target_ts))
            selected.extend(self._take_with_padding(domestic_candidates, 3, "domestic", target_ts))
            selected.extend(self._take_with_padding(international_candidates, 2, "international", target_ts))

            rows: list[dict[str, Any]] = []
            for index, item in enumerate(selected, start=1):
                title = str(item.get("title") or "").strip() or f"{item['category']}_{index}"
                publish_time = item.get("publish_time") or datetime.combine(target_ts.date(), time(hour=12))
                news_id = hashlib.md5(
                    f"{trade_date}|{item['source']}|{item['category']}|{index}|{title}".encode("utf-8")
                ).hexdigest()
                rows.append(
                    {
                        "news_id": news_id,
                        "publish_time": publish_time,
                        "source": item["source"],
                        "title": title,
                        "summary": str(item.get("summary") or "")[:512],
                        "content": item.get("content"),
                        "url": item.get("url"),
                        "mentioned_symbols": [],
                        "importance_score_raw": item.get("importance_score_raw"),
                        "normalized_flag": False,
                        "data_version": self.config.project.get("data_version", "v1"),
                        "target_trade_date": trade_date,
                    }
                )
            return rows

        return self.wrap_fetch("news_raw_daily", {"trade_date": trade_date}, _fetch)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        numeric = pd.to_numeric(value, errors="coerce")
        return float(default if pd.isna(numeric) else numeric)

    @staticmethod
    def _safe_datetime(trade_date: pd.Timestamp, value: Any, fallback: time) -> datetime:
        if value is not None and str(value).strip() and str(value).strip() != "--":
            parsed = pd.to_datetime(f"{trade_date.strftime('%Y-%m-%d')} {value}", errors="coerce")
            if not pd.isna(parsed):
                return parsed.to_pydatetime()
        return datetime.combine(trade_date.date(), fallback)

    def _build_financial_candidates(self, ak: Any, trade_date: pd.Timestamp) -> list[dict[str, Any]]:
        try:
            economic_df = ak.news_economic_baidu(date=trade_date.strftime("%Y%m%d"))
        except Exception:
            economic_df = pd.DataFrame()

        if economic_df is None or economic_df.empty:
            return []

        df = economic_df.copy()
        df["重要性"] = pd.to_numeric(df.get("重要性"), errors="coerce").fillna(0)
        if "时间" in df.columns:
            df["时间"] = df["时间"].fillna("08:00")
        df = df.sort_values(["重要性", "时间"], ascending=[False, True])

        rows: list[dict[str, Any]] = []
        for row in df.to_dict("records"):
            region = str(row.get("地区") or "全球").strip()
            event_name = str(row.get("事件") or "").strip()
            if not event_name:
                continue
            publish_time = self._safe_datetime(trade_date, row.get("时间"), fallback=time(hour=8))
            summary = (
                f"地区: {region}; 公布: {row.get('公布')}; 预期: {row.get('预期')}; "
                f"前值: {row.get('前值')}; 重要性: {row.get('重要性')}"
            )
            rows.append(
                {
                    "category": "financial",
                    "source": "daily_financial",
                    "title": f"[金融热点][{region}] {event_name}",
                    "summary": summary,
                    "content": summary,
                    "url": None,
                    "publish_time": publish_time,
                    "importance_score_raw": self._safe_float(row.get("重要性")),
                }
            )
        return rows

    def _build_domestic_candidates(self, ak: Any, trade_date: pd.Timestamp) -> list[dict[str, Any]]:
        try:
            cctv_df = ak.news_cctv(date=trade_date.strftime("%Y%m%d"))
        except Exception:
            cctv_df = pd.DataFrame()

        if cctv_df is None or cctv_df.empty:
            return []

        rows: list[dict[str, Any]] = []
        for index, row in enumerate(cctv_df.to_dict("records"), start=1):
            title = str(row.get("title") or "").strip()
            content = str(row.get("content") or "").strip()
            if not title and not content:
                continue
            publish_time = datetime.combine(trade_date.date(), time(hour=19, minute=min(index, 50)))
            rows.append(
                {
                    "category": "domestic",
                    "source": "daily_domestic",
                    "title": f"[国内热点] {title or content[:40]}",
                    "summary": content[:256],
                    "content": content,
                    "url": None,
                    "publish_time": publish_time,
                    "importance_score_raw": float(max(0.0, 10 - index)),
                }
            )
        return rows

    def _build_international_candidates(self, ak: Any, trade_date: pd.Timestamp) -> list[dict[str, Any]]:
        try:
            economic_df = ak.news_economic_baidu(date=trade_date.strftime("%Y%m%d"))
        except Exception:
            economic_df = pd.DataFrame()

        if economic_df is None or economic_df.empty:
            return []

        df = economic_df.copy()
        if "地区" in df.columns:
            df["地区"] = df["地区"].fillna("").astype(str)
        else:
            df["地区"] = ""
        international_mask = ~df["地区"].str.contains("中国|香港|澳门|台湾", regex=True)
        df = df[international_mask].copy()
        if df.empty:
            return []

        df["重要性"] = pd.to_numeric(df.get("重要性"), errors="coerce").fillna(0)
        if "时间" in df.columns:
            df["时间"] = df["时间"].fillna("09:00")
        df = df.sort_values(["重要性", "时间"], ascending=[False, True])

        rows: list[dict[str, Any]] = []
        for row in df.to_dict("records"):
            region = str(row.get("地区") or "海外").strip()
            event_name = str(row.get("事件") or "").strip()
            if not event_name:
                continue
            publish_time = self._safe_datetime(trade_date, row.get("时间"), fallback=time(hour=9))
            summary = (
                f"地区: {region}; 公布: {row.get('公布')}; 预期: {row.get('预期')}; "
                f"前值: {row.get('前值')}; 重要性: {row.get('重要性')}"
            )
            rows.append(
                {
                    "category": "international",
                    "source": "daily_international",
                    "title": f"[国际热点][{region}] {event_name}",
                    "summary": summary,
                    "content": summary,
                    "url": None,
                    "publish_time": publish_time,
                    "importance_score_raw": self._safe_float(row.get("重要性")),
                }
            )
        return rows

    def _take_with_padding(
        self,
        candidates: list[dict[str, Any]],
        target_count: int,
        category: str,
        trade_date: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        selected = candidates[:target_count]
        while len(selected) < target_count:
            index = len(selected) + 1
            publish_time = datetime.combine(trade_date.date(), time(hour=12, minute=min(index, 59)))
            selected.append(
                {
                    "category": category,
                    "source": f"daily_{category}",
                    "title": f"[{category}] {trade_date.strftime('%Y-%m-%d')} 暂无足量热点，使用占位事件补齐",
                    "summary": f"{trade_date.strftime('%Y-%m-%d')} {category} category placeholder event",
                    "content": f"{trade_date.strftime('%Y-%m-%d')} {category} category placeholder event",
                    "url": None,
                    "publish_time": publish_time,
                    "importance_score_raw": 0.0,
                }
            )
        return selected
