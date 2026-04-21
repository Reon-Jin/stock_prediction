from __future__ import annotations

from typing import Any

import pandas as pd

from providers.base import BaseProvider
from utils.symbols import normalize_symbol


class FinancialProvider(BaseProvider):
    provider_name = "provider_financial"

    def fetch_financial_snapshot(self, symbol: str, start: str, end: str) -> list[dict[str, Any]]:
        normalized_symbol = normalize_symbol(symbol)
        lookback_start = pd.Timestamp(start) - pd.Timedelta(days=450)

        def _fetch() -> list[dict[str, Any]]:
            fin = self._fetch_financial_analysis_em(normalized_symbol)
            if fin.empty:
                return []

            fin["report_date"] = pd.to_datetime(fin.get("REPORT_DATE"), errors="coerce").dt.date
            notice_col = "NOTICE_DATE" if "NOTICE_DATE" in fin.columns else "UPDATE_DATE"
            fin["asof_date"] = pd.to_datetime(fin.get(notice_col), errors="coerce").dt.date
            fin["asof_date"] = fin["asof_date"].fillna(fin["report_date"])
            fin = fin[
                (pd.to_datetime(fin["asof_date"], errors="coerce") >= lookback_start)
                & (pd.to_datetime(fin["asof_date"], errors="coerce") <= pd.Timestamp(end))
            ].copy()
            if fin.empty:
                return []

            field_map = {
                "ROEJQ": "roe",
                "TOTALOPERATEREVETZ": "revenue_yoy",
                "PARENTNETPROFITTZ": "profit_yoy",
                "XSMLL": "gross_margin",
                "ZCFZL": "debt_ratio",
            }
            for src, dst in field_map.items():
                fin[dst] = pd.to_numeric(fin.get(src), errors="coerce")

            # Free Eastmoney financial analysis exposes strong report-period metrics,
            # but not stable daily historical valuation series comparable to the old
            # removed akshare.stock_a_indicator_lg API. Keep these nullable instead
            # of leaking current valuations backward in time.
            fin["pe_ttm"] = None
            fin["pb"] = None
            fin["ps_ttm"] = None
            fin["industry_percentile"] = None
            fin["symbol"] = normalized_symbol
            fin["source"] = "eastmoney_datacenter"
            fin["data_version"] = self.config.project.get("data_version", "v1")

            fin = fin.dropna(subset=["report_date", "asof_date"])
            fin = fin.sort_values(["asof_date", "report_date"]).drop_duplicates(["symbol", "report_date", "asof_date"])
            return fin[
                [
                    "symbol",
                    "report_date",
                    "asof_date",
                    "pe_ttm",
                    "pb",
                    "ps_ttm",
                    "roe",
                    "revenue_yoy",
                    "profit_yoy",
                    "gross_margin",
                    "debt_ratio",
                    "industry_percentile",
                    "source",
                    "data_version",
                ]
            ].to_dict("records")

        return self.wrap_fetch("financial_snapshot", {"symbol": symbol, "start": start, "end": end}, _fetch)

    def _fetch_financial_analysis_em(self, symbol: str) -> pd.DataFrame:
        data_json = self.http_get_json(
            "https://datacenter.eastmoney.com/securities/api/data/get",
            params={
                "type": "RPT_F10_FINANCE_MAINFINADATA",
                "sty": "APP_F10_MAINFINADATA",
                "quoteColumns": "",
                "filter": f'(SECUCODE="{symbol}")',
                "p": "1",
                "ps": "200",
                "sr": "-1",
                "st": "REPORT_DATE",
                "source": "HSF10",
                "client": "PC",
            },
        )
        rows = (((data_json or {}).get("result") or {}).get("data")) or []
        return pd.DataFrame(rows)
