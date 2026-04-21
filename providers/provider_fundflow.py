from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from providers.base import BaseProvider
from utils.symbols import symbol_to_akshare, normalize_symbol


class FundFlowProvider(BaseProvider):
    provider_name = "provider_fundflow"

    def fetch_capital_flow(self, symbol: str, start: str, end: str) -> list[dict[str, Any]]:
        code = symbol_to_akshare(symbol)

        def _fetch() -> list[dict[str, Any]]:
            session = self.build_http_session()
            response = session.get(f"https://stockpage.10jqka.com.cn/{code}/Funds/", timeout=float(self.provider_conf.get("request_timeout", 20)))
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            if not tables:
                return []

            # The first table on the THS funds page contains the historical
            # daily fund-flow breakdown for the current stock.
            df = tables[0].copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    "_".join([str(part) for part in column if str(part) != "nan"]).strip("_")
                    for column in df.columns.to_flat_index()
                ]
            df.columns = [str(column).strip() for column in df.columns]
            rename_map = {
                "日期_日期": "trade_date",
                "收盘价_收盘价": "close",
                "涨跌幅_涨跌幅": "pct_chg",
                "资金净流入_资金净流入": "main_net_inflow",
                "5日主力净额_5日主力净额": "main_net_inflow_5d",
                "大单(主力)_净额": "large_net_inflow",
                "大单(主力)_净占比": "main_inflow_ratio",
                "中单_净额": "medium_net_inflow",
                "小单_净额": "small_net_inflow",
            }
            df = df.rename(columns=rename_map)
            if "trade_date" not in df.columns:
                return []

            df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce").dt.date
            df = df[(df["trade_date"] >= pd.to_datetime(start).date()) & (df["trade_date"] <= pd.to_datetime(end).date())]
            if df.empty:
                return []

            for column in ["main_net_inflow", "large_net_inflow", "medium_net_inflow", "small_net_inflow", "main_inflow_ratio"]:
                if column not in df.columns:
                    df[column] = None
                series = df[column].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
                df[column] = pd.to_numeric(series, errors="coerce")

            # THS page does not expose separate super-large order history in the
            # rendered table, so keep it nullable instead of inventing values.
            df["super_large_net_inflow"] = None
            df["symbol"] = normalize_symbol(symbol)
            df["source"] = "ths_funds_page"
            df["data_version"] = self.config.project.get("data_version", "v1")
            return df[
                [
                    "symbol",
                    "trade_date",
                    "main_net_inflow",
                    "super_large_net_inflow",
                    "large_net_inflow",
                    "medium_net_inflow",
                    "small_net_inflow",
                    "main_inflow_ratio",
                    "source",
                    "data_version",
                ]
            ].to_dict("records")

        return self.wrap_fetch("capital_flow_daily", {"symbol": symbol, "start": start, "end": end}, _fetch)
