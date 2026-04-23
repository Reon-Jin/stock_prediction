from __future__ import annotations
from typing import Any

import pandas as pd

from providers.base import BaseProvider, ProviderError
from utils.symbols import normalize_symbol, symbol_to_akshare


class MarketProvider(BaseProvider):
    provider_name = "provider_market"

    def fetch_security_profile(self, symbol: str) -> dict[str, Any]:
        ak = self.require_akshare()
        normalized_symbol = normalize_symbol(symbol)
        code = normalized_symbol.split(".")[0]

        def _fetch() -> dict[str, Any]:
            df = ak.stock_individual_info_em(symbol=code)
            if df is None or df.empty:
                return {"symbol": normalized_symbol, "industry": None, "list_date": None, "name": None}
            columns = [str(column) for column in df.columns]
            if "item" not in columns or "value" not in columns:
                renamed = df.copy()
                renamed.columns = ["item", "value"][: len(renamed.columns)]
                df = renamed
            mapping = {
                str(row["item"]).strip(): row.get("value")
                for row in df[["item", "value"]].to_dict("records")
                if str(row.get("item", "")).strip()
            }
            list_date = mapping.get("上市时间")
            if pd.notna(list_date) and str(list_date).strip():
                list_date = pd.to_datetime(str(list_date)).date()
            else:
                list_date = None
            name = mapping.get("股票简称")
            name = str(name).strip() if pd.notna(name) and str(name).strip() else None
            industry = mapping.get("行业")
            industry = str(industry).strip() if pd.notna(industry) and str(industry).strip() else None
            return {
                "symbol": normalized_symbol,
                "name": name,
                "industry": industry,
                "list_date": list_date,
            }

        return self.wrap_fetch("security_profile", {"symbol": normalized_symbol}, _fetch)

    def fetch_security_profile_cninfo(self, symbol: str) -> dict[str, Any]:
        ak = self.require_akshare()
        normalized_symbol = normalize_symbol(symbol)
        code = normalized_symbol.split(".")[0]

        def _fetch() -> dict[str, Any]:
            df = ak.stock_profile_cninfo(symbol=code)
            if df is None or df.empty:
                return {"symbol": normalized_symbol, "industry": None, "list_date": None, "name": None}
            row = df.iloc[0].to_dict()
            name = row.get("A股简称") or row.get("公司名称")
            name = str(name).strip() if pd.notna(name) and str(name).strip() else None
            industry = row.get("所属行业")
            industry = str(industry).strip() if pd.notna(industry) and str(industry).strip() else None
            list_date = row.get("上市日期")
            if pd.notna(list_date) and str(list_date).strip():
                list_date = pd.to_datetime(str(list_date)).date()
            else:
                list_date = None
            return {
                "symbol": normalized_symbol,
                "name": name,
                "industry": industry,
                "list_date": list_date,
            }

        return self.wrap_fetch("security_profile_cninfo", {"symbol": normalized_symbol}, _fetch)

    def fetch_trade_calendar(self, start: str, end: str) -> list[dict[str, Any]]:
        ak = self.require_akshare()

        def _fetch() -> list[dict[str, Any]]:
            df = ak.tool_trade_date_hist_sina()
            if df is None or df.empty:
                return []
            col = "trade_date" if "trade_date" in df.columns else df.columns[0]
            df = df.rename(columns={col: "trade_date"})
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            df = df[(df["trade_date"] >= pd.to_datetime(start).date()) & (df["trade_date"] <= pd.to_datetime(end).date())]
            df["is_open"] = True
            df["exchange"] = "CN"
            return df[["trade_date", "exchange", "is_open"]].to_dict("records")

        return self.wrap_fetch("trade_calendar", {"start": start, "end": end}, _fetch)

    def fetch_securities(self) -> list[dict[str, Any]]:
        ak = self.require_akshare()

        def _fetch() -> list[dict[str, Any]]:
            basic = ak.stock_info_a_code_name()
            merged = basic.rename(columns={"code": "symbol", "name": "name"}).copy()
            merged["industry"] = None
            merged["list_date"] = None

            try:
                spot = ak.stock_zh_a_spot_em()
                rename_map = {
                    "代码": "symbol",
                    "名称": "name",
                    "总市值": "market_cap",
                    "所处行业": "industry",
                    "所属行业": "industry",
                    "上市时间": "list_date",
                }
                spot = spot.rename(columns=rename_map)
                keep_cols = [col for col in ["symbol", "name", "industry", "list_date"] if col in spot.columns]
                if keep_cols:
                    merged = merged.merge(spot[keep_cols], on=["symbol", "name"], how="left", suffixes=("", "_spot"))
                    if "industry_spot" in merged.columns:
                        merged["industry"] = merged["industry_spot"].combine_first(merged["industry"])
                    if "list_date_spot" in merged.columns:
                        merged["list_date"] = merged["list_date_spot"].combine_first(merged["list_date"])
            except Exception as exc:
                self.logger.warning("stock_zh_a_spot_em unavailable, fallback to basic code-name list only: %s", exc)

            records: list[dict[str, Any]] = []
            for row in merged.to_dict("records"):
                symbol = normalize_symbol(row["symbol"])
                name = str(row["name"])
                board = self._infer_board(symbol)
                list_date = row.get("list_date")
                if pd.notna(list_date) and str(list_date).strip():
                    list_date = pd.to_datetime(str(list_date)).date()
                else:
                    list_date = None
                records.append(
                    {
                        "symbol": symbol,
                        "name": name,
                        "exchange": symbol.split(".")[1],
                        "board": board,
                        "industry": row.get("industry"),
                        "list_date": list_date,
                        "delist_date": None,
                        "is_st": name.upper().startswith("ST") or "ST" in name.upper(),
                        "status": "active",
                        "data_version": self.config.project.get("data_version", "v1"),
                    }
                )
            return records

        return self.wrap_fetch("securities", {}, _fetch)

    def fetch_daily_bars(self, symbol: str, start: str, end: str) -> list[dict[str, Any]]:
        code = symbol_to_akshare(symbol)
        normalized_symbol = normalize_symbol(symbol)
        market_symbol = self._to_market_prefixed_symbol(normalized_symbol)

        def _fetch() -> list[dict[str, Any]]:
            source_order = list(self.provider_conf.get("daily_bar_source_order", ["sina", "eastmoney", "tx"]))
            errors: list[str] = []
            for source_name in source_order:
                try:
                    if source_name == "eastmoney_http":
                        normalized = self._fetch_eastmoney_http_daily(normalized_symbol, start, end)
                    elif source_name == "eastmoney":
                        ak = self.require_akshare()
                        df = ak.stock_zh_a_hist(
                            symbol=code,
                            period="daily",
                            start_date=start.replace("-", ""),
                            end_date=end.replace("-", ""),
                            adjust="qfq",
                        )
                        normalized = self._normalize_eastmoney_hist(df, normalized_symbol)
                    elif source_name == "sina":
                        ak = self.require_akshare()
                        df = ak.stock_zh_a_daily(
                            symbol=market_symbol,
                            start_date=start.replace("-", ""),
                            end_date=end.replace("-", ""),
                            adjust="qfq",
                        )
                        normalized = self._normalize_sina_daily(df, normalized_symbol)
                    elif source_name == "tx":
                        ak = self.require_akshare()
                        df = ak.stock_zh_a_hist_tx(
                            symbol=market_symbol,
                            start_date=start.replace("-", ""),
                            end_date=end.replace("-", ""),
                            adjust="qfq",
                            timeout=float(self.provider_conf.get("request_timeout", 20)),
                        )
                        normalized = self._normalize_tx_hist(df, normalized_symbol)
                    else:
                        continue
                    if normalized:
                        return normalized
                except Exception as exc:
                    errors.append(f"{source_name}: {exc}")
                    continue
            raise ProviderError(f"all daily bar sources failed for {normalized_symbol}: {' | '.join(errors)}")

        return self.wrap_fetch("daily_bars", {"symbol": symbol, "start": start, "end": end}, _fetch)

    def fetch_index_bars(self, index_code: str, start: str, end: str) -> list[dict[str, Any]]:
        ak = self.require_akshare()
        market_map = {"000300.SH": "sh000300", "399006.SZ": "sz399006", "000001.SH": "sh000001"}
        em_symbol = market_map.get(index_code, index_code.lower().replace(".", ""))

        def _fetch() -> list[dict[str, Any]]:
            errors: list[str] = []
            try:
                df = ak.index_zh_a_hist(
                    symbol=em_symbol,
                    period="daily",
                    start_date=start.replace("-", ""),
                    end_date=end.replace("-", ""),
                )
                normalized = self._normalize_index_eastmoney(df, index_code)
                if normalized:
                    return normalized
            except Exception as exc:
                errors.append(f"eastmoney: {exc}")

            try:
                df = ak.stock_zh_index_daily(symbol=em_symbol)
                normalized = self._normalize_index_sina(df, index_code, start, end)
                if normalized:
                    return normalized
            except Exception as exc:
                errors.append(f"sina: {exc}")

            try:
                df = ak.stock_zh_index_daily_tx(symbol=em_symbol)
                normalized = self._normalize_index_tx(df, index_code, start, end)
                if normalized:
                    return normalized
            except Exception as exc:
                errors.append(f"tx: {exc}")

            raise ProviderError(f"failed to fetch index {index_code}: {' | '.join(errors)}")

        return self.wrap_fetch("index_bars", {"index_code": index_code, "start": start, "end": end}, _fetch)

    def fetch_sector_daily(self, start: str, end: str, limit: int | None = None) -> list[dict[str, Any]]:
        ak = self.require_akshare()

        def _fetch() -> list[dict[str, Any]]:
            sectors = ak.stock_board_industry_name_ths().rename(columns={"name": "sector_name", "code": "sector_id"})
            if limit:
                sectors = sectors.head(limit)
            records: list[dict[str, Any]] = []
            for row in sectors.to_dict("records"):
                try:
                    sector_name = str(row["sector_name"])
                    sector_id = str(row["sector_id"])
                    hist = ak.stock_board_industry_index_ths(
                        symbol=sector_name,
                        start_date=start.replace("-", ""),
                        end_date=end.replace("-", ""),
                    )
                    if hist is None or hist.empty:
                        continue
                    hist = hist.rename(
                        columns={
                            "日期": "trade_date",
                            "收盘价": "close",
                            "成交额": "amount",
                        }
                    )
                    hist["trade_date"] = pd.to_datetime(hist["trade_date"], errors="coerce").dt.date
                    hist["close"] = pd.to_numeric(hist["close"], errors="coerce")
                    hist["amount"] = pd.to_numeric(hist["amount"], errors="coerce")
                    hist["pct_chg"] = pd.to_numeric(hist["close"], errors="coerce").pct_change() * 100
                    hist["turnover"] = None
                    hist["sector_id"] = sector_id
                    hist["sector_name"] = sector_name
                    hist["up_limit_count"] = None
                    hist["source"] = "ths_industry_index"
                    hist["data_version"] = self.config.project.get("data_version", "v1")
                    records.extend(
                        hist[
                            [
                                "sector_id",
                                "sector_name",
                                "trade_date",
                                "close",
                                "pct_chg",
                                "amount",
                                "turnover",
                                "up_limit_count",
                                "source",
                                "data_version",
                            ]
                        ].to_dict("records")
                    )
                except Exception as exc:
                    self.logger.warning("skip sector %s due to fetch error: %s", row.get("sector_name"), exc)
                    continue
            return records

        return self.wrap_fetch("sector_daily", {"start": start, "end": end, "limit": limit}, _fetch)

    @staticmethod
    def _infer_board(symbol: str) -> str:
        code = normalize_symbol(symbol).split(".")[0]
        if code.startswith("688"):
            return "STAR"
        if code.startswith("300"):
            return "ChiNext"
        if code.startswith(("000", "001", "002", "003")):
            return "MainBoard"
        if code.startswith("8") or code.startswith("43"):
            return "BSE"
        return "MainBoard"

    @staticmethod
    def _to_market_prefixed_symbol(symbol: str) -> str:
        normalized = normalize_symbol(symbol)
        code, exchange = normalized.split(".")
        market = {"SH": "sh", "SZ": "sz", "BJ": "bj"}.get(exchange, "sz")
        return f"{market}{code}"

    @staticmethod
    def _to_eastmoney_secid(symbol: str) -> str:
        normalized = normalize_symbol(symbol)
        code, exchange = normalized.split(".")
        market = "1" if exchange == "SH" else "0"
        return f"{market}.{code}"

    def _fetch_eastmoney_http_daily(self, symbol: str, start: str, end: str) -> list[dict[str, Any]]:
        payload = self.http_get_json(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={
                "secid": self._to_eastmoney_secid(symbol),
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
                "klt": "101",
                "fqt": "1",
                "beg": start.replace("-", ""),
                "end": end.replace("-", ""),
            },
            timeout=float(self.provider_conf.get("request_timeout", 20)),
        )
        klines = ((payload or {}).get("data") or {}).get("klines") or []
        rows: list[dict[str, Any]] = []
        for item in klines:
            parts = str(item).split(",")
            if len(parts) < 11:
                continue
            rows.append(
                {
                    "trade_date": parts[0],
                    "open": parts[1],
                    "close": parts[2],
                    "high": parts[3],
                    "low": parts[4],
                    "volume": float(parts[5]) * 100,
                    "amount": parts[6],
                    "amplitude": parts[7],
                    "pct_chg": parts[8],
                    "turnover_rate": parts[10],
                }
            )
        return self._finalize_daily_bar_df(pd.DataFrame(rows), symbol, "eastmoney_http")

    def _finalize_daily_bar_df(self, df: pd.DataFrame, symbol: str, source: str) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        data = df.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.date
        data["symbol"] = symbol
        data["open"] = pd.to_numeric(data["open"], errors="coerce")
        data["high"] = pd.to_numeric(data["high"], errors="coerce")
        data["low"] = pd.to_numeric(data["low"], errors="coerce")
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data["volume"] = pd.to_numeric(data.get("volume"), errors="coerce")
        data["amount"] = pd.to_numeric(data.get("amount"), errors="coerce")
        data["turnover_rate"] = pd.to_numeric(data.get("turnover_rate"), errors="coerce")
        data["pre_close"] = data["close"].shift(1)
        data["pct_chg"] = (data["close"] / data["pre_close"] - 1) * 100
        data["amplitude"] = ((data["high"] - data["low"]) / data["pre_close"]) * 100
        data["adj_factor"] = 1.0
        data["adj_close"] = data["close"]
        data["is_suspended"] = False
        data["source"] = source
        data["data_version"] = self.config.project.get("data_version", "v1")
        data = data.dropna(subset=["trade_date", "open", "high", "low", "close"])
        return data[
            [
                "symbol",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "pre_close",
                "volume",
                "amount",
                "turnover_rate",
                "amplitude",
                "pct_chg",
                "adj_factor",
                "adj_close",
                "is_suspended",
                "source",
                "data_version",
            ]
        ].to_dict("records")

    def _normalize_eastmoney_hist(self, df: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        data = df.rename(
            columns={
                "日期": "trade_date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover_rate",
            }
        )
        return self._finalize_daily_bar_df(data, symbol, "akshare_eastmoney")

    def _normalize_sina_daily(self, df: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        data = df.rename(columns={"date": "trade_date"})
        if "turnover" in data.columns and "turnover_rate" not in data.columns:
            data["turnover_rate"] = pd.to_numeric(data["turnover"], errors="coerce") * 100
        return self._finalize_daily_bar_df(data, symbol, "akshare_sina")

    def _normalize_tx_hist(self, df: pd.DataFrame, symbol: str) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        data = df.rename(columns={"date": "trade_date", "amount": "volume"})
        data["volume"] = pd.to_numeric(data.get("volume"), errors="coerce") * 100
        data["close"] = pd.to_numeric(data.get("close"), errors="coerce")
        data["amount"] = data["volume"] * data["close"]
        data["turnover_rate"] = None
        return self._finalize_daily_bar_df(data, symbol, "akshare_tx")

    def _finalize_index_df(self, df: pd.DataFrame, index_code: str, source: str) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        data = df.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"]).dt.date
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            if col not in data.columns:
                data[col] = None
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data["index_code"] = index_code
        data["source"] = source
        data["data_version"] = self.config.project.get("data_version", "v1")
        data = data.dropna(subset=["trade_date", "close"])
        return data[
            ["index_code", "trade_date", "open", "high", "low", "close", "volume", "amount", "source", "data_version"]
        ].to_dict("records")

    def _normalize_index_eastmoney(self, df: pd.DataFrame, index_code: str) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        data = df.rename(
            columns={
                "日期": "trade_date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
            }
        )
        return self._finalize_index_df(data, index_code, "akshare_index_eastmoney")

    def _normalize_index_sina(self, df: pd.DataFrame, index_code: str, start: str, end: str) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        data = df.rename(columns={"date": "trade_date"})
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data[(data["trade_date"] >= pd.Timestamp(start)) & (data["trade_date"] <= pd.Timestamp(end))]
        return self._finalize_index_df(data, index_code, "akshare_index_sina")

    def _normalize_index_tx(self, df: pd.DataFrame, index_code: str, start: str, end: str) -> list[dict[str, Any]]:
        if df is None or df.empty:
            return []
        data = df.rename(columns={"date": "trade_date", "amount": "volume"})
        if "amount" not in data.columns:
            data["amount"] = None
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        data = data[(data["trade_date"] >= pd.Timestamp(start)) & (data["trade_date"] <= pd.Timestamp(end))]
        return self._finalize_index_df(data, index_code, "akshare_index_tx")
