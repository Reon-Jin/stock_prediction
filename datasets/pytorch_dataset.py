from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_SEQ_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "amount",
    "turnover_rate",
    "pct_chg",
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma5_gap",
    "ma10_gap",
    "ma20_gap",
    "ma60_gap",
    "rsi_6",
    "rsi_14",
    "macd_dif",
    "macd_dea",
    "macd_hist",
    "atr_14",
    "boll_pos",
    "volatility_5",
    "volatility_20",
    "vol_ratio_5",
    "vol_ratio_20",
    "intraday_range",
    "candle_body",
    "upper_shadow",
    "lower_shadow",
    "gap_open_to_prev_close",
    "close_to_prev_close",
    "high_to_prev_close",
    "low_to_prev_close",
    "amount_log1p",
    "volume_log1p",
    "turnover_rate_delta",
    "ret_spread_5_20",
    "volatility_ratio_5_20",
]

DEFAULT_TAB_COLUMNS = [
    "list_days",
    "turnover_rank_industry",
    "amount_rank_market",
    "ret5_vs_hs300",
    "ret10_vs_hs300",
    "ret20_vs_industry",
    "stock_rank_in_industry",
    "industry_rank_5d",
    "industry_rank_20d",
    "roe",
    "revenue_yoy",
    "profit_yoy",
    "industry_roe_percentile",
]

EVENT_EMBEDDING_DIM = 256
EVENT_EMBEDDING_COLUMN = "event_embedding"
DEFAULT_EVENT_COLUMNS = [f"event_embedding_{idx:03d}" for idx in range(EVENT_EMBEDDING_DIM)]

DEFAULT_MKT_COLUMNS = [
    "up_limit_count",
    "down_limit_count",
    "broken_limit_rate",
    "consecutive_limit_height",
    "market_turnover",
    "hs300_ret_1",
    "cyb_ret_1",
    "market_volatility_5",
    "sector_hotness_top1",
    "sector_hotness_top3_mean",
    "risk_on_flag",
    "risk_off_flag",
    "limit_count_spread",
    "limit_count_ratio",
    "sector_hotness_spread",
    "market_regime_score",
]

DEFAULT_COMPANY_ID_COLUMNS = ["symbol_id", "industry_id", "board_id"]

DEFAULT_COMPANY_PROFILE_COLUMNS = [
    "market_cap_log",
    "volatility_120",
    "beta_120",
    "turnover_mean_120",
    "amount_mean_120",
    "ret_20",
    "ret_60",
    "roe",
    "revenue_yoy",
    "profit_yoy",
    "debt_ratio",
    "gross_margin",
]

DEFAULT_LABEL_GROUPS = {
    "p_win": ["label_win_3", "label_win_5", "label_win_10", "label_win_20", "label_win_40"],
    "ret_mu": ["label_ret_3", "label_ret_5", "label_ret_10", "label_ret_20", "label_ret_40"],
    "risk_dd": ["label_maxdd_3", "label_maxdd_5", "label_maxdd_10", "label_maxdd_20", "label_maxdd_40"],
    "bigloss": ["label_bigloss_5", "label_bigloss_20"],
    "rank_score": ["label_rank_score"],
}

META_COLUMNS = ["symbol", "trade_date", "name", "industry_sw", "board"]

INPUT_GROUP_DESCRIPTIONS = {
    "X_seq": "Recent rolling sequence features for one stock over the latest seq_length trading days.",
    "X_tab": "Point-in-time tabular features describing the stock on the sample trade_date.",
    "X_event": "Shared daily event vector encoded from 10 curated news items on the sample trade_date.",
    "X_mkt": "Shared market-state features for the whole market on the sample trade_date.",
    "X_company_ids": "Discrete company identity ids for symbol, industry, and board.",
    "X_company_profile": "Continuous slow-moving company profile features used to build company-aware embeddings.",
    "neighbors": "Optional top-k similar company ids and similarity weights for similarity regularization.",
}

LABEL_GROUP_DESCRIPTIONS = {
    "p_win": "Binary win heads for 3/5/10/20/40 trading-day horizons.",
    "ret_mu": "Future return regression heads for 3/5/10/20/40 trading-day horizons.",
    "risk_dd": "Future max-drawdown regression heads for 3/5/10/20/40 trading-day horizons.",
    "bigloss": "Binary downside-event heads for 5/20 trading-day horizons based on severe future drawdown.",
    "rank_score": "Same-day cross-sectional percentile rank derived from label_alpha_5.",
}

COLUMN_DESCRIPTIONS = {
    "symbol_id": "个股唯一整数ID",
    "industry_id": "行业唯一整数ID",
    "board_id": "板块唯一整数ID",
    "market_cap_log": "公司规模对数近似值",
    "volatility_120": "120日收益波动率",
    "beta_120": "相对沪深300的120日beta",
    "turnover_mean_120": "120日平均换手率",
    "amount_mean_120": "120日平均成交额",
    "ret_60": "过去60个交易日收益率",
    "debt_ratio": "资产负债率",
    "gross_margin": "毛利率",
    "neighbor_symbol_ids": "相似公司symbol_id列表",
    "neighbor_scores": "相似公司相似度权重列表",
    "open": "当日开盘价",
    "high": "当日最高价",
    "low": "当日最低价",
    "close": "当日收盘价",
    "adj_close": "前复权收盘价",
    "volume": "当日成交量",
    "amount": "当日成交额",
    "turnover_rate": "当日换手率",
    "pct_chg": "当日涨跌幅",
    "ret_1": "过去1个交易日收益率",
    "ret_3": "过去3个交易日收益率",
    "ret_5": "过去5个交易日收益率",
    "ret_10": "过去10个交易日收益率",
    "ret_20": "过去20个交易日收益率",
    "ma5_gap": "收盘价相对5日均线偏离",
    "ma10_gap": "收盘价相对10日均线偏离",
    "ma20_gap": "收盘价相对20日均线偏离",
    "ma60_gap": "收盘价相对60日均线偏离",
    "rsi_6": "6日RSI",
    "rsi_14": "14日RSI",
    "macd_dif": "MACD DIF",
    "macd_dea": "MACD DEA",
    "macd_hist": "MACD柱",
    "atr_14": "14日平均真实波幅",
    "boll_pos": "布林带相对位置",
    "volatility_5": "5日收益波动率",
    "volatility_20": "20日收益波动率",
    "vol_ratio_5": "成交量相对5日均量比",
    "vol_ratio_20": "成交量相对20日均量比",
    "list_days": "上市天数",
    "turnover_rank_industry": "行业内换手率分位数",
    "amount_rank_market": "全市场成交额分位数",
    "ret5_vs_hs300": "5日相对沪深300超额收益",
    "ret10_vs_hs300": "10日相对沪深300超额收益",
    "ret20_vs_industry": "20日相对行业超额收益",
    "stock_rank_in_industry": "当日个股在行业内涨跌幅分位",
    "industry_rank_5d": "行业短期强弱分位",
    "industry_rank_20d": "行业中期强弱分位",
    "roe": "净资产收益率",
    "revenue_yoy": "营收同比增速",
    "profit_yoy": "利润同比增速",
    "industry_roe_percentile": "行业内ROE分位数",
    "up_limit_count": "全市场涨停家数",
    "down_limit_count": "全市场跌停家数",
    "broken_limit_rate": "炸板率",
    "consecutive_limit_height": "近期连板高度",
    "market_turnover": "全市场成交额",
    "hs300_ret_1": "沪深300单日收益率",
    "cyb_ret_1": "创业板指单日收益率",
    "market_volatility_5": "市场5日波动率",
    "sector_hotness_top1": "最强板块涨幅",
    "sector_hotness_top3_mean": "前三强板块平均涨幅",
    "risk_on_flag": "市场风险偏好打开标记",
    "risk_off_flag": "市场风险偏好收缩标记",
    "label_win_3": "未来3日是否盈利",
    "label_win_5": "未来5日是否盈利",
    "label_win_10": "未来10日是否盈利",
    "label_win_20": "未来20日是否盈利",
    "label_win_40": "未来40日是否盈利",
    "label_ret_3": "未来3日收益率",
    "label_ret_5": "未来5日收益率",
    "label_ret_10": "未来10日收益率",
    "label_ret_20": "未来20日收益率",
    "label_ret_40": "未来40日收益率",
    "label_maxdd_3": "未来3日最大回撤",
    "label_maxdd_5": "未来5日最大回撤",
    "label_maxdd_10": "未来10日最大回撤",
    "label_maxdd_20": "未来20日最大回撤",
    "label_maxdd_40": "未来40日最大回撤",
    "label_rank_score": "基于5日alpha的横截面分位标签",
}


@dataclass(slots=True)
class MultiInputSample:
    x_seq: torch.Tensor
    x_tab: torch.Tensor
    x_event: torch.Tensor
    x_mkt: torch.Tensor
    x_company_ids: dict[str, torch.Tensor]
    x_company_profile: torch.Tensor
    neighbor_symbol_ids: torch.Tensor
    neighbor_scores: torch.Tensor
    y: dict[str, torch.Tensor]
    meta: dict[str, object]


@dataclass(slots=True)
class MultiInputInferenceSample:
    x_seq: torch.Tensor
    x_tab: torch.Tensor
    x_event: torch.Tensor
    x_mkt: torch.Tensor
    x_company_ids: dict[str, torch.Tensor]
    x_company_profile: torch.Tensor
    neighbor_symbol_ids: torch.Tensor
    neighbor_scores: torch.Tensor
    meta: dict[str, object]


class MultiInputTrainingDataset(Dataset[MultiInputSample]):
    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        seq_length: int = 20,
        seq_columns: Sequence[str] | None = None,
        tab_columns: Sequence[str] | None = None,
        event_columns: Sequence[str] | None = None,
        mkt_columns: Sequence[str] | None = None,
        company_id_columns: Sequence[str] | None = None,
        company_profile_columns: Sequence[str] | None = None,
        label_groups: dict[str, Sequence[str]] | None = None,
        fillna_value: float = 0.0,
        drop_incomplete_sequence: bool = True,
        neighbor_topk: int = 10,
        profile_scaling: str = "robust",
    ):
        if isinstance(source, pd.DataFrame):
            df = source.copy()
        else:
            df = pd.read_parquet(source)
        self.df = df.copy()
        self.df["trade_date"] = pd.to_datetime(self.df["trade_date"])
        self.df = self.df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        self.df = self._augment_aligned_features(self.df)

        self.seq_length = int(seq_length)
        self.fillna_value = float(fillna_value)
        self.neighbor_topk = int(neighbor_topk)
        self.profile_scaling = profile_scaling

        self.seq_columns = self._existing_columns(seq_columns or DEFAULT_SEQ_COLUMNS)
        self.tab_columns = self._existing_columns(tab_columns or DEFAULT_TAB_COLUMNS)
        requested_event_columns = list(event_columns or DEFAULT_EVENT_COLUMNS)
        self.event_source_columns = self._existing_columns(requested_event_columns)
        self.event_vector_mode = False
        if self.event_source_columns:
            self.event_columns = list(self.event_source_columns)
        elif EVENT_EMBEDDING_COLUMN in self.df.columns:
            self.event_vector_mode = True
            self.event_columns = list(requested_event_columns)
        else:
            self.event_columns = []
        self.mkt_columns = self._existing_columns(mkt_columns or DEFAULT_MKT_COLUMNS)
        self.company_id_columns = self._existing_columns(company_id_columns or DEFAULT_COMPANY_ID_COLUMNS)
        self.company_profile_columns = self._existing_columns(company_profile_columns or DEFAULT_COMPANY_PROFILE_COLUMNS)

        raw_label_groups = label_groups or DEFAULT_LABEL_GROUPS
        self.label_groups = {
            head_name: [column for column in columns if column in self.df.columns]
            for head_name, columns in raw_label_groups.items()
        }

        numeric_columns = list(
            {
                *self.seq_columns,
                *self.tab_columns,
                *self.event_source_columns,
                *self.mkt_columns,
                *self.company_profile_columns,
                *(column for columns in self.label_groups.values() for column in columns),
            }
        )
        if numeric_columns:
            self.df[numeric_columns] = self.df[numeric_columns].apply(pd.to_numeric, errors="coerce")

        for column in self.company_id_columns:
            self.df[column] = pd.to_numeric(self.df.get(column), errors="coerce").fillna(0).astype(int)

        self.profile_scaler_stats = self._build_profile_scaler_stats(self.company_profile_columns)
        self._row_positions: list[tuple[list[int], int, int]] = []
        grouped = self.df.groupby("symbol", sort=False)
        for _, group in grouped:
            indices = group.index.to_list()
            for offset, row_idx in enumerate(indices):
                if drop_incomplete_sequence and offset + 1 < self.seq_length:
                    continue
                if not self._has_complete_labels(row_idx):
                    continue
                self._row_positions.append((indices, offset, row_idx))

    def _augment_aligned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        frame = df.copy()

        def numeric_series(column_name: str) -> pd.Series:
            if column_name in frame.columns:
                return pd.to_numeric(frame[column_name], errors="coerce")
            return pd.Series(np.nan, index=frame.index, dtype=float)

        numeric_candidates = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover_rate",
            "ret_5",
            "ret_20",
            "volatility_5",
            "volatility_20",
            "up_limit_count",
            "down_limit_count",
            "sector_hotness_top1",
            "sector_hotness_top3_mean",
            "risk_on_flag",
            "risk_off_flag",
        ]
        existing_numeric = [column for column in numeric_candidates if column in frame.columns]
        if existing_numeric:
            frame[existing_numeric] = frame[existing_numeric].apply(pd.to_numeric, errors="coerce")

        close_price = numeric_series("close")
        open_price = numeric_series("open")
        high_price = numeric_series("high")
        low_price = numeric_series("low")
        amount = numeric_series("amount")
        volume = numeric_series("volume")
        turnover_rate = numeric_series("turnover_rate")
        prev_close = close_price.groupby(frame["symbol"]).shift(1)

        close_base = close_price.replace(0, np.nan)
        open_base = open_price.replace(0, np.nan)
        prev_base = prev_close.replace(0, np.nan)
        max_oc = pd.concat([open_price, close_price], axis=1).max(axis=1)
        min_oc = pd.concat([open_price, close_price], axis=1).min(axis=1)

        frame["intraday_range"] = (high_price - low_price) / close_base
        frame["candle_body"] = (close_price - open_price) / open_base
        frame["upper_shadow"] = (high_price - max_oc) / open_base
        frame["lower_shadow"] = (min_oc - low_price) / open_base
        frame["gap_open_to_prev_close"] = (open_price - prev_close) / prev_base
        frame["close_to_prev_close"] = (close_price - prev_close) / prev_base
        frame["high_to_prev_close"] = (high_price - prev_close) / prev_base
        frame["low_to_prev_close"] = (low_price - prev_close) / prev_base
        frame["amount_log1p"] = np.log1p(amount.clip(lower=0))
        frame["volume_log1p"] = np.log1p(volume.clip(lower=0))
        frame["turnover_rate_delta"] = turnover_rate.groupby(frame["symbol"]).diff()
        frame["ret_spread_5_20"] = numeric_series("ret_5") - numeric_series("ret_20")
        frame["volatility_ratio_5_20"] = numeric_series("volatility_5") / numeric_series("volatility_20").replace(0, np.nan)

        up_limit = numeric_series("up_limit_count").fillna(0)
        down_limit = numeric_series("down_limit_count").fillna(0)
        top1 = numeric_series("sector_hotness_top1").fillna(0)
        top3_mean = numeric_series("sector_hotness_top3_mean").fillna(0)
        risk_on = numeric_series("risk_on_flag").fillna(0)
        risk_off = numeric_series("risk_off_flag").fillna(0)
        frame["limit_count_spread"] = up_limit - down_limit
        frame["limit_count_ratio"] = (up_limit - down_limit) / (up_limit + down_limit + 1.0)
        frame["sector_hotness_spread"] = top1 - top3_mean
        frame["market_regime_score"] = risk_on - risk_off
        return frame

    def _existing_columns(self, requested: Sequence[str]) -> list[str]:
        return [column for column in requested if column in self.df.columns]

    def _build_profile_scaler_stats(self, columns: Sequence[str]) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        for column in columns:
            series = pd.to_numeric(self.df[column], errors="coerce")
            median = float(series.median()) if not series.dropna().empty else 0.0
            q1 = float(series.quantile(0.25)) if not series.dropna().empty else 0.0
            q3 = float(series.quantile(0.75)) if not series.dropna().empty else 1.0
            iqr = q3 - q1
            mean = float(series.mean()) if not series.dropna().empty else 0.0
            std = float(series.std(ddof=0)) if not series.dropna().empty else 1.0
            stats[column] = {
                "median": median,
                "iqr": iqr if np.isfinite(iqr) and iqr != 0 else 1.0,
                "mean": mean,
                "std": std if np.isfinite(std) and std != 0 else 1.0,
            }
        return stats

    def _scale_company_profile_row(self, row: pd.Series) -> np.ndarray:
        values: list[float] = []
        for column in self.company_profile_columns:
            raw_value = pd.to_numeric(row.get(column), errors="coerce")
            stats = self.profile_scaler_stats.get(column, {"median": 0.0, "iqr": 1.0, "mean": 0.0, "std": 1.0})
            if pd.isna(raw_value):
                raw_value = stats["median"]
            if self.profile_scaling == "zscore":
                scaled = (float(raw_value) - stats["mean"]) / (stats["std"] or 1.0)
            else:
                scaled = (float(raw_value) - stats["median"]) / (stats["iqr"] or 1.0)
            values.append(float(scaled))
        return np.asarray(values, dtype=np.float32)

    def _parse_neighbor_array(self, value: Any, dtype: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if pd.isna(value):
            return []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                return []
        return []

    def _parse_event_vector(self, value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            vector = value.astype(np.float32, copy=False).flatten()
        elif isinstance(value, list):
            vector = np.asarray(value, dtype=np.float32).flatten()
        elif pd.isna(value):
            vector = np.zeros(0, dtype=np.float32)
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
                vector = np.asarray(parsed if isinstance(parsed, list) else [], dtype=np.float32).flatten()
            except json.JSONDecodeError:
                vector = np.zeros(0, dtype=np.float32)
        else:
            vector = np.zeros(0, dtype=np.float32)

        output = np.full(len(self.event_columns), self.fillna_value, dtype=np.float32)
        if vector.size:
            output[: min(len(output), vector.size)] = vector[: len(output)]
        return output

    def _build_event_tensor(self, row: pd.Series) -> torch.Tensor:
        if self.event_vector_mode:
            return torch.tensor(self._parse_event_vector(row.get(EVENT_EMBEDDING_COLUMN)), dtype=torch.float32)
        if not self.event_source_columns:
            return torch.zeros(len(self.event_columns), dtype=torch.float32)
        values = (
            pd.to_numeric(row[self.event_source_columns], errors="coerce")
            .fillna(self.fillna_value)
            .to_numpy(dtype=np.float32)
        )
        return torch.tensor(values, dtype=torch.float32)

    def _event_column_descriptions(self) -> list[dict[str, str]]:
        if self.event_vector_mode:
            return [
                {"name": column, "description": "One dimension of the shared 256-dim daily news event embedding."}
                for column in self.event_columns
            ]
        return [{"name": column, "description": COLUMN_DESCRIPTIONS.get(column, column)} for column in self.event_columns]

    def _build_neighbor_tensors(self, row: pd.Series) -> tuple[torch.Tensor, torch.Tensor]:
        neighbor_ids = self._parse_neighbor_array(row.get("neighbor_symbol_ids"), int)
        neighbor_scores = self._parse_neighbor_array(row.get("neighbor_scores"), float)
        ids = np.zeros(self.neighbor_topk, dtype=np.int64)
        scores = np.zeros(self.neighbor_topk, dtype=np.float32)
        for idx, value in enumerate(neighbor_ids[: self.neighbor_topk]):
            numeric = pd.to_numeric(value, errors="coerce")
            ids[idx] = int(0 if pd.isna(numeric) else numeric)
        for idx, value in enumerate(neighbor_scores[: self.neighbor_topk]):
            numeric = pd.to_numeric(value, errors="coerce")
            scores[idx] = float(0.0 if pd.isna(numeric) else numeric)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(scores, dtype=torch.float32)

    def _has_complete_labels(self, row_idx: int) -> bool:
        row = self.df.loc[row_idx]
        for columns in self.label_groups.values():
            if columns and row[columns].isna().any():
                return False
        return True

    def __len__(self) -> int:
        return len(self._row_positions)

    def __getitem__(self, idx: int) -> MultiInputSample:
        indices, offset, row_idx = self._row_positions[idx]
        window_indices = indices[max(0, offset - self.seq_length + 1) : offset + 1]
        window = self.df.loc[window_indices, self.seq_columns].copy()
        window = window.apply(pd.to_numeric, errors="coerce").fillna(self.fillna_value)
        if len(window) < self.seq_length:
            pad = pd.DataFrame(
                np.full((self.seq_length - len(window), len(self.seq_columns)), self.fillna_value),
                columns=self.seq_columns,
            )
            window = pd.concat([pad, window], ignore_index=True)

        row = self.df.loc[row_idx]
        x_seq = torch.tensor(window.to_numpy(dtype=np.float32), dtype=torch.float32)
        x_tab = torch.tensor(
            pd.to_numeric(row[self.tab_columns], errors="coerce").fillna(self.fillna_value).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        x_event = self._build_event_tensor(row)
        x_mkt = torch.tensor(
            pd.to_numeric(row[self.mkt_columns], errors="coerce").fillna(self.fillna_value).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        x_company_ids = {
            column: torch.tensor(int(row.get(column, 0) or 0), dtype=torch.long) for column in self.company_id_columns
        }
        company_profile_values = self._scale_company_profile_row(row)
        x_company_profile = torch.tensor(company_profile_values, dtype=torch.float32)
        neighbor_symbol_ids, neighbor_scores = self._build_neighbor_tensors(row)

        y = {
            head_name: torch.tensor(row[columns].to_numpy(dtype=np.float32), dtype=torch.float32)
            for head_name, columns in self.label_groups.items()
            if columns
        }
        meta = {column: row.get(column) for column in META_COLUMNS}
        meta["trade_date"] = str(meta.get("trade_date"))
        return MultiInputSample(
            x_seq=x_seq,
            x_tab=x_tab,
            x_event=x_event,
            x_mkt=x_mkt,
            x_company_ids=x_company_ids,
            x_company_profile=x_company_profile,
            neighbor_symbol_ids=neighbor_symbol_ids,
            neighbor_scores=neighbor_scores,
            y=y,
            meta=meta,
        )

    @property
    def input_dims(self) -> dict[str, int]:
        return {
            "seq_length": self.seq_length,
            "f_seq": len(self.seq_columns),
            "f_tab": len(self.tab_columns),
            "f_event": len(self.event_columns),
            "f_mkt": len(self.mkt_columns),
            "f_company_profile": len(self.company_profile_columns),
            "neighbor_topk": self.neighbor_topk,
        }

    @property
    def head_dims(self) -> dict[str, int]:
        return {head_name: len(columns) for head_name, columns in self.label_groups.items()}

    @property
    def vocab_sizes(self) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for column in self.company_id_columns:
            sizes[column] = int(pd.to_numeric(self.df[column], errors="coerce").fillna(0).max()) + 1
        return sizes

    def describe_inputs(self) -> dict[str, dict[str, Any]]:
        return {
            "X_seq": {
                "shape": (self.seq_length, len(self.seq_columns)),
                "description": INPUT_GROUP_DESCRIPTIONS["X_seq"],
                "columns": [{"name": column, "description": COLUMN_DESCRIPTIONS.get(column, column)} for column in self.seq_columns],
            },
            "X_tab": {
                "shape": (len(self.tab_columns),),
                "description": INPUT_GROUP_DESCRIPTIONS["X_tab"],
                "columns": [{"name": column, "description": COLUMN_DESCRIPTIONS.get(column, column)} for column in self.tab_columns],
            },
            "X_event": {
                "shape": (len(self.event_columns),),
                "description": INPUT_GROUP_DESCRIPTIONS["X_event"],
                "columns": self._event_column_descriptions(),
            },
            "X_mkt": {
                "shape": (len(self.mkt_columns),),
                "description": INPUT_GROUP_DESCRIPTIONS["X_mkt"],
                "columns": [{"name": column, "description": COLUMN_DESCRIPTIONS.get(column, column)} for column in self.mkt_columns],
            },
            "X_company_ids": {
                "shape": (len(self.company_id_columns),),
                "description": INPUT_GROUP_DESCRIPTIONS["X_company_ids"],
                "columns": [
                    {"name": column, "description": COLUMN_DESCRIPTIONS.get(column, column)} for column in self.company_id_columns
                ],
            },
            "X_company_profile": {
                "shape": (len(self.company_profile_columns),),
                "description": INPUT_GROUP_DESCRIPTIONS["X_company_profile"],
                "columns": [
                    {"name": column, "description": COLUMN_DESCRIPTIONS.get(column, column)}
                    for column in self.company_profile_columns
                ],
            },
            "neighbors": {
                "shape": (self.neighbor_topk,),
                "description": INPUT_GROUP_DESCRIPTIONS["neighbors"],
                "columns": [
                    {"name": "neighbor_symbol_ids", "description": COLUMN_DESCRIPTIONS["neighbor_symbol_ids"]},
                    {"name": "neighbor_scores", "description": COLUMN_DESCRIPTIONS["neighbor_scores"]},
                ],
            },
        }

    def describe_outputs(self) -> dict[str, dict[str, Any]]:
        return {
            head_name: {
                "shape": (len(columns),),
                "description": LABEL_GROUP_DESCRIPTIONS.get(head_name, head_name),
                "columns": [{"name": column, "description": COLUMN_DESCRIPTIONS.get(column, column)} for column in columns],
            }
            for head_name, columns in self.label_groups.items()
            if columns
        }

    def describe_sample(self) -> dict[str, Any]:
        return {
            "inputs": self.describe_inputs(),
            "outputs": self.describe_outputs(),
            "meta_columns": META_COLUMNS,
            "vocab_sizes": self.vocab_sizes,
        }


class MultiInputInferenceDataset(MultiInputTrainingDataset):
    def __init__(
        self,
        source: str | Path | pd.DataFrame,
        seq_length: int = 20,
        seq_columns: Sequence[str] | None = None,
        tab_columns: Sequence[str] | None = None,
        event_columns: Sequence[str] | None = None,
        mkt_columns: Sequence[str] | None = None,
        company_id_columns: Sequence[str] | None = None,
        company_profile_columns: Sequence[str] | None = None,
        fillna_value: float = 0.0,
        drop_incomplete_sequence: bool = True,
        neighbor_topk: int = 10,
        profile_scaling: str = "robust",
    ):
        super().__init__(
            source=source,
            seq_length=seq_length,
            seq_columns=seq_columns,
            tab_columns=tab_columns,
            event_columns=event_columns,
            mkt_columns=mkt_columns,
            company_id_columns=company_id_columns,
            company_profile_columns=company_profile_columns,
            label_groups={},
            fillna_value=fillna_value,
            drop_incomplete_sequence=drop_incomplete_sequence,
            neighbor_topk=neighbor_topk,
            profile_scaling=profile_scaling,
        )

    def _has_complete_labels(self, row_idx: int) -> bool:
        return True

    def __getitem__(self, idx: int) -> MultiInputInferenceSample:
        indices, offset, row_idx = self._row_positions[idx]
        window_indices = indices[max(0, offset - self.seq_length + 1) : offset + 1]
        window = self.df.loc[window_indices, self.seq_columns].copy()
        window = window.apply(pd.to_numeric, errors="coerce").fillna(self.fillna_value)
        if len(window) < self.seq_length:
            pad = pd.DataFrame(
                np.full((self.seq_length - len(window), len(self.seq_columns)), self.fillna_value),
                columns=self.seq_columns,
            )
            window = pd.concat([pad, window], ignore_index=True)

        row = self.df.loc[row_idx]
        x_seq = torch.tensor(window.to_numpy(dtype=np.float32), dtype=torch.float32)
        x_tab = torch.tensor(
            pd.to_numeric(row[self.tab_columns], errors="coerce").fillna(self.fillna_value).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        x_event = self._build_event_tensor(row)
        x_mkt = torch.tensor(
            pd.to_numeric(row[self.mkt_columns], errors="coerce").fillna(self.fillna_value).to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
        x_company_ids = {
            column: torch.tensor(int(row.get(column, 0) or 0), dtype=torch.long) for column in self.company_id_columns
        }
        x_company_profile = torch.tensor(self._scale_company_profile_row(row), dtype=torch.float32)
        neighbor_symbol_ids, neighbor_scores = self._build_neighbor_tensors(row)
        meta = {column: row.get(column) for column in META_COLUMNS}
        meta["trade_date"] = str(meta.get("trade_date"))
        return MultiInputInferenceSample(
            x_seq=x_seq,
            x_tab=x_tab,
            x_event=x_event,
            x_mkt=x_mkt,
            x_company_ids=x_company_ids,
            x_company_profile=x_company_profile,
            neighbor_symbol_ids=neighbor_symbol_ids,
            neighbor_scores=neighbor_scores,
            meta=meta,
        )

    def describe_sample(self) -> dict[str, Any]:
        return {
            "inputs": self.describe_inputs(),
            "meta_columns": META_COLUMNS,
            "vocab_sizes": self.vocab_sizes,
        }


def multi_input_collate_fn(batch: Sequence[MultiInputSample]) -> dict[str, Any]:
    x_seq = torch.stack([item.x_seq for item in batch], dim=0)
    x_tab = torch.stack([item.x_tab for item in batch], dim=0)
    x_event = torch.stack([item.x_event for item in batch], dim=0)
    x_mkt = torch.stack([item.x_mkt for item in batch], dim=0)
    x_company_profile = torch.stack([item.x_company_profile for item in batch], dim=0)
    x_company_ids = {
        key: torch.stack([item.x_company_ids[key] for item in batch], dim=0)
        for key in batch[0].x_company_ids.keys()
    }
    neighbors = {
        "neighbor_symbol_ids": torch.stack([item.neighbor_symbol_ids for item in batch], dim=0),
        "neighbor_scores": torch.stack([item.neighbor_scores for item in batch], dim=0),
    }
    y = {
        head_name: torch.stack([item.y[head_name] for item in batch], dim=0)
        for head_name in batch[0].y.keys()
    }
    meta = [item.meta for item in batch]
    return {
        "X_seq": x_seq,
        "X_tab": x_tab,
        "X_event": x_event,
        "X_mkt": x_mkt,
        "X_company_ids": x_company_ids,
        "X_company_profile": x_company_profile,
        "neighbors": neighbors,
        "y": y,
        "meta": meta,
    }
