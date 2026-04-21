from __future__ import annotations

import json

import numpy as np
import pandas as pd

from datasets.pytorch_dataset import (
    DEFAULT_COMPANY_ID_COLUMNS,
    DEFAULT_COMPANY_PROFILE_COLUMNS,
    DEFAULT_EVENT_COLUMNS,
    DEFAULT_MKT_COLUMNS,
    DEFAULT_SEQ_COLUMNS,
    DEFAULT_TAB_COLUMNS,
    EVENT_EMBEDDING_COLUMN,
    EVENT_EMBEDDING_DIM,
)


RANK_LIKE_COLUMNS = {
    "turnover_rank_industry",
    "amount_rank_market",
    "stock_rank_in_industry",
    "industry_rank_5d",
    "industry_rank_20d",
    "industry_roe_percentile",
}

ZERO_DEFAULT_COLUMNS = {
    *DEFAULT_EVENT_COLUMNS,
    *DEFAULT_MKT_COLUMNS,
    "ret5_vs_hs300",
    "ret10_vs_hs300",
    "ret20_vs_industry",
    "shrink_volume_flag",
    "surge_volume_flag",
}

FLAG_COLUMNS = {
    "shrink_volume_flag",
    "surge_volume_flag",
    "risk_on_flag",
    "risk_off_flag",
}

BOUNDED_COLUMNS = {
    "boll_pos",
    "broken_limit_rate",
    "industry_roe_percentile",
    *RANK_LIKE_COLUMNS,
}


def _fill_by_trade_date(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not columns:
        return frame
    trade_date_series = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame[column] = frame[column].fillna(frame.groupby(trade_date_series)[column].transform("median"))
        frame[column] = frame[column].fillna(frame[column].median())
        frame[column] = frame[column].fillna(0.0)
    return frame


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _winsorize(frame: pd.DataFrame, columns: list[str], lower: float = 0.005, upper: float = 0.995) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.dropna().empty:
            continue
        lo = float(series.quantile(lower))
        hi = float(series.quantile(upper))
        if not np.isfinite(lo) or not np.isfinite(hi):
            continue
        frame[column] = series.clip(lower=lo, upper=hi)
    return frame


def finalize_training_samples(samples: pd.DataFrame) -> pd.DataFrame:
    if samples.empty:
        return samples

    output = samples.copy()
    output["trade_date"] = pd.to_datetime(output["trade_date"], errors="coerce")
    output = output.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    for column in ["name", "industry_sw", "board", "sample_status", "data_version", "feature_version", "label_version"]:
        if column in output.columns:
            output[column] = output[column].fillna("Unknown")

    seq_columns = [column for column in DEFAULT_SEQ_COLUMNS if column in output.columns]
    company_profile_columns = [column for column in DEFAULT_COMPANY_PROFILE_COLUMNS if column in output.columns]
    tab_columns = [column for column in DEFAULT_TAB_COLUMNS if column in output.columns]
    event_columns = [column for column in DEFAULT_EVENT_COLUMNS if column in output.columns]
    mkt_columns = [column for column in DEFAULT_MKT_COLUMNS if column in output.columns]
    id_columns = [column for column in DEFAULT_COMPANY_ID_COLUMNS if column in output.columns]
    label_columns = [column for column in output.columns if column.startswith("label_")]
    numeric_columns = [*seq_columns, *tab_columns, *company_profile_columns, *event_columns, *mkt_columns, *label_columns]

    output = _coerce_numeric(output, numeric_columns)

    if seq_columns or company_profile_columns:
        grouped = output.groupby("symbol", group_keys=False)
        for column in [*seq_columns, *company_profile_columns]:
            output[column] = grouped[column].transform(lambda s: s.ffill().bfill())

    rank_columns = [column for column in tab_columns if column in RANK_LIKE_COLUMNS]
    zero_default_columns = [column for column in output.columns if column in ZERO_DEFAULT_COLUMNS]
    trade_date_median_columns = [
        column
        for column in [*tab_columns, *company_profile_columns]
        if column not in rank_columns and column not in zero_default_columns
    ]

    for column in rank_columns:
        output[column] = output[column].fillna(0.5).clip(lower=0.0, upper=1.0)
    for column in zero_default_columns:
        output[column] = output[column].fillna(0.0)

    for column in [name for name in FLAG_COLUMNS if name in output.columns]:
        output[column] = output[column].fillna(0.0).clip(lower=0.0, upper=1.0)

    for column in [name for name in BOUNDED_COLUMNS if name in output.columns]:
        output[column] = output[column].fillna(0.5).clip(lower=0.0, upper=1.0)

    if EVENT_EMBEDDING_COLUMN in output.columns:
        zero_event = [0.0] * EVENT_EMBEDDING_DIM

        def normalize_event_embedding(value):
            if isinstance(value, list):
                vector = value
            elif isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    vector = parsed if isinstance(parsed, list) else []
                except Exception:
                    vector = []
            else:
                vector = []
            normalized = zero_event.copy()
            for idx, item in enumerate(vector[:EVENT_EMBEDDING_DIM]):
                numeric = pd.to_numeric(item, errors="coerce")
                normalized[idx] = 0.0 if pd.isna(numeric) else float(numeric)
            return normalized

        output[EVENT_EMBEDDING_COLUMN] = output[EVENT_EMBEDDING_COLUMN].apply(normalize_event_embedding)

    output = _fill_by_trade_date(output, trade_date_median_columns)
    clip_candidates = [
        column
        for column in [*seq_columns, *tab_columns, *company_profile_columns, *mkt_columns]
        if column not in rank_columns and column not in zero_default_columns and column not in FLAG_COLUMNS
    ]
    output = _winsorize(output, clip_candidates)

    for column in id_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0).astype(int)

    if "list_days" in output.columns:
        output["list_days"] = pd.to_numeric(output["list_days"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    return output
