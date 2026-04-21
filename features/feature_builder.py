from __future__ import annotations

import pandas as pd

from features.company_profile_builder import (
    COMPANY_PROFILE_COLUMNS,
    build_company_id_maps,
    build_company_profiles,
    merge_company_profile_features,
)
from features.fundamental_features import build_fundamental_features
from features.market_features import build_market_features
from features.price_features import build_price_features
from features.relative_features import build_relative_features
from features.volume_features import build_volume_features


def _normalize_trade_date_column(frame: pd.DataFrame, column: str = "trade_date") -> pd.DataFrame:
    normalized = frame.copy()
    if column in normalized.columns:
        normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
    return normalized


def _company_profiles_usable(company_profiles: pd.DataFrame | None) -> bool:
    if company_profiles is None or company_profiles.empty:
        return False
    available_columns = [column for column in COMPANY_PROFILE_COLUMNS if column in company_profiles.columns]
    if not available_columns:
        return False
    numeric = company_profiles.loc[:, available_columns].apply(pd.to_numeric, errors="coerce")
    if int(numeric.notna().sum().sum()) == 0:
        return False
    core_columns = [column for column in ["volatility_120", "beta_120", "turnover_mean_120", "amount_mean_120"] if column in numeric.columns]
    if core_columns and float(numeric.loc[:, core_columns].abs().sum().sum()) == 0.0:
        return False
    return True


def build_all_features(
    daily_bars: pd.DataFrame,
    securities: pd.DataFrame,
    index_bars: pd.DataFrame,
    sector_daily: pd.DataFrame,
    financial_snapshot: pd.DataFrame,
    event_features_daily: pd.DataFrame,
    company_profiles: pd.DataFrame | None = None,
    identity_securities: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = daily_bars[
        ["symbol", "trade_date", "open", "high", "low", "close", "adj_close", "volume", "amount", "turnover_rate", "pct_chg"]
    ].copy()
    base = _normalize_trade_date_column(base)
    feature_frames = [
        build_price_features(daily_bars),
        build_volume_features(daily_bars, securities),
        build_relative_features(daily_bars, index_bars, securities, sector_daily),
        build_fundamental_features(daily_bars, financial_snapshot, securities),
    ]
    merged = base.copy()
    for frame in feature_frames:
        if frame is None or frame.empty:
            continue
        frame = _normalize_trade_date_column(frame)
        merged = merged.merge(frame, on=["symbol", "trade_date"], how="left")
    if event_features_daily is not None and not event_features_daily.empty:
        event_frame = _normalize_trade_date_column(event_features_daily.copy())
        event_merge_columns = [column for column in event_frame.columns if column != "symbol"]
        merged = merged.merge(event_frame[event_merge_columns], on="trade_date", how="left")
    market = build_market_features(daily_bars, index_bars, sector_daily)
    market = _normalize_trade_date_column(market)
    merged = merged.merge(market, on="trade_date", how="left")
    security_cols = securities[["symbol", "name", "industry", "board", "list_date", "is_st"]].copy()
    security_cols = security_cols.rename(columns={"industry": "industry_sw"})
    merged = merged.merge(security_cols, on="symbol", how="left")
    first_trade_dates = (
        daily_bars[["symbol", "trade_date"]]
        .copy()
        .assign(trade_date=lambda df: pd.to_datetime(df["trade_date"], errors="coerce"))
        .dropna(subset=["trade_date"])
        .groupby("symbol", as_index=False)["trade_date"]
        .min()
        .rename(columns={"trade_date": "first_trade_date"})
    )
    merged = merged.merge(first_trade_dates, on="symbol", how="left")
    identity_frame = build_company_id_maps(identity_securities if identity_securities is not None else securities)
    merged = merged.merge(identity_frame[["symbol", "symbol_id", "industry_id", "board_id"]], on="symbol", how="left")
    company_profile_frame = company_profiles if _company_profiles_usable(company_profiles) else pd.DataFrame()
    if company_profile_frame.empty:
        normalized_dates = pd.to_datetime(daily_bars["trade_date"], errors="coerce").dropna()
        profile_start = normalized_dates.min().strftime("%Y-%m-%d") if not normalized_dates.empty else None
        profile_end = normalized_dates.max().strftime("%Y-%m-%d") if not normalized_dates.empty else None
        company_profile_frame = build_company_profiles(
            daily_bars=daily_bars,
            index_bars=index_bars,
            financial_snapshot=financial_snapshot,
            securities=securities,
            start=profile_start,
            end=profile_end,
        )
    merged = merge_company_profile_features(merged, company_profile_frame)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"])
    merged["list_date"] = pd.to_datetime(merged["list_date"], errors="coerce")
    merged["list_date"] = merged["list_date"].fillna(merged["first_trade_date"])
    merged["list_days"] = (merged["trade_date"] - merged["list_date"]).dt.days
    merged = merged.drop(columns=["first_trade_date"])
    return merged
