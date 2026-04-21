from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


COMPANY_PROFILE_COLUMNS = [
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

COMPANY_PROFILE_TABLE_COLUMNS = [
    "symbol",
    "asof_date",
    "industry_name",
    "board",
    *COMPANY_PROFILE_COLUMNS,
    "profile_version",
]


def build_company_id_maps(securities: pd.DataFrame) -> pd.DataFrame:
    if securities.empty:
        return pd.DataFrame(columns=["symbol", "industry_name", "board", "symbol_id", "industry_id", "board_id"])

    df = securities.copy()
    df["symbol"] = df["symbol"].astype(str)
    df["industry_name"] = df.get("industry")
    df["board"] = df.get("board")

    symbol_values = sorted(df["symbol"].dropna().astype(str).unique().tolist())
    symbol_map = {symbol: idx for idx, symbol in enumerate(symbol_values, start=1)}

    industry_values = sorted(
        value for value in df["industry_name"].dropna().astype(str).str.strip().unique().tolist() if value
    )
    industry_map = {value: idx for idx, value in enumerate(industry_values, start=1)}

    board_values = sorted(value for value in df["board"].dropna().astype(str).str.strip().unique().tolist() if value)
    board_map = {value: idx for idx, value in enumerate(board_values, start=1)}

    out = df[["symbol", "industry_name", "board"]].drop_duplicates("symbol").copy()
    out["symbol_id"] = out["symbol"].map(symbol_map).fillna(0).astype(int)
    out["industry_id"] = out["industry_name"].map(industry_map).fillna(0).astype(int)
    out["board_id"] = out["board"].map(board_map).fillna(0).astype(int)
    return out


def _rolling_beta(group: pd.DataFrame, benchmark_col: str, window: int) -> pd.Series:
    stock_ret = pd.to_numeric(group["stock_ret_1"], errors="coerce")
    bench_ret = pd.to_numeric(group[benchmark_col], errors="coerce")
    cov = stock_ret.rolling(window).cov(bench_ret)
    var = bench_ret.rolling(window).var()
    return cov / var.replace(0, np.nan)


def _build_beta_series(df: pd.DataFrame, benchmark_col: str, window: int) -> pd.Series:
    parts: list[pd.Series] = []
    for _, group in df.groupby("symbol", sort=False):
        beta = _rolling_beta(group, benchmark_col=benchmark_col, window=window)
        beta.index = group.index
        parts.append(beta)
    if not parts:
        return pd.Series(index=df.index, dtype=float)
    return pd.concat(parts).reindex(df.index)


def _estimate_market_cap(df: pd.DataFrame) -> pd.Series:
    # TODO: if upstream providers expose reliable total/free-float market cap,
    # replace this turnover-based approximation with the direct market-cap field.
    turnover_rate = pd.to_numeric(df.get("turnover_rate"), errors="coerce")
    amount = pd.to_numeric(df.get("amount"), errors="coerce")
    close = pd.to_numeric(df.get("close"), errors="coerce")
    volume = pd.to_numeric(df.get("volume"), errors="coerce")

    turnover_fraction = turnover_rate / 100.0
    market_cap_from_turnover = amount / turnover_fraction.replace(0, np.nan)
    market_cap_from_trading_value = close * volume
    market_cap_est = market_cap_from_turnover.fillna(market_cap_from_trading_value)
    market_cap_est = market_cap_est.clip(lower=0)
    return np.log1p(market_cap_est)


def _merge_financial_snapshot(base: pd.DataFrame, financial_snapshot: pd.DataFrame) -> pd.DataFrame:
    required_columns = [
        "symbol",
        "asof_date",
        "roe",
        "revenue_yoy",
        "profit_yoy",
        "debt_ratio",
        "gross_margin",
    ]
    if financial_snapshot.empty:
        for column in required_columns[2:]:
            base[column] = np.nan
        return base

    snapshot = financial_snapshot.copy()
    snapshot["asof_date"] = pd.to_datetime(snapshot["asof_date"])
    snapshot = snapshot.sort_values(["symbol", "asof_date"])
    outputs: list[pd.DataFrame] = []
    for symbol, group in base.groupby("symbol", sort=False):
        group = group.sort_values("asof_date").copy()
        symbol_snapshot = snapshot[snapshot["symbol"] == symbol][required_columns].sort_values("asof_date").copy()
        if symbol_snapshot.empty:
            for column in required_columns[2:]:
                group[column] = np.nan
            outputs.append(group)
            continue
        merged = pd.merge_asof(
            group,
            symbol_snapshot,
            by="symbol",
            left_on="asof_date",
            right_on="asof_date",
            direction="backward",
            allow_exact_matches=True,
        )
        outputs.append(merged)
    if not outputs:
        return base
    return pd.concat(outputs, ignore_index=True)


def merge_company_profile_features(base_df: pd.DataFrame, company_profiles: pd.DataFrame) -> pd.DataFrame:
    output = base_df.copy()
    if "trade_date" in output.columns:
        output["trade_date"] = pd.to_datetime(output["trade_date"])
    if company_profiles.empty:
        for column in COMPANY_PROFILE_COLUMNS:
            if column not in output.columns:
                output[column] = np.nan
        if "company_profile_version" not in output.columns:
            output["company_profile_version"] = None
        return output

    profile_df = company_profiles.copy()
    profile_df["asof_date"] = pd.to_datetime(profile_df["asof_date"])
    rename_map = {"asof_date": "trade_date", "profile_version": "company_profile_version"}
    profile_df = profile_df.rename(columns=rename_map)
    merge_columns = ["symbol", "trade_date", "company_profile_version", *COMPANY_PROFILE_COLUMNS]
    for optional in ["industry_name", "board"]:
        if optional in profile_df.columns:
            merge_columns.append(optional)
    merged = output.merge(profile_df[merge_columns], on=["symbol", "trade_date"], how="left", suffixes=("", "_profile"))

    for column in COMPANY_PROFILE_COLUMNS:
        profile_column = f"{column}_profile"
        if profile_column in merged.columns:
            if column in merged.columns:
                merged[column] = merged[column].combine_first(merged[profile_column])
            else:
                merged[column] = merged[profile_column]
            merged = merged.drop(columns=[profile_column])

    if "industry_name" in merged.columns and "industry_sw" in merged.columns:
        merged["industry_sw"] = merged["industry_sw"].combine_first(merged["industry_name"])
        merged = merged.drop(columns=["industry_name"])
    if "board_profile" in merged.columns and "board" in merged.columns:
        merged["board"] = merged["board"].combine_first(merged["board_profile"])
        merged = merged.drop(columns=["board_profile"])
    return merged


def build_company_profiles(
    daily_bars: pd.DataFrame,
    index_bars: pd.DataFrame,
    financial_snapshot: pd.DataFrame,
    securities: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    lookback_days: int = 120,
    profile_version: str = "cp1",
) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame(columns=COMPANY_PROFILE_TABLE_COLUMNS)

    df = daily_bars.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values(["symbol", "trade_date"])
    for column in ["close", "adj_close", "amount", "turnover_rate", "volume"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["stock_ret_1"] = df.groupby("symbol")["adj_close"].pct_change()
    df["ret_20"] = df.groupby("symbol")["adj_close"].pct_change(20)
    df["ret_60"] = df.groupby("symbol")["adj_close"].pct_change(60)
    df["volatility_120"] = df.groupby("symbol")["stock_ret_1"].transform(lambda s: s.rolling(lookback_days).std())
    df["turnover_mean_120"] = df.groupby("symbol")["turnover_rate"].transform(lambda s: s.rolling(lookback_days).mean())
    df["amount_mean_120"] = df.groupby("symbol")["amount"].transform(lambda s: s.rolling(lookback_days).mean())
    df["market_cap_log"] = _estimate_market_cap(df)

    index_df = index_bars.copy()
    benchmark = index_df[index_df["index_code"] == "000300.SH"].copy()
    if benchmark.empty:
        df["beta_120"] = np.nan
    else:
        benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
        benchmark = benchmark.sort_values("trade_date")
        benchmark["bench_ret_1"] = pd.to_numeric(benchmark["close"], errors="coerce").pct_change()
        df = df.merge(benchmark[["trade_date", "bench_ret_1"]], on="trade_date", how="left")
        df["beta_120"] = _build_beta_series(df, benchmark_col="bench_ret_1", window=lookback_days)

    base = df[
        [
            "symbol",
            "trade_date",
            "market_cap_log",
            "volatility_120",
            "beta_120",
            "turnover_mean_120",
            "amount_mean_120",
            "ret_20",
            "ret_60",
        ]
    ].copy()
    base = base.rename(columns={"trade_date": "asof_date"})
    base = _merge_financial_snapshot(base, financial_snapshot)

    security_cols = securities.copy()
    security_cols["industry_name"] = security_cols.get("industry")
    security_cols = security_cols[["symbol", "industry_name", "board"]].drop_duplicates("symbol")
    base = base.merge(security_cols, on="symbol", how="left")

    if start:
        base = base[base["asof_date"] >= pd.Timestamp(start)]
    if end:
        base = base[base["asof_date"] <= pd.Timestamp(end)]

    for column in COMPANY_PROFILE_COLUMNS:
        if column not in base.columns:
            base[column] = np.nan
        base[column] = pd.to_numeric(base[column], errors="coerce")

    base["profile_version"] = profile_version
    base["asof_date"] = pd.to_datetime(base["asof_date"]).dt.date
    return base[COMPANY_PROFILE_TABLE_COLUMNS].copy()
