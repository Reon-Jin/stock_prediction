from __future__ import annotations

import pandas as pd


def build_fundamental_features(
    daily_bars: pd.DataFrame,
    financial_snapshot: pd.DataFrame,
    securities: pd.DataFrame,
) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame()
    base = daily_bars[["symbol", "trade_date"]].sort_values(["symbol", "trade_date"]).copy()
    snap = financial_snapshot.copy()
    if snap.empty:
        for col in [
            "roe",
            "revenue_yoy",
            "profit_yoy",
            "industry_roe_percentile",
        ]:
            base[col] = None
        return base
    snap["asof_date"] = pd.to_datetime(snap["asof_date"])
    base["trade_date"] = pd.to_datetime(base["trade_date"])
    snap = snap.sort_values(["symbol", "asof_date"])
    merged = pd.merge_asof(
        base.sort_values("trade_date"),
        snap.sort_values("asof_date"),
        by="symbol",
        left_on="trade_date",
        right_on="asof_date",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.merge(securities[["symbol", "industry"]], on="symbol", how="left")
    merged["industry_roe_percentile"] = merged.groupby(["trade_date", "industry"])["roe"].rank(pct=True)
    return merged[
        [
            "symbol",
            "trade_date",
            "roe",
            "revenue_yoy",
            "profit_yoy",
            "industry_roe_percentile",
        ]
    ]
