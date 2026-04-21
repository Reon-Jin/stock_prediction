from __future__ import annotations

import pandas as pd


def build_volume_features(daily_bars: pd.DataFrame, securities: pd.DataFrame) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame()
    df = daily_bars.sort_values(["symbol", "trade_date"]).copy()
    industry_map = securities[["symbol", "industry"]] if "industry" in securities.columns else pd.DataFrame(columns=["symbol", "industry"])
    df = df.merge(industry_map, on="symbol", how="left")
    group = df.groupby("symbol", group_keys=False)
    vol = group["volume"]
    amount = df.groupby("trade_date")["amount"]

    df["vol_ratio_5"] = df["volume"] / vol.transform(lambda s: s.rolling(5).mean())
    df["vol_ratio_20"] = df["volume"] / vol.transform(lambda s: s.rolling(20).mean())
    df["turnover_rank_industry"] = df.groupby(["trade_date", "industry"])["turnover_rate"].rank(pct=True)
    df["amount_rank_market"] = df.groupby("trade_date")["amount"].rank(pct=True)
    df["shrink_volume_flag"] = (df["vol_ratio_5"] < 0.7).astype(float)
    df["surge_volume_flag"] = (df["vol_ratio_5"] > 1.8).astype(float)
    return df[
        [
            "symbol",
            "trade_date",
            "vol_ratio_5",
            "vol_ratio_20",
            "turnover_rank_industry",
            "amount_rank_market",
            "shrink_volume_flag",
            "surge_volume_flag",
        ]
    ]
