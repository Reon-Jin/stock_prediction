from __future__ import annotations

import pandas as pd


def build_relative_features(
    daily_bars: pd.DataFrame,
    index_bars: pd.DataFrame,
    securities: pd.DataFrame,
    sector_daily: pd.DataFrame,
) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame()
    df = daily_bars.sort_values(["symbol", "trade_date"]).copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.merge(securities[["symbol", "industry"]].copy(), on="symbol", how="left")
    index_df = index_bars.copy()
    index_df["trade_date"] = pd.to_datetime(index_df["trade_date"], errors="coerce").dt.normalize()
    hs300 = index_df[index_df["index_code"] == "000300.SH"][["trade_date", "close"]].rename(columns={"close": "hs300_close"})
    hs300["hs300_close"] = pd.to_numeric(hs300["hs300_close"], errors="coerce")
    hs300 = hs300.sort_values("trade_date")
    hs300["hs300_ret_5"] = hs300["hs300_close"].pct_change(5)
    hs300["hs300_ret_10"] = hs300["hs300_close"].pct_change(10)
    df = df.merge(hs300[["trade_date", "hs300_ret_5", "hs300_ret_10"]], on="trade_date", how="left")

    group = df.groupby("symbol", group_keys=False)["adj_close"]
    df["stock_ret_5"] = group.pct_change(5)
    df["stock_ret_10"] = group.pct_change(10)
    df["stock_ret_20"] = group.pct_change(20)
    df["ret5_vs_hs300"] = df["stock_ret_5"] - df["hs300_ret_5"]
    df["ret10_vs_hs300"] = df["stock_ret_10"] - df["hs300_ret_10"]
    df["stock_rank_in_industry"] = df.groupby(["trade_date", "industry"])["pct_chg"].rank(pct=True)

    industry_daily = (
        df.dropna(subset=["industry"])
        .groupby(["trade_date", "industry"], as_index=False)
        .agg(
            industry_ret_5=("stock_ret_5", "mean"),
            industry_ret_20=("stock_ret_20", "mean"),
        )
        .sort_values(["industry", "trade_date"])
    )
    if industry_daily.empty:
        df["ret20_vs_industry"] = None
        df["industry_rank_5d"] = None
        df["industry_rank_20d"] = None
    else:
        industry_daily["industry_rank_5d"] = industry_daily.groupby("trade_date")["industry_ret_5"].rank(pct=True)
        industry_daily["industry_rank_20d"] = industry_daily.groupby("trade_date")["industry_ret_20"].rank(pct=True)
        df = df.merge(
            industry_daily[["industry", "trade_date", "industry_ret_20", "industry_rank_5d", "industry_rank_20d"]],
            on=["industry", "trade_date"],
            how="left",
        )
        df["ret20_vs_industry"] = df["stock_ret_20"] - df["industry_ret_20"]

    return df[
        [
            "symbol",
            "trade_date",
            "ret5_vs_hs300",
            "ret10_vs_hs300",
            "ret20_vs_industry",
            "stock_rank_in_industry",
            "industry_rank_5d",
            "industry_rank_20d",
        ]
    ]
