from __future__ import annotations

import pandas as pd


def build_market_features(daily_bars: pd.DataFrame, index_bars: pd.DataFrame, sector_daily: pd.DataFrame) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame()
    market = daily_bars.copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    grouped = market.groupby("trade_date")
    summary = grouped.agg(
        up_limit_count=("pct_chg", lambda s: (pd.to_numeric(s, errors="coerce") >= 9.5).sum()),
        down_limit_count=("pct_chg", lambda s: (pd.to_numeric(s, errors="coerce") <= -9.5).sum()),
        market_turnover=("amount", "sum"),
    ).reset_index()
    market["up_touch_flag"] = ((pd.to_numeric(market["high"], errors="coerce") / pd.to_numeric(market["pre_close"], errors="coerce") - 1) >= 0.095).astype(int)
    touched_up = market.groupby("trade_date", as_index=False)["up_touch_flag"].sum().rename(columns={"up_touch_flag": "up_touch_count"})
    broken = summary.merge(touched_up, on="trade_date", how="left")
    broken["broken_limit_rate"] = (
        (broken["up_touch_count"] - broken["up_limit_count"]).clip(lower=0) / broken["up_touch_count"].replace(0, pd.NA)
    ).fillna(0.0)
    broken["consecutive_limit_height"] = broken["up_limit_count"].rolling(5).max()

    idx = index_bars.copy()
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    hs300 = idx[idx["index_code"] == "000300.SH"][["trade_date", "close"]].rename(columns={"close": "hs300_close"})
    cyb = idx[idx["index_code"] == "399006.SZ"][["trade_date", "close"]].rename(columns={"close": "cyb_close"})
    summary = broken.merge(hs300, on="trade_date", how="left").merge(cyb, on="trade_date", how="left")
    summary["hs300_ret_1"] = summary["hs300_close"].pct_change()
    summary["cyb_ret_1"] = summary["cyb_close"].pct_change()
    summary["market_volatility_5"] = summary["hs300_ret_1"].rolling(5).std()

    sectors = sector_daily.copy()
    if sectors.empty:
        summary["sector_hotness_top1"] = 0.0
        summary["sector_hotness_top3_mean"] = 0.0
    else:
        sectors["trade_date"] = pd.to_datetime(sectors["trade_date"])
        rank = sectors.groupby("trade_date")["pct_chg"].apply(lambda s: s.nlargest(3).tolist()).reset_index(name="top3")
        rank["sector_hotness_top1"] = rank["top3"].apply(lambda x: x[0] if x else 0.0)
        rank["sector_hotness_top3_mean"] = rank["top3"].apply(lambda x: sum(x) / len(x) if x else 0.0)
        summary = summary.merge(rank[["trade_date", "sector_hotness_top1", "sector_hotness_top3_mean"]], on="trade_date", how="left")

    breadth_spread = summary["up_limit_count"] - summary["down_limit_count"]
    sector_top1 = pd.to_numeric(summary["sector_hotness_top1"], errors="coerce").fillna(0.0)
    sector_top3 = pd.to_numeric(summary["sector_hotness_top3_mean"], errors="coerce").fillna(0.0)
    summary["risk_on_flag"] = (
        (summary["hs300_ret_1"] > 0)
        & ((breadth_spread > 0) | (sector_top1 > 0) | (sector_top3 > 0))
    ).astype(float)
    summary["risk_off_flag"] = (
        (summary["hs300_ret_1"] < 0)
        & ((breadth_spread < 0) | (sector_top1 < 0) | (sector_top3 < 0))
    ).astype(float)
    return summary[
        [
            "trade_date",
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
        ]
    ]
