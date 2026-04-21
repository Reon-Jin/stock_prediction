from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def build_price_features(daily_bars: pd.DataFrame) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame()
    df = daily_bars.sort_values(["symbol", "trade_date"]).copy()
    group = df.groupby("symbol", group_keys=False)
    price = group["adj_close"]

    for window in (1, 3, 5, 10, 20, 60):
        df[f"ret_{window}"] = price.pct_change(window)

    for window in (5, 10, 20, 60):
        ma = price.transform(lambda s: s.rolling(window).mean())
        df[f"ma{window}_gap"] = df["adj_close"] / ma - 1

    df["rsi_6"] = price.transform(lambda s: _rsi(s, 6))
    df["rsi_14"] = price.transform(lambda s: _rsi(s, 14))

    ema12 = price.transform(lambda s: _ema(s, 12))
    ema26 = price.transform(lambda s: _ema(s, 26))
    dif = ema12 - ema26
    dea = dif.groupby(df["symbol"]).transform(lambda s: _ema(s, 9))
    df["macd_dif"] = dif
    df["macd_dea"] = dea
    df["macd_hist"] = (dif - dea) * 2

    prev_close = group["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.groupby(df["symbol"]).transform(lambda s: s.rolling(14).mean())

    ma20 = price.transform(lambda s: s.rolling(20).mean())
    std20 = price.transform(lambda s: s.rolling(20).std())
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    spread = (upper - lower).replace(0, np.nan)
    df["boll_pos"] = (df["adj_close"] - lower) / spread

    returns = group["adj_close"].pct_change()
    df["volatility_5"] = returns.groupby(df["symbol"]).transform(lambda s: s.rolling(5).std())
    df["volatility_20"] = returns.groupby(df["symbol"]).transform(lambda s: s.rolling(20).std())

    return df[["symbol", "trade_date"] + [c for c in df.columns if c not in daily_bars.columns and c not in {"id"}]]
