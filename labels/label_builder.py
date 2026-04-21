from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _max_drawdown_from_path(path: np.ndarray) -> float:
    if len(path) == 0 or np.isnan(path).all():
        return np.nan
    running_max = np.maximum.accumulate(path)
    drawdowns = path / running_max - 1
    return float(np.nanmin(drawdowns))


def _build_symbol_labels(
    df: pd.DataFrame,
    periods: Iterable[int],
    bigloss_threshold: float,
    target_start: pd.Timestamp | None = None,
    target_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = df.sort_values("trade_date").copy()
    df["entry_open"] = df["open"].shift(-1)
    target_mask = pd.Series(True, index=df.index)
    if target_start is not None:
        target_mask &= df["trade_date"] >= target_start
    if target_end is not None:
        target_mask &= df["trade_date"] <= target_end
    target_indices = set(df.index[target_mask].tolist())
    for k in periods:
        exit_close = df["close"].shift(-k)
        ret = exit_close / df["entry_open"] - 1
        df[f"label_ret_{k}"] = ret
        df[f"label_win_{k}"] = (ret > 0).astype("Int64")
        drawdowns = []
        valid_mask = []
        close_values = df["close"].to_numpy(dtype=float)
        open_values = df["entry_open"].to_numpy(dtype=float)
        for i in range(len(df)):
            row_index = df.index[i]
            if row_index not in target_indices:
                drawdowns.append(np.nan)
                valid_mask.append(False)
                continue
            if i + k >= len(df) or np.isnan(open_values[i]):
                drawdowns.append(np.nan)
                valid_mask.append(False)
                continue
            future_path = np.concatenate(([open_values[i]], close_values[i + 1 : i + k + 1]))
            drawdowns.append(_max_drawdown_from_path(future_path))
            valid_mask.append(True)
        df[f"label_maxdd_{k}"] = drawdowns
        df[f"_valid_{k}"] = valid_mask
        if k in (5, 20):
            df[f"label_bigloss_{k}"] = (df[f"label_maxdd_{k}"] <= bigloss_threshold).astype("Int64")
    return df


def build_labels(
    daily_bars: pd.DataFrame,
    benchmark_index: pd.DataFrame,
    periods: Iterable[int],
    bigloss_threshold: float = -0.05,
    target_start: str | None = None,
    target_end: str | None = None,
) -> pd.DataFrame:
    if daily_bars.empty:
        return pd.DataFrame()
    periods = list(periods)
    df = daily_bars[["symbol", "trade_date", "open", "close"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    target_start_ts = pd.Timestamp(target_start) if target_start else None
    target_end_ts = pd.Timestamp(target_end) if target_end else None
    labeled_frames = []
    for _, group in df.groupby("symbol", sort=False):
        labeled_group = _build_symbol_labels(
            group.copy(),
            periods=periods,
            bigloss_threshold=bigloss_threshold,
            target_start=target_start_ts,
            target_end=target_end_ts,
        )
        if target_start_ts is not None or target_end_ts is not None:
            mask = pd.Series(True, index=labeled_group.index)
            if target_start_ts is not None:
                mask &= labeled_group["trade_date"] >= target_start_ts
            if target_end_ts is not None:
                mask &= labeled_group["trade_date"] <= target_end_ts
            labeled_group = labeled_group[mask]
        if not labeled_group.empty:
            labeled_frames.append(labeled_group)
    labeled = pd.concat(labeled_frames, ignore_index=True) if labeled_frames else pd.DataFrame()

    benchmark = benchmark_index.copy()
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"])
    benchmark = benchmark.sort_values("trade_date")
    benchmark["bench_entry_open"] = benchmark["open"].shift(-1)
    for k in periods:
        benchmark[f"bench_ret_{k}"] = benchmark["close"].shift(-k) / benchmark["bench_entry_open"] - 1
    labeled = labeled.merge(
        benchmark[["trade_date"] + [f"bench_ret_{k}" for k in periods]],
        on="trade_date",
        how="left",
    )
    for k in periods:
        labeled[f"label_alpha_{k}"] = labeled[f"label_ret_{k}"] - labeled[f"bench_ret_{k}"]

    valid_flags = [f"_valid_{k}" for k in periods]
    labeled["enough_future_window"] = labeled[valid_flags].all(axis=1)
    labeled = labeled[labeled["enough_future_window"]].copy()
    cols = ["symbol", "trade_date"]
    for k in periods:
        cols.extend(
            [
                f"label_ret_{k}",
                f"label_win_{k}",
                f"label_alpha_{k}",
                f"label_maxdd_{k}",
            ]
        )
    for k in (5, 20):
        if f"label_bigloss_{k}" in labeled.columns:
            cols.append(f"label_bigloss_{k}")
    return labeled[cols]
