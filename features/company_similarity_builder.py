from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from features.company_profile_builder import COMPANY_PROFILE_COLUMNS


SIMILARITY_FUNDAMENTAL_COLUMNS = [
    "roe",
    "revenue_yoy",
    "profit_yoy",
    "gross_margin",
    "debt_ratio",
]

SIMILARITY_BEHAVIOR_COLUMNS = [
    "volatility_120",
    "beta_120",
    "turnover_mean_120",
    "amount_mean_120",
    "ret_20",
    "ret_60",
]


def _fill_and_standardize(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = df[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        median = frame[column].median()
        if pd.isna(median):
            median = 0.0
        frame[column] = frame[column].fillna(median)
        std = frame[column].std(ddof=0)
        if pd.isna(std) or std == 0:
            std = 1.0
        frame[column] = (frame[column] - frame[column].mean()) / std
    return frame


def _cosine_similarity_matrix(frame: pd.DataFrame) -> np.ndarray:
    if frame.empty:
        return np.zeros((0, 0), dtype=float)
    values = frame.to_numpy(dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = values / norms
    return normalized @ normalized.T


def _size_similarity(values: pd.Series) -> np.ndarray:
    safe_values = pd.to_numeric(values, errors="coerce").fillna(values.median() if not values.dropna().empty else 0.0)
    diff = np.abs(np.subtract.outer(safe_values.to_numpy(dtype=float), safe_values.to_numpy(dtype=float)))
    return np.exp(-diff)


def _industry_similarity(latest_profiles: pd.DataFrame) -> np.ndarray:
    industry = latest_profiles["industry_name"].fillna("").astype(str).to_numpy()
    board = latest_profiles["board"].fillna("").astype(str).to_numpy()
    same_industry = (industry[:, None] == industry[None, :]) & (industry[:, None] != "")
    same_board = (board[:, None] == board[None, :]) & (board[:, None] != "")
    score = np.where(same_industry, 1.0, np.where(same_board, 0.25, 0.0))
    return score.astype(float)


def build_company_similarity(
    company_profiles: pd.DataFrame,
    topk: int = 10,
    similarity_version: str = "cs1",
) -> pd.DataFrame:
    if company_profiles.empty:
        return pd.DataFrame(
            columns=["symbol", "neighbor_symbol", "sim_score", "sim_rank", "similarity_version", "asof_date"]
        )

    profiles = company_profiles.copy()
    profiles["asof_date"] = pd.to_datetime(profiles["asof_date"])
    profiles = profiles.sort_values(["symbol", "asof_date"])
    latest = profiles.groupby("symbol", as_index=False).tail(1).reset_index(drop=True)
    if latest.empty:
        return pd.DataFrame(
            columns=["symbol", "neighbor_symbol", "sim_score", "sim_rank", "similarity_version", "asof_date"]
        )

    industry_sim = _industry_similarity(latest)
    size_sim = _size_similarity(latest["market_cap_log"])
    fundamental_sim = _cosine_similarity_matrix(_fill_and_standardize(latest, SIMILARITY_FUNDAMENTAL_COLUMNS))
    behavior_sim = _cosine_similarity_matrix(_fill_and_standardize(latest, SIMILARITY_BEHAVIOR_COLUMNS))

    score = 0.40 * industry_sim + 0.20 * size_sim + 0.20 * fundamental_sim + 0.20 * behavior_sim
    np.fill_diagonal(score, -np.inf)

    symbols = latest["symbol"].astype(str).tolist()
    asof_dates = latest["asof_date"].tolist()
    rows: list[dict[str, Any]] = []
    for row_idx, symbol in enumerate(symbols):
        ranking = np.argsort(score[row_idx])[::-1]
        neighbor_rank = 1
        for col_idx in ranking:
            sim_score = float(score[row_idx, col_idx])
            if not np.isfinite(sim_score) or sim_score <= 0:
                continue
            neighbor_symbol = symbols[col_idx]
            if neighbor_symbol == symbol:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "neighbor_symbol": neighbor_symbol,
                    "sim_score": max(0.0, sim_score),
                    "sim_rank": neighbor_rank,
                    "similarity_version": similarity_version,
                    "asof_date": max(asof_dates[row_idx], asof_dates[col_idx]).date(),
                }
            )
            neighbor_rank += 1
            if neighbor_rank > topk:
                break
    return pd.DataFrame(rows)


def build_company_neighbor_frame(
    company_similarity: pd.DataFrame,
    symbol_id_map: pd.DataFrame,
    topk: int = 10,
) -> pd.DataFrame:
    if company_similarity.empty or symbol_id_map.empty:
        return pd.DataFrame(columns=["symbol", "neighbor_symbol_ids", "neighbor_scores"])

    mapping = symbol_id_map[["symbol", "symbol_id"]].drop_duplicates("symbol").copy()
    mapping["symbol_id"] = pd.to_numeric(mapping["symbol_id"], errors="coerce").fillna(0).astype(int)

    sim = company_similarity.copy()
    sim = sim[sim["sim_rank"] <= topk].copy()
    sim = sim.merge(mapping.rename(columns={"symbol": "neighbor_symbol"}), on="neighbor_symbol", how="left")
    sim["symbol_id"] = pd.to_numeric(sim["symbol_id"], errors="coerce").fillna(0).astype(int)

    rows: list[dict[str, Any]] = []
    for symbol, group in sim.sort_values(["symbol", "sim_rank"]).groupby("symbol"):
        neighbor_ids = group["symbol_id"].tolist()
        neighbor_scores = pd.to_numeric(group["sim_score"], errors="coerce").fillna(0.0).astype(float).tolist()
        rows.append(
            {
                "symbol": symbol,
                "neighbor_symbol_ids": neighbor_ids[:topk],
                "neighbor_scores": neighbor_scores[:topk],
            }
        )
    return pd.DataFrame(rows)
