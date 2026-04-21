from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pytorch_dataset import MultiInputInferenceDataset
from features.company_profile_builder import build_company_id_maps
from features.company_similarity_builder import build_company_neighbor_frame
from features.feature_builder import build_all_features
from features.sample_finalize import finalize_training_samples
from utils.cli import build_common_parser
from utils.io import dump_json, dump_parquet

from jobs.common import bootstrap, filter_securities_by_symbols, resolve_start_end


PREDICTION_BASE_COLUMNS = [
    "symbol",
    "trade_date",
    "name",
    "industry_sw",
    "board",
    "symbol_id",
    "industry_id",
    "board_id",
    "company_profile_version",
    "list_days",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "amount",
    "turnover_rate",
    "pct_chg",
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "ma5_gap",
    "ma10_gap",
    "ma20_gap",
    "ma60_gap",
    "rsi_6",
    "rsi_14",
    "macd_dif",
    "macd_dea",
    "macd_hist",
    "atr_14",
    "boll_pos",
    "volatility_5",
    "volatility_20",
    "volatility_120",
    "beta_120",
    "vol_ratio_5",
    "vol_ratio_20",
    "market_cap_log",
    "turnover_mean_120",
    "amount_mean_120",
    "turnover_rank_industry",
    "amount_rank_market",
    "shrink_volume_flag",
    "surge_volume_flag",
    "ret5_vs_hs300",
    "ret10_vs_hs300",
    "ret20_vs_industry",
    "stock_rank_in_industry",
    "industry_rank_5d",
    "industry_rank_20d",
    "roe",
    "revenue_yoy",
    "profit_yoy",
    "debt_ratio",
    "gross_margin",
    "industry_roe_percentile",
    "event_embedding",
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
    "neighbor_symbol_ids",
    "neighbor_scores",
    "sample_status",
    "data_version",
    "feature_version",
]

PREDICTION_EXPORT_COLUMNS = [
    "symbol",
    "trade_date",
    "name",
    "industry_sw",
    "board",
    "symbol_id",
    "industry_id",
    "board_id",
    "x_seq",
    "x_tab",
    "x_event",
    "x_mkt",
    "x_company_profile",
    "neighbor_symbol_ids",
    "neighbor_scores",
    "seq_columns",
    "tab_columns",
    "event_columns",
    "mkt_columns",
    "company_id_columns",
    "company_profile_columns",
    "neighbor_topk",
    "seq_length",
    "sample_status",
    "data_version",
    "feature_version",
]


def _resolve_today_target(config_path: str, end: str | None) -> str:
    ctx = bootstrap("resolve_prediction_target", config_path)
    _, resolved_end = resolve_start_end(ctx.config, None, end)
    return resolved_end


def _normalize_symbol(symbol: str | None) -> str | None:
    if symbol is None:
        return None
    normalized = str(symbol).strip().upper()
    return normalized or None


def _prediction_file_stem(target_symbol: str | None) -> str:
    return "today" if not target_symbol else f"today_{target_symbol}"


def _prediction_infer_file_stem(target_symbol: str | None) -> str:
    return "today_infer" if not target_symbol else f"today_infer_{target_symbol}"


def _resolve_effective_trade_date(
    daily_bars: pd.DataFrame,
    requested_target_ts: pd.Timestamp,
) -> pd.Timestamp | None:
    if daily_bars.empty or "trade_date" not in daily_bars.columns:
        return None
    trade_dates = pd.to_datetime(daily_bars["trade_date"], errors="coerce").dropna()
    if trade_dates.empty:
        return None
    eligible = trade_dates[trade_dates <= requested_target_ts]
    if eligible.empty:
        return None
    return pd.Timestamp(eligible.max()).normalize()


def build_prediction_frame(
    config_path: str,
    target_date: str | None = None,
    target_symbol: str | None = None,
    target_symbols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    ctx = bootstrap("build_prediction_samples", config_path)
    _, resolved_end = resolve_start_end(ctx.config, None, target_date)
    target_date = resolved_end
    target_symbol = _normalize_symbol(target_symbol)
    normalized_target_symbols = [
        normalized for normalized in (_normalize_symbol(symbol) for symbol in (target_symbols or [])) if normalized
    ]
    target_ts = pd.Timestamp(target_date)
    feature_lookback_days = max(140, int(ctx.config.project.get("seq_length", 20)) + 120)
    start_buffer = (target_ts - pd.Timedelta(days=feature_lookback_days)).strftime("%Y-%m-%d")
    end_buffer = target_ts.strftime("%Y-%m-%d")
    symbol_filter = normalized_target_symbols or ([target_symbol] if target_symbol else None)

    securities_all = ctx.repo.get_securities(active_only=False)
    securities = securities_all
    if symbol_filter:
        securities = filter_securities_by_symbols(securities_all, symbol_filter)
        if securities.empty:
            empty = pd.DataFrame(columns=PREDICTION_BASE_COLUMNS)
            return empty, empty, None

    company_similarity = ctx.repo.fetch_table("company_similarity", end=target_date, symbols=symbol_filter)
    daily_bars = ctx.repo.fetch_table("daily_bars", start=start_buffer, end=end_buffer, symbols=symbol_filter)
    index_bars = ctx.repo.fetch_table("index_bars", start=start_buffer, end=end_buffer)
    sector_daily = ctx.repo.fetch_table("sector_daily", start=start_buffer, end=target_date)
    financial_snapshot = ctx.repo.fetch_table("financial_snapshot", end=target_date, symbols=symbol_filter)
    event_features = ctx.repo.fetch_table("event_features_daily", start=start_buffer, end=target_date, symbols=symbol_filter)
    company_profiles = ctx.repo.fetch_table("company_profiles", start=start_buffer, end=target_date, symbols=symbol_filter)

    if daily_bars.empty:
        empty = pd.DataFrame(columns=PREDICTION_BASE_COLUMNS)
        return empty, empty, None

    effective_trade_ts = _resolve_effective_trade_date(daily_bars, target_ts)
    if effective_trade_ts is None:
        empty = pd.DataFrame(columns=PREDICTION_BASE_COLUMNS)
        return empty, empty, None
    if effective_trade_ts != target_ts:
        ctx.logger.warning(
            "requested prediction date %s has no daily bars; fallback to latest available trade_date=%s",
            target_ts.strftime("%Y-%m-%d"),
            effective_trade_ts.strftime("%Y-%m-%d"),
        )

    features = build_all_features(
        daily_bars=daily_bars,
        securities=securities,
        index_bars=index_bars,
        sector_daily=sector_daily,
        financial_snapshot=financial_snapshot,
        event_features_daily=event_features,
        company_profiles=company_profiles,
        identity_securities=securities_all,
    )
    features["trade_date"] = pd.to_datetime(features["trade_date"])
    min_list_days = int(ctx.config.project.get("min_list_days", 120))
    samples = features[
        (~features["is_st"].fillna(False))
        & (features["list_days"].fillna(0) >= min_list_days)
    ].copy()
    if samples.empty:
        empty = pd.DataFrame(columns=PREDICTION_BASE_COLUMNS)
        return empty, empty, effective_trade_ts.strftime("%Y-%m-%d")

    samples["sample_status"] = "predict_ready"
    samples["data_version"] = ctx.config.project.get("data_version", "v1")
    samples["feature_version"] = ctx.config.project.get("feature_version", "f1")

    company_id_map = build_company_id_maps(securities_all)
    neighbor_frame = build_company_neighbor_frame(
        company_similarity=company_similarity,
        symbol_id_map=company_id_map,
        topk=int(ctx.config.company_encoding.get("similarity_topk", 10)),
    )
    if not neighbor_frame.empty:
        samples = samples.merge(neighbor_frame, on="symbol", how="left")

    if "is_suspended" in daily_bars.columns:
        suspended_map = daily_bars[["symbol", "trade_date", "is_suspended"]].copy()
        suspended_map["trade_date"] = pd.to_datetime(suspended_map["trade_date"])
        samples = samples.merge(suspended_map, on=["symbol", "trade_date"], how="left")
        samples = samples[~samples["is_suspended"].fillna(False)].copy()

    for date_col in ("trade_date", "list_date"):
        if date_col in samples.columns:
            samples[date_col] = pd.to_datetime(samples[date_col], errors="coerce").dt.date

    context_output = samples[[column for column in PREDICTION_BASE_COLUMNS if column in samples.columns]].copy()
    context_output = finalize_training_samples(context_output)
    raw_output = context_output[
        pd.to_datetime(context_output["trade_date"], errors="coerce") == effective_trade_ts
    ].copy()
    return raw_output, context_output, effective_trade_ts.strftime("%Y-%m-%d")


def _build_prediction_export_frame(
    context_prediction_df: pd.DataFrame,
    effective_trade_date: str | None,
    seq_length: int = 20,
) -> pd.DataFrame:
    if context_prediction_df.empty or not effective_trade_date:
        return pd.DataFrame(columns=PREDICTION_EXPORT_COLUMNS)

    target_ts = pd.Timestamp(effective_trade_date).normalize()
    dataset = MultiInputInferenceDataset(context_prediction_df, seq_length=seq_length)
    rows: list[dict[str, object]] = []
    for sample in dataset:
        sample_trade_ts = pd.Timestamp(sample.meta.get("trade_date")).normalize()
        if sample_trade_ts != target_ts:
            continue
        row = {
            "symbol": sample.meta.get("symbol"),
            "trade_date": sample.meta.get("trade_date"),
            "name": sample.meta.get("name"),
            "industry_sw": sample.meta.get("industry_sw"),
            "board": sample.meta.get("board"),
            "symbol_id": int(sample.x_company_ids.get("symbol_id", 0).item()),
            "industry_id": int(sample.x_company_ids.get("industry_id", 0).item()),
            "board_id": int(sample.x_company_ids.get("board_id", 0).item()),
            "x_seq": sample.x_seq.tolist(),
            "x_tab": sample.x_tab.tolist(),
            "x_event": sample.x_event.tolist(),
            "x_mkt": sample.x_mkt.tolist(),
            "x_company_profile": sample.x_company_profile.tolist(),
            "neighbor_symbol_ids": sample.neighbor_symbol_ids.tolist(),
            "neighbor_scores": sample.neighbor_scores.tolist(),
            "seq_columns": list(dataset.seq_columns),
            "tab_columns": list(dataset.tab_columns),
            "event_columns": list(dataset.event_columns),
            "mkt_columns": list(dataset.mkt_columns),
            "company_id_columns": list(dataset.company_id_columns),
            "company_profile_columns": list(dataset.company_profile_columns),
            "neighbor_topk": int(dataset.neighbor_topk),
            "seq_length": int(dataset.seq_length),
            "sample_status": context_prediction_df.iloc[-1].get("sample_status", "predict_ready"),
            "data_version": context_prediction_df.iloc[-1].get("data_version"),
            "feature_version": context_prediction_df.iloc[-1].get("feature_version"),
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=PREDICTION_EXPORT_COLUMNS)


def export_prediction_frame(
    raw_prediction_df: pd.DataFrame,
    inference_prediction_df: pd.DataFrame,
    export_root: str | Path,
    requested_target_date: str,
    effective_trade_date: str | None,
    target_symbol: str | None = None,
) -> dict[str, Path]:
    base_dir = Path(export_root)
    archive_dir = base_dir / "predictions"
    file_stem = _prediction_file_stem(_normalize_symbol(target_symbol))
    infer_file_stem = _prediction_infer_file_stem(_normalize_symbol(target_symbol))
    latest_path = base_dir / f"{file_stem}.parquet"
    archive_path = archive_dir / f"{file_stem}_{requested_target_date}.parquet"
    infer_latest_path = base_dir / f"{infer_file_stem}.parquet"
    infer_archive_path = archive_dir / f"{infer_file_stem}_{requested_target_date}.parquet"
    manifest_path = archive_dir / f"{file_stem}_manifest.json"

    dump_parquet(raw_prediction_df, latest_path)
    dump_parquet(raw_prediction_df, archive_path)
    dump_parquet(inference_prediction_df, infer_latest_path)
    dump_parquet(inference_prediction_df, infer_archive_path)
    dump_json(
        manifest_path,
        {
            "requested_target_date": requested_target_date,
            "effective_trade_date": effective_trade_date,
            "target_symbol": _normalize_symbol(target_symbol),
            "latest_path": str(latest_path),
            "archive_path": str(archive_path),
            "inference_latest_path": str(infer_latest_path),
            "inference_archive_path": str(infer_archive_path),
            "raw_rows": int(len(raw_prediction_df)),
            "inference_rows": int(len(inference_prediction_df)),
        },
    )
    return {
        "latest": latest_path,
        "archive": archive_path,
        "inference_latest": infer_latest_path,
        "inference_archive": infer_archive_path,
        "manifest": manifest_path,
    }


def run(
    config_path: str,
    target_date: str | None = None,
    target_symbol: str | None = None,
) -> dict[str, Path]:
    ctx = bootstrap("build_prediction_samples", config_path)
    _, resolved_end = resolve_start_end(ctx.config, None, target_date)
    target_date = resolved_end
    target_symbol = _normalize_symbol(target_symbol)
    job_id = ctx.repo.record_job_start(
        "build_prediction_samples",
        {"target_date": target_date, "target_symbol": target_symbol},
    )
    try:
        raw_prediction_df, context_prediction_df, effective_trade_date = build_prediction_frame(
            config_path=config_path,
            target_date=target_date,
            target_symbol=target_symbol,
        )
        inference_prediction_df = _build_prediction_export_frame(
            context_prediction_df=context_prediction_df,
            effective_trade_date=effective_trade_date,
            seq_length=int(ctx.config.project.get("seq_length", 20)),
        )
        export_root = ctx.config.storage.get("export_dir", "data/exports")
        base_data_dir = Path(export_root).resolve().parent
        paths = export_prediction_frame(
            raw_prediction_df=raw_prediction_df,
            inference_prediction_df=inference_prediction_df,
            export_root=base_data_dir,
            requested_target_date=target_date,
            effective_trade_date=effective_trade_date,
            target_symbol=target_symbol,
        )
        ctx.logger.info(
            "built prediction samples raw_rows=%s inference_rows=%s requested_target_date=%s effective_trade_date=%s target_symbol=%s",
            len(raw_prediction_df),
            len(inference_prediction_df),
            target_date,
            effective_trade_date,
            target_symbol or "ALL",
        )
        ctx.repo.record_job_end(
            job_id,
            "success",
            len(raw_prediction_df),
            f"prediction export ready for requested_date={target_date} effective_date={effective_trade_date} symbol={target_symbol or 'ALL'}",
        )
        return paths
    except Exception as exc:
        ctx.logger.exception("build prediction samples failed")
        ctx.repo.record_job_end(job_id, "failed", message=str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Build prediction samples for one target date")
    parser.add_argument("--symbol", default=None, help="Optional single symbol to export, e.g. 000001.SZ")
    args = parser.parse_args()
    run(args.config, target_date=args.end or args.start, target_symbol=args.symbol)


if __name__ == "__main__":
    main()
