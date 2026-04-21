from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.splitter import (
    UNSAFE_RANDOM_SPLIT_MESSAGE,
    build_time_purged_split_config,
    export_time_purged_splits,
)
from features.company_profile_builder import build_company_id_maps
from features.company_similarity_builder import build_company_neighbor_frame
from features.feature_builder import build_all_features
from features.sample_finalize import finalize_training_samples
from labels.label_builder import build_labels
from utils.cli import build_common_parser
from warehouse.models import TrainingSample

from jobs.common import bootstrap, resolve_start_end


def _iter_date_chunks(start: str, end: str, chunk_days: int) -> list[tuple[str, str]]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    chunks: list[tuple[str, str]] = []
    current = start_ts
    while current <= end_ts:
        chunk_end = min(current + pd.Timedelta(days=chunk_days - 1), end_ts)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + pd.Timedelta(days=1)
    return chunks


def _build_sample_chunk(
    ctx,
    securities: pd.DataFrame,
    neighbor_frame: pd.DataFrame,
    full_start: str,
    full_end: str,
    chunk_start: str,
    chunk_end: str,
    progress_bar: Any | None = None,
    chunk_label: str | None = None,
) -> pd.DataFrame:
    label = chunk_label or f"{chunk_start}->{chunk_end}"

    def advance(stage: str) -> None:
        if progress_bar is not None:
            progress_bar.set_postfix_str(f"{label} | {stage}")
            progress_bar.update(1)

    feature_lookback_days = max(140, int(ctx.config.project.get("seq_length", 20)) + 120)
    future_label_days = max(int(max(ctx.config.labels.get("holding_periods", [3, 5, 10, 20, 40]))), 45)
    start_buffer = (pd.Timestamp(chunk_start) - pd.Timedelta(days=feature_lookback_days)).strftime("%Y-%m-%d")
    end_buffer = (pd.Timestamp(chunk_end) + pd.Timedelta(days=future_label_days)).strftime("%Y-%m-%d")

    daily_bars = ctx.repo.fetch_table("daily_bars", start=start_buffer, end=end_buffer)
    index_bars = ctx.repo.fetch_table("index_bars", start=start_buffer, end=end_buffer)
    sector_daily = ctx.repo.fetch_table("sector_daily", start=start_buffer, end=chunk_end)
    financial_snapshot = ctx.repo.fetch_table("financial_snapshot", end=chunk_end)
    event_features = ctx.repo.fetch_table("event_features_daily", start=chunk_start, end=chunk_end)
    company_profiles = ctx.repo.fetch_table("company_profiles", start=start_buffer, end=chunk_end)
    advance("fetched source tables")

    if daily_bars.empty:
        advance("skipped empty daily_bars")
        advance("skipped labels")
        advance("skipped sample assembly")
        return pd.DataFrame()

    features = build_all_features(
        daily_bars=daily_bars,
        securities=securities,
        index_bars=index_bars,
        sector_daily=sector_daily,
        financial_snapshot=financial_snapshot,
        event_features_daily=event_features,
        company_profiles=company_profiles,
    )
    advance("built features")
    benchmark = index_bars[index_bars["index_code"] == ctx.config.labels.get("benchmark_index", "000300.SH")].copy()
    labels = build_labels(
        daily_bars=daily_bars,
        benchmark_index=benchmark,
        periods=ctx.config.labels.get("holding_periods", [3, 5, 10, 20, 40]),
        bigloss_threshold=float(ctx.config.labels.get("bigloss_threshold", -0.05)),
        target_start=chunk_start,
        target_end=chunk_end,
    )
    advance("built labels")
    if labels.empty:
        advance("skipped sample assembly")
        return pd.DataFrame()

    samples = features.merge(labels, on=["symbol", "trade_date"], how="inner")
    samples["trade_date"] = pd.to_datetime(samples["trade_date"])
    min_list_days = int(ctx.config.project.get("min_list_days", 120))
    samples = samples[
        (samples["trade_date"] >= pd.Timestamp(chunk_start))
        & (samples["trade_date"] <= pd.Timestamp(chunk_end))
        & (samples["trade_date"] >= pd.Timestamp(full_start))
        & (samples["trade_date"] <= pd.Timestamp(full_end))
        & (~samples["is_st"].fillna(False))
        & (samples["list_days"].fillna(0) >= min_list_days)
    ].copy()
    if samples.empty:
        advance("assembled 0 rows")
        return samples

    samples["sample_status"] = "ready"
    samples["data_version"] = ctx.config.project.get("data_version", "v1")
    samples["feature_version"] = ctx.config.project.get("feature_version", "f1")
    samples["label_version"] = ctx.config.project.get("label_version", "l1")
    samples["label_rank_score"] = samples.groupby("trade_date")["label_alpha_5"].rank(pct=True)

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

    model_columns = [column.name for column in TrainingSample.__table__.columns]
    result = samples[[column for column in model_columns if column in samples.columns]].copy()
    result = finalize_training_samples(result)
    advance(f"assembled {len(result)} rows")
    return result


def _configure_runtime_warnings() -> None:
    # Leading all-NaN windows in rolling features are expected and only create noisy console output.
    warnings.filterwarnings(
        "ignore",
        message="Mean of empty slice",
        category=RuntimeWarning,
        module=r"numpy\.lib\._nanfunctions_impl",
    )


def run(
    config_path: str,
    start: str | None = None,
    end: str | None = None,
    export: bool = False,
    chunk_days: int = 90,
    build_samples: bool = True,
) -> None:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be a positive integer")

    _configure_runtime_warnings()
    ctx = bootstrap("build_training_samples", config_path)
    start, end = resolve_start_end(ctx.config, start, end)
    job_id = ctx.repo.record_job_start(
        "build_training_samples",
        {
            "start": start,
            "end": end,
            "export": export,
            "chunk_days": chunk_days,
            "build_samples": build_samples,
        },
    )
    try:
        affected = 0
        if build_samples:
            ctx.logger.info("loading securities")
            securities = ctx.repo.get_securities(active_only=False)
            ctx.logger.info("loaded securities rows=%s", len(securities))
            ctx.logger.info("loading company similarity")
            company_similarity = ctx.repo.fetch_table("company_similarity", end=end)
            ctx.logger.info("loaded company similarity rows=%s", len(company_similarity))
            company_id_map = build_company_id_maps(securities)
            neighbor_frame = build_company_neighbor_frame(
                company_similarity=company_similarity,
                symbol_id_map=company_id_map,
                topk=int(ctx.config.company_encoding.get("similarity_topk", 10)),
            )
            upsert_batch_size = int(ctx.config.project.get("training_sample_upsert_batch_size", 2000))
            chunks = _iter_date_chunks(start, end, chunk_days=chunk_days)
            total_chunks = len(chunks)
            steps_per_chunk = 6
            with tqdm(total=total_chunks * steps_per_chunk, desc="Build training samples", unit="step") as progress_bar:
                for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
                    chunk_label = f"chunk {index}/{total_chunks} {chunk_start}->{chunk_end}"
                    progress_bar.set_postfix_str(f"{chunk_label} | starting")
                    ctx.logger.info(
                        "building training samples chunk %s/%s start=%s end=%s",
                        index,
                        total_chunks,
                        chunk_start,
                        chunk_end,
                    )
                    deleted = ctx.repo.delete_training_samples(chunk_start, chunk_end)
                    ctx.logger.info(
                        "deleted existing training samples chunk %s/%s start=%s end=%s rows=%s",
                        index,
                        total_chunks,
                        chunk_start,
                        chunk_end,
                        deleted,
                    )
                    progress_bar.set_postfix_str(f"{chunk_label} | deleted {deleted} old rows")
                    progress_bar.update(1)
                    samples = _build_sample_chunk(
                        ctx=ctx,
                        securities=securities,
                        neighbor_frame=neighbor_frame,
                        full_start=start,
                        full_end=end,
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        progress_bar=progress_bar,
                        chunk_label=chunk_label,
                    )
                    if samples.empty:
                        progress_bar.set_postfix_str(f"{chunk_label} | skipped empty chunk")
                        progress_bar.update(1)
                        continue
                    affected += ctx.repo.upsert_dataframe(
                        TrainingSample,
                        samples,
                        batch_size=upsert_batch_size,
                        show_progress=False,
                        progress_desc=f"Upsert {chunk_start}->{chunk_end}",
                    )
                    progress_bar.set_postfix_str(f"{chunk_label} | upserted {len(samples)} rows")
                    progress_bar.update(1)
            ctx.logger.info("built training samples rows=%s", affected)
        else:
            ctx.logger.info("skip build requested, exporting existing training_samples only")
        if export:
            export_chunk_rows = int(ctx.config.storage.get("export_chunk_rows", 50000))
            export_dir = ctx.config.storage.get("export_dir", "data/exports")
            split_method = str(ctx.config.project.get("split_method", "")).strip().lower()
            if split_method == "random":
                raise ValueError(UNSAFE_RANDOM_SPLIT_MESSAGE)

            split_config = build_time_purged_split_config(ctx.config)
            export_frames = {}
            expected_rows: dict[str, int] = {}
            expected_chunks_total = 0
            for split_name, window in split_config.windows.items():
                split_start = window.start.strftime("%Y-%m-%d")
                split_end = window.end.strftime("%Y-%m-%d")
                split_rows = ctx.repo.count_training_samples(split_start, split_end)
                expected_rows[split_name] = split_rows
                split_chunks = max(1, (split_rows + export_chunk_rows - 1) // export_chunk_rows) if split_rows > 0 else 1
                expected_chunks_total += split_chunks
                ctx.logger.info(
                    "exporting parquet split=%s start=%s end=%s gap_days=%s chunksize=%s rows=%s est_chunks=%s",
                    split_name,
                    split_start,
                    split_end,
                    split_config.gap_days,
                    export_chunk_rows,
                    split_rows,
                    split_chunks,
                )
                export_frames[split_name] = ctx.repo.iter_export_training_samples(
                    split_start,
                    split_end,
                    chunksize=export_chunk_rows,
                )
            with tqdm(total=expected_chunks_total, desc="Export parquet splits", unit="chunk") as export_bar:
                _, split_summary = export_time_purged_splits(
                    export_frames,
                    export_dir,
                    split_config=split_config,
                    progress_bar=export_bar,
                    expected_rows=expected_rows,
                )
            for split_name, stats in split_summary["splits"].items():
                ctx.logger.info(
                    "split=%s rows=%s unique_symbols=%s unique_dates=%s range=%s..%s estimated_samples=%s event_rows=%s",
                    split_name,
                    stats["rows"],
                    stats["unique_symbols"],
                    stats["unique_dates"],
                    stats["min_trade_date"],
                    stats["max_trade_date"],
                    stats["estimated_samples"],
                    stats.get("event_rows"),
                )
            ctx.logger.info(
                "leakage check passed required_gap_days=%s summary=%s",
                split_summary["leakage_check"]["required_gap_days"],
                split_summary["leakage_check"]["boundary_checks"],
            )
        ctx.repo.record_job_end(job_id, "success", affected, "build completed")
    except Exception as exc:
        ctx.logger.exception("build training samples failed")
        ctx.repo.record_job_end(job_id, "failed", message=str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Build training samples")
    parser.add_argument("--export", action="store_true", help="Export train/valid/test parquet")
    parser.add_argument("--export-only", action="store_true", help="Skip rebuild and export existing training_samples")
    parser.add_argument("--chunk-days", type=int, default=90, help="Build samples in date chunks to reduce memory")
    args = parser.parse_args()
    run(
        args.config,
        args.start,
        args.end,
        args.export or args.export_only,
        args.chunk_days,
        build_samples=not args.export_only,
    )


if __name__ == "__main__":
    main()
