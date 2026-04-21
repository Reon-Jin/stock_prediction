from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.event_features import DailyNewsEncoder, build_event_features
from utils.cli import build_common_parser
from warehouse.models import EventFeaturesDaily

from jobs.common import bootstrap, normalize_symbols, resolve_start_end


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


def run(
    config_path: str,
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
    chunk_days: int = 30,
) -> None:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be a positive integer")

    ctx = bootstrap("build_event_features", config_path)
    start, end = resolve_start_end(ctx.config, start, end)
    normalized_symbols = normalize_symbols(symbols)
    job_id = ctx.repo.record_job_start(
        "build_event_features",
        {"start": start, "end": end, "symbols": normalized_symbols, "chunk_days": chunk_days},
    )
    try:
        encoder = DailyNewsEncoder(ctx.config.events)
        chunks = _iter_date_chunks(start, end, chunk_days=chunk_days)
        ctx.repo.delete_event_features(start, end)
        affected = 0

        with tqdm(total=len(chunks), desc="Build event features", unit="chunk") as progress_bar:
            for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
                progress_bar.set_postfix_str(f"{chunk_start}->{chunk_end} loading")
                ctx.logger.info(
                    "building event features chunk %s/%s start=%s end=%s",
                    index,
                    len(chunks),
                    chunk_start,
                    chunk_end,
                )
                daily_bars = ctx.repo.fetch_table("daily_bars", start=chunk_start, end=chunk_end, symbols=normalized_symbols)
                if daily_bars.empty:
                    progress_bar.set_postfix_str(f"{chunk_start}->{chunk_end} skipped empty bars")
                    progress_bar.update(1)
                    continue

                news_norm = ctx.repo.fetch_table("news_norm", start=chunk_start, end=chunk_end, symbols=normalized_symbols)
                features = build_event_features(news_norm, daily_bars, encoder)
                features = features[
                    (pd.to_datetime(features["trade_date"]) >= pd.Timestamp(chunk_start))
                    & (pd.to_datetime(features["trade_date"]) <= pd.Timestamp(chunk_end))
                ].copy()
                if features.empty:
                    progress_bar.set_postfix_str(f"{chunk_start}->{chunk_end} built 0 rows")
                    progress_bar.update(1)
                    continue

                features["data_version"] = ctx.config.project.get("data_version", "v1")
                features["feature_version"] = ctx.config.project.get("feature_version", "f1")
                chunk_affected = ctx.repo.upsert(EventFeaturesDaily, features.to_dict("records"))
                affected += chunk_affected
                progress_bar.set_postfix_str(f"{chunk_start}->{chunk_end} upserted {chunk_affected} rows")
                progress_bar.update(1)

        ctx.logger.info("built event features rows=%s", affected)
        ctx.repo.record_job_end(job_id, "success", affected, "build completed")
    except Exception as exc:
        ctx.logger.exception("build event features failed")
        ctx.repo.record_job_end(job_id, "failed", message=str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Build event features")
    parser.add_argument("--chunk-days", type=int, default=30, help="Build event features in date chunks to reduce memory")
    args = parser.parse_args()
    run(args.config, args.start, args.end, None, args.chunk_days)


if __name__ == "__main__":
    main()
