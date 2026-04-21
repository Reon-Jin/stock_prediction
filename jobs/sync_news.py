from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.event_features import build_event_extractor, normalize_event_records
from utils.cli import build_common_parser
from warehouse.models import NewsNorm, NewsRaw

from jobs.common import bootstrap, resolve_start_end


def run(
    config_path: str,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    symbols: list[str] | None = None,
) -> None:
    ctx = bootstrap("sync_news", config_path)
    start, end = resolve_start_end(ctx.config, start, end)
    job_id = ctx.repo.record_job_start(
        "sync_news",
        {"start": start, "end": end, "limit": limit, "symbols": symbols},
    )
    affected = 0
    failed_dates: list[str] = []
    try:
        calendar = ctx.repo.get_trading_calendar()
        extractor = build_event_extractor(ctx.config.events)
        trade_dates = [item.strftime("%Y-%m-%d") for item in calendar.range(start, end)]
        if limit:
            trade_dates = trade_dates[:limit]
        if symbols:
            ctx.logger.warning("sync_news now ignores symbol filters because news is rebuilt at trade-date level")

        for trade_date in trade_dates:
            try:
                raw_rows = ctx.news_provider.fetch_trade_date_news(trade_date)
                affected += ctx.repo.upsert(NewsRaw, raw_rows)
                raw_df = pd.DataFrame(raw_rows)
                if raw_df.empty:
                    continue
                norm_df = normalize_event_records(raw_df, id_column="news_id", calendar=calendar, extractor=extractor)
                norm_df["data_version"] = ctx.config.project.get("data_version", "v1")
                affected += ctx.repo.upsert(NewsNorm, norm_df.to_dict("records"))
                ctx.logger.info("synced news %s raw rows for trade_date=%s", len(raw_rows), trade_date)
            except Exception as exc:
                failed_dates.append(trade_date)
                ctx.repo.save_bad_records(
                    "news_fetch_or_normalize",
                    [{"trade_date": trade_date, "start": start, "end": end}],
                    str(exc)[:4000],
                )
                ctx.logger.warning("failed syncing news for trade_date=%s, skipped: %s", trade_date, exc)
        status = "success" if not failed_dates else "partial_success"
        message = "sync completed" if not failed_dates else f"skipped {len(failed_dates)} trade_dates: {', '.join(failed_dates[:20])}"
        ctx.repo.record_job_end(job_id, status, affected, message)
    except Exception as exc:
        ctx.logger.exception("sync news failed")
        ctx.repo.record_job_end(job_id, "failed", affected, str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Sync stock news")
    args = parser.parse_args()
    run(args.config, args.start, args.end, args.limit)


if __name__ == "__main__":
    main()
