from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.event_features import build_event_extractor, normalize_event_records
from utils.cli import build_common_parser
from warehouse.models import AnnouncementNorm, AnnouncementRaw

from jobs.common import bootstrap, choose_symbols, resolve_start_end


def run(
    config_path: str,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    symbols: list[str] | None = None,
) -> None:
    ctx = bootstrap("sync_announcements", config_path)
    start, end = resolve_start_end(ctx.config, start, end)
    job_id = ctx.repo.record_job_start(
        "sync_announcements",
        {"start": start, "end": end, "limit": limit, "symbols": symbols},
    )
    affected = 0
    try:
        securities = ctx.repo.get_securities()
        calendar = ctx.repo.get_trading_calendar()
        extractor = build_event_extractor(ctx.config.events)
        for symbol in choose_symbols(securities, limit, target_symbols=symbols):
            raw_rows = ctx.disclosure_provider.fetch_announcements(symbol, start, end)
            affected += ctx.repo.upsert(AnnouncementRaw, raw_rows)
            raw_df = pd.DataFrame(raw_rows)
            if raw_df.empty:
                continue
            norm_df = normalize_event_records(
                raw_df.rename(columns={"announcement_id": "news_id"}),
                id_column="news_id",
                calendar=calendar,
                extractor=extractor,
            ).rename(columns={"news_id": "announcement_id"})
            norm_df["data_version"] = ctx.config.project.get("data_version", "v1")
            affected += ctx.repo.upsert(AnnouncementNorm, norm_df.to_dict("records"))
            ctx.logger.info("synced announcements %s raw rows for %s", len(raw_rows), symbol)
        ctx.repo.record_job_end(job_id, "success", affected, "sync completed")
    except Exception as exc:
        ctx.logger.exception("sync announcements failed")
        ctx.repo.record_job_end(job_id, "failed", affected, str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Sync stock announcements")
    args = parser.parse_args()
    run(args.config, args.start, args.end, args.limit)


if __name__ == "__main__":
    main()
