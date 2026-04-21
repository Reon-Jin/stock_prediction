from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import build_common_parser
from warehouse.models import SectorDaily

from jobs.common import bootstrap, resolve_start_end


def run(config_path: str, start: str | None = None, end: str | None = None, limit: int | None = None) -> None:
    ctx = bootstrap("sync_sector_daily", config_path)
    start, end = resolve_start_end(ctx.config, start, end)
    job_id = ctx.repo.record_job_start("sync_sector_daily", {"start": start, "end": end, "limit": limit})
    continue_on_error = bool(ctx.config.jobs.get("continue_on_symbol_error", True))
    try:
        rows = ctx.market_provider.fetch_sector_daily(start, end, limit)
        affected = ctx.repo.upsert(SectorDaily, rows)
        ctx.logger.info("synced sector rows=%s", affected)
        ctx.repo.record_job_end(job_id, "success", affected, "sync completed")
    except Exception as exc:
        ctx.logger.exception("sync sector daily failed")
        existing_rows = ctx.repo.fetch_table("sector_daily", start=start, end=end)
        if continue_on_error:
            reused = len(existing_rows)
            message = (
                f"sector sync skipped after upstream failure; reused existing rows={reused}; error={str(exc)[:500]}"
            )
            ctx.logger.warning(message)
            ctx.repo.record_job_end(job_id, "partial_success", reused, message)
            return
        ctx.repo.record_job_end(job_id, "failed", message=str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Sync sector daily")
    args = parser.parse_args()
    run(args.config, args.start, args.end, args.limit)


if __name__ == "__main__":
    main()
