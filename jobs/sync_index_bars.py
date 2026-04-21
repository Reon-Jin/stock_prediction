from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import build_common_parser
from warehouse.models import IndexBar

from jobs.common import bootstrap, resolve_start_end


DEFAULT_INDICES = ["000300.SH", "399006.SZ", "000001.SH"]


def run(config_path: str, start: str | None = None, end: str | None = None) -> None:
    ctx = bootstrap("sync_index_bars", config_path)
    start, end = resolve_start_end(ctx.config, start, end)
    job_id = ctx.repo.record_job_start("sync_index_bars", {"start": start, "end": end})
    affected = 0
    try:
        for index_code in DEFAULT_INDICES:
            rows = ctx.market_provider.fetch_index_bars(index_code, start, end)
            affected += ctx.repo.upsert(IndexBar, rows)
            ctx.logger.info("synced index %s rows=%s", index_code, len(rows))
        ctx.repo.record_job_end(job_id, "success", affected, "sync completed")
    except Exception as exc:
        ctx.logger.exception("sync index bars failed")
        ctx.repo.record_job_end(job_id, "failed", affected, str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Sync index bars")
    args = parser.parse_args()
    run(args.config, args.start, args.end)


if __name__ == "__main__":
    main()
