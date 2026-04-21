from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import build_common_parser
from warehouse.models import FinancialSnapshot

from jobs.common import bootstrap, choose_symbols, resolve_start_end


def run(
    config_path: str,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    symbols: list[str] | None = None,
) -> None:
    ctx = bootstrap("sync_financial_snapshot", config_path)
    start, end = resolve_start_end(ctx.config, start, end)
    job_id = ctx.repo.record_job_start(
        "sync_financial_snapshot",
        {"start": start, "end": end, "limit": limit, "symbols": symbols},
    )
    affected = 0
    failed_symbols: list[str] = []
    continue_on_symbol_error = bool(ctx.config.jobs.get("continue_on_symbol_error", True))
    try:
        securities = ctx.repo.get_securities()
        for symbol in choose_symbols(securities, limit, target_symbols=symbols):
            try:
                rows = ctx.financial_provider.fetch_financial_snapshot(symbol, start, end)
                affected += ctx.repo.upsert(FinancialSnapshot, rows)
                ctx.logger.info("synced financial snapshot %s rows for %s", len(rows), symbol)
            except Exception as exc:
                failed_symbols.append(symbol)
                ctx.repo.save_bad_records(
                    "financial_snapshot_fetch",
                    [{"symbol": symbol, "start": start, "end": end}],
                    str(exc)[:4000],
                )
                ctx.logger.warning("failed syncing financial snapshot for %s, skipped: %s", symbol, exc)
                if not continue_on_symbol_error:
                    raise
        status = "success" if not failed_symbols else "partial_success"
        message = "sync completed" if not failed_symbols else f"skipped {len(failed_symbols)} symbols: {', '.join(failed_symbols[:20])}"
        ctx.repo.record_job_end(job_id, status, affected, message)
    except Exception as exc:
        ctx.logger.exception("sync financial snapshot failed")
        ctx.repo.record_job_end(job_id, "failed", affected, str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Sync financial snapshot")
    args = parser.parse_args()
    run(args.config, args.start, args.end, args.limit)


if __name__ == "__main__":
    main()
