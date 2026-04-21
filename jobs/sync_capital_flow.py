from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import build_common_parser
from warehouse.models import CapitalFlowDaily

from jobs.common import bootstrap, choose_symbols, resolve_start_end


def run(
    config_path: str,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    symbols: list[str] | None = None,
) -> None:
    ctx = bootstrap("sync_capital_flow", config_path)
    start, end = resolve_start_end(ctx.config, start, end)
    job_id = ctx.repo.record_job_start(
        "sync_capital_flow",
        {"start": start, "end": end, "limit": limit, "symbols": symbols},
    )
    affected = 0
    failed_symbols: list[str] = []
    total_attempts = 0
    consecutive_failures = 0
    continue_on_symbol_error = bool(ctx.config.jobs.get("continue_on_symbol_error", True))
    fail_fast_threshold = int(ctx.config.jobs.get("capital_flow_fail_fast_threshold", 20))
    public_window_days = int(ctx.config.jobs.get("capital_flow_public_window_days", 60))
    if public_window_days > 0:
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=public_window_days)
        if pd.Timestamp(end) < cutoff:
            message = (
                f"public THS fund-flow source only covers recent window; "
                f"requested end={end} older than cutoff={cutoff.date()}; skipped sync"
            )
            ctx.logger.warning(message)
            ctx.repo.record_job_end(job_id, "partial_success", 0, message)
            return
    try:
        securities = ctx.repo.get_securities()
        for symbol in choose_symbols(securities, limit, target_symbols=symbols):
            total_attempts += 1
            try:
                rows = ctx.fundflow_provider.fetch_capital_flow(symbol, start, end)
                affected += ctx.repo.upsert(CapitalFlowDaily, rows)
                consecutive_failures = 0
                ctx.logger.info("synced capital flow %s rows for %s", len(rows), symbol)
            except Exception as exc:
                failed_symbols.append(symbol)
                consecutive_failures += 1
                ctx.repo.save_bad_records(
                    "capital_flow_fetch",
                    [{"symbol": symbol, "start": start, "end": end}],
                    str(exc)[:4000],
                )
                ctx.logger.warning("failed syncing capital flow for %s, skipped: %s", symbol, exc)
                if not continue_on_symbol_error:
                    raise
                if fail_fast_threshold > 0 and consecutive_failures >= fail_fast_threshold:
                    ctx.logger.warning(
                        "capital flow upstream appears unavailable; stop early after %s consecutive failures and continue pipeline",
                        consecutive_failures,
                    )
                    break
        status = "success" if not failed_symbols else "partial_success"
        if not failed_symbols:
            message = "sync completed"
        else:
            message = (
                f"attempted {total_attempts} symbols, skipped {len(failed_symbols)} symbols: "
                f"{', '.join(failed_symbols[:20])}"
            )
        ctx.repo.record_job_end(job_id, status, affected, message)
    except Exception as exc:
        ctx.logger.exception("sync capital flow failed")
        ctx.repo.record_job_end(job_id, "failed", affected, str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Sync capital flow")
    args = parser.parse_args()
    run(args.config, args.start, args.end, args.limit)


if __name__ == "__main__":
    main()
