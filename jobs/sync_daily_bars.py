from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import build_common_parser
from utils.normalizers import normalize_daily_bars
from warehouse.models import DailyBar

from jobs.common import bootstrap, choose_symbols, resolve_start_end


def run(
    config_path: str,
    start: str | None = None,
    end: str | None = None,
    limit: int | None = None,
    symbols: list[str] | None = None,
    progress_callback: Callable[[int, int, str | None], None] | None = None,
    source_order: list[str] | None = None,
    request_sleep_seconds: float | None = None,
    max_workers_override: int | None = None,
) -> None:
    ctx = bootstrap("sync_daily_bars", config_path)
    if source_order:
        ctx.config.providers["daily_bar_source_order"] = list(source_order)
    start, end = resolve_start_end(ctx.config, start, end)
    job_id = ctx.repo.record_job_start(
        "sync_daily_bars",
        {"start": start, "end": end, "limit": limit, "symbols": symbols, "source_order": source_order},
    )
    affected = 0
    failed_symbols: list[str] = []
    continue_on_symbol_error = bool(ctx.config.jobs.get("continue_on_symbol_error", True))
    if request_sleep_seconds is None:
        request_sleep_seconds = float(ctx.config.jobs.get("request_sleep_seconds", 0.2))
    progress_log_every = int(ctx.config.jobs.get("progress_log_every", 1))
    skip_completed_symbols = bool(ctx.config.jobs.get("skip_completed_symbols", True))
    max_workers = max(
        1,
        int(
            max_workers_override
            if max_workers_override is not None
            else ctx.config.jobs.get("daily_bar_max_workers", ctx.config.jobs.get("max_workers", 1))
            or 1
        ),
    )

    def fetch_symbol(symbol: str) -> tuple[str, list[dict], list[dict]]:
        rows = ctx.market_provider.fetch_daily_bars(symbol, start, end)
        valid_rows, bad_rows = normalize_daily_bars(rows)
        return symbol, valid_rows, bad_rows

    def save_symbol_result(symbol: str, valid_rows: list[dict], bad_rows: list[dict]) -> None:
        nonlocal affected
        if bad_rows:
            ctx.repo.save_bad_records("daily_bars", bad_rows, "bar validation failed")
        affected += ctx.repo.upsert(DailyBar, valid_rows)
        ctx.logger.info("synced daily bars %s rows for %s", len(valid_rows), symbol)

    try:
        securities = ctx.repo.get_securities()
        selected_symbols = choose_symbols(securities, limit, target_symbols=symbols)
        if skip_completed_symbols and selected_symbols:
            calendar = ctx.repo.get_trading_calendar()
            trading_days = calendar.range(start, end)
            existing_counts = ctx.repo.get_symbol_trade_counts("daily_bars", start, end, selected_symbols)
            sec_list_date = {}
            for row in securities[["symbol", "list_date"]].to_dict("records"):
                sec_list_date[str(row["symbol"])] = row.get("list_date")
            start_ts = pd.Timestamp(start)
            trading_day_set = set(trading_days)
            filtered_symbols: list[str] = []
            skipped_completed = 0
            for symbol in selected_symbols:
                list_date = sec_list_date.get(symbol)
                symbol_start = max(start_ts, pd.Timestamp(list_date)) if pd.notna(list_date) else start_ts
                expected_days = sum(1 for day in trading_days if day >= symbol_start)
                existing_days = existing_counts.get(symbol, 0)
                if expected_days > 0 and existing_days >= expected_days:
                    skipped_completed += 1
                    continue
                filtered_symbols.append(symbol)
            selected_symbols = filtered_symbols
            ctx.logger.info("skip completed symbols=%s, remaining=%s", skipped_completed, len(selected_symbols))
        total = len(selected_symbols)
        if progress_callback:
            progress_callback(0, total, None)
        if max_workers <= 1 or total <= 1:
            for idx, symbol in enumerate(selected_symbols, start=1):
                try:
                    if progress_callback:
                        progress_callback(idx - 1, total, symbol)
                    if progress_log_every > 0 and (idx == 1 or idx % progress_log_every == 0):
                        ctx.logger.info("progress %s/%s fetching %s", idx, total, symbol)
                    _, valid_rows, bad_rows = fetch_symbol(symbol)
                    save_symbol_result(symbol, valid_rows, bad_rows)
                    if progress_callback:
                        progress_callback(idx, total, symbol)
                    if request_sleep_seconds > 0:
                        time.sleep(request_sleep_seconds)
                except Exception as exc:
                    failed_symbols.append(symbol)
                    ctx.repo.save_bad_records(
                        "daily_bars_fetch",
                        [{"symbol": symbol, "start": start, "end": end}],
                        str(exc)[:4000],
                    )
                    ctx.logger.warning("failed syncing %s, skipped: %s", symbol, exc)
                    if progress_callback:
                        progress_callback(idx, total, symbol)
                    if not continue_on_symbol_error:
                        raise
        else:
            ctx.logger.info("fetching %s symbols with %s workers", total, max_workers)
            completed = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {executor.submit(fetch_symbol, symbol): symbol for symbol in selected_symbols}
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    completed += 1
                    try:
                        _, valid_rows, bad_rows = future.result()
                        save_symbol_result(symbol, valid_rows, bad_rows)
                    except Exception as exc:
                        failed_symbols.append(symbol)
                        ctx.repo.save_bad_records(
                            "daily_bars_fetch",
                            [{"symbol": symbol, "start": start, "end": end}],
                            str(exc)[:4000],
                        )
                        ctx.logger.warning("failed syncing %s, skipped: %s", symbol, exc)
                        if not continue_on_symbol_error:
                            raise
                    if progress_log_every > 0 and (completed == 1 or completed % progress_log_every == 0):
                        ctx.logger.info("progress %s/%s completed %s", completed, total, symbol)
                    if progress_callback:
                        progress_callback(completed, total, symbol)
                    if request_sleep_seconds > 0:
                        time.sleep(request_sleep_seconds)
        status = "success" if not failed_symbols else "partial_success"
        message = "sync completed" if not failed_symbols else f"skipped {len(failed_symbols)} symbols: {', '.join(failed_symbols[:20])}"
        ctx.repo.record_job_end(job_id, status, affected, message)
    except Exception as exc:
        ctx.logger.exception("sync daily bars failed")
        ctx.repo.record_job_end(job_id, "failed", affected, str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Sync daily bars")
    args = parser.parse_args()
    run(args.config, args.start, args.end, args.limit)


if __name__ == "__main__":
    main()
