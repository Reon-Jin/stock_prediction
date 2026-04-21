from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jobs.build_company_profiles import run as run_build_company_profiles
from jobs.build_prediction_samples import run as run_build_prediction_samples
from jobs.build_company_similarity import run as run_build_company_similarity
from jobs.build_event_features import run as run_build_event_features
from jobs.build_training_samples import run as run_build_training_samples
from jobs.sync_announcements import run as run_sync_announcements
from jobs.sync_capital_flow import run as run_sync_capital_flow
from jobs.sync_daily_bars import run as run_sync_daily_bars
from jobs.sync_financial_snapshot import run as run_sync_financial_snapshot
from jobs.sync_index_bars import run as run_sync_index_bars
from jobs.sync_news import run as run_sync_news
from jobs.sync_sector_daily import run as run_sync_sector_daily
from jobs.sync_securities import run as run_sync_securities
from utils.logger import get_logger
from warehouse.schema_init import init_schema


def _resolve_today_window(target_date: str) -> tuple[str, str]:
    end_ts = pd.Timestamp(target_date)
    start_ts = end_ts - pd.Timedelta(days=180)
    return start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full A-share dataset generation pipeline")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config yaml")
    parser.add_argument("--start", default=None, help="Start date, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date, YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=None, help="Optional symbol or sector limit for debugging")
    parser.add_argument("--skip-text", action="store_true", help="Skip news and announcement sync")
    parser.add_argument("--chunk-days", type=int, default=90, help="Build training samples in date chunks")
    parser.add_argument(
        "--today",
        nargs="?",
        const="__ALL__",
        default=None,
        help="Build prediction-only samples for the target date; optionally pass one symbol, e.g. --today 000001.SZ",
    )
    args = parser.parse_args()

    logger = get_logger("run_full_pipeline")
    logger.info("initializing schema")
    init_schema(args.config)

    effective_start = args.start
    effective_end = args.end
    target_symbol = None if args.today in (None, "__ALL__") else str(args.today).strip()
    target_symbols = [target_symbol] if target_symbol else None
    if args.today is not None:
        target_date = args.end or args.start or pd.Timestamp.now().strftime("%Y-%m-%d")
        effective_start, effective_end = _resolve_today_window(target_date)
        logger.info(
            "today mode enabled target_date=%s sync_window=%s..%s target_symbol=%s",
            target_date,
            effective_start,
            effective_end,
            target_symbol or "ALL",
        )
        text_start = target_date
        text_end = target_date
    else:
        text_start = effective_start
        text_end = effective_end

    logger.info("syncing securities and calendar")
    run_sync_securities(args.config, effective_start, effective_end, target_symbol=target_symbol)

    logger.info("syncing bars and factors")
    run_sync_daily_bars(args.config, effective_start, effective_end, args.limit, symbols=target_symbols)
    if target_symbol:
        logger.info("single-symbol today mode skips index and sector sync; reusing cached market context if available")
    else:
        run_sync_index_bars(args.config, effective_start, effective_end)
        run_sync_sector_daily(args.config, effective_start, effective_end, args.limit)
    run_sync_capital_flow(args.config, effective_start, effective_end, args.limit, symbols=target_symbols)
    run_sync_financial_snapshot(args.config, effective_start, effective_end, args.limit, symbols=target_symbols)

    if not args.skip_text:
        logger.info("syncing news and announcements")
        run_sync_news(args.config, text_start, text_end, args.limit, symbols=target_symbols)
        if args.today is not None:
            logger.info("today mode skips announcement sync because daily event vectors no longer depend on announcements")
        else:
            run_sync_announcements(args.config, text_start, text_end, args.limit, symbols=target_symbols)

    logger.info("building event features")
    run_build_event_features(args.config, text_start, text_end, symbols=target_symbols)

    logger.info("building company profiles")
    run_build_company_profiles(args.config, effective_start, effective_end, symbols=target_symbols)
    if target_symbol:
        logger.info("single-symbol today mode skips company similarity rebuild and reuses existing similarity data if available")
    else:
        logger.info("building company similarity graph")
        run_build_company_similarity(args.config, effective_start, effective_end)

    if args.today is not None:
        logger.info("building prediction samples and exporting today parquet")
        run_build_prediction_samples(args.config, target_date=effective_end, target_symbol=target_symbol)
        logger.info("today prediction pipeline finished")
        return

    logger.info("building training samples and exporting parquet")
    run_build_training_samples(args.config, effective_start, effective_end, export=True, chunk_days=args.chunk_days)
    logger.info("pipeline finished")


if __name__ == "__main__":
    main()
