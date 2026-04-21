from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.company_profile_builder import build_company_profiles
from utils.cli import build_common_parser
from warehouse.models import CompanyProfile

from jobs.common import bootstrap, filter_securities_by_symbols, normalize_symbols, resolve_start_end


def run(
    config_path: str,
    start: str | None = None,
    end: str | None = None,
    symbols: list[str] | None = None,
) -> None:
    ctx = bootstrap("build_company_profiles", config_path)
    if not bool(ctx.config.company_encoding.get("enabled", True)):
        ctx.logger.info("company encoding disabled, skipping company profile build")
        return
    start, end = resolve_start_end(ctx.config, start, end)
    lookback_days = int(ctx.config.company_encoding.get("profile_lookback_days", 120))
    profile_version = str(ctx.config.company_encoding.get("profile_version", "cp1"))
    normalized_symbols = normalize_symbols(symbols)
    job_id = ctx.repo.record_job_start(
        "build_company_profiles",
        {
            "start": start,
            "end": end,
            "lookback_days": lookback_days,
            "profile_version": profile_version,
            "symbols": normalized_symbols,
        },
    )
    try:
        start_buffer = (pd.Timestamp(start) - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
        daily_bars = ctx.repo.fetch_table("daily_bars", start=start_buffer, end=end, symbols=normalized_symbols)
        index_bars = ctx.repo.fetch_table("index_bars", start=start_buffer, end=end)
        financial_snapshot = ctx.repo.fetch_table("financial_snapshot", symbols=normalized_symbols)
        securities = filter_securities_by_symbols(ctx.repo.get_securities(active_only=False), normalized_symbols)
        profiles = build_company_profiles(
            daily_bars=daily_bars,
            index_bars=index_bars,
            financial_snapshot=financial_snapshot,
            securities=securities,
            start=start,
            end=end,
            lookback_days=lookback_days,
            profile_version=profile_version,
        )
        affected = ctx.repo.upsert(CompanyProfile, profiles.to_dict("records"))
        ctx.logger.info("built company profiles rows=%s", affected)
        ctx.repo.record_job_end(job_id, "success", affected, "build completed")
    except Exception as exc:
        ctx.logger.exception("build company profiles failed")
        ctx.repo.record_job_end(job_id, "failed", message=str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Build company profile snapshots")
    args = parser.parse_args()
    run(args.config, args.start, args.end)


if __name__ == "__main__":
    main()
