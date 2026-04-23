from __future__ import annotations

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.cli import build_common_parser
from utils.normalizers import normalize_securities
from utils.symbols import normalize_symbol
from warehouse.models import Security, TradeCalendar

from jobs.common import bootstrap, resolve_start_end


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = [str(column).strip() for column in columns]
    for candidate in candidates:
        for column in normalized:
            if column == candidate:
                return column
    for candidate in candidates:
        for column in normalized:
            if candidate in column:
                return column
    return None


def _enrich_with_bulk_spot(ctx, securities: pd.DataFrame) -> pd.DataFrame:
    try:
        ak = ctx.market_provider.require_akshare()
        spot = ak.stock_zh_a_spot_em()
        if spot is None or spot.empty:
            return securities
        spot_columns = [str(column).strip() for column in spot.columns]
        rename_map: dict[str, str] = {}
        symbol_col = _find_column(spot_columns, ["代码"])
        name_col = _find_column(spot_columns, ["名称"])
        industry_col = _find_column(spot_columns, ["所属行业", "所处行业", "行业"])
        list_date_col = _find_column(spot_columns, ["上市时间"])
        if symbol_col:
            rename_map[symbol_col] = "symbol"
        if name_col:
            rename_map[name_col] = "name_spot"
        if industry_col:
            rename_map[industry_col] = "industry_spot"
        if list_date_col:
            rename_map[list_date_col] = "list_date_spot"
        spot = spot.rename(columns=rename_map)
        keep_cols = [column for column in ["symbol", "name_spot", "industry_spot", "list_date_spot"] if column in spot.columns]
        if "symbol" not in keep_cols:
            return securities
        spot = spot[keep_cols].copy()
        spot["symbol"] = spot["symbol"].astype(str).str.strip()
        enriched = securities.merge(spot, on="symbol", how="left")
        if "name_spot" in enriched.columns:
            enriched["name"] = enriched["name"].combine_first(enriched["name_spot"])
        if "industry_spot" in enriched.columns:
            enriched["industry"] = enriched["industry"].combine_first(enriched["industry_spot"])
        if "list_date_spot" in enriched.columns:
            enriched["list_date"] = enriched["list_date"].combine_first(enriched["list_date_spot"])
        return enriched[[column for column in enriched.columns if not column.endswith("_spot")]]
    except Exception as exc:
        ctx.logger.warning("bulk spot enrichment unavailable, continue with current securities snapshot: %s", exc)
        return securities


def _prepare_single_symbol_security(ctx, target_symbol: str, existing: pd.DataFrame) -> pd.DataFrame:
    normalized_symbol = normalize_symbol(target_symbol)
    security = pd.DataFrame()
    if not existing.empty and "symbol" in existing.columns:
        security = existing[existing["symbol"].astype(str).str.upper() == normalized_symbol].copy()

    if security.empty:
        ctx.logger.info("target symbol %s not found in local securities snapshot; fetching universe once", normalized_symbol)
        fetched = pd.DataFrame(normalize_securities(ctx.market_provider.fetch_securities()))
        if fetched.empty:
            raise ValueError(f"failed to locate security metadata for {normalized_symbol}")
        security = fetched[fetched["symbol"].astype(str).str.upper() == normalized_symbol].copy()
        if security.empty:
            raise ValueError(f"target symbol {normalized_symbol} not found in fetched securities universe")
    else:
        ctx.logger.info("reusing existing security snapshot for %s", normalized_symbol)

    detail = {}
    if bool(ctx.config.providers.get("security_detail_enabled", False)):
        try:
            detail = ctx.market_provider.fetch_security_profile_cninfo(normalized_symbol)
        except Exception as exc:
            ctx.logger.warning("single-symbol profile enrichment unavailable for %s: %s", normalized_symbol, exc)

    if detail:
        for column in ["name", "industry", "list_date"]:
            value = detail.get(column)
            if value is None or pd.isna(value):
                continue
            if column not in security.columns:
                security[column] = None
            security[column] = security[column].fillna(value)

    if "exchange" not in security.columns:
        security["exchange"] = normalized_symbol.split(".")[1]
    if "board" not in security.columns:
        security["board"] = ctx.market_provider._infer_board(normalized_symbol)
    if "delist_date" not in security.columns:
        security["delist_date"] = None
    if "is_st" not in security.columns:
        security["is_st"] = False
    if "status" not in security.columns:
        security["status"] = "active"

    security["symbol"] = normalized_symbol
    security["industry"] = security["industry"].fillna("Unknown")
    security["board"] = security["board"].fillna("Unknown")
    security["status"] = security["status"].fillna("active")
    security["is_st"] = security["is_st"].fillna(False).astype(bool)
    security["data_version"] = ctx.config.project.get("data_version", "v1")

    keep_columns = [
        "symbol",
        "name",
        "exchange",
        "board",
        "industry",
        "list_date",
        "delist_date",
        "is_st",
        "status",
        "data_version",
    ]
    for column in keep_columns:
        if column not in security.columns:
            security[column] = None
    return security[keep_columns].drop_duplicates("symbol").copy()


def run(
    config_path: str,
    start: str | None = None,
    end: str | None = None,
    target_symbol: str | None = None,
) -> None:
    ctx = bootstrap("sync_securities", config_path)
    start, end = resolve_start_end(ctx.config, start, end)
    normalized_target = normalize_symbol(target_symbol) if target_symbol else None
    job_id = ctx.repo.record_job_start(
        "sync_securities",
        {"start": start, "end": end, "target_symbol": normalized_target},
    )
    try:
        existing = ctx.repo.get_securities(active_only=False)
        if normalized_target:
            securities = _prepare_single_symbol_security(ctx, normalized_target, existing)
        else:
            securities = pd.DataFrame(normalize_securities(ctx.market_provider.fetch_securities()))
            if securities.empty:
                securities = pd.DataFrame(columns=["symbol", "name", "exchange", "board", "industry", "list_date", "delist_date", "is_st", "status", "data_version"])
            if not existing.empty:
                existing = existing.rename(columns={column: f"{column}_existing" for column in existing.columns if column != "symbol"})
                securities = securities.merge(existing, on="symbol", how="left")
                for column in ["name", "exchange", "board", "industry", "list_date", "delist_date", "is_st", "status"]:
                    existing_column = f"{column}_existing"
                    if existing_column in securities.columns:
                        securities[column] = securities[column].combine_first(securities[existing_column])
                securities = securities[[column for column in securities.columns if not column.endswith("_existing")]]

            securities = _enrich_with_bulk_spot(ctx, securities)

            detail_enabled = bool(ctx.config.providers.get("security_detail_enabled", False))
            if detail_enabled:
                refresh_all_profiles = bool(ctx.config.providers.get("security_detail_refresh_all", False))
                missing_profile_mask = (
                    refresh_all_profiles
                    | securities["industry"].isna()
                    | securities["list_date"].isna()
                    | securities["name"].isna()
                )
                detail_symbols = securities.loc[missing_profile_mask, "symbol"].dropna().astype(str).tolist()
                max_detail_symbols = int(ctx.config.providers.get("security_detail_max_symbols", 50))
                detail_fail_fast_threshold = int(ctx.config.providers.get("security_detail_fail_fast_threshold", 5))
                detail_max_workers = int(ctx.config.providers.get("security_detail_max_workers", 1))
                if detail_max_workers > 1:
                    ctx.logger.warning(
                        "security detail enrichment uses cninfo/mini_racer; forcing max_workers=1 to avoid native crashes"
                    )
                    detail_max_workers = 1
                if max_detail_symbols >= 0 and len(detail_symbols) > max_detail_symbols:
                    ctx.logger.warning(
                        "detail enrichment symbols=%s exceeds cap=%s; only enriching the first capped subset after bulk spot fill",
                        len(detail_symbols),
                        max_detail_symbols,
                    )
                    detail_symbols = detail_symbols[:max_detail_symbols]
                if detail_symbols:
                    details: list[dict[str, object]] = []
                    consecutive_detail_failures = 0
                    with ThreadPoolExecutor(max_workers=max(1, detail_max_workers)) as executor:
                        future_map = {
                            executor.submit(ctx.market_provider.fetch_security_profile_cninfo, symbol): symbol
                            for symbol in detail_symbols
                        }
                        progress = tqdm(total=len(future_map), desc="Enrich security profiles", unit="stock")
                        try:
                            for future in as_completed(future_map):
                                symbol = future_map[future]
                                try:
                                    details.append(future.result())
                                    consecutive_detail_failures = 0
                                except Exception as exc:
                                    consecutive_detail_failures += 1
                                    ctx.logger.warning("failed enriching security profile for %s: %s", symbol, exc)
                                    if detail_fail_fast_threshold > 0 and consecutive_detail_failures >= detail_fail_fast_threshold:
                                        ctx.logger.warning(
                                            "security detail source appears unavailable; stop detail enrichment early after %s consecutive failures",
                                            consecutive_detail_failures,
                                        )
                                        for pending_future in future_map:
                                            pending_future.cancel()
                                        break
                                finally:
                                    progress.update(1)
                        finally:
                            progress.close()
                    if details:
                        detail_df = pd.DataFrame(details).drop_duplicates("symbol")
                        securities = securities.merge(detail_df, on="symbol", how="left", suffixes=("", "_detail"))
                        for column in ["name", "industry", "list_date"]:
                            detail_column = f"{column}_detail"
                            if detail_column in securities.columns:
                                securities[column] = securities[detail_column].combine_first(securities[column])
                        securities = securities[[column for column in securities.columns if not column.endswith("_detail")]]
            else:
                ctx.logger.info("security detail enrichment disabled; using bulk/basic security snapshot only")

            securities["industry"] = securities["industry"].fillna("Unknown")
            securities["board"] = securities["board"].fillna("Unknown")
            securities["status"] = securities["status"].fillna("active")
            securities["is_st"] = securities["is_st"].fillna(False).astype(bool)
        securities_records = securities.to_dict("records")
        calendar_rows = ctx.market_provider.fetch_trade_calendar(start, end)
        sec_count = ctx.repo.upsert(Security, securities_records)
        cal_count = ctx.repo.upsert(TradeCalendar, calendar_rows)
        ctx.logger.info("upserted %s securities and %s trade calendar rows", sec_count, cal_count)
        ctx.repo.record_job_end(job_id, "success", sec_count + cal_count, "sync completed")
    except Exception as exc:
        ctx.logger.exception("sync securities failed")
        ctx.repo.record_job_end(job_id, "failed", message=str(exc))
        raise


def main() -> None:
    parser = build_common_parser("Sync A-share securities and trade calendar")
    args = parser.parse_args()
    run(args.config, args.start, args.end)


if __name__ == "__main__":
    main()
