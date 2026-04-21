from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from providers.provider_disclosure import DisclosureProvider
from providers.provider_financial import FinancialProvider
from providers.provider_fundflow import FundFlowProvider
from providers.provider_market import MarketProvider
from providers.provider_news import NewsProvider
from utils.config import AppConfig, load_config
from utils.logger import get_logger
from utils.symbols import normalize_symbol
from warehouse.db import build_engine
from warehouse.repository import WarehouseRepository


@dataclass(slots=True)
class JobContext:
    config: AppConfig
    repo: WarehouseRepository
    logger: object
    market_provider: MarketProvider
    fundflow_provider: FundFlowProvider
    financial_provider: FinancialProvider
    news_provider: NewsProvider
    disclosure_provider: DisclosureProvider


def bootstrap(job_name: str, config_path: str) -> JobContext:
    from warehouse.schema_init import init_schema

    init_schema(config_path)
    config = load_config(config_path)
    engine = build_engine(config)
    repo = WarehouseRepository(engine)
    logger = get_logger(job_name)
    return JobContext(
        config=config,
        repo=repo,
        logger=logger,
        market_provider=MarketProvider(config),
        fundflow_provider=FundFlowProvider(config),
        financial_provider=FinancialProvider(config),
        news_provider=NewsProvider(config),
        disclosure_provider=DisclosureProvider(config),
    )


def resolve_start_end(config: AppConfig, start: str | None, end: str | None) -> tuple[str, str]:
    project = config.project
    return start or project["default_start"], end or project["default_end"]


def normalize_symbols(symbols: str | Sequence[str] | None) -> list[str]:
    if symbols is None:
        return []
    if isinstance(symbols, str):
        raw_symbols = [symbols]
    else:
        raw_symbols = list(symbols)

    normalized: list[str] = []
    seen: set[str] = set()
    for symbol in raw_symbols:
        value = normalize_symbol(symbol)
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def filter_securities_by_symbols(securities: pd.DataFrame, symbols: str | Sequence[str] | None) -> pd.DataFrame:
    normalized_symbols = normalize_symbols(symbols)
    if not normalized_symbols or securities.empty or "symbol" not in securities.columns:
        return securities.copy()
    mask = securities["symbol"].astype(str).str.upper().isin(normalized_symbols)
    return securities.loc[mask].copy()


def choose_symbols(
    securities: pd.DataFrame,
    limit: int | None = None,
    target_symbols: str | Sequence[str] | None = None,
) -> list[str]:
    df = securities.copy()
    df = df[df["status"] == "active"]
    normalized_targets = normalize_symbols(target_symbols)
    if normalized_targets:
        df = filter_securities_by_symbols(df, normalized_targets)
        available = set(df["symbol"].dropna().astype(str).str.upper().tolist())
        symbols = [symbol for symbol in normalized_targets if symbol in available]
    else:
        symbols = df["symbol"].dropna().astype(str).tolist()
    return symbols[:limit] if limit else symbols


def chunked(items: Iterable[str], size: int) -> list[list[str]]:
    items = list(items)
    return [items[i : i + size] for i in range(0, len(items), size)]
