from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import Select, func, select
from sqlalchemy import Boolean as SABoolean
from sqlalchemy import Date as SADate
from sqlalchemy import DateTime as SADateTime
from sqlalchemy import Float as SAFloat
from sqlalchemy import Integer as SAInteger
from sqlalchemy import JSON as SAJSON
from sqlalchemy import String as SAString
from sqlalchemy import Text as SAText
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from tqdm.auto import tqdm

from datasets.pytorch_dataset import (
    DEFAULT_COMPANY_ID_COLUMNS,
    DEFAULT_COMPANY_PROFILE_COLUMNS,
    DEFAULT_LABEL_GROUPS,
    DEFAULT_MKT_COLUMNS,
    DEFAULT_SEQ_COLUMNS,
    DEFAULT_TAB_COLUMNS,
    EVENT_EMBEDDING_COLUMN,
)
from utils.dates import build_calendar_from_series, default_business_calendar
from warehouse.db import build_engine, build_session_factory, session_scope
from warehouse.models import (
    AnnouncementNorm,
    AnnouncementRaw,
    BadRecord,
    CapitalFlowDaily,
    CompanyProfile,
    CompanySimilarity,
    DailyBar,
    EventFeaturesDaily,
    FinancialSnapshot,
    IndexBar,
    JobRun,
    NewsNorm,
    NewsRaw,
    Security,
    SectorDaily,
    TradeCalendar,
    TrainingSample,
)


MODEL_MAP = {
    "securities": Security,
    "trade_calendar": TradeCalendar,
    "daily_bars": DailyBar,
    "index_bars": IndexBar,
    "sector_daily": SectorDaily,
    "capital_flow_daily": CapitalFlowDaily,
    "financial_snapshot": FinancialSnapshot,
    "news_raw": NewsRaw,
    "news_norm": NewsNorm,
    "announcement_raw": AnnouncementRaw,
    "announcement_norm": AnnouncementNorm,
    "event_features_daily": EventFeaturesDaily,
    "company_profiles": CompanyProfile,
    "company_similarity": CompanySimilarity,
    "training_samples": TrainingSample,
}

TRAINING_SAMPLE_EXPORT_COLUMNS = [
    "symbol",
    "trade_date",
    "name",
    "industry_sw",
    "board",
    *DEFAULT_COMPANY_ID_COLUMNS,
    "company_profile_version",
    "list_days",
    *DEFAULT_SEQ_COLUMNS,
    *DEFAULT_TAB_COLUMNS,
    *DEFAULT_COMPANY_PROFILE_COLUMNS,
    EVENT_EMBEDDING_COLUMN,
    *DEFAULT_MKT_COLUMNS,
    *[column for columns in DEFAULT_LABEL_GROUPS.values() for column in columns],
    "neighbor_symbol_ids",
    "neighbor_scores",
    "sample_status",
    "data_version",
    "feature_version",
]


class WarehouseRepository:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.session_factory = build_session_factory(engine)

    @classmethod
    def from_config_path(cls, config_path: str = "configs/config.yaml") -> "WarehouseRepository":
        from utils.config import load_config

        config = load_config(config_path)
        return cls(build_engine(config))

    def upsert(self, model: Any, rows: list[dict[str, Any]]) -> int:
        if not rows:
            return 0
        table = model.__table__
        table_name = table.name
        table_columns = [column.name for column in table.columns]
        primary_keys = {column.name for column in table.primary_key.columns}
        auto_primary_keys = {
            column.name
            for column in table.primary_key.columns
            if getattr(column, "autoincrement", False) is True
        }
        timestamp_now = datetime.utcnow()

        normalized_rows: list[dict[str, Any]] = []
        for row in rows:
            normalized_row = dict(row)
            if "created_at" in table_columns and "created_at" not in normalized_row:
                normalized_row["created_at"] = timestamp_now
            if "updated_at" in table_columns and "updated_at" not in normalized_row:
                normalized_row["updated_at"] = timestamp_now
            normalized_rows.append(normalized_row)

        provided_columns = [
            column
            for column in table_columns
            if column not in auto_primary_keys and any(column in row for row in normalized_rows)
        ]
        if not provided_columns:
            return 0

        update_columns = [column for column in provided_columns if column not in primary_keys and column != "created_at"]
        column_sql = ", ".join(f"`{column}`" for column in provided_columns)
        placeholder_sql = ", ".join(["%s"] * len(provided_columns))
        update_sql = ", ".join(f"`{column}`=VALUES(`{column}`)" for column in update_columns)
        sql = f"INSERT INTO `{table_name}` ({column_sql}) VALUES ({placeholder_sql})"
        if update_sql:
            sql += f" ON DUPLICATE KEY UPDATE {update_sql}"

        batch_size = 200
        conn = self.engine.raw_connection()
        try:
            cursor = conn.cursor()
            for left in range(0, len(normalized_rows), batch_size):
                batch = normalized_rows[left : left + batch_size]
                values = [tuple(self._normalize_value(row.get(column)) for column in provided_columns) for row in batch]
                cursor.executemany(sql, values)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return len(normalized_rows)

    def upsert_dataframe(
        self,
        model: Any,
        df: pd.DataFrame,
        batch_size: int = 1000,
        show_progress: bool = False,
        progress_desc: str | None = None,
    ) -> int:
        if df.empty:
            return 0

        table = model.__table__
        table_name = table.name
        table_columns = [column.name for column in table.columns]
        primary_keys = {column.name for column in table.primary_key.columns}
        auto_primary_keys = {
            column.name
            for column in table.primary_key.columns
            if getattr(column, "autoincrement", False) is True
        }
        timestamp_now = datetime.utcnow()

        provided_columns = [column for column in table_columns if column in df.columns and column not in auto_primary_keys]
        has_created_at = "created_at" in table_columns and "created_at" not in df.columns
        has_updated_at = "updated_at" in table_columns and "updated_at" not in df.columns
        if has_created_at:
            provided_columns.append("created_at")
        if has_updated_at:
            provided_columns.append("updated_at")
        if not provided_columns:
            return 0

        update_columns = [column for column in provided_columns if column not in primary_keys and column != "created_at"]
        column_sql = ", ".join(f"`{column}`" for column in provided_columns)
        placeholder_sql = ", ".join(["%s"] * len(provided_columns))
        update_sql = ", ".join(f"`{column}`=VALUES(`{column}`)" for column in update_columns)
        sql = f"INSERT INTO `{table_name}` ({column_sql}) VALUES ({placeholder_sql})"
        if update_sql:
            sql += f" ON DUPLICATE KEY UPDATE {update_sql}"

        total = 0
        total_batches = (len(df) + batch_size - 1) // batch_size
        conn = self.engine.raw_connection()
        try:
            cursor = conn.cursor()
            batch_iter = range(0, len(df), batch_size)
            if show_progress and total_batches > 1:
                batch_iter = tqdm(
                    batch_iter,
                    total=total_batches,
                    desc=progress_desc or f"Upsert {table_name}",
                    unit="batch",
                    leave=False,
                )
            for left in batch_iter:
                batch_df = df.iloc[left : left + batch_size]
                if has_created_at or has_updated_at:
                    batch_df = batch_df.copy()
                    if has_created_at:
                        batch_df["created_at"] = timestamp_now
                    if has_updated_at:
                        batch_df["updated_at"] = timestamp_now
                batch_df = batch_df.loc[:, provided_columns]
                values = [
                    tuple(self._normalize_value(value) for value in row)
                    for row in batch_df.itertuples(index=False, name=None)
                ]
                cursor.executemany(sql, values)
                total += len(batch_df)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return total

    @staticmethod
    def _normalize_value(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, default=str)
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        return value

    @classmethod
    def _normalize_json_payload(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): cls._normalize_json_payload(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._normalize_json_payload(item) for item in value]
        if isinstance(value, (pd.Timestamp, datetime)):
            return pd.Timestamp(value).isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    def record_job_start(self, job_name: str, params: dict[str, Any]) -> int:
        with session_scope(self.session_factory) as session:
            job = JobRun(
                job_name=job_name,
                start_time=datetime.utcnow(),
                status="running",
                params_json=self._normalize_json_payload(params),
            )
            session.add(job)
            session.flush()
            return int(job.id)

    def record_job_end(
        self,
        job_run_id: int,
        status: str,
        rows_affected: int | None = None,
        message: str | None = None,
    ) -> None:
        if message is not None and len(message) > 4000:
            message = message[:4000] + "... [truncated]"
        with session_scope(self.session_factory) as session:
            job = session.get(JobRun, job_run_id)
            if job is None:
                return
            job.end_time = datetime.utcnow()
            job.status = status
            job.rows_affected = rows_affected
            job.message = message

    def save_bad_records(self, domain: str, records: list[dict[str, Any]], error_message: str) -> int:
        rows = [
            {
                "domain": domain,
                "business_key": str(record.get("symbol") or record.get("news_id") or record.get("announcement_id") or ""),
                "payload": record,
                "error_message": error_message,
            }
            for record in records
        ]
        return self.upsert(BadRecord, rows) if rows else 0

    def fetch_df(self, stmt: Select[Any]) -> pd.DataFrame:
        with self.engine.begin() as conn:
            return pd.read_sql(stmt, conn)

    def fetch_table(
        self,
        table_name: str,
        start: str | None = None,
        end: str | None = None,
        symbols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        model = MODEL_MAP[table_name]
        stmt = select(model)
        if hasattr(model, "trade_date"):
            if start:
                stmt = stmt.where(model.trade_date >= start)
            if end:
                stmt = stmt.where(model.trade_date <= end)
        elif hasattr(model, "asof_date"):
            if start:
                stmt = stmt.where(model.asof_date >= start)
            if end:
                stmt = stmt.where(model.asof_date <= end)
        if hasattr(model, "symbol") and symbols:
            stmt = stmt.where(model.symbol.in_(symbols))
        if hasattr(model, "trade_date"):
            stmt = stmt.order_by(model.trade_date)
        elif hasattr(model, "asof_date"):
            stmt = stmt.order_by(model.asof_date)
        return self.fetch_df(stmt)

    def get_securities(self, active_only: bool = True) -> pd.DataFrame:
        stmt = select(Security)
        if active_only:
            stmt = stmt.where(Security.status == "active")
        return self.fetch_df(stmt)

    def get_trading_calendar(self) -> Any:
        stmt = select(TradeCalendar.trade_date).where(TradeCalendar.is_open.is_(True)).order_by(TradeCalendar.trade_date)
        df = self.fetch_df(stmt)
        if df.empty:
            return default_business_calendar()
        return build_calendar_from_series(df["trade_date"].tolist())

    def get_latest_trade_date(self, table_name: str) -> str | None:
        model = MODEL_MAP[table_name]
        if not hasattr(model, "trade_date"):
            return None
        stmt = select(func.max(model.trade_date))
        with self.engine.begin() as conn:
            value = conn.execute(stmt).scalar()
        return value.isoformat() if value else None

    def get_symbol_trade_counts(
        self,
        table_name: str,
        start: str,
        end: str,
        symbols: Sequence[str] | None = None,
    ) -> dict[str, int]:
        model = MODEL_MAP[table_name]
        if not hasattr(model, "trade_date") or not hasattr(model, "symbol"):
            return {}
        stmt = (
            select(model.symbol, func.count().label("cnt"))
            .where(model.trade_date >= start, model.trade_date <= end)
            .group_by(model.symbol)
        )
        if symbols:
            stmt = stmt.where(model.symbol.in_(symbols))
        with self.engine.begin() as conn:
            rows = conn.execute(stmt).fetchall()
        return {str(symbol): int(cnt) for symbol, cnt in rows}

    def delete_training_samples(self, start: str, end: str) -> int:
        with session_scope(self.session_factory) as session:
            affected = (
                session.query(TrainingSample)
                .filter(TrainingSample.trade_date >= start, TrainingSample.trade_date <= end)
                .delete(synchronize_session=False)
            )
        return int(affected)

    def count_training_samples(self, start: str, end: str) -> int:
        stmt = select(func.count()).select_from(TrainingSample).where(
            TrainingSample.trade_date >= start,
            TrainingSample.trade_date <= end,
        )
        with self.engine.begin() as conn:
            value = conn.execute(stmt).scalar()
        return int(value or 0)

    def delete_event_features(self, start: str, end: str) -> int:
        with session_scope(self.session_factory) as session:
            affected = (
                session.query(EventFeaturesDaily)
                .filter(EventFeaturesDaily.trade_date >= start, EventFeaturesDaily.trade_date <= end)
                .delete(synchronize_session=False)
            )
        return int(affected)

    def delete_company_similarity(self, similarity_version: str) -> int:
        with session_scope(self.session_factory) as session:
            affected = (
                session.query(CompanySimilarity)
                .filter(CompanySimilarity.similarity_version == similarity_version)
                .delete(synchronize_session=False)
            )
        return int(affected)

    def delete_company_profiles(
        self,
        start: str | None = None,
        end: str | None = None,
        profile_version: str | None = None,
    ) -> int:
        with session_scope(self.session_factory) as session:
            query = session.query(CompanyProfile)
            if start:
                query = query.filter(CompanyProfile.asof_date >= start)
            if end:
                query = query.filter(CompanyProfile.asof_date <= end)
            if profile_version:
                query = query.filter(CompanyProfile.profile_version == profile_version)
            affected = query.delete(synchronize_session=False)
        return int(affected)

    def export_training_samples(self, start: str, end: str) -> pd.DataFrame:
        stmt = self._build_training_sample_export_stmt(start, end)
        return self._normalize_dataframe_for_parquet(self.fetch_df(stmt), TrainingSample)

    def iter_export_training_samples(self, start: str, end: str, chunksize: int = 50000):
        stmt = self._build_training_sample_export_stmt(start, end)
        with self.engine.connect().execution_options(stream_results=True) as conn:
            for chunk in pd.read_sql(stmt, conn, chunksize=chunksize):
                yield self._normalize_dataframe_for_parquet(chunk, TrainingSample)

    @staticmethod
    def _build_training_sample_export_stmt(start: str, end: str) -> Select[Any]:
        export_columns = [
            getattr(TrainingSample, column.name)
            for column in TrainingSample.__table__.columns
            if column.name in TRAINING_SAMPLE_EXPORT_COLUMNS
        ]
        return (
            select(*export_columns)
            .where(TrainingSample.trade_date >= start, TrainingSample.trade_date <= end)
            .order_by(TrainingSample.symbol, TrainingSample.trade_date)
        )

    @staticmethod
    def _normalize_dataframe_for_parquet(df: pd.DataFrame, model: Any) -> pd.DataFrame:
        if df.empty:
            return df

        output = df.copy()
        model_columns = {column.name: column for column in model.__table__.columns}
        for column_name, column in model_columns.items():
            if column_name not in output.columns:
                continue
            column_type = column.type
            if isinstance(column_type, (SAString, SAText)):
                output[column_name] = output[column_name].astype("string")
            elif isinstance(column_type, SAJSON):
                output[column_name] = output[column_name].map(
                    lambda value: json.dumps(value, ensure_ascii=False, default=str)
                    if isinstance(value, (dict, list))
                    else (None if pd.isna(value) else str(value))
                ).astype("string")
            elif isinstance(column_type, SADateTime):
                output[column_name] = pd.to_datetime(output[column_name], errors="coerce")
            elif isinstance(column_type, SADate):
                output[column_name] = pd.to_datetime(output[column_name], errors="coerce")
            elif isinstance(column_type, SABoolean):
                output[column_name] = output[column_name].astype("boolean")
            elif isinstance(column_type, SAInteger):
                output[column_name] = pd.to_numeric(output[column_name], errors="coerce").astype("Int64")
            elif isinstance(column_type, SAFloat):
                output[column_name] = pd.to_numeric(output[column_name], errors="coerce").astype("float64")
        return output
