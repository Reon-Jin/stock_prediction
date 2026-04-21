from __future__ import annotations

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from utils.config import load_config
from utils.logger import get_logger
from warehouse.db import build_engine
from warehouse.models import Base

OBSOLETE_COLUMN_PLAN: dict[str, tuple[str, ...]] = {
    "company_profiles": (
        "pe_ttm",
        "pb",
        "event_pos_rate_60",
        "event_neg_rate_60",
    ),
    "training_samples": (
        "pe_ttm",
        "pb",
        "industry_pe_percentile",
        "pos_event_count_1d",
        "neg_event_count_1d",
        "pos_event_count_3d",
        "neg_event_count_3d",
        "max_pos_strength_3d",
        "max_neg_strength_3d",
        "earnings_event_score",
        "policy_event_score",
        "buyback_event_score",
        "reduction_event_score",
        "event_recency_score",
        "event_consistency_score",
        "event_pos_rate_60",
        "event_neg_rate_60",
        "main_net_inflow_1d",
        "main_net_inflow_3d",
        "main_net_inflow_5d",
        "main_net_inflow_10d",
        "super_large_net_inflow_3d",
        "flow_reversal_flag",
    ),
}


def ensure_database_exists(config_path: str = "configs/config.yaml") -> None:
    logger = get_logger("schema_init")
    config = load_config(config_path)
    db_url = config.database["url"]
    url = make_url(db_url)
    database_name = url.database
    if not database_name:
        raise ValueError("database name is missing in config url")

    # Connect to the built-in `mysql` database first, because the target
    # project database may not exist yet.
    server_url = url.set(database="mysql")
    engine = create_engine(
        server_url,
        echo=bool(config.database.get("echo", False)),
        pool_pre_ping=bool(config.database.get("pool_pre_ping", True)),
        future=True,
    )
    create_sql = text(f"CREATE DATABASE IF NOT EXISTS `{database_name}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    with engine.begin() as conn:
        logger.info("ensuring database %s exists", database_name)
        conn.execute(create_sql)
    engine.dispose()


def init_schema(config_path: str = "configs/config.yaml") -> None:
    logger = get_logger("schema_init")
    ensure_database_exists(config_path)
    config = load_config(config_path)
    engine = build_engine(config)
    logger.info("creating database schema")
    Base.metadata.create_all(engine)
    _ensure_compatible_columns(engine)
    logger.info("schema ready")


def _ensure_compatible_columns(engine) -> None:
    """
    Lightweight dev-time schema repair.
    `create_all` will not add missing columns to existing tables, so when the
    local schema evolves we patch a small set of critical columns here.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    compatibility_plan: dict[str, dict[str, object]] = {
        "securities": {
            "add": {
                "feature_version": "VARCHAR(32) NULL",
                "label_version": "VARCHAR(32) NULL",
                "created_at": "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
            }
        },
        "daily_bars": {
            "add": {
                "feature_version": "VARCHAR(32) NULL",
                "label_version": "VARCHAR(32) NULL",
            }
        },
        "index_bars": {
            "add": {
                "feature_version": "VARCHAR(32) NULL",
                "label_version": "VARCHAR(32) NULL",
            }
        },
        "sector_daily": {
            "add": {
                "feature_version": "VARCHAR(32) NULL",
                "label_version": "VARCHAR(32) NULL",
            }
        },
        "capital_flow_daily": {
            "add": {
                "feature_version": "VARCHAR(32) NULL",
                "label_version": "VARCHAR(32) NULL",
            }
        },
        "financial_snapshot": {
            "add": {
                "feature_version": "VARCHAR(32) NULL",
                "label_version": "VARCHAR(32) NULL",
            }
        },
        "news_raw": {
            "add": {
                "feature_version": "VARCHAR(32) NULL",
                "label_version": "VARCHAR(32) NULL",
            }
        },
        "news_norm": {
            "add": {
                "feature_version": "VARCHAR(32) NULL",
                "label_version": "VARCHAR(32) NULL",
                "event_confidence": "DOUBLE NULL",
                "event_backend": "VARCHAR(64) NULL",
                "event_model": "VARCHAR(128) NULL",
                "event_details": "JSON NULL",
            }
        },
        "announcement_raw": {
            "add": {
                "feature_version": "VARCHAR(32) NULL",
                "label_version": "VARCHAR(32) NULL",
            }
        },
        "announcement_norm": {
            "add": {
                "feature_version": "VARCHAR(32) NULL",
                "label_version": "VARCHAR(32) NULL",
                "event_confidence": "DOUBLE NULL",
                "event_backend": "VARCHAR(64) NULL",
                "event_model": "VARCHAR(128) NULL",
                "event_details": "JSON NULL",
            }
        },
        "event_features_daily": {
            "add": {
                "label_version": "VARCHAR(32) NULL",
                "event_embedding": "JSON NULL",
            },
        },
        "training_samples": {
            "add": {
                "sample_status": "VARCHAR(32) NULL DEFAULT 'ready'",
                "label_rank_score": "DOUBLE NULL",
                "symbol_id": "INT NULL",
                "industry_id": "INT NULL",
                "board_id": "INT NULL",
                "company_profile_version": "VARCHAR(32) NULL",
                "market_cap_log": "DOUBLE NULL",
                "volatility_120": "DOUBLE NULL",
                "beta_120": "DOUBLE NULL",
                "turnover_mean_120": "DOUBLE NULL",
                "amount_mean_120": "DOUBLE NULL",
                "debt_ratio": "DOUBLE NULL",
                "gross_margin": "DOUBLE NULL",
                "event_embedding": "JSON NULL",
                "neighbor_symbol_ids": "JSON NULL",
                "neighbor_scores": "JSON NULL",
            }
        },
        "company_profiles": {
            "add": {
                "created_at": "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
            }
        },
        "company_similarity": {
            "add": {
                "created_at": "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP",
            }
        },
        "job_runs": {
            "modify": {
                "message": "LONGTEXT NULL",
            }
        },
    }
    with engine.begin() as conn:
        _prepare_event_features_daily_table(conn, inspector)
        for table_name, table_plan in compatibility_plan.items():
            if table_name not in existing_tables:
                continue
            existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
            for column_name, column_sql in table_plan.get("add", {}).items():
                if column_name in existing_columns:
                    continue
                conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))
            for column_name, column_sql in table_plan.get("modify", {}).items():
                if column_name not in existing_columns:
                    continue
                conn.execute(text(f"ALTER TABLE {table_name} MODIFY COLUMN {column_name} {column_sql}"))
        _migrate_event_features_daily_table(conn, inspector)
        _drop_obsolete_columns(conn)


def _prepare_event_features_daily_table(conn, inspector) -> None:
    if "event_features_daily" not in set(inspector.get_table_names()):
        return
    total_rows = conn.execute(text("SELECT COUNT(*) FROM event_features_daily")).scalar() or 0
    distinct_dates = conn.execute(text("SELECT COUNT(DISTINCT trade_date) FROM event_features_daily")).scalar() or 0
    if total_rows and total_rows != distinct_dates:
        conn.execute(text("TRUNCATE TABLE event_features_daily"))


def _migrate_event_features_daily_table(conn, inspector) -> None:
    if "event_features_daily" not in set(inspector.get_table_names()):
        return

    indexes = {item["name"] for item in inspector.get_indexes("event_features_daily")}
    unique_constraints = {item["name"] for item in inspector.get_unique_constraints("event_features_daily")}
    column_defs = {item["name"]: item for item in inspector.get_columns("event_features_daily")}
    columns = set(column_defs)

    if "symbol" in columns and not bool(column_defs["symbol"].get("nullable", False)):
        conn.execute(text("ALTER TABLE event_features_daily MODIFY COLUMN symbol VARCHAR(16) NULL"))

    has_daily_unique = "uq_event_features_trade_date" in unique_constraints
    if not has_daily_unique:
        if "uq_event_features_symbol_date" in unique_constraints:
            conn.execute(text("ALTER TABLE event_features_daily DROP INDEX uq_event_features_symbol_date"))
        unique_constraints = {item["name"] for item in inspect(conn).get_unique_constraints("event_features_daily")}
        if "uq_event_features_trade_date" not in unique_constraints:
            conn.execute(text("ALTER TABLE event_features_daily ADD CONSTRAINT uq_event_features_trade_date UNIQUE (trade_date)"))

    if "idx_event_features_trade_date" not in indexes:
        conn.execute(text("CREATE INDEX idx_event_features_trade_date ON event_features_daily (trade_date)"))


def _drop_obsolete_columns(conn) -> None:
    inspector = inspect(conn)
    existing_tables = set(inspector.get_table_names())
    for table_name, column_names in OBSOLETE_COLUMN_PLAN.items():
        if table_name not in existing_tables:
            continue
        existing_columns = {column["name"] for column in inspector.get_columns(table_name)}
        for column_name in column_names:
            if column_name not in existing_columns:
                continue
            conn.execute(text(f"ALTER TABLE {table_name} DROP COLUMN {column_name}"))
            existing_columns.remove(column_name)


if __name__ == "__main__":
    init_schema()
