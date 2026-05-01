from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    """Return naive datetime representing current UTC time."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class VersionMixin:
    data_version: Mapped[str] = mapped_column(String(32), nullable=False, default="v1")
    feature_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    label_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )


class Security(Base, VersionMixin):
    __tablename__ = "securities"
    symbol: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    exchange: Mapped[str] = mapped_column(String(8), nullable=False)
    board: Mapped[str | None] = mapped_column(String(32))
    industry: Mapped[str | None] = mapped_column(String(128))
    list_date: Mapped[date | None] = mapped_column(Date)
    delist_date: Mapped[date | None] = mapped_column(Date)
    is_st: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="active", nullable=False)


class TradeCalendar(Base):
    __tablename__ = "trade_calendar"
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    is_open: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    exchange: Mapped[str] = mapped_column(String(8), primary_key=True, default="CN")


class DailyBar(Base, VersionMixin):
    __tablename__ = "daily_bars"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    pre_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    amount: Mapped[float | None] = mapped_column(Float)
    turnover_rate: Mapped[float | None] = mapped_column(Float)
    amplitude: Mapped[float | None] = mapped_column(Float)
    pct_chg: Mapped[float | None] = mapped_column(Float)
    adj_factor: Mapped[float | None] = mapped_column(Float, default=1.0)
    adj_close: Mapped[float | None] = mapped_column(Float)
    is_suspended: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", name="uq_daily_bars_symbol_date"),
        Index("idx_daily_bars_symbol_trade_date", "symbol", "trade_date"),
        Index("idx_daily_bars_trade_date", "trade_date"),
    )


class IndexBar(Base, VersionMixin):
    __tablename__ = "index_bars"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    index_code: Mapped[str] = mapped_column(String(16), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    amount: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    __table_args__ = (
        UniqueConstraint("index_code", "trade_date", name="uq_index_bars_code_date"),
        Index("idx_index_bars_trade_date", "trade_date"),
    )


class SectorDaily(Base, VersionMixin):
    __tablename__ = "sector_daily"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    sector_id: Mapped[str] = mapped_column(String(32), nullable=False)
    sector_name: Mapped[str] = mapped_column(String(128), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    close: Mapped[float | None] = mapped_column(Float)
    pct_chg: Mapped[float | None] = mapped_column(Float)
    amount: Mapped[float | None] = mapped_column(Float)
    turnover: Mapped[float | None] = mapped_column(Float)
    up_limit_count: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    __table_args__ = (
        UniqueConstraint("sector_id", "trade_date", name="uq_sector_daily_id_date"),
        Index("idx_sector_daily_trade_date", "trade_date"),
    )


class CapitalFlowDaily(Base, VersionMixin):
    __tablename__ = "capital_flow_daily"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    main_net_inflow: Mapped[float | None] = mapped_column(Float)
    super_large_net_inflow: Mapped[float | None] = mapped_column(Float)
    large_net_inflow: Mapped[float | None] = mapped_column(Float)
    medium_net_inflow: Mapped[float | None] = mapped_column(Float)
    small_net_inflow: Mapped[float | None] = mapped_column(Float)
    main_inflow_ratio: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", name="uq_capital_flow_symbol_date"),
        Index("idx_capital_flow_trade_date", "trade_date"),
    )


class FinancialSnapshot(Base, VersionMixin):
    __tablename__ = "financial_snapshot"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    report_date: Mapped[date] = mapped_column(Date, nullable=False)
    asof_date: Mapped[date] = mapped_column(Date, nullable=False)
    pe_ttm: Mapped[float | None] = mapped_column(Float)
    pb: Mapped[float | None] = mapped_column(Float)
    ps_ttm: Mapped[float | None] = mapped_column(Float)
    roe: Mapped[float | None] = mapped_column(Float)
    revenue_yoy: Mapped[float | None] = mapped_column(Float)
    profit_yoy: Mapped[float | None] = mapped_column(Float)
    gross_margin: Mapped[float | None] = mapped_column(Float)
    debt_ratio: Mapped[float | None] = mapped_column(Float)
    industry_percentile: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    __table_args__ = (
        UniqueConstraint("symbol", "report_date", "asof_date", name="uq_financial_snapshot_symbol_report_asof"),
        Index("idx_financial_snapshot_symbol_asof", "symbol", "asof_date"),
    )


class NewsRaw(Base, VersionMixin):
    __tablename__ = "news_raw"
    news_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    publish_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1024))
    mentioned_symbols: Mapped[list[str] | None] = mapped_column(JSON)
    importance_score_raw: Mapped[float | None] = mapped_column(Float)
    normalized_flag: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    __table_args__ = (
        Index("idx_news_raw_publish_time", "publish_time"),
        Index("idx_news_raw_source", "source"),
    )


class NewsNorm(Base, VersionMixin):
    __tablename__ = "news_norm"
    news_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    publish_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    trade_date: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    url: Mapped[str | None] = mapped_column(String(1024))
    mentioned_symbols: Mapped[list[str] | None] = mapped_column(JSON)
    importance_score_raw: Mapped[float | None] = mapped_column(Float)
    normalized_flag: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    event_type: Mapped[str | None] = mapped_column(String(64))
    event_direction: Mapped[str | None] = mapped_column(String(16))
    event_strength: Mapped[float | None] = mapped_column(Float)
    event_confidence: Mapped[float | None] = mapped_column(Float)
    event_backend: Mapped[str | None] = mapped_column(String(64))
    event_model: Mapped[str | None] = mapped_column(String(128))
    event_details: Mapped[dict | None] = mapped_column(JSON)
    __table_args__ = (
        Index("idx_news_norm_trade_date", "trade_date"),
        Index("idx_news_norm_publish_time", "publish_time"),
    )


class AnnouncementRaw(Base, VersionMixin):
    __tablename__ = "announcement_raw"
    announcement_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    publish_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str | None] = mapped_column(String(128))
    summary: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    mentioned_symbols: Mapped[list[str] | None] = mapped_column(JSON)
    __table_args__ = (
        Index("idx_announcement_raw_publish_time", "publish_time"),
        Index("idx_announcement_raw_source", "source"),
    )


class AnnouncementNorm(Base, VersionMixin):
    __tablename__ = "announcement_norm"
    announcement_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    publish_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    trade_date: Mapped[date | None] = mapped_column(Date)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str | None] = mapped_column(String(128))
    summary: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    mentioned_symbols: Mapped[list[str] | None] = mapped_column(JSON)
    event_type: Mapped[str | None] = mapped_column(String(64))
    event_direction: Mapped[str | None] = mapped_column(String(16))
    event_strength: Mapped[float | None] = mapped_column(Float)
    event_confidence: Mapped[float | None] = mapped_column(Float)
    event_backend: Mapped[str | None] = mapped_column(String(64))
    event_model: Mapped[str | None] = mapped_column(String(128))
    event_details: Mapped[dict | None] = mapped_column(JSON)
    __table_args__ = (
        Index("idx_announcement_norm_trade_date", "trade_date"),
        Index("idx_announcement_norm_publish_time", "publish_time"),
    )


class EventFeaturesDaily(Base, VersionMixin):
    __tablename__ = "event_features_daily"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    event_embedding: Mapped[list[float] | None] = mapped_column(JSON)
    __table_args__ = (
        UniqueConstraint("trade_date", name="uq_event_features_trade_date"),
        Index("idx_event_features_trade_date", "trade_date"),
    )


class CompanyProfile(Base):
    __tablename__ = "company_profiles"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    asof_date: Mapped[date] = mapped_column(Date, nullable=False)
    industry_name: Mapped[str | None] = mapped_column(String(128))
    board: Mapped[str | None] = mapped_column(String(32))
    market_cap_log: Mapped[float | None] = mapped_column(Float)
    volatility_120: Mapped[float | None] = mapped_column(Float)
    beta_120: Mapped[float | None] = mapped_column(Float)
    turnover_mean_120: Mapped[float | None] = mapped_column(Float)
    amount_mean_120: Mapped[float | None] = mapped_column(Float)
    ret_20: Mapped[float | None] = mapped_column(Float)
    ret_60: Mapped[float | None] = mapped_column(Float)
    roe: Mapped[float | None] = mapped_column(Float)
    revenue_yoy: Mapped[float | None] = mapped_column(Float)
    profit_yoy: Mapped[float | None] = mapped_column(Float)
    debt_ratio: Mapped[float | None] = mapped_column(Float)
    gross_margin: Mapped[float | None] = mapped_column(Float)
    profile_version: Mapped[str] = mapped_column(String(32), nullable=False, default="cp1")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )
    __table_args__ = (
        UniqueConstraint("symbol", "asof_date", name="uq_company_profiles_symbol_asof"),
        Index("idx_company_profiles_symbol_asof", "symbol", "asof_date"),
        Index("idx_company_profiles_asof_date", "asof_date"),
    )


class CompanySimilarity(Base):
    __tablename__ = "company_similarity"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    neighbor_symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    sim_score: Mapped[float] = mapped_column(Float, nullable=False)
    sim_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity_version: Mapped[str] = mapped_column(String(32), nullable=False, default="cs1")
    asof_date: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )
    __table_args__ = (
        UniqueConstraint("symbol", "neighbor_symbol", "similarity_version", name="uq_company_similarity_pair_version"),
        Index("idx_company_similarity_symbol_rank", "symbol", "sim_rank"),
        Index("idx_company_similarity_neighbor", "neighbor_symbol"),
    )


class UserAccount(Base):
    __tablename__ = "user_accounts"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    email: Mapped[str] = mapped_column(String(128), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(64))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime)
    __table_args__ = (
        UniqueConstraint("username", name="uq_user_accounts_username"),
        UniqueConstraint("email", name="uq_user_accounts_email"),
        Index("idx_user_accounts_active", "is_active"),
    )


class AnalysisSession(Base):
    __tablename__ = "analysis_sessions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("user_accounts.id"), nullable=False)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    stock_name: Mapped[str | None] = mapped_column(String(64))
    title: Mapped[str | None] = mapped_column(String(128))
    latest_trade_date: Mapped[date | None] = mapped_column(Date)
    is_holding: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    holding_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    risk_preference: Mapped[str] = mapped_column(String(24), nullable=False, default="balanced")
    latest_analysis_json: Mapped[dict | None] = mapped_column(JSON)
    last_user_message: Mapped[str | None] = mapped_column(Text)
    last_assistant_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )
    __table_args__ = (
        Index("idx_analysis_sessions_user_updated", "user_id", "updated_at"),
        Index("idx_analysis_sessions_user_symbol", "user_id", "symbol"),
    )


class AnalysisMessage(Base):
    __tablename__ = "analysis_messages"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("analysis_sessions.id"), nullable=False)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("user_accounts.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, onupdate=_utcnow, nullable=False
    )
    __table_args__ = (
        Index("idx_analysis_messages_session_created", "session_id", "created_at"),
        Index("idx_analysis_messages_user_created", "user_id", "created_at"),
    )


class TrainingSample(Base, VersionMixin):
    __tablename__ = "training_samples"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(16), nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    name: Mapped[str | None] = mapped_column(String(64))
    industry_sw: Mapped[str | None] = mapped_column(String(128))
    board: Mapped[str | None] = mapped_column(String(32))
    symbol_id: Mapped[int | None] = mapped_column(Integer)
    industry_id: Mapped[int | None] = mapped_column(Integer)
    board_id: Mapped[int | None] = mapped_column(Integer)
    company_profile_version: Mapped[str | None] = mapped_column(String(32))
    list_days: Mapped[int | None] = mapped_column(Integer)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    amount: Mapped[float | None] = mapped_column(Float)
    turnover_rate: Mapped[float | None] = mapped_column(Float)
    pct_chg: Mapped[float | None] = mapped_column(Float)
    ret_1: Mapped[float | None] = mapped_column(Float)
    ret_3: Mapped[float | None] = mapped_column(Float)
    ret_5: Mapped[float | None] = mapped_column(Float)
    ret_10: Mapped[float | None] = mapped_column(Float)
    ret_20: Mapped[float | None] = mapped_column(Float)
    ret_60: Mapped[float | None] = mapped_column(Float)
    ma5_gap: Mapped[float | None] = mapped_column(Float)
    ma10_gap: Mapped[float | None] = mapped_column(Float)
    ma20_gap: Mapped[float | None] = mapped_column(Float)
    ma60_gap: Mapped[float | None] = mapped_column(Float)
    rsi_6: Mapped[float | None] = mapped_column(Float)
    rsi_14: Mapped[float | None] = mapped_column(Float)
    macd_dif: Mapped[float | None] = mapped_column(Float)
    macd_dea: Mapped[float | None] = mapped_column(Float)
    macd_hist: Mapped[float | None] = mapped_column(Float)
    atr_14: Mapped[float | None] = mapped_column(Float)
    boll_pos: Mapped[float | None] = mapped_column(Float)
    volatility_5: Mapped[float | None] = mapped_column(Float)
    volatility_20: Mapped[float | None] = mapped_column(Float)
    volatility_120: Mapped[float | None] = mapped_column(Float)
    beta_120: Mapped[float | None] = mapped_column(Float)
    vol_ratio_5: Mapped[float | None] = mapped_column(Float)
    vol_ratio_20: Mapped[float | None] = mapped_column(Float)
    market_cap_log: Mapped[float | None] = mapped_column(Float)
    turnover_mean_120: Mapped[float | None] = mapped_column(Float)
    amount_mean_120: Mapped[float | None] = mapped_column(Float)
    turnover_rank_industry: Mapped[float | None] = mapped_column(Float)
    amount_rank_market: Mapped[float | None] = mapped_column(Float)
    shrink_volume_flag: Mapped[float | None] = mapped_column(Float)
    surge_volume_flag: Mapped[float | None] = mapped_column(Float)
    ret5_vs_hs300: Mapped[float | None] = mapped_column(Float)
    ret10_vs_hs300: Mapped[float | None] = mapped_column(Float)
    ret20_vs_industry: Mapped[float | None] = mapped_column(Float)
    stock_rank_in_industry: Mapped[float | None] = mapped_column(Float)
    industry_rank_5d: Mapped[float | None] = mapped_column(Float)
    industry_rank_20d: Mapped[float | None] = mapped_column(Float)
    roe: Mapped[float | None] = mapped_column(Float)
    revenue_yoy: Mapped[float | None] = mapped_column(Float)
    profit_yoy: Mapped[float | None] = mapped_column(Float)
    debt_ratio: Mapped[float | None] = mapped_column(Float)
    gross_margin: Mapped[float | None] = mapped_column(Float)
    industry_roe_percentile: Mapped[float | None] = mapped_column(Float)
    event_embedding: Mapped[list[float] | None] = mapped_column(JSON)
    up_limit_count: Mapped[float | None] = mapped_column(Float)
    down_limit_count: Mapped[float | None] = mapped_column(Float)
    broken_limit_rate: Mapped[float | None] = mapped_column(Float)
    consecutive_limit_height: Mapped[float | None] = mapped_column(Float)
    market_turnover: Mapped[float | None] = mapped_column(Float)
    hs300_ret_1: Mapped[float | None] = mapped_column(Float)
    cyb_ret_1: Mapped[float | None] = mapped_column(Float)
    market_volatility_5: Mapped[float | None] = mapped_column(Float)
    sector_hotness_top1: Mapped[float | None] = mapped_column(Float)
    sector_hotness_top3_mean: Mapped[float | None] = mapped_column(Float)
    risk_on_flag: Mapped[float | None] = mapped_column(Float)
    risk_off_flag: Mapped[float | None] = mapped_column(Float)
    label_ret_3: Mapped[float | None] = mapped_column(Float)
    label_ret_5: Mapped[float | None] = mapped_column(Float)
    label_ret_10: Mapped[float | None] = mapped_column(Float)
    label_ret_20: Mapped[float | None] = mapped_column(Float)
    label_ret_40: Mapped[float | None] = mapped_column(Float)
    label_win_3: Mapped[int | None] = mapped_column(Integer)
    label_win_5: Mapped[int | None] = mapped_column(Integer)
    label_win_10: Mapped[int | None] = mapped_column(Integer)
    label_win_20: Mapped[int | None] = mapped_column(Integer)
    label_win_40: Mapped[int | None] = mapped_column(Integer)
    label_alpha_3: Mapped[float | None] = mapped_column(Float)
    label_alpha_5: Mapped[float | None] = mapped_column(Float)
    label_alpha_10: Mapped[float | None] = mapped_column(Float)
    label_alpha_20: Mapped[float | None] = mapped_column(Float)
    label_alpha_40: Mapped[float | None] = mapped_column(Float)
    label_maxdd_3: Mapped[float | None] = mapped_column(Float)
    label_maxdd_5: Mapped[float | None] = mapped_column(Float)
    label_maxdd_10: Mapped[float | None] = mapped_column(Float)
    label_maxdd_20: Mapped[float | None] = mapped_column(Float)
    label_maxdd_40: Mapped[float | None] = mapped_column(Float)
    label_bigloss_5: Mapped[int | None] = mapped_column(Integer)
    label_bigloss_20: Mapped[int | None] = mapped_column(Integer)
    label_rank_score: Mapped[float | None] = mapped_column(Float)
    neighbor_symbol_ids: Mapped[list[int] | None] = mapped_column(JSON)
    neighbor_scores: Mapped[list[float] | None] = mapped_column(JSON)
    sample_status: Mapped[str | None] = mapped_column(String(32), default="ready")
    __table_args__ = (
        UniqueConstraint("symbol", "trade_date", name="uq_training_samples_symbol_date"),
        Index("idx_training_samples_trade_date", "trade_date"),
        Index("idx_training_samples_symbol_trade_date", "symbol", "trade_date"),
    )


class JobRun(Base):
    __tablename__ = "job_runs"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_name: Mapped[str] = mapped_column(String(128), nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=_utcnow)
    end_time: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    message: Mapped[str | None] = mapped_column(Text)
    params_json: Mapped[dict | None] = mapped_column(JSON)
    rows_affected: Mapped[int | None] = mapped_column(Integer)


class BadRecord(Base):
    __tablename__ = "bad_records"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    domain: Mapped[str] = mapped_column(String(64), nullable=False)
    business_key: Mapped[str | None] = mapped_column(String(256))
    payload: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow, nullable=False)
    __table_args__ = (Index("idx_bad_records_domain", "domain"),)
