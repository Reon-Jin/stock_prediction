from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UserPublic(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str
    display_name: str | None = None
    is_active: bool


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class RegisterRequest(BaseModel):
    username: str = Field(max_length=24)
    email: str = Field(max_length=128)
    password: str = Field(max_length=128)
    display_name: str | None = Field(default=None, max_length=64)


class LoginRequest(BaseModel):
    account: str = Field(max_length=128)
    password: str = Field(max_length=128)


class SingleAnalysisRequest(BaseModel):
    symbol: str = Field(min_length=6, max_length=16)
    is_holding: bool = False
    holding_days: int = Field(default=0, ge=0, le=3650)
    target_date: str | None = None
    risk_preference: str = Field(default="balanced", max_length=24)


class SingleAnalysisChatRequest(BaseModel):
    symbol: str | None = Field(default=None, min_length=6, max_length=16)
    is_holding: bool = False
    holding_days: int = Field(default=0, ge=0, le=3650)
    target_date: str | None = None
    risk_preference: str = Field(default="balanced", max_length=24)
    session_id: int | None = Field(default=None, ge=1)
    message: str | None = Field(default=None, max_length=4000)
    refresh_analysis: bool = True


class AnalysisSessionSummary(BaseModel):
    id: int
    symbol: str
    stock_name: str | None = None
    title: str | None = None
    latest_trade_date: str | None = None
    is_holding: bool
    holding_days: int
    risk_preference: str
    last_user_message: str | None = None
    last_assistant_message: str | None = None
    updated_at: str | None = None


class AnalysisMessageItem(BaseModel):
    id: int
    role: str
    content: str
    created_at: str | None = None


class AnalysisSessionDetail(AnalysisSessionSummary):
    messages: list[AnalysisMessageItem] = Field(default_factory=list)
    latest_analysis: dict[str, Any] | None = None


class MarketScanRequest(BaseModel):
    top_n: int = Field(default=12, ge=1, le=100)
    target_date: str | None = None
    risk_preference: str = Field(default="balanced", max_length=24)
    scan_mode: str = Field(default="market", pattern="^(market|quick)$")


class MarketScanChatRequest(BaseModel):
    top_n: int = Field(default=12, ge=1, le=100)
    target_date: str | None = None
    scan_mode: str = Field(default="market", pattern="^(market|quick)$")
    session_id: int | None = Field(default=None, ge=1)
    message: str | None = Field(default=None, max_length=4000)
    refresh_analysis: bool = True


class DashboardSummary(BaseModel):
    latest_trade_date: str | None
    training_sample_count: int
    security_count: int
    active_user_count: int
    latest_checkpoint: str | None
    latest_checkpoint_time: str | None
    current_modules: list[dict[str, Any]]
    planned_modules: list[dict[str, Any]]


class ModelRunSummary(BaseModel):
    run_name: str
    checkpoint_path: str | None
    updated_at: str | None
    train_minutes: float | None
    best_valid_loss: float | None
    test_p_win_acc: float | None
    test_rank_score_mae: float | None
    has_test_metrics: bool
