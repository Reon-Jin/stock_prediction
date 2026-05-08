from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from utils.symbols import normalize_symbol
from warehouse.models import UserAccount
from webapp.analysis_chat import (
    delete_analysis_session,
    delete_market_scan_session,
    get_analysis_session_detail,
    get_market_scan_session_detail,
    list_analysis_sessions,
    list_market_scan_sessions,
    stream_market_scan_chat,
    stream_single_stock_chat,
)
from webapp.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    validate_email,
    validate_username,
    verify_password,
)
from webapp.schemas import (
    AnalysisSessionDetail,
    AnalysisSessionSummary,
    AuthTokenResponse,
    DashboardSummary,
    LoginRequest,
    MarketScanChatRequest,
    MarketScanRequest,
    ModelRunSummary,
    RegisterRequest,
    SingleAnalysisChatRequest,
    SingleAnalysisRequest,
    UserPublic,
)
from webapp.services import (
    bootstrap_web_app,
    build_market_scan_v2,
    build_single_stock_analysis,
    find_user_by_account,
    get_dashboard_summary,
    get_secret_key,
    iter_db,
    list_model_runs,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    bootstrap_web_app()
    yield


app = FastAPI(
    title="A股智能分析平台",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

_DEFAULT_ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173"]
_cors_env = os.environ.get("CORS_ORIGINS")
_allow_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] if _cors_env else _DEFAULT_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_current_user(
    authorization: str | None = Header(default=None),
    session: Session = Depends(iter_db),
) -> UserAccount:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="请先登录。")
    token = authorization.split(" ", 1)[1].strip()
    payload = decode_access_token(token, get_secret_key())
    if not payload or "uid" not in payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录状态已失效，请重新登录。")
    user = session.get(UserAccount, int(payload["uid"]))
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="当前账号不可用。")
    return user


def build_auth_response(user: UserAccount) -> AuthTokenResponse:
    token = create_access_token(
        {"uid": int(user.id), "username": user.username},
        get_secret_key(),
        expires_in_hours=48,
    )
    return AuthTokenResponse(access_token=token, user=UserPublic.model_validate(user))


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok", "server_time": datetime.now(timezone.utc).isoformat()}


@app.post("/api/auth/register", response_model=AuthTokenResponse)
def register(payload: RegisterRequest, session: Session = Depends(iter_db)) -> AuthTokenResponse:
    username = payload.username.strip()
    email = payload.email.strip().lower()
    password = payload.password
    if not username:
        raise HTTPException(status_code=400, detail="请输入用户名。")
    if not email:
        raise HTTPException(status_code=400, detail="请输入邮箱地址。")
    if not password:
        raise HTTPException(status_code=400, detail="请输入密码。")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少需要 6 位。")
    if not validate_username(username):
        raise HTTPException(status_code=400, detail="用户名需为 3-24 位，可包含中文、字母、数字或下划线。")
    if not validate_email(email):
        raise HTTPException(status_code=400, detail="请输入有效邮箱地址。")

    duplicate_stmt = select(UserAccount).where((UserAccount.username == username) | (UserAccount.email == email))
    duplicate = session.execute(duplicate_stmt).scalar_one_or_none()
    if duplicate is not None:
        raise HTTPException(status_code=409, detail="用户名或邮箱已存在。")

    user = UserAccount(
        username=username,
        email=email,
        password_hash=hash_password(password),
        display_name=(payload.display_name or username).strip(),
        is_active=True,
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return build_auth_response(user)


@app.post("/api/auth/login", response_model=AuthTokenResponse)
def login(payload: LoginRequest, session: Session = Depends(iter_db)) -> AuthTokenResponse:
    if not payload.account.strip():
        raise HTTPException(status_code=400, detail="请输入用户名或邮箱。")
    if not payload.password:
        raise HTTPException(status_code=400, detail="请输入密码。")
    user = find_user_by_account(session, payload.account)
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="账号或密码错误。")
    user.last_login_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(user)
    return build_auth_response(user)


@app.get("/api/auth/me", response_model=UserPublic)
def me(current_user: UserAccount = Depends(get_current_user)) -> UserPublic:
    return UserPublic.model_validate(current_user)


@app.get("/api/dashboard/summary", response_model=DashboardSummary)
def dashboard_summary(
    _: UserAccount = Depends(get_current_user),
    session: Session = Depends(iter_db),
) -> DashboardSummary:
    return DashboardSummary.model_validate(get_dashboard_summary(session))


@app.get("/api/models/runs", response_model=list[ModelRunSummary])
def model_runs(_: UserAccount = Depends(get_current_user)) -> list[ModelRunSummary]:
    return [ModelRunSummary.model_validate(item) for item in list_model_runs()]


@app.post("/api/analysis/single")
def analyze_single(
    payload: SingleAnalysisRequest,
    _: UserAccount = Depends(get_current_user),
) -> dict:
    try:
        return build_single_stock_analysis(
            symbol=normalize_symbol(payload.symbol),
            target_date=payload.target_date,
            is_holding=payload.is_holding,
            holding_days=payload.holding_days,
            risk_preference=payload.risk_preference,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/analysis/single/stream")
def analyze_single_stream(
    payload: SingleAnalysisChatRequest,
    current_user: UserAccount = Depends(get_current_user),
) -> StreamingResponse:
    try:
        stream = stream_single_stock_chat(
            user=current_user,
            session_id=payload.session_id,
            symbol=payload.symbol,
            is_holding=payload.is_holding,
            holding_days=payload.holding_days,
            target_date=payload.target_date,
            risk_preference=payload.risk_preference,
            message=payload.message,
            refresh_analysis=payload.refresh_analysis,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(stream, media_type="text/event-stream")


@app.get("/api/analysis/single/sessions", response_model=list[AnalysisSessionSummary])
def single_analysis_sessions(current_user: UserAccount = Depends(get_current_user)) -> list[AnalysisSessionSummary]:
    return [AnalysisSessionSummary.model_validate(item) for item in list_analysis_sessions(int(current_user.id))]


@app.get("/api/analysis/single/sessions/{session_id}", response_model=AnalysisSessionDetail)
def single_analysis_session_detail(
    session_id: int,
    current_user: UserAccount = Depends(get_current_user),
) -> AnalysisSessionDetail:
    try:
        return AnalysisSessionDetail.model_validate(get_analysis_session_detail(int(current_user.id), int(session_id)))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/analysis/single/sessions/{session_id}")
def single_analysis_session_delete(
    session_id: int,
    current_user: UserAccount = Depends(get_current_user),
) -> dict[str, bool]:
    try:
        delete_analysis_session(int(current_user.id), int(session_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"success": True}


@app.post("/api/analysis/market-scan")
def analyze_market_scan(
    payload: MarketScanRequest,
    _: UserAccount = Depends(get_current_user),
) -> dict:
    try:
        return build_market_scan_v2(
            top_n=payload.top_n,
            target_date=payload.target_date,
            risk_preference=payload.risk_preference,
            scan_mode=payload.scan_mode,
            holding_days=payload.holding_days,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/analysis/market-scan/stream")
def analyze_market_scan_stream(
    payload: MarketScanChatRequest,
    current_user: UserAccount = Depends(get_current_user),
) -> StreamingResponse:
    try:
        stream = stream_market_scan_chat(
            user=current_user,
            session_id=payload.session_id,
            top_n=payload.top_n,
            target_date=payload.target_date,
            scan_mode=payload.scan_mode,
            holding_days=payload.holding_days,
            message=payload.message,
            refresh_analysis=payload.refresh_analysis,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StreamingResponse(stream, media_type="text/event-stream")


@app.get("/api/analysis/market-scan/sessions", response_model=list[AnalysisSessionSummary])
def market_scan_sessions(current_user: UserAccount = Depends(get_current_user)) -> list[AnalysisSessionSummary]:
    return [AnalysisSessionSummary.model_validate(item) for item in list_market_scan_sessions(int(current_user.id))]


@app.get("/api/analysis/market-scan/sessions/{session_id}", response_model=AnalysisSessionDetail)
def market_scan_session_detail(
    session_id: int,
    current_user: UserAccount = Depends(get_current_user),
) -> AnalysisSessionDetail:
    try:
        return AnalysisSessionDetail.model_validate(get_market_scan_session_detail(int(current_user.id), int(session_id)))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/analysis/market-scan/sessions/{session_id}")
def market_scan_session_delete(
    session_id: int,
    current_user: UserAccount = Depends(get_current_user),
) -> dict[str, bool]:
    try:
        delete_market_scan_session(int(current_user.id), int(session_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"success": True}


@app.get("/api/platform/placeholders")
def platform_placeholders(_: UserAccount = Depends(get_current_user)) -> dict:
    return {
        "decision_model": {
            "status": "ready",
            "title": "决策引擎",
            "description": "基于预测结果、持仓上下文和风控规则输出买卖建议。",
        },
        "llm_summary": {
            "status": "ready",
            "title": "大模型总结",
            "description": "将结构化数据、模型结果和决策结论交给 DeepSeek 做最终文本分析。",
        },
        "more_features": {
            "status": "planned",
            "title": "更多扩展能力",
            "description": "为后续策略、回测和组合分析预留入口。",
        },
    }
