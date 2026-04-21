from __future__ import annotations

import json
import os
import queue
import threading
from datetime import date, datetime
from typing import Any, Iterator

import requests
from sqlalchemy import delete, select

from utils.symbols import normalize_symbol
from warehouse.models import AnalysisMessage, AnalysisSession, UserAccount
from webapp.services import build_market_scan_v2, build_single_stock_analysis, get_app_config, get_session_factory


MARKET_SCAN_SYMBOL = "__MARKET_SCAN__"

STAGE_LABELS = {
    "candidate_select": "候选股票",
    "extract_data": "提取数据",
    "news_extraction": "新闻提取",
    "model_predict": "模型预测",
    "decision_engine": "决策引擎",
    "llm_analysis": "AI分析",
}


def _json_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _trim_text(value: str | None, limit: int = 180) -> str | None:
    if not value:
        return None
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _serialize_message(message: AnalysisMessage) -> dict[str, Any]:
    return {
        "id": int(message.id),
        "role": message.role,
        "content": message.content,
        "created_at": _iso(message.created_at),
    }


def _serialize_session(session_obj: AnalysisSession, messages: list[AnalysisMessage] | None = None) -> dict[str, Any]:
    payload = {
        "id": int(session_obj.id),
        "symbol": session_obj.symbol,
        "stock_name": session_obj.stock_name,
        "title": session_obj.title,
        "latest_trade_date": _iso(session_obj.latest_trade_date),
        "is_holding": bool(session_obj.is_holding),
        "holding_days": int(session_obj.holding_days or 0),
        "risk_preference": session_obj.risk_preference,
        "last_user_message": session_obj.last_user_message,
        "last_assistant_message": session_obj.last_assistant_message,
        "updated_at": _iso(session_obj.updated_at),
    }
    if messages is not None:
        payload["messages"] = [_serialize_message(message) for message in messages]
        payload["latest_analysis"] = session_obj.latest_analysis_json
    return payload


def _list_sessions(user_id: int, *, symbol_filter: str | None = None, exclude_symbol: str | None = None) -> list[dict[str, Any]]:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        stmt = (
            select(AnalysisSession)
            .where(AnalysisSession.user_id == int(user_id))
            .order_by(AnalysisSession.updated_at.desc())
        )
        rows = session.execute(stmt).scalars().all()
        output: list[dict[str, Any]] = []
        for item in rows:
            if symbol_filter is not None and item.symbol != symbol_filter:
                continue
            if exclude_symbol is not None and item.symbol == exclude_symbol:
                continue
            output.append(_serialize_session(item))
        return output
    finally:
        session.close()


def _get_session_detail(user_id: int, session_id: int, *, symbol_filter: str | None = None) -> dict[str, Any]:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        session_obj = session.get(AnalysisSession, int(session_id))
        if session_obj is None or int(session_obj.user_id) != int(user_id):
            raise ValueError("会话不存在或无权访问。")
        if symbol_filter is not None and session_obj.symbol != symbol_filter:
            raise ValueError("会话不存在或无权访问。")
        messages = session.execute(
            select(AnalysisMessage)
            .where(
                AnalysisMessage.user_id == int(user_id),
                AnalysisMessage.session_id == int(session_id),
            )
            .order_by(AnalysisMessage.created_at.asc(), AnalysisMessage.id.asc())
        ).scalars().all()
        return _serialize_session(session_obj, messages)
    finally:
        session.close()


def _delete_session(user_id: int, session_id: int, *, symbol_filter: str | None = None) -> None:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        session_obj = session.get(AnalysisSession, int(session_id))
        if session_obj is None or int(session_obj.user_id) != int(user_id):
            raise ValueError("会话不存在或无权访问。")
        if symbol_filter is not None and session_obj.symbol != symbol_filter:
            raise ValueError("会话不存在或无权访问。")
        session.execute(
            delete(AnalysisMessage).where(
                AnalysisMessage.user_id == int(user_id),
                AnalysisMessage.session_id == int(session_id),
            )
        )
        session.delete(session_obj)
        session.commit()
    finally:
        session.close()


def list_analysis_sessions(user_id: int) -> list[dict[str, Any]]:
    return _list_sessions(user_id, exclude_symbol=MARKET_SCAN_SYMBOL)


def get_analysis_session_detail(user_id: int, session_id: int) -> dict[str, Any]:
    payload = _get_session_detail(user_id, session_id, symbol_filter=None)
    if payload.get("symbol") == MARKET_SCAN_SYMBOL:
        raise ValueError("会话不存在或无权访问。")
    return payload


def delete_analysis_session(user_id: int, session_id: int) -> None:
    payload = _get_session_detail(user_id, session_id, symbol_filter=None)
    if payload.get("symbol") == MARKET_SCAN_SYMBOL:
        raise ValueError("会话不存在或无权访问。")
    _delete_session(user_id, session_id, symbol_filter=None)


def list_market_scan_sessions(user_id: int) -> list[dict[str, Any]]:
    return _list_sessions(user_id, symbol_filter=MARKET_SCAN_SYMBOL)


def get_market_scan_session_detail(user_id: int, session_id: int) -> dict[str, Any]:
    return _get_session_detail(user_id, session_id, symbol_filter=MARKET_SCAN_SYMBOL)


def delete_market_scan_session(user_id: int, session_id: int) -> None:
    _delete_session(user_id, session_id, symbol_filter=MARKET_SCAN_SYMBOL)


def _resolve_llm_settings() -> dict[str, Any]:
    llm_conf = get_app_config().llm
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("LLM_API_KEY") or llm_conf.get("api_key", "")
    return {
        "base_url": str(llm_conf.get("base_url", "https://api.deepseek.com")).rstrip("/"),
        "model": str(llm_conf.get("model", "deepseek-chat")),
        "api_key": api_key,
        "temperature": float(llm_conf.get("temperature", 0.35)),
        "max_context_messages": int(llm_conf.get("max_context_messages", 12)),
    }


def _build_default_user_prompt(symbol: str, analysis: dict[str, Any]) -> str:
    stock_name = analysis.get("stock", {}).get("name") or symbol
    return f"请结合这次完整分析结果，给我 {stock_name}（{symbol}）的综合判断、关键风险和操作建议。"


def _scan_mode_label(scan_mode: str | None) -> str:
    return "快速推荐" if str(scan_mode or "").lower() == "quick" else "全市场推荐"

def _build_market_scan_default_user_prompt(top_n: int, analysis: dict[str, Any]) -> str:
    candidates = analysis.get("candidates") or []
    mode_label = _scan_mode_label(analysis.get("scan_mode"))
    if candidates:
        first = candidates[0]
        name = first.get("name") or first.get("symbol") or "首选股票"
        hold_days = first.get("recommended_hold_days") or (((first.get("decision_result") or {}).get("decision") or {}).get("suggested_hold_days")) or "?"
        return f"请解释这次{mode_label}前 {top_n} 名结果，重点说明为什么 {name} 排名靠前，以及为什么建议持有 {hold_days} 天。"
    return f"请解释这次{mode_label}前 {top_n} 名结果，包括哪些股票值得优先关注，以及主要风险是什么。"

def _build_title(symbol: str, stock_name: str | None) -> str:
    if stock_name:
        return f"{stock_name} {symbol}"
    return symbol


def _build_market_scan_title(top_n: int, effective_trade_date: str | None, scan_mode: str | None = None) -> str:
    suffix = f" · {effective_trade_date}" if effective_trade_date else ""
    return f"{_scan_mode_label(scan_mode)} Top {top_n}{suffix}"


def _prepare_session(
    user_id: int,
    *,
    session_id: int | None,
    symbol: str | None,
    is_holding: bool,
    holding_days: int,
    risk_preference: str,
    title: str | None = None,
    stock_name: str | None = None,
) -> tuple[AnalysisSession, bool]:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        created = False
        session_obj: AnalysisSession | None = None
        if session_id is not None:
            session_obj = session.get(AnalysisSession, int(session_id))
            if session_obj is None or int(session_obj.user_id) != int(user_id):
                raise ValueError("会话不存在或无权访问。")
        if session_obj is None:
            if not symbol:
                raise ValueError("新建会话时必须提供标识。")
            session_obj = AnalysisSession(
                user_id=int(user_id),
                symbol=symbol,
                stock_name=stock_name,
                title=title or symbol,
                is_holding=bool(is_holding),
                holding_days=int(holding_days),
                risk_preference=risk_preference,
            )
            session.add(session_obj)
            session.flush()
            created = True
        else:
            if symbol:
                session_obj.symbol = symbol
            if stock_name is not None:
                session_obj.stock_name = stock_name
            if title is not None:
                session_obj.title = title
            session_obj.is_holding = bool(is_holding)
            session_obj.holding_days = int(holding_days)
            session_obj.risk_preference = risk_preference
            session.flush()
        session.commit()
        session.refresh(session_obj)
        return session_obj, created
    finally:
        session.close()


def _append_message(user_id: int, session_id: int, role: str, content: str, payload_json: dict[str, Any] | None = None) -> AnalysisMessage:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        message = AnalysisMessage(
            user_id=int(user_id),
            session_id=int(session_id),
            role=role,
            content=content,
            payload_json=payload_json,
        )
        session.add(message)
        session.flush()
        session.commit()
        session.refresh(message)
        return message
    finally:
        session.close()


def _update_session_after_analysis(
    user_id: int,
    session_id: int,
    *,
    analysis: dict[str, Any],
    user_message: str,
) -> AnalysisSession:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        session_obj = session.get(AnalysisSession, int(session_id))
        if session_obj is None or int(session_obj.user_id) != int(user_id):
            raise ValueError("会话不存在或无权访问。")
        stock = analysis.get("stock", {})
        session_obj.symbol = stock.get("symbol") or session_obj.symbol
        session_obj.stock_name = stock.get("name")
        session_obj.title = _build_title(session_obj.symbol, session_obj.stock_name)
        effective_trade_date = analysis.get("effective_trade_date")
        if effective_trade_date:
            session_obj.latest_trade_date = date.fromisoformat(str(effective_trade_date))
        session_obj.latest_analysis_json = analysis
        session_obj.last_user_message = _trim_text(user_message)
        session.flush()
        session.commit()
        session.refresh(session_obj)
        return session_obj
    finally:
        session.close()


def _update_market_scan_session_after_analysis(
    user_id: int,
    session_id: int,
    *,
    analysis: dict[str, Any],
    user_message: str,
) -> AnalysisSession:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        session_obj = session.get(AnalysisSession, int(session_id))
        if session_obj is None or int(session_obj.user_id) != int(user_id):
            raise ValueError("会话不存在或无权访问。")
        session_obj.symbol = MARKET_SCAN_SYMBOL
        session_obj.stock_name = "股票推荐"
        effective_trade_date = analysis.get("effective_trade_date")
        if effective_trade_date:
            session_obj.latest_trade_date = date.fromisoformat(str(effective_trade_date))
        top_n = int(analysis.get("top_n") or session_obj.holding_days or 12)
        scan_mode = str(analysis.get("scan_mode") or "market")
        session_obj.holding_days = top_n
        session_obj.title = _build_market_scan_title(top_n, _iso(session_obj.latest_trade_date), scan_mode)
        session_obj.latest_analysis_json = analysis
        session_obj.last_user_message = _trim_text(user_message)
        session.flush()
        session.commit()
        session.refresh(session_obj)
        return session_obj
    finally:
        session.close()


def _update_session_after_assistant(user_id: int, session_id: int, assistant_text: str) -> None:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        session_obj = session.get(AnalysisSession, int(session_id))
        if session_obj is None or int(session_obj.user_id) != int(user_id):
            raise ValueError("会话不存在或无权访问。")
        session_obj.last_assistant_message = _trim_text(assistant_text)
        session.flush()
        session.commit()
    finally:
        session.close()


def _load_history_messages(session_id: int) -> list[AnalysisMessage]:
    session_factory = get_session_factory()
    session = session_factory()
    try:
        return session.execute(
            select(AnalysisMessage)
            .where(AnalysisMessage.session_id == int(session_id))
            .order_by(AnalysisMessage.created_at.asc(), AnalysisMessage.id.asc())
        ).scalars().all()
    finally:
        session.close()


def _build_llm_messages(session_id: int, analysis: dict[str, Any], max_context_messages: int) -> list[dict[str, str]]:
    history = _load_history_messages(session_id)[-max(1, max_context_messages) :]
    stock = analysis.get("stock", {})
    model_info = analysis.get("model_info", {})
    system_prompt = (
        "你是A股个股分析助手。"
        "你会结合结构化样本、模型输出、决策引擎结果和新闻事件，用中文给出清晰、克制、可执行的建议。"
        "请明确区分数据事实、模型倾向和风险提示，不要夸大确定性。"
    )
    context_prompt = (
        f"当前分析对象：{stock.get('name') or stock.get('symbol')}（{stock.get('symbol')}）。"
        f"模型来源：{model_info.get('source')}。"
        "下面是当前完整结构化分析 JSON，请基于它回答：\n"
        f"{json.dumps(analysis, ensure_ascii=False)}"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context_prompt},
    ]
    for row in history:
        role = row.role if row.role in {"user", "assistant"} else "user"
        messages.append({"role": role, "content": row.content})
    return messages


def _build_market_scan_llm_messages(session_id: int, analysis: dict[str, Any], max_context_messages: int) -> list[dict[str, str]]:
    history = _load_history_messages(session_id)[-max(1, max_context_messages) :]
    candidates = analysis.get("candidates") or []
    first = candidates[0] if candidates else {}
    system_prompt = (
        "你是A股市场推荐助手。"
        "请用清晰中文解释批量预测排序结果。"
        "需要具体说明推荐股票、建议持有天数、预测胜率、主要风险和后续观察点。"
        "请使用适合聊天界面渲染的 Markdown。"
    )
    context_prompt = (
        f"推荐日期：{analysis.get('effective_trade_date')}。"
        f"推荐数量：{analysis.get('top_n')}。"
        f"首选股票：{first.get('name') or first.get('symbol') or '无'}。"
        "下面是完整的市场扫描 JSON，请基于它回答：\n"
        f"{json.dumps(analysis, ensure_ascii=False)}"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context_prompt},
    ]
    for row in history:
        role = row.role if row.role in {"user", "assistant"} else "user"
        messages.append({"role": role, "content": row.content})
    return messages

def _stream_llm_response(messages: list[dict[str, str]]) -> Iterator[str]:
    settings = _resolve_llm_settings()
    if not settings["api_key"]:
        raise RuntimeError("DeepSeek API key 未配置。")
    response = requests.post(
        f"{settings['base_url']}/chat/completions",
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        json={
            "model": settings["model"],
            "messages": messages,
            "stream": True,
            "temperature": settings["temperature"],
        },
        stream=True,
        timeout=(20, 300),
    )
    try:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            delta = (((payload.get("choices") or [{}])[0]).get("delta") or {})
            content = delta.get("content") or ""
            if content:
                yield content
    finally:
        response.close()


def _build_fallback_text(analysis: dict[str, Any], user_message: str, exc: Exception) -> str:
    stock = analysis.get("stock", {})
    prediction = analysis.get("prediction", {})
    decision = analysis.get("decision_result", {}).get("decision", {})
    reasons = analysis.get("decision_result", {}).get("reasons", [])[:3]
    lines = [
        f"本次先用本地兜底分析返回结果，原因是大模型调用失败：{exc}",
        f"{stock.get('name') or stock.get('symbol')}（{stock.get('symbol')}）当前系统建议：{decision.get('action_cn', '待观察')}。",
        f"5日胜率约 {prediction.get('p_win_prob_5', 0.0) * 100:.1f}%，信号分 {prediction.get('signal_score', 0.0):.3f}。",
    ]
    if reasons:
        lines.append("核心依据：" + "；".join(str(item) for item in reasons))
    if user_message:
        lines.append(f"你本轮追问的是：{user_message}")
    return "\n".join(lines)


def _build_market_scan_fallback_text(analysis: dict[str, Any], user_message: str, exc: Exception) -> str:
    candidates = analysis.get("candidates") or []
    lines = [f"本次先用本地兜底分析返回结果，原因是 DeepSeek 调用失败：{exc}。"]
    if candidates:
        first = candidates[0]
        decision = ((first.get("decision_result") or {}).get("decision") or {})
        hold_days = decision.get("suggested_hold_days", "?")
        reasons = first.get("reasons") or []
        lines.append(
            f"当前首选是 {first.get('name') or first.get('symbol')}（{first.get('symbol')}），建议动作：{decision.get('action_cn', '买入')}，建议持有 {hold_days} 天。"
        )
        if reasons:
            lines.append("首选依据：" + "；".join(str(item) for item in reasons[:3]))
    lines.append(f"本次共返回前 {analysis.get('top_n', 0)} 只股票，样本日期 {analysis.get('effective_trade_date') or '未知'}。")
    if user_message:
        lines.append(f"你本轮追问的是：{user_message}")
    return "\n".join(lines)


def stream_single_stock_chat(
    *,
    user: UserAccount,
    session_id: int | None,
    symbol: str | None,
    is_holding: bool,
    holding_days: int,
    target_date: str | None,
    risk_preference: str,
    message: str | None,
    refresh_analysis: bool,
) -> Iterator[str]:
    normalized_symbol = normalize_symbol(symbol) if symbol else None
    session_obj, _ = _prepare_session(
        int(user.id),
        session_id=session_id,
        symbol=normalized_symbol,
        is_holding=is_holding,
        holding_days=holding_days,
        risk_preference=risk_preference,
    )
    yield _json_event("session", {"session_id": int(session_obj.id)})

    analysis = session_obj.latest_analysis_json
    analysis_error: Exception | None = None
    if refresh_analysis or not analysis:
        event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def progress(stage: str) -> None:
            event_queue.put(
                (
                    "stage",
                    {
                        "stage": stage,
                        "label": STAGE_LABELS.get(stage, stage),
                        "status": "running",
                    },
                )
            )

        def worker() -> None:
            try:
                result = build_single_stock_analysis(
                    symbol=session_obj.symbol,
                    target_date=target_date,
                    is_holding=bool(is_holding),
                    holding_days=int(holding_days),
                    risk_preference=risk_preference,
                    progress=progress,
                )
                event_queue.put(("analysis", result))
            except Exception as exc:
                event_queue.put(("error", exc))
            finally:
                event_queue.put(("done", None))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        finished = False
        while not finished:
            event_type, payload = event_queue.get()
            if event_type == "stage":
                yield _json_event("stage", payload)
            elif event_type == "analysis":
                analysis = payload
                yield _json_event("analysis_result", payload)
            elif event_type == "error":
                analysis_error = payload
            elif event_type == "done":
                finished = True
        if analysis_error is not None:
            yield _json_event("error", {"detail": str(analysis_error)})
            return

    if not analysis:
        yield _json_event("error", {"detail": "未获取到分析结果。"})
        return

    user_message = (message or "").strip() or _build_default_user_prompt(session_obj.symbol, analysis)
    _update_session_after_analysis(int(user.id), int(session_obj.id), analysis=analysis, user_message=user_message)
    user_msg_row = _append_message(int(user.id), int(session_obj.id), "user", user_message)
    yield _json_event("message", _serialize_message(user_msg_row))

    yield _json_event(
        "stage",
        {
            "stage": "llm_analysis",
            "label": STAGE_LABELS["llm_analysis"],
            "status": "running",
        },
    )

    assistant_text = ""
    try:
        llm_messages = _build_llm_messages(
            int(session_obj.id),
            analysis=analysis,
            max_context_messages=_resolve_llm_settings()["max_context_messages"],
        )
        for chunk in _stream_llm_response(llm_messages):
            assistant_text += chunk
            yield _json_event("delta", {"content": chunk})
    except Exception as exc:
        fallback = _build_fallback_text(analysis, user_message, exc)
        assistant_text = fallback
        yield _json_event("delta", {"content": fallback})

    assistant_row = _append_message(
        int(user.id),
        int(session_obj.id),
        "assistant",
        assistant_text,
        payload_json={"analysis_trade_date": analysis.get("effective_trade_date")},
    )
    _update_session_after_assistant(int(user.id), int(session_obj.id), assistant_text)
    yield _json_event("assistant_message", _serialize_message(assistant_row))
    yield _json_event("done", {"session_id": int(session_obj.id)})


def stream_market_scan_chat(
    *,
    user: UserAccount,
    session_id: int | None,
    top_n: int,
    target_date: str | None,
    scan_mode: str,
    message: str | None,
    refresh_analysis: bool,
) -> Iterator[str]:
    session_obj, _ = _prepare_session(
        int(user.id),
        session_id=session_id,
        symbol=MARKET_SCAN_SYMBOL,
        is_holding=False,
        holding_days=int(top_n),
        risk_preference="balanced",
        title=_build_market_scan_title(top_n, None, scan_mode),
        stock_name="股票推荐",
    )
    yield _json_event("session", {"session_id": int(session_obj.id)})

    analysis = session_obj.latest_analysis_json
    if analysis and (
        int(analysis.get("top_n") or 0) != int(top_n)
        or str(analysis.get("scan_mode") or "market") != str(scan_mode or "market")
    ):
        refresh_analysis = True

    analysis_error: Exception | None = None
    if refresh_analysis or not analysis:
        event_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        def progress(payload: dict[str, Any]) -> None:
            stage = str(payload.get("stage") or "")
            event_queue.put(
                (
                    "stage",
                    {
                        "stage": stage,
                        "label": STAGE_LABELS.get(stage, stage),
                        "status": "running",
                        "current": payload.get("current"),
                        "total": payload.get("total"),
                        "message": payload.get("message"),
                    },
                )
            )

        def worker() -> None:
            try:
                result = build_market_scan_v2(
                    top_n=int(top_n),
                    target_date=target_date,
                    risk_preference="balanced",
                    scan_mode=scan_mode,
                    progress=progress,
                )
                event_queue.put(("analysis", result))
            except Exception as exc:
                event_queue.put(("error", exc))
            finally:
                event_queue.put(("done", None))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        finished = False
        while not finished:
            event_type, payload = event_queue.get()
            if event_type == "stage":
                yield _json_event("stage", payload)
            elif event_type == "analysis":
                analysis = payload
                yield _json_event("analysis_result", payload)
            elif event_type == "error":
                analysis_error = payload
            elif event_type == "done":
                finished = True
        if analysis_error is not None:
            yield _json_event("error", {"detail": str(analysis_error)})
            return

    if not analysis:
        yield _json_event("error", {"detail": "未获取到推荐结果。"})
        return

    user_message = (message or "").strip() or _build_market_scan_default_user_prompt(int(top_n), analysis)
    _update_market_scan_session_after_analysis(int(user.id), int(session_obj.id), analysis=analysis, user_message=user_message)
    user_msg_row = _append_message(int(user.id), int(session_obj.id), "user", user_message)
    yield _json_event("message", _serialize_message(user_msg_row))

    yield _json_event(
        "stage",
        {
            "stage": "llm_analysis",
            "label": STAGE_LABELS["llm_analysis"],
            "status": "running",
        },
    )

    assistant_text = ""
    try:
        llm_messages = _build_market_scan_llm_messages(
            int(session_obj.id),
            analysis=analysis,
            max_context_messages=_resolve_llm_settings()["max_context_messages"],
        )
        for chunk in _stream_llm_response(llm_messages):
            assistant_text += chunk
            yield _json_event("delta", {"content": chunk})
    except Exception as exc:
        fallback = _build_market_scan_fallback_text(analysis, user_message, exc)
        assistant_text = fallback
        yield _json_event("delta", {"content": fallback})

    assistant_row = _append_message(
        int(user.id),
        int(session_obj.id),
        "assistant",
        assistant_text,
        payload_json={"analysis_trade_date": analysis.get("effective_trade_date"), "top_n": int(top_n)},
    )
    _update_session_after_assistant(int(user.id), int(session_obj.id), assistant_text)
    yield _json_event("assistant_message", _serialize_message(assistant_row))
    yield _json_event("done", {"session_id": int(session_obj.id)})
