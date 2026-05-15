from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, sessionmaker
from torch.utils.data import DataLoader, Dataset

from decision import ENGINE_VERSION, evaluate_stock_decision, rank_market_candidates_quick
from jobs.build_company_profiles import run as run_build_company_profiles
from jobs.build_event_features import run as run_build_event_features
from jobs.build_prediction_samples import _build_prediction_export_frame, build_prediction_frame
from jobs.sync_daily_bars import run as run_sync_daily_bars
from jobs.sync_financial_snapshot import run as run_sync_financial_snapshot
from jobs.sync_index_bars import run as run_sync_index_bars
from jobs.sync_news import run as run_sync_news
from jobs.sync_sector_daily import run as run_sync_sector_daily
from jobs.sync_securities import run as run_sync_securities
from train.data import FeatureNormalizer
from train.model import TinyMultiInputModel
from train.predict import (
    _parse_array,
    collate_inference_batch,
    load_model,
    resolve_device_name,
    resolve_p_win_thresholds,
    run_prediction,
)
from utils.config import AppConfig, load_config
from utils.symbols import normalize_symbol
from warehouse.db import build_engine, build_session_factory
from warehouse.models import (
    AnnouncementNorm,
    CompanyProfile,
    DailyBar,
    EventFeaturesDaily,
    FinancialSnapshot,
    IndexBar,
    NewsNorm,
    Security,
    SectorDaily,
    TradeCalendar,
    TrainingSample,
    UserAccount,
)
from warehouse.repository import WarehouseRepository
from warehouse.schema_init import init_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
ARTIFACTS_DIR = PROJECT_ROOT / "train" / "artifacts"
CACHE_DIR = PROJECT_ROOT / "tmp" / "webapp_cache"
P_WIN_HORIZONS = (3, 5, 10, 20, 40)
FRESH_SAMPLE_LOCK = Lock()
CN_TZ = ZoneInfo("Asia/Shanghai")
MARKET_SCAN_QUICK_SAMPLE_SIZE = 100
MARKET_SCAN_QUICK_MIN_WIN_RATE = 0.55
MARKET_SCAN_QUICK_MIN_RET_MU = 0.01
MARKET_SCAN_QUICK_BATCH_SIZE = 32
MARKET_SCAN_FULL_DAILY_COVERAGE_RATIO = 0.9
MARKET_SCAN_FULL_MIN_SAMPLE_RATIO = 0.05


def _normalize_market_scan_holding_days(value: int | None) -> int | None:
    if value is None:
        return None
    holding_days = int(value)
    if holding_days not in P_WIN_HORIZONS:
        raise ValueError("推荐持有天数必须是 3、5、10、20、40 之一")
    return holding_days


@dataclass(frozen=True)
class PredictionBundle:
    model: Any
    normalizer: FeatureNormalizer
    checkpoint: dict[str, Any]
    device: torch.device
    thresholds: np.ndarray
    source: str
    descriptor: str


class PackagedDataFrameDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        source_df: pd.DataFrame,
        normalizer: FeatureNormalizer,
        vocab_limits: Mapping[str, int] | None = None,
    ):
        self.df = source_df.reset_index(drop=True)
        self.normalizer = normalizer
        self.vocab_limits = dict(vocab_limits or {})

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        symbol_id = self._clip_id(pd.to_numeric(row.get("symbol_id"), errors="coerce"), "symbol_id")
        industry_id = self._clip_id(pd.to_numeric(row.get("industry_id"), errors="coerce"), "industry_id")
        board_id = self._clip_id(pd.to_numeric(row.get("board_id"), errors="coerce"), "board_id")
        neighbor_symbol_ids = torch.tensor(self._clip_neighbor_ids(row.get("neighbor_symbol_ids", [])), dtype=torch.long)
        neighbor_scores = torch.tensor(_parse_array(row.get("neighbor_scores", []), np.float32), dtype=torch.float32)
        return {
            "X_seq": self.normalizer.normalize_seq(
                torch.tensor(_parse_array(row["x_seq"], np.float32), dtype=torch.float32)
            ),
            "X_tab": self.normalizer.normalize_tab(
                torch.tensor(_parse_array(row["x_tab"], np.float32), dtype=torch.float32)
            ),
            "X_event": self.normalizer.normalize_event(
                torch.tensor(_parse_array(row["x_event"], np.float32), dtype=torch.float32)
            ),
            "X_mkt": self.normalizer.normalize_mkt(
                torch.tensor(_parse_array(row["x_mkt"], np.float32), dtype=torch.float32)
            ),
            "X_company_profile": self.normalizer.normalize_profile(
                torch.tensor(_parse_array(row["x_company_profile"], np.float32), dtype=torch.float32)
            ),
            "X_company_ids": {
                "symbol_id": torch.tensor(symbol_id, dtype=torch.long),
                "industry_id": torch.tensor(industry_id, dtype=torch.long),
                "board_id": torch.tensor(board_id, dtype=torch.long),
            },
            "neighbors": {
                "neighbor_symbol_ids": neighbor_symbol_ids,
                "neighbor_scores": neighbor_scores,
            },
            "meta": {
                "symbol": row.get("symbol"),
                "trade_date": row.get("trade_date"),
                "name": row.get("name"),
                "industry_sw": row.get("industry_sw"),
                "board": row.get("board"),
            },
        }

    def _clip_id(self, value: Any, key: str) -> int:
        numeric = _safe_int(value, 0)
        limit = int(self.vocab_limits.get(key, 0) or 0)
        if limit > 0:
            return max(0, min(numeric, limit - 1))
        return max(0, numeric)

    def _clip_neighbor_ids(self, value: Any) -> np.ndarray:
        arr = _parse_array(value, np.int64)
        limit = int(self.vocab_limits.get("symbol_id", 0) or 0)
        if limit > 0 and arr.size > 0:
            arr = np.clip(arr, 0, limit - 1)
        return arr


def bootstrap_web_app() -> None:
    init_schema(str(CONFIG_PATH))


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    return load_config(CONFIG_PATH)


@lru_cache(maxsize=1)
def get_engine():
    return build_engine(get_app_config())


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    return build_session_factory(get_engine())


@lru_cache(maxsize=1)
def get_repository() -> WarehouseRepository:
    return WarehouseRepository(get_engine())


def iter_db() -> Session:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def get_secret_key() -> str:
    import os
    key = os.environ.get("JWT_SECRET_KEY")
    if not key:
        raise RuntimeError(
            "JWT_SECRET_KEY 环境变量未设置。"
            "请运行: export JWT_SECRET_KEY=<your-secret-key>"
        )
    return key


def _to_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    return str(value)


def _normalize_trade_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "trade_date" in output.columns:
        output["trade_date"] = pd.to_datetime(output["trade_date"], errors="coerce")
    return output


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(numeric) or np.isinf(numeric):
        return default
    return numeric


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        numeric = int(float(value))
    except (TypeError, ValueError):
        return default
    return numeric


def _normalize_scan_mode(scan_mode: str | None) -> str:
    value = str(scan_mode or "market").strip().lower()
    if value not in {"market", "quick"}:
        raise ValueError("扫描模式必须是 market 或 quick")
    return value


def _sample_market_symbols(raw_df: pd.DataFrame, sample_size: int) -> list[str]:
    if raw_df.empty or "symbol" not in raw_df.columns:
        return []
    unique_symbols = pd.Series(raw_df["symbol"], dtype="object").dropna().astype(str).drop_duplicates().tolist()
    if len(unique_symbols) <= sample_size:
        return unique_symbols
    rng = np.random.default_rng()
    selected = rng.choice(unique_symbols, size=sample_size, replace=False)
    return [str(symbol) for symbol in selected.tolist()]


def _sample_active_market_symbols(sample_size: int) -> list[str]:
    securities = get_repository().get_securities(active_only=True)
    if securities.empty or "symbol" not in securities.columns:
        return []
    unique_symbols = securities["symbol"].dropna().astype(str).drop_duplicates().tolist()
    if len(unique_symbols) <= sample_size:
        return unique_symbols
    rng = np.random.default_rng()
    selected = rng.choice(unique_symbols, size=sample_size, replace=False)
    return [str(symbol) for symbol in selected.tolist()]


def _list_active_market_symbols() -> list[str]:
    securities = get_repository().get_securities(active_only=True)
    if securities.empty or "symbol" not in securities.columns:
        return []
    return sorted(securities["symbol"].dropna().astype(str).drop_duplicates().tolist())


def _list_active_market_symbols_for_trade_date(target_date: str) -> list[str]:
    session = get_session_factory()()
    try:
        target = date.fromisoformat(str(target_date))
        rows = session.execute(
            select(Security.symbol)
            .join(DailyBar, DailyBar.symbol == Security.symbol)
            .where(Security.status == "active", DailyBar.trade_date == target)
            .distinct()
            .order_by(Security.symbol.asc())
        ).all()
        return [str(symbol) for (symbol,) in rows if symbol]
    finally:
        session.close()


def _filter_prediction_frame_by_symbols(frame: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    if frame.empty or not symbols or "symbol" not in frame.columns:
        return frame.copy()
    symbol_set = {str(symbol) for symbol in symbols}
    return frame[frame["symbol"].astype(str).isin(symbol_set)].copy()


def _best_short_horizon(record: Mapping[str, Any]) -> tuple[int, float, float]:
    best_horizon = 3
    best_p_win = -1.0
    best_ret_mu = 0.0
    for horizon in (3, 5, 10):
        p_win = _safe_float(record.get(f"p_win_prob_{horizon}"), -1.0)
        ret_mu = _safe_float(record.get(f"ret_mu_pred_{horizon}"))
        candidate = (p_win, ret_mu, -horizon)
        if candidate > (best_p_win, best_ret_mu, -best_horizon):
            best_horizon = int(horizon)
            best_p_win = float(p_win)
            best_ret_mu = float(ret_mu)
    return best_horizon, max(0.0, float(best_p_win)), float(best_ret_mu)


def _packaged_column_mean(sequence: Any, columns: Any, target_column: str, window: int = 5) -> float:
    if target_column not in columns:
        return 0.0
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        return 0.0
    column_index = list(columns).index(target_column)
    subset = values[max(0, values.shape[0] - window) :, column_index]
    subset = subset[np.isfinite(subset)]
    if subset.size == 0:
        return 0.0
    return float(subset.mean())


def _feature_missing_rate(packaged_row: Mapping[str, Any]) -> float:
    arrays = []
    for key in ("x_seq", "x_tab", "x_event", "x_mkt", "x_company_profile"):
        raw = packaged_row.get(key)
        if raw is None:
            continue
        arrays.append(np.asarray(raw, dtype=np.float32).reshape(-1))
    if not arrays:
        return 0.0
    flat = np.concatenate(arrays)
    if flat.size == 0:
        return 0.0
    return float((~np.isfinite(flat)).sum() / max(1, flat.size))


def _build_packaged_aux_frame(packaged_df: pd.DataFrame) -> pd.DataFrame:
    if packaged_df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "trade_date",
                "name",
                "industry_sw",
                "board",
                "avg_amount_5",
                "feature_missing_rate",
                "neighbor_symbol_ids",
                "neighbor_scores",
            ]
        )
    rows: list[dict[str, Any]] = []
    for row in packaged_df.to_dict(orient="records"):
        rows.append(
            {
                "symbol": row.get("symbol"),
                "trade_date": row.get("trade_date"),
                "name": row.get("name"),
                "industry_sw": row.get("industry_sw"),
                "board": row.get("board"),
                "avg_amount_5": _packaged_column_mean(row.get("x_seq"), row.get("seq_columns") or [], "amount", 5),
                "feature_missing_rate": _feature_missing_rate(row),
                "neighbor_symbol_ids": row.get("neighbor_symbol_ids") or [],
                "neighbor_scores": row.get("neighbor_scores") or [],
            }
        )
    return _normalize_trade_date_column(pd.DataFrame(rows))


def get_latest_checkpoint_path() -> Path | None:
    candidates = sorted(ARTIFACTS_DIR.glob("best/best_model.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _build_cache_path(kind: str, checkpoint_path: Path, effective_trade_date: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    run_name = checkpoint_path.parent.name
    safe_date = effective_trade_date.replace("-", "")
    return CACHE_DIR / f"{kind}_{safe_date}_{run_name}.parquet"


@lru_cache(maxsize=4)
def load_prediction_bundle(checkpoint_path: str) -> PredictionBundle:
    resolved = Path(checkpoint_path)
    # Web inference prefers stability over speed; force CPU to avoid GPU index-assert crashes.
    device = torch.device(resolve_device_name("cpu"))
    model, normalizer, checkpoint = load_model(resolved, device)
    thresholds = resolve_p_win_thresholds(checkpoint)
    return PredictionBundle(
        model=model,
        normalizer=normalizer,
        checkpoint=checkpoint,
        device=device,
        thresholds=np.asarray(thresholds, dtype=np.float32).reshape(-1),
        source="checkpoint",
        descriptor=str(resolved),
    )


def _array_column_stats(df: pd.DataFrame, column: str, reduce_axis: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.stack([_parse_array(item, np.float32) for item in df[column].tolist()], axis=0)
    mean = values.mean(axis=reduce_axis).astype(np.float32, copy=False)
    std = values.std(axis=reduce_axis).astype(np.float32, copy=False)
    std = np.clip(std, 1e-3, None)
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


def build_random_prediction_bundle(df: pd.DataFrame) -> PredictionBundle:
    if df.empty:
        raise ValueError("cannot build random prediction bundle from empty dataframe")
    # Keep random-init inference path deterministic and stable for web usage.
    device = torch.device(resolve_device_name("cpu"))
    seq_mean, seq_std = _array_column_stats(df, "x_seq", (0, 1))
    tab_mean, tab_std = _array_column_stats(df, "x_tab", (0,))
    event_mean, event_std = _array_column_stats(df, "x_event", (0,))
    mkt_mean, mkt_std = _array_column_stats(df, "x_mkt", (0,))
    profile_mean, profile_std = _array_column_stats(df, "x_company_profile", (0,))
    normalizer = FeatureNormalizer(
        seq_mean=seq_mean,
        seq_std=seq_std,
        tab_mean=tab_mean,
        tab_std=tab_std,
        event_mean=event_mean,
        event_std=event_std,
        mkt_mean=mkt_mean,
        mkt_std=mkt_std,
        profile_mean=profile_mean,
        profile_std=profile_std,
        clip_value=10.0,
    )
    first_row = df.iloc[0]
    x_seq = np.asarray(first_row["x_seq"], dtype=np.float32)
    x_tab = np.asarray(first_row["x_tab"], dtype=np.float32)
    x_event = np.asarray(first_row["x_event"], dtype=np.float32)
    x_mkt = np.asarray(first_row["x_mkt"], dtype=np.float32)
    x_profile = np.asarray(first_row["x_company_profile"], dtype=np.float32)
    input_dims = {
        "seq_length": int(x_seq.shape[0]),
        "f_seq": int(x_seq.shape[1]),
        "f_tab": int(x_tab.shape[0]),
        "f_event": int(x_event.shape[0]),
        "f_mkt": int(x_mkt.shape[0]),
        "f_company_profile": int(x_profile.shape[0]),
        "neighbor_topk": int(len(_parse_array(first_row.get("neighbor_symbol_ids", []), np.int64))),
    }
    vocab_sizes = {
        "symbol_id": int(pd.to_numeric(df["symbol_id"], errors="coerce").fillna(0).max()) + 2,
        "industry_id": int(pd.to_numeric(df["industry_id"], errors="coerce").fillna(0).max()) + 2,
        "board_id": int(pd.to_numeric(df["board_id"], errors="coerce").fillna(0).max()) + 2,
    }
    head_dims = {"p_win": 5, "ret_mu": 5, "risk_dd": 5, "bigloss": 2, "rank_score": 1}
    torch.manual_seed(42)
    model = TinyMultiInputModel(input_dims=input_dims, vocab_sizes=vocab_sizes, head_dims=head_dims).to(device)
    model.eval()
    checkpoint = {
        "input_dims": input_dims,
        "vocab_sizes": vocab_sizes,
        "head_dims": head_dims,
        "model_source": "random_init",
        "model_structure": "train.model.TinyMultiInputModel",
    }
    return PredictionBundle(
        model=model,
        normalizer=normalizer,
        checkpoint=checkpoint,
        device=device,
        thresholds=np.full(len(P_WIN_HORIZONS), 0.5, dtype=np.float32),
        source="random_init",
        descriptor="random://train/model.py#TinyMultiInputModel",
    )


def resolve_prediction_bundle(df: pd.DataFrame, checkpoint_path: Path | None) -> PredictionBundle:
    if checkpoint_path is not None and checkpoint_path.exists():
        return load_prediction_bundle(str(checkpoint_path.resolve()))
    return build_random_prediction_bundle(df)


def predict_packaged_dataframe(
    df: pd.DataFrame,
    checkpoint_path: Path | None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, PredictionBundle]:
    bundle = resolve_prediction_bundle(df, checkpoint_path)
    vocab_limits = bundle.checkpoint.get("vocab_sizes", {}) if isinstance(bundle.checkpoint, dict) else {}
    dataset = PackagedDataFrameDataset(df, bundle.normalizer, vocab_limits=vocab_limits)
    batch_size = 2048 if bundle.device.type == "cuda" else 512
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=bundle.device.type == "cuda",
        collate_fn=collate_inference_batch,
    )
    predicted = run_prediction(
        bundle.model,
        loader,
        bundle.device,
        p_win_thresholds=bundle.thresholds,
        progress_callback=progress,
    )
    return predicted, bundle


def list_model_runs(limit: int = 8) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not ARTIFACTS_DIR.exists():
        return runs
    for run_dir in sorted(ARTIFACTS_DIR.glob("run_*"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        checkpoint_path = run_dir / "best_model.pt"
        metrics_path = run_dir / "test_metrics.json"
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        test_metrics_raw = metrics.get("test_metrics_raw") or {}
        test_evaluation_skipped = bool(metrics.get("test_evaluation_skipped"))
        runs.append(
            {
                "run_name": run_dir.name,
                "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
                "updated_at": _to_iso(datetime.fromtimestamp(run_dir.stat().st_mtime)),
                "train_minutes": metrics.get("train_minutes"),
                "best_valid_loss": metrics.get("best_valid_loss"),
                "test_p_win_acc": None if test_evaluation_skipped else test_metrics_raw.get("p_win_acc"),
                "test_rank_score_mae": None if test_evaluation_skipped else test_metrics_raw.get("rank_score_mae"),
                "has_test_metrics": metrics_path.exists() and not test_evaluation_skipped,
                "validation_includes_test": bool(metrics.get("validation_includes_test")),
            }
        )
    return runs


def get_dashboard_summary(session: Session) -> dict[str, Any]:
    repo = get_repository()
    latest_checkpoint = get_latest_checkpoint_path()
    latest_trade_date = repo.get_latest_trade_date("training_samples")
    training_sample_count = 0
    if latest_trade_date:
        latest_trade_value = date.fromisoformat(latest_trade_date)
        stmt = select(func.count()).select_from(TrainingSample).where(TrainingSample.trade_date == latest_trade_value)
        training_sample_count = int(session.execute(stmt).scalar() or 0)
    security_count = int(session.execute(select(func.count()).select_from(Security)).scalar() or 0)
    active_user_count = int(
        session.execute(select(func.count()).select_from(UserAccount).where(UserAccount.is_active.is_(True))).scalar() or 0
    )
    current_modules = [
        {"key": "dataset", "title": "数据集管理", "status": "ready"},
        {"key": "prediction", "title": "模型预测", "status": "ready"},
        {"key": "today_data", "title": "今日数据", "status": "ready"},
        {"key": "decision", "title": "决策引擎", "status": "ready"},
        {"key": "llm", "title": "大语言模型", "status": "ready"},
    ]
    planned_modules = [
        {"key": "strategy", "title": "策略回测", "status": "planned"},
    ]
    return {
        "latest_trade_date": latest_trade_date,
        "training_sample_count": training_sample_count,
        "security_count": security_count,
        "active_user_count": active_user_count,
        "latest_checkpoint": str(latest_checkpoint) if latest_checkpoint else None,
        "latest_checkpoint_time": _to_iso(datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)) if latest_checkpoint else None,
        "current_modules": current_modules,
        "planned_modules": planned_modules,
    }


def _merge_prediction_frames(
    raw_df: pd.DataFrame,
    predicted_df: pd.DataFrame,
    packaged_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    raw_output = _normalize_trade_date_column(raw_df)
    predicted_output = _normalize_trade_date_column(predicted_df)
    merged = raw_output.merge(
        predicted_output,
        on=["symbol", "trade_date", "name", "industry_sw", "board"],
        how="left",
    )
    if packaged_df is not None and not packaged_df.empty:
        aux_output = _build_packaged_aux_frame(packaged_df)
        merged = merged.merge(
            aux_output,
            on=["symbol", "trade_date", "name", "industry_sw", "board"],
            how="left",
        )
    if "signal_score" in merged.columns:
        merged = merged.sort_values("signal_score", ascending=False).reset_index(drop=True)
    return merged


@lru_cache(maxsize=1)
def get_active_security_count() -> int:
    session = get_session_factory()()
    try:
        stmt = select(func.count()).select_from(Security).where(Security.status == "active")
        return int(session.execute(stmt).scalar() or 0)
    finally:
        session.close()


@lru_cache(maxsize=1)
def _industry_training_percentiles() -> dict[str, float]:
    session = get_session_factory()()
    try:
        rows = session.execute(select(TrainingSample.industry_sw, func.count()).group_by(TrainingSample.industry_sw)).all()
        if not rows:
            return {}
        count_map = {str(industry or "Unknown"): int(count) for industry, count in rows}
        sorted_counts = sorted(count_map.values())
        total = len(sorted_counts)
        percentiles: dict[str, float] = {}
        for industry, count in count_map.items():
            rank = sum(1 for value in sorted_counts if value <= count)
            percentiles[industry] = rank / max(1, total)
        return percentiles
    finally:
        session.close()


def load_training_coverage(symbols: list[str], industries: list[str]) -> dict[str, Any]:
    session = get_session_factory()()
    try:
        symbol_counts: dict[str, int] = {}
        normalized_symbols = sorted({symbol for symbol in symbols if symbol})
        if normalized_symbols:
            rows = session.execute(
                select(TrainingSample.symbol, func.count())
                .where(TrainingSample.symbol.in_(normalized_symbols))
                .group_by(TrainingSample.symbol)
            ).all()
            symbol_counts = {str(symbol): int(count) for symbol, count in rows}
        percentiles = _industry_training_percentiles()
        industry_pctiles = {
            str(industry): float(percentiles.get(str(industry), 1.0))
            for industry in industries
            if industry
        }
        return {"symbol_counts": symbol_counts, "industry_pctiles": industry_pctiles}
    finally:
        session.close()


def _augment_decision_record(
    row: Mapping[str, Any],
    *,
    model_descriptor: str,
    market_symbol_count: int,
    coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = dict(row)
    record["checkpoint_path"] = str(model_descriptor)
    record["market_symbol_count"] = market_symbol_count
    down_limit_count = _safe_float(record.get("down_limit_count"))
    up_limit_count = _safe_float(record.get("up_limit_count"))
    if market_symbol_count > 0:
        record["down_limit_ratio"] = down_limit_count / market_symbol_count
        record["up_limit_ratio"] = up_limit_count / market_symbol_count
    record["major_index_ret_1"] = min(_safe_float(record.get("hs300_ret_1")), _safe_float(record.get("cyb_ret_1")))
    symbol_counts = (coverage or {}).get("symbol_counts", {})
    industry_pctiles = (coverage or {}).get("industry_pctiles", {})
    record["train_symbol_samples"] = int(symbol_counts.get(str(record.get("symbol")), 0))
    record["train_industry_pctile"] = float(industry_pctiles.get(str(record.get("industry_sw")), 1.0))
    return record


def _decision_metadata_from_bundle(bundle: PredictionBundle, feature_version: Any) -> dict[str, Any]:
    model_version = Path(bundle.descriptor).parent.name if bundle.source == "checkpoint" else "random_init"
    return {
        "model_version": model_version,
        "feature_version": str(feature_version or "unknown_feature"),
        "inference_ts": _to_iso(datetime.now().astimezone()),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    return value


def _build_sample_payload(
    packaged_row: Mapping[str, Any],
    model_input: Mapping[str, Any],
) -> dict[str, Any]:
    company_ids = model_input.get("X_company_ids", {})
    neighbors = model_input.get("neighbors", {})
    meta = model_input.get("meta", {})
    return {
        "format": "fast_stock_inference_sample_v2",
        "meta": {
            "symbol": meta.get("symbol") or packaged_row.get("symbol"),
            "trade_date": _to_iso(meta.get("trade_date") or packaged_row.get("trade_date")),
            "name": meta.get("name") or packaged_row.get("name"),
            "industry_sw": meta.get("industry_sw") or packaged_row.get("industry_sw"),
            "board": meta.get("board") or packaged_row.get("board"),
        },
        "X_seq": _json_ready(model_input.get("X_seq")),
        "X_tab": _json_ready(model_input.get("X_tab")),
        "X_event": _json_ready(model_input.get("X_event")),
        "X_mkt": _json_ready(model_input.get("X_mkt")),
        "X_company_ids": {
            "symbol_id": _safe_int(company_ids.get("symbol_id").item() if company_ids.get("symbol_id") is not None else 0),
            "industry_id": _safe_int(company_ids.get("industry_id").item() if company_ids.get("industry_id") is not None else 0),
            "board_id": _safe_int(company_ids.get("board_id").item() if company_ids.get("board_id") is not None else 0),
        },
        "X_company_profile": _json_ready(model_input.get("X_company_profile")),
        "neighbors": {
            "neighbor_symbol_ids": _json_ready(neighbors.get("neighbor_symbol_ids")),
            "neighbor_scores": _json_ready(neighbors.get("neighbor_scores")),
        },
        "schema": {
            "seq_columns": list(packaged_row.get("seq_columns") or []),
            "tab_columns": list(packaged_row.get("tab_columns") or []),
            "event_columns": list(packaged_row.get("event_columns") or []),
            "mkt_columns": list(packaged_row.get("mkt_columns") or []),
            "company_id_columns": list(packaged_row.get("company_id_columns") or []),
            "company_profile_columns": list(packaged_row.get("company_profile_columns") or []),
            "neighbor_topk": _safe_int(pd.to_numeric(packaged_row.get("neighbor_topk"), errors="coerce")),
            "seq_length": _safe_int(pd.to_numeric(packaged_row.get("seq_length"), errors="coerce")),
        },
    }


def _extract_symbol_events(symbol: str, effective_trade_date: str, limit: int = 8) -> list[dict[str, Any]]:
    session = get_session_factory()()
    try:
        end_dt = datetime.fromisoformat(f"{effective_trade_date}T23:59:59")
        start_dt = end_dt - timedelta(days=7)
        event_items: list[dict[str, Any]] = []

        # First pass: news that explicitly mentions this symbol
        news_rows = list(session.execute(
            select(NewsNorm)
            .where(NewsNorm.publish_time >= start_dt, NewsNorm.publish_time <= end_dt)
            .order_by(NewsNorm.publish_time.desc())
            .limit(80)
        ).scalars())
        for item in news_rows:
            mentioned = {str(entry).upper() for entry in (item.mentioned_symbols or [])}
            if symbol not in mentioned:
                continue
            event_items.append(
                {
                    "kind": "news",
                    "id": item.news_id,
                    "publish_time": _to_iso(item.publish_time),
                    "title": item.title,
                    "summary": item.summary,
                    "event_type": item.event_type,
                    "event_direction": item.event_direction,
                    "event_strength": _safe_float(item.event_strength),
                    "event_confidence": _safe_float(item.event_confidence),
                    "source": item.source,
                    "url": item.url,
                }
            )
            if len(event_items) >= limit:
                break

        # Fallback: show trade-date daily news when no symbol-specific matches
        # The news provider builds macro/daily news with empty mentioned_symbols.
        if not event_items:
            date_start = datetime.fromisoformat(effective_trade_date)
            date_end = datetime.fromisoformat(f"{effective_trade_date}T23:59:59")
            day_news_rows = list(session.execute(
                select(NewsNorm)
                .where(NewsNorm.publish_time >= date_start, NewsNorm.publish_time <= date_end)
                .order_by(NewsNorm.publish_time.desc())
                .limit(limit)
            ).scalars())
            for item in day_news_rows:
                event_items.append(
                    {
                        "kind": "news",
                        "id": item.news_id,
                        "publish_time": _to_iso(item.publish_time),
                        "title": item.title,
                        "summary": item.summary,
                        "event_type": item.event_type,
                        "event_direction": item.event_direction,
                        "event_strength": _safe_float(item.event_strength),
                        "event_confidence": _safe_float(item.event_confidence),
                        "source": item.source,
                        "url": item.url,
                    }
                )

        if len(event_items) < limit:
            announcement_rows = session.execute(
                select(AnnouncementNorm)
                .where(AnnouncementNorm.publish_time >= start_dt, AnnouncementNorm.publish_time <= end_dt)
                .order_by(AnnouncementNorm.publish_time.desc())
                .limit(80)
            ).scalars()
            for item in announcement_rows:
                mentioned = {str(entry).upper() for entry in (item.mentioned_symbols or [])}
                if symbol not in mentioned:
                    continue
                event_items.append(
                    {
                        "kind": "announcement",
                        "id": item.announcement_id,
                        "publish_time": _to_iso(item.publish_time),
                        "title": item.title,
                        "summary": item.summary,
                        "event_type": item.event_type,
                        "event_direction": item.event_direction,
                        "event_strength": _safe_float(item.event_strength),
                        "event_confidence": _safe_float(item.event_confidence),
                        "source": item.source,
                    }
                )
                if len(event_items) >= limit:
                    break
        event_items.sort(key=lambda item: item.get("publish_time") or "", reverse=True)
        return event_items[:limit]
    finally:
        session.close()


def _has_exact_trade_date_inputs(symbol: str, target_date: str) -> bool:
    session = get_session_factory()()
    try:
        target = date.fromisoformat(str(target_date))
        news_exists = bool(
            session.execute(select(NewsNorm.news_id).where(NewsNorm.trade_date == target).limit(1)).scalar_one_or_none()
        )
        daily_exists = bool(
            session.execute(
                select(DailyBar.id).where(DailyBar.symbol == symbol, DailyBar.trade_date == target).limit(1)
            ).scalar_one_or_none()
        )
        index_exists = bool(
            session.execute(select(IndexBar.id).where(IndexBar.trade_date == target).limit(1)).scalar_one_or_none()
        )
        sector_exists = bool(
            session.execute(select(SectorDaily.id).where(SectorDaily.trade_date == target).limit(1)).scalar_one_or_none()
        )
        event_exists = bool(
            session.execute(
                select(EventFeaturesDaily.id).where(EventFeaturesDaily.trade_date == target).limit(1)
            ).scalar_one_or_none()
        )
        profile_exists = bool(
            session.execute(
                select(CompanyProfile.id)
                .where(CompanyProfile.symbol == symbol, CompanyProfile.asof_date == target)
                .limit(1)
            ).scalar_one_or_none()
        )
        financial_exists = bool(
            session.execute(
                select(FinancialSnapshot.id)
                .where(FinancialSnapshot.symbol == symbol, FinancialSnapshot.asof_date <= target)
                .order_by(FinancialSnapshot.asof_date.desc())
                .limit(1)
            ).scalar_one_or_none()
        )
        return daily_exists and index_exists and sector_exists and event_exists and profile_exists and financial_exists and news_exists
    finally:
        session.close()


def _trade_date_open_state(target_date: str) -> bool | None:
    session = get_session_factory()()
    try:
        target = date.fromisoformat(str(target_date))
        return session.execute(
            select(TradeCalendar.is_open).where(TradeCalendar.trade_date == target, TradeCalendar.exchange == "CN")
        ).scalar_one_or_none()
    finally:
        session.close()


def _latest_open_trade_date_on_or_before(anchor_date: date) -> date:
    session = get_session_factory()()
    try:
        resolved = session.execute(
            select(func.max(TradeCalendar.trade_date)).where(
                TradeCalendar.exchange == "CN",
                TradeCalendar.is_open.is_(True),
                TradeCalendar.trade_date <= anchor_date,
            )
        ).scalar_one_or_none()
        if resolved is not None:
            return resolved
    finally:
        session.close()

    fallback = anchor_date
    while fallback.weekday() >= 5:
        fallback -= timedelta(days=1)
    return fallback


def _resolve_latest_required_trade_date(anchor_date: date) -> date:
    open_state = _trade_date_open_state(anchor_date.isoformat())
    if open_state is True:
        return anchor_date
    if open_state is False:
        return _latest_open_trade_date_on_or_before(anchor_date - timedelta(days=1))
    if anchor_date.weekday() >= 5:
        return _latest_open_trade_date_on_or_before(anchor_date)
    return anchor_date


def _latest_available_symbol_trade_date_on_or_before(symbol: str, anchor_date: str) -> str | None:
    normalized_symbol = normalize_symbol(symbol)
    session = get_session_factory()()
    try:
        target = date.fromisoformat(str(anchor_date))
        resolved = session.execute(
            select(func.max(DailyBar.trade_date)).where(
                DailyBar.symbol == normalized_symbol,
                DailyBar.trade_date <= target,
            )
        ).scalar_one_or_none()
        return resolved.isoformat() if resolved is not None else None
    finally:
        session.close()


def _latest_available_market_trade_date_on_or_before(anchor_date: str) -> str | None:
    session = get_session_factory()()
    try:
        target = date.fromisoformat(str(anchor_date))
        window_start = target - timedelta(days=14)
        rows = session.execute(
            select(DailyBar.trade_date, func.count(func.distinct(DailyBar.symbol)).label("symbol_count"))
            .join(Security, Security.symbol == DailyBar.symbol)
            .where(
                Security.status == "active",
                DailyBar.trade_date <= target,
                DailyBar.trade_date >= window_start,
            )
            .group_by(DailyBar.trade_date)
            .order_by(DailyBar.trade_date.desc())
        ).all()
        if not rows:
            return None
        best_count = max(int(symbol_count or 0) for _, symbol_count in rows)
        coverage_floor = max(1, int(best_count * 0.95))
        for trade_date, symbol_count in rows:
            if int(symbol_count or 0) >= coverage_floor:
                return trade_date.isoformat()
        return rows[0][0].isoformat()
    finally:
        session.close()


def _get_security_snapshot(symbol: str) -> dict[str, Any] | None:
    normalized_symbol = normalize_symbol(symbol)
    session = get_session_factory()()
    try:
        row = session.execute(
            select(Security.symbol, Security.name, Security.status, Security.is_st, Security.list_date).where(
                Security.symbol == normalized_symbol
            )
        ).one_or_none()
        if row is None:
            return None
        return {
            "symbol": str(row.symbol),
            "name": str(row.name),
            "status": str(row.status or ""),
            "is_st": bool(row.is_st),
            "list_date": row.list_date,
        }
    finally:
        session.close()


def _active_daily_bar_count_for_date(target_date: str) -> int:
    session = get_session_factory()()
    try:
        target = date.fromisoformat(str(target_date))
        return int(
            session.execute(
                select(func.count(func.distinct(DailyBar.symbol)))
                .join(Security, Security.symbol == DailyBar.symbol)
                .where(Security.status == "active", DailyBar.trade_date == target)
            ).scalar()
            or 0
        )
    finally:
        session.close()


def _full_market_daily_coverage_floor(market_total_candidates: int) -> int:
    return max(1, int(int(market_total_candidates or 0) * MARKET_SCAN_FULL_DAILY_COVERAGE_RATIO))


def _ensure_full_market_daily_inputs(
    target_date: str,
    *,
    market_total_candidates: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    if _is_known_non_trade_date(target_date):
        raise ValueError(f"{target_date} 不是交易日，无法构建市场样本")

    coverage_floor = _full_market_daily_coverage_floor(market_total_candidates)
    coverage = _active_daily_bar_count_for_date(target_date)
    if coverage >= coverage_floor:
        return coverage

    if progress:
        progress(
            {
                "stage": "candidate_select",
                "current": int(coverage),
                "total": int(market_total_candidates),
                "message": f"正在补齐 {target_date} 的全市场行情数据，已有 {coverage}/{market_total_candidates}",
            }
        )
    with FRESH_SAMPLE_LOCK:
        coverage = _active_daily_bar_count_for_date(target_date)
        if coverage < coverage_floor:
            _refresh_market_inputs_for_date(target_date, progress=progress)

    coverage = _active_daily_bar_count_for_date(target_date)
    if coverage < coverage_floor:
        raise ValueError(
            f"{target_date} 全市场行情数据覆盖不足，当前仅 {coverage}/{market_total_candidates}，"
            "请检查数据源或稍后重试"
        )
    return coverage


def _resolve_analysis_target_date(target_date: str | None, now: datetime | None = None) -> str:
    if target_date:
        return date.fromisoformat(str(target_date)).isoformat()

    current = now or datetime.now(CN_TZ)
    if current.tzinfo is None:
        current = current.replace(tzinfo=CN_TZ)
    else:
        current = current.astimezone(CN_TZ)

    previous_date = current.date() - timedelta(days=1)
    return _resolve_latest_required_trade_date(previous_date).isoformat()


def _is_known_non_trade_date(target_date: str) -> bool:
    open_state = _trade_date_open_state(target_date)
    if open_state is False:
        return True
    if open_state is True:
        return False
    return pd.Timestamp(target_date).dayofweek >= 5


def _refresh_single_stock_inputs_for_date(
    symbol: str, target_date: str, *, progress: Callable[[str], None] | None = None
) -> None:
    sync_start = (pd.Timestamp(target_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    normalized_symbol = normalize_symbol(symbol)
    symbols = [normalized_symbol]
    config_path = str(CONFIG_PATH)

    run_sync_daily_bars(config_path, sync_start, target_date, symbols=symbols)
    run_sync_index_bars(config_path, sync_start, target_date)
    run_sync_sector_daily(config_path, sync_start, target_date, None)
    run_sync_financial_snapshot(config_path, sync_start, target_date, None, symbols=symbols)
    run_sync_news(config_path, target_date, target_date, None, symbols=symbols)
    if progress:
        progress("news_extraction")
    run_build_event_features(config_path, target_date, target_date, symbols=symbols)
    run_build_company_profiles(config_path, sync_start, target_date, symbols=symbols)


def _refresh_market_inputs_for_date(
    target_date: str,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    sync_start = (pd.Timestamp(target_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    config_path = str(CONFIG_PATH)
    config = get_app_config()
    full_market_source_order = list(config.providers.get("full_market_daily_bar_source_order", ["tx"]))
    full_market_sleep_seconds = float(config.providers.get("full_market_request_sleep_seconds", 0))
    full_market_max_workers = int(config.providers.get("full_market_daily_bar_max_workers", 4))
    total = int(get_active_security_count())
    base_daily_coverage = _active_daily_bar_count_for_date(target_date)

    def emit_extract_progress(current: int, total_count: int, symbol: str | None) -> None:
        if not progress:
            return
        display_total = int(total or total_count or 0)
        display_current = min(int(base_daily_coverage) + int(current), display_total) if display_total else int(current)
        progress(
            {
                "stage": "extract_data",
                "current": int(display_current),
                "total": display_total,
                "message": f"正在提取 {symbol} 数据" if symbol else "正在提取市场数据",
            }
        )

    run_sync_securities(config_path, sync_start, target_date, target_symbol=None)
    run_sync_daily_bars(
        config_path,
        sync_start,
        target_date,
        symbols=None,
        progress_callback=emit_extract_progress,
        source_order=full_market_source_order,
        request_sleep_seconds=full_market_sleep_seconds,
        max_workers_override=full_market_max_workers,
    )
    if progress:
        progress(
            {
                "stage": "extract_data",
                "current": int(total),
                "total": int(total),
                "message": "数据提取完成",
            }
        )
    run_sync_index_bars(config_path, sync_start, target_date)
    run_sync_sector_daily(config_path, sync_start, target_date, None)
    run_sync_financial_snapshot(config_path, sync_start, target_date, None, symbols=None)
    run_sync_news(config_path, target_date, target_date, None, symbols=None)
    run_build_event_features(config_path, target_date, target_date, symbols=None)
    run_build_company_profiles(config_path, sync_start, target_date, symbols=None)


def _refresh_market_inputs_for_symbols(
    target_date: str,
    symbols: list[str],
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    normalized_symbols = [normalize_symbol(symbol) for symbol in symbols if symbol]
    if not normalized_symbols:
        return
    sync_start = (pd.Timestamp(target_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    config_path = str(CONFIG_PATH)
    total = len(normalized_symbols)

    run_sync_index_bars(config_path, sync_start, target_date)
    run_sync_sector_daily(config_path, sync_start, target_date, None)
    run_sync_financial_snapshot(config_path, sync_start, target_date, None, symbols=normalized_symbols)
    run_sync_news(config_path, target_date, target_date, None, symbols=normalized_symbols)

    def emit_extract_progress(current: int, total_count: int, symbol: str | None) -> None:
        if not progress:
            return
        completed_before_sync = max(total - int(total_count), 0)
        display_current = min(completed_before_sync + int(current), total)
        message = f"正在提取 {symbol} 数据" if symbol else "正在提取市场数据"
        progress(
            {
                "stage": "extract_data",
                "current": int(display_current),
                "total": int(total),
                "message": message,
            }
        )

    run_sync_daily_bars(
        config_path,
        sync_start,
        target_date,
        symbols=normalized_symbols,
        progress_callback=emit_extract_progress,
    )
    if progress:
        progress(
            {
                "stage": "extract_data",
                "current": int(total),
                "total": int(total),
                "message": "数据提取完成",
            }
        )
    run_build_event_features(config_path, target_date, target_date, symbols=normalized_symbols)
    run_build_company_profiles(config_path, sync_start, target_date, symbols=normalized_symbols)


def _ensure_quick_scan_shared_inputs(target_date: str) -> None:
    session = get_session_factory()()
    try:
        target = date.fromisoformat(str(target_date))
        index_exists = bool(session.execute(select(IndexBar.id).where(IndexBar.trade_date == target).limit(1)).scalar_one_or_none())
        sector_exists = bool(
            session.execute(select(SectorDaily.id).where(SectorDaily.trade_date == target).limit(1)).scalar_one_or_none()
        )
        news_exists = bool(session.execute(select(NewsNorm.news_id).where(NewsNorm.trade_date == target).limit(1)).scalar_one_or_none())
        event_exists = bool(
            session.execute(select(EventFeaturesDaily.id).where(EventFeaturesDaily.trade_date == target).limit(1)).scalar_one_or_none()
        )
    finally:
        session.close()

    if index_exists and sector_exists and news_exists and event_exists:
        return

    sync_start = (pd.Timestamp(target_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    config_path = str(CONFIG_PATH)
    if not index_exists:
        run_sync_index_bars(config_path, sync_start, target_date)
    if not sector_exists:
        run_sync_sector_daily(config_path, sync_start, target_date, None)
    if not news_exists:
        run_sync_news(config_path, target_date, target_date, None, symbols=None)
    if not event_exists:
        run_build_event_features(config_path, target_date, target_date, symbols=None)


def _refresh_quick_scan_symbol_inputs(target_date: str, symbol: str) -> None:
    normalized_symbol = normalize_symbol(symbol)
    if not normalized_symbol:
        return
    sync_start = (pd.Timestamp(target_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    config_path = str(CONFIG_PATH)
    symbols = [normalized_symbol]
    run_sync_daily_bars(config_path, sync_start, target_date, symbols=symbols)
    run_sync_financial_snapshot(config_path, sync_start, target_date, None, symbols=symbols)
    run_sync_news(config_path, target_date, target_date, None, symbols=symbols)
    run_build_event_features(config_path, target_date, target_date, symbols=symbols)
    run_build_company_profiles(config_path, sync_start, target_date, symbols=symbols)


def _refresh_quick_scan_batch_symbol_inputs(
    target_date: str,
    symbols: list[str],
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    normalized_symbols = [normalize_symbol(symbol) for symbol in symbols if symbol]
    if not normalized_symbols:
        return
    sync_start = (pd.Timestamp(target_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    config_path = str(CONFIG_PATH)
    total = len(normalized_symbols)

    def emit_extract_progress(current: int, total_count: int, symbol: str | None) -> None:
        if not progress:
            return
        completed_before_sync = max(total - int(total_count), 0)
        display_current = min(completed_before_sync + int(current), total)
        progress(
            {
                "stage": "extract_data",
                "current": int(display_current),
                "total": int(total),
                "message": f"正在提取 {symbol} 数据" if symbol else "正在提取快速推荐候选数据",
            }
        )

    run_sync_daily_bars(
        config_path,
        sync_start,
        target_date,
        symbols=normalized_symbols,
        progress_callback=emit_extract_progress,
    )
    session = get_session_factory()()
    try:
        target = date.fromisoformat(str(target_date))
        event_exists = bool(
            session.execute(select(EventFeaturesDaily.id).where(EventFeaturesDaily.trade_date == target).limit(1)).scalar_one_or_none()
        )
    finally:
        session.close()
    if not event_exists:
        run_build_event_features(config_path, target_date, target_date, symbols=normalized_symbols)
    run_sync_financial_snapshot(config_path, sync_start, target_date, None, symbols=normalized_symbols)
    run_build_company_profiles(config_path, sync_start, target_date, symbols=normalized_symbols)


def _has_quick_scan_batch_daily_rows(target_date: str, symbols: list[str]) -> bool:
    normalized_symbols = [normalize_symbol(symbol) for symbol in symbols if symbol]
    if not normalized_symbols:
        return False
    session = get_session_factory()()
    try:
        target = date.fromisoformat(str(target_date))
        count = (
            session.execute(
                select(func.count(func.distinct(DailyBar.symbol))).where(
                    DailyBar.trade_date == target,
                    DailyBar.symbol.in_(normalized_symbols),
                )
            ).scalar()
            or 0
        )
        return int(count) > 0
    finally:
        session.close()
def _ensure_exact_trade_date_inputs(symbol: str, target_date: str) -> None:
    normalized_symbol = normalize_symbol(symbol)
    if _has_exact_trade_date_inputs(normalized_symbol, target_date):
        return
    if _is_known_non_trade_date(target_date):
        raise ValueError(f"{target_date} 不是交易日，无法获取数据")

    with FRESH_SAMPLE_LOCK:
        if _has_exact_trade_date_inputs(normalized_symbol, target_date):
            return
        sync_start = (pd.Timestamp(target_date) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        run_sync_securities(str(CONFIG_PATH), sync_start, target_date, target_symbol=normalized_symbol)
        if _is_known_non_trade_date(target_date):
            raise ValueError(f"{target_date} 不是交易日，无法获取数据")
        _refresh_single_stock_inputs_for_date(normalized_symbol, target_date)

    if not _has_exact_trade_date_inputs(normalized_symbol, target_date):
        raise ValueError(
            f"无法为 {target_date} 获取完整的输入数据，请检查数据同步任务"
        )
def _ensure_exact_market_trade_date_inputs(
    target_date: str,
    target_symbols: list[str] | None = None,
    *,
    min_context_rows: int = 1,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    if _is_known_non_trade_date(target_date):
        raise ValueError(f"{target_date} 不是交易日，无法构建市场样本")

    raw_df, context_df, effective_trade_date = build_prediction_frame(
        str(CONFIG_PATH),
        target_date=target_date,
        target_symbol=None,
        target_symbols=target_symbols,
        strict_target_date=True,
    )
    if len(raw_df) >= min_context_rows and len(context_df) >= min_context_rows and effective_trade_date == target_date:
        return

    with FRESH_SAMPLE_LOCK:
        raw_df, context_df, effective_trade_date = build_prediction_frame(
            str(CONFIG_PATH),
            target_date=target_date,
            target_symbol=None,
            target_symbols=target_symbols,
            strict_target_date=True,
        )
        if len(raw_df) >= min_context_rows and len(context_df) >= min_context_rows and effective_trade_date == target_date:
            return
        if target_symbols:
            _refresh_market_inputs_for_symbols(target_date, target_symbols, progress=progress)
        else:
            _refresh_market_inputs_for_date(target_date, progress=progress)

    raw_df, context_df, effective_trade_date = build_prediction_frame(
        str(CONFIG_PATH),
        target_date=target_date,
        target_symbol=None,
        target_symbols=target_symbols,
        strict_target_date=True,
    )
    if len(raw_df) < min_context_rows or len(context_df) < min_context_rows or effective_trade_date != target_date:
        raise ValueError(f"无法为 {target_date} 构建市场样本")
def build_single_stock_analysis(
    symbol: str,
    target_date: str | None,
    is_holding: bool,
    holding_days: int,
    risk_preference: str = "balanced",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    checkpoint_path = get_latest_checkpoint_path()
    config = get_app_config()
    analysis_date = _resolve_analysis_target_date(target_date)
    min_list_days = int(config.project.get("min_list_days", 120))
    security_snapshot = _get_security_snapshot(normalized_symbol)
    if security_snapshot is not None:
        if security_snapshot["status"] != "active":
            raise ValueError(f"{normalized_symbol} 当前不是可分析状态")
        list_date = security_snapshot.get("list_date")
        if list_date is not None:
            listed_days = (pd.Timestamp(analysis_date).date() - list_date).days
            if listed_days < min_list_days:
                raise ValueError(
                    f"{normalized_symbol} 上市时间不足 {min_list_days} 天，当前暂不支持个股分析"
                )
    if progress:
        progress("extract_data")
    _ensure_exact_trade_date_inputs(normalized_symbol, analysis_date)
    raw_df, context_df, effective_trade_date = build_prediction_frame(
        str(CONFIG_PATH),
        target_date=analysis_date,
        target_symbol=normalized_symbol,
        strict_target_date=True,
        include_st=True,
    )
    if effective_trade_date != analysis_date:
        raise ValueError(
            f"无法为 {analysis_date} 构建预测样本，实际获取到的是 {effective_trade_date}，请检查数据同步"
        )
    if raw_df.empty or context_df.empty or not effective_trade_date:
        raise ValueError("当前股票没有可用的预测数据")

    packaged_df = _build_prediction_export_frame(
        context_prediction_df=context_df,
        effective_trade_date=effective_trade_date,
        seq_length=int(config.project.get("seq_length", 20)),
    )
    if packaged_df.empty:
        raise ValueError("无法构建特征数据包")

    if progress:
        progress("model_predict")
    predicted_df, bundle = predict_packaged_dataframe(packaged_df, checkpoint_path)
    merged = _merge_prediction_frames(raw_df, predicted_df, packaged_df)
    if merged.empty:
        raise ValueError("预测结果合并失败")

    row = merged.iloc[0].to_dict()
    packaged_row = packaged_df.iloc[0].to_dict()
    vocab_limits = bundle.checkpoint.get("vocab_sizes", {}) if isinstance(bundle.checkpoint, dict) else {}
    model_input = PackagedDataFrameDataset(
        packaged_df.iloc[[0]].copy(),
        bundle.normalizer,
        vocab_limits=vocab_limits,
    )[0]
    coverage = load_training_coverage(
        symbols=[str(row.get("symbol") or "")],
        industries=[str(row.get("industry_sw") or "")],
    )
    decision_record = _augment_decision_record(
        row,
        model_descriptor=bundle.descriptor,
        market_symbol_count=get_active_security_count(),
        coverage=coverage,
    )
    if progress:
        progress("decision_engine")
    decision_result = evaluate_stock_decision(
        decision_record,
        is_holding=is_holding,
        holding_days=holding_days,
        risk_preference=risk_preference,
        metadata=_decision_metadata_from_bundle(bundle, row.get("feature_version")),
    )
    return {
        "analysis_date": analysis_date,
        "effective_trade_date": effective_trade_date,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "engine_version": ENGINE_VERSION,
        "model_info": {
            "source": bundle.source,
            "descriptor": bundle.descriptor,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "feature_version": str(row.get("feature_version") or "unknown_feature"),
        },
        "stock": {
            "symbol": row.get("symbol"),
            "name": row.get("name"),
            "industry_sw": row.get("industry_sw"),
            "board": row.get("board"),
        },
        "holding_context": {
            "is_holding": bool(is_holding),
            "holding_days": int(holding_days),
            "used_by_decision_model": True,
            "risk_preference": risk_preference,
        },
        "sample": _build_sample_payload(packaged_row, model_input),
        "company_info": {
            "list_days": _safe_int(pd.to_numeric(row.get("list_days"), errors="coerce")),
            "market_cap_log": _safe_float(row.get("market_cap_log")),
            "volatility_120": _safe_float(row.get("volatility_120")),
            "beta_120": _safe_float(row.get("beta_120")),
            "turnover_mean_120": _safe_float(row.get("turnover_mean_120")),
            "amount_mean_120": _safe_float(row.get("amount_mean_120")),
            "roe": _safe_float(row.get("roe")),
            "revenue_yoy": _safe_float(row.get("revenue_yoy")),
            "profit_yoy": _safe_float(row.get("profit_yoy")),
            "debt_ratio": _safe_float(row.get("debt_ratio")),
            "gross_margin": _safe_float(row.get("gross_margin")),
        },
        "feature_record": _json_ready(row),
        "prediction": {
            "signal_score": _safe_float(row.get("signal_score")),
            "rank_score_pred": _safe_float(row.get("rank_score_pred")),
            "p_win_prob_3": _safe_float(row.get("p_win_prob_3")),
            "p_win_prob_5": _safe_float(row.get("p_win_prob_5")),
            "p_win_prob_10": _safe_float(row.get("p_win_prob_10")),
            "p_win_prob_20": _safe_float(row.get("p_win_prob_20")),
            "p_win_prob_40": _safe_float(row.get("p_win_prob_40")),
            "ret_mu_pred_5": _safe_float(row.get("ret_mu_pred_5")),
            "risk_dd_pred_5": _safe_float(row.get("risk_dd_pred_5")),
            "ret_mu_pred_20": _safe_float(row.get("ret_mu_pred_20")),
            "risk_dd_pred_20": _safe_float(row.get("risk_dd_pred_20")),
            "market_regime_prob": _safe_float(row.get("market_regime_prob"), 0.5),
            "bigloss_prob_5": _safe_float(row.get("bigloss_prob_5")),
        },
        "market_snapshot": {
            "close": _safe_float(row.get("close")),
            "pct_chg": _safe_float(row.get("pct_chg")),
            "turnover_rate": _safe_float(row.get("turnover_rate")),
            "vol_ratio_5": _safe_float(row.get("vol_ratio_5")),
            "ret_20": _safe_float(row.get("ret_20")),
            "ret_5": _safe_float(row.get("ret_5")),
            "industry_rank_20d": _safe_float(row.get("industry_rank_20d")),
            "market_volatility_5": _safe_float(row.get("market_volatility_5")),
            "up_limit_count": _safe_float(row.get("up_limit_count")),
            "down_limit_count": _safe_float(row.get("down_limit_count")),
            "avg_amount_5": _safe_float(row.get("avg_amount_5")),
        },
        "decision_result": decision_result,
    }


def build_market_scan(
    top_n: int,
    target_date: str | None,
    risk_preference: str = "balanced",
    scan_mode: str = "market",
    holding_days: int = 10,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return build_market_scan_v2(
        top_n=top_n,
        target_date=target_date,
        risk_preference=risk_preference,
        scan_mode=scan_mode,
        holding_days=holding_days,
        progress=progress,
    )
def _simple_rank_candidates(
    records: list[dict[str, Any]],
    top_n: int,
    risk_preference: str = "balanced",
    metadata: dict[str, Any] | None = None,
    holding_days: int | None = None,
) -> dict[str, Any]:
    if not records:
        return {
            "effective_trade_date": None,
            "top_n": int(top_n),
            "pool_size": 0,
            "selected_count": 0,
            "market_regime_counts": {},
            "candidates": [],
        }

    horizons = tuple(int(item) for item in P_WIN_HORIZONS)
    selected_holding_days = _normalize_market_scan_holding_days(holding_days)
    buy_actions = {"STRONG_BUY", "BUY", "RECOMMEND_BUY"}

    def _horizon_label(days: int) -> str:
        return f"{int(days)}d"

    enriched: list[dict[str, Any]] = []
    for record in records:
        actual_decision_result = evaluate_stock_decision(
            record,
            is_holding=False,
            holding_days=0,
            risk_preference=risk_preference,
            metadata=metadata,
        )
        best_horizon = selected_holding_days or max(
            horizons,
            key=lambda horizon: (
                _safe_float(record.get(f"p_win_prob_{horizon}"), -1.0),
                _safe_float(record.get(f"ret_mu_pred_{horizon}")),
            ),
        )
        best_p_win = max(0.0, _safe_float(record.get(f"p_win_prob_{best_horizon}"), 0.0))
        signal_score = _safe_float(record.get("signal_score"))
        rank_score = _safe_float(record.get("rank_score_pred"))
        ret_mu = _safe_float(record.get(f"ret_mu_pred_{best_horizon}"))
        action = str(actual_decision_result.get("decision", {}).get("action") or "")
        action_priority = -int(actual_decision_result.get("decision", {}).get("priority") or 99)
        enriched.append(
            {
                "record": record,
                "actual_decision_result": actual_decision_result,
                "best_horizon": int(best_horizon),
                "best_p_win": float(best_p_win),
                "eligible": action in buy_actions and best_p_win >= 0.5,
                "ranking_score": (action_priority, float(best_p_win), ret_mu, signal_score, rank_score),
            }
        )

    ranked_pool = [item for item in enriched if item["eligible"]]
    ranked_pool.sort(key=lambda item: item["ranking_score"], reverse=True)
    selected = ranked_pool[: int(top_n)]

    candidates: list[dict[str, Any]] = []
    for rank_index, item in enumerate(selected, start=1):
        record = item["record"]
        best_horizon = int(item["best_horizon"])
        best_p_win = float(item["best_p_win"])
        model_output: dict[str, float] = {"rank_score": _safe_float(record.get("rank_score_pred"))}
        for horizon in horizons:
            model_output[f"p_win_{horizon}"] = _safe_float(record.get(f"p_win_prob_{horizon}"))
            model_output[f"ret_mu_{horizon}"] = _safe_float(record.get(f"ret_mu_pred_{horizon}"))
            model_output[f"risk_dd_{horizon}"] = _safe_float(record.get(f"risk_dd_pred_{horizon}"))
        for horizon in (5, 20):
            model_output[f"bigloss_{horizon}"] = _safe_float(record.get(f"bigloss_prob_{horizon}"))

        decision_result = {
            "symbol": str(record.get("symbol") or ""),
            "symbol_name": str(record.get("name") or ""),
            "trade_date": str(record.get("trade_date") or ""),
            "decision": {
                "action": "buy",
                "action_cn": "模型推荐",
                "confidence": best_p_win,
                "suggested_hold_days": best_horizon,
                "best_horizon": str(best_horizon),
                "path": "market_scan_model_rank",
                "priority": 1,
            },
            "scores": {
                "S_final": _safe_float(record.get("signal_score")),
                "S_3": _safe_float(record.get("p_win_prob_3")),
                "S_5": _safe_float(record.get("p_win_prob_5")),
                "S_10": _safe_float(record.get("p_win_prob_10")),
                "S_20": _safe_float(record.get("p_win_prob_20")),
                "S_40": _safe_float(record.get("p_win_prob_40")),
                "consistency": 0.0,
                "R_score": best_p_win,
            },
            "model_output": model_output,
            "market_regime": "neutral",
            "risk_flags": list(record.get("risk_flags", [])),
            "risk_review": {
                "passed": True,
                "original_action": "buy",
                "final_action": "buy",
                "downgraded": False,
                "risk_flags": list(record.get("risk_flags", [])),
                "risk_warnings": [],
                "blocked_rules": [],
            },
            "reasons": [
                f"预测未来{_horizon_label(best_horizon)}上涨概率 {best_p_win * 100:.1f}%",
                f"信号得分 {_safe_float(record.get('signal_score')):.3f}，排名得分 {_safe_float(record.get('rank_score_pred')):.3f}",
                f"预期收益 {_safe_float(record.get(f'ret_mu_pred_{best_horizon}')) * 100:.2f}%，预期回撤 {_safe_float(record.get(f'risk_dd_pred_{best_horizon}')) * 100:.2f}%",
            ],
            "metadata": _json_ready(metadata or {}),
        }
        decision_result = _json_ready(item["actual_decision_result"])
        decision_result["model_output"] = model_output
        decision_result.setdefault("scores", {})
        decision_result["scores"]["R_score"] = best_p_win
        decision_result.setdefault("decision", {})
        decision_result["decision"]["confidence"] = best_p_win
        decision_result["decision"]["suggested_hold_days"] = best_horizon
        decision_result["decision"]["best_horizon"] = str(best_horizon)
        reasons = list(decision_result.get("reasons", []))
        reasons.insert(0, f"Predicted {_horizon_label(best_horizon)} win probability: {best_p_win * 100:.1f}%.")
        decision_result["reasons"] = reasons
        candidates.append(
            {
                "rank": rank_index,
                "symbol": str(record.get("symbol") or ""),
                "name": str(record.get("name") or ""),
                "industry_sw": str(record.get("industry_sw") or ""),
                "board": str(record.get("board") or ""),
                "close": _safe_float(record.get("close")),
                "pct_chg": _safe_float(record.get("pct_chg")),
                "avg_amount_5": _safe_float(record.get("avg_amount_5")),
                "recommended_hold_days": best_horizon,
                "recommended_hold_label": _horizon_label(best_horizon),
                "predicted_win_rate": best_p_win,
                "raw_predicted_win_rate": best_p_win,
                "win_rate_source": "raw_model_output",
                "signal_score": _safe_float(record.get("signal_score")),
                "rank_score_pred": _safe_float(record.get("rank_score_pred")),
                "ret_mu_pred": _safe_float(record.get(f"ret_mu_pred_{best_horizon}")),
                "risk_dd_pred": _safe_float(record.get(f"risk_dd_pred_{best_horizon}")),
                "bigloss_prob": _safe_float(record.get(f"bigloss_prob_{best_horizon}")),
                "market_regime_prob": _safe_float(record.get("market_regime_prob"), 0.5),
                "feature_missing_rate": _safe_float(record.get("feature_missing_rate")),
                "company_info": {
                    "list_days": _safe_int(pd.to_numeric(record.get("list_days"), errors="coerce")),
                    "market_cap_log": _safe_float(record.get("market_cap_log")),
                    "volatility_120": _safe_float(record.get("volatility_120")),
                    "beta_120": _safe_float(record.get("beta_120")),
                    "turnover_mean_120": _safe_float(record.get("turnover_mean_120")),
                    "amount_mean_120": _safe_float(record.get("amount_mean_120")),
                    "roe": _safe_float(record.get("roe")),
                    "revenue_yoy": _safe_float(record.get("revenue_yoy")),
                    "profit_yoy": _safe_float(record.get("profit_yoy")),
                    "debt_ratio": _safe_float(record.get("debt_ratio")),
                    "gross_margin": _safe_float(record.get("gross_margin")),
                },
                "prediction": {
                    "signal_score": _safe_float(record.get("signal_score")),
                    "rank_score_pred": _safe_float(record.get("rank_score_pred")),
                    **{f"p_win_prob_{horizon}": _safe_float(record.get(f"p_win_prob_{horizon}")) for horizon in horizons},
                    **{f"ret_mu_pred_{horizon}": _safe_float(record.get(f"ret_mu_pred_{horizon}")) for horizon in horizons},
                    **{f"risk_dd_pred_{horizon}": _safe_float(record.get(f"risk_dd_pred_{horizon}")) for horizon in horizons},
                    "market_regime_prob": _safe_float(record.get("market_regime_prob"), 0.5),
                    "bigloss_prob_5": _safe_float(record.get("bigloss_prob_5")),
                    "bigloss_prob_20": _safe_float(record.get("bigloss_prob_20")),
                },
                "market_snapshot": {
                    "close": _safe_float(record.get("close")),
                    "pct_chg": _safe_float(record.get("pct_chg")),
                    "turnover_rate": _safe_float(record.get("turnover_rate")),
                    "vol_ratio_5": _safe_float(record.get("vol_ratio_5")),
                    "ret_20": _safe_float(record.get("ret_20")),
                    "ret_5": _safe_float(record.get("ret_5")),
                    "industry_rank_20d": _safe_float(record.get("industry_rank_20d")),
                    "market_volatility_5": _safe_float(record.get("market_volatility_5")),
                    "up_limit_count": _safe_float(record.get("up_limit_count")),
                    "down_limit_count": _safe_float(record.get("down_limit_count")),
                    "avg_amount_5": _safe_float(record.get("avg_amount_5")),
                },
                "decision_result": decision_result,
                "risk_flags": list(decision_result.get("risk_flags", [])),
                "reasons": decision_result["reasons"],
            }
        )

    return {
        "effective_trade_date": str(records[0].get("trade_date") or ""),
        "holding_days": selected_holding_days,
        "top_n": int(top_n),
        "pool_size": len(ranked_pool),
        "selected_count": len(candidates),
        "market_regime_counts": {},
        "candidates": candidates,
    }


def _build_market_scan_market_full(
    *,
    top_n: int,
    analysis_date: str,
    checkpoint_path: Path | None,
    config: AppConfig,
    risk_preference: str,
    holding_days: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    coverage_before_refresh = _active_daily_bar_count_for_date(analysis_date)
    market_total_candidates = int(get_active_security_count())
    coverage_floor = _full_market_daily_coverage_floor(market_total_candidates)
    if progress:
        progress(
            {
                "stage": "candidate_select",
                "current": int(coverage_before_refresh),
                "total": market_total_candidates,
                "message": (
                    f"正在补齐 {analysis_date} 的全市场数据，已有 {coverage_before_refresh}/{market_total_candidates}"
                    if coverage_before_refresh < coverage_floor
                    else "正在一次性提取全市场股票样本"
                ),
            }
        )
    daily_coverage = _ensure_full_market_daily_inputs(
        analysis_date,
        market_total_candidates=market_total_candidates,
        progress=progress,
    )
    raw_df, context_df, effective_trade_date = build_prediction_frame(
        str(CONFIG_PATH),
        target_date=analysis_date,
        target_symbol=None,
        target_symbols=None,
        strict_target_date=True,
    )
    if effective_trade_date != analysis_date:
        raise ValueError(f"昨天 {analysis_date} 的市场数据还没有准备完整，当前实际拿到的是 {effective_trade_date}")
    if raw_df.empty or context_df.empty:
        raise ValueError("当前数据库中没有可用于市场扫描的预测样本。")
    min_expected_samples = max(1, int(market_total_candidates * MARKET_SCAN_FULL_MIN_SAMPLE_RATIO))
    if len(raw_df) < min_expected_samples:
        if progress:
            progress(
                {
                    "stage": "candidate_select",
                    "current": int(len(raw_df)),
                    "total": int(market_total_candidates),
                    "message": (
                        f"有效样本仅 {len(raw_df)}/{market_total_candidates}，正在补齐静态、市场、公司和新闻事件数据"
                    ),
                }
            )
        with FRESH_SAMPLE_LOCK:
            _refresh_market_inputs_for_date(analysis_date, progress=progress)
        daily_coverage = _active_daily_bar_count_for_date(analysis_date)
        raw_df, context_df, effective_trade_date = build_prediction_frame(
            str(CONFIG_PATH),
            target_date=analysis_date,
            target_symbol=None,
            target_symbols=None,
            strict_target_date=True,
        )
        if effective_trade_date != analysis_date:
            raise ValueError(f"昨天 {analysis_date} 的市场数据还没有准备完整，当前实际拿到的是 {effective_trade_date}")
        if raw_df.empty or context_df.empty:
            raise ValueError("当前数据库中没有可用于市场扫描的预测样本。")
        if len(raw_df) < min_expected_samples:
            raise ValueError(
                f"{analysis_date} 全市场样本构建数量异常偏少，仅 {len(raw_df)}/{market_total_candidates}，"
                f"当日日线覆盖 {daily_coverage}/{market_total_candidates}，请检查样本构建所需的静态、市场、公司、新闻数据"
            )
    if progress:
        progress(
            {
                "stage": "candidate_select",
                "current": int(len(raw_df)),
                "total": market_total_candidates,
                "message": f"全市场样本提取完成，有效样本 {len(raw_df)}/{market_total_candidates}",
            }
        )
        progress(
            {
                "stage": "model_predict",
                "current": 0,
                "total": int(len(raw_df)),
                "message": "正在对全市场样本批量预测",
            }
        )

    packaged_df = _build_prediction_export_frame(
        context_prediction_df=context_df,
        effective_trade_date=effective_trade_date,
        seq_length=int(config.project.get("seq_length", 20)),
    )
    if packaged_df.empty:
        raise ValueError("当前交易日未能构建市场推荐样本。")
    if progress:
        progress(
            {
                "stage": "model_predict",
                "current": 0,
                "total": int(len(packaged_df)),
                "message": f"模型输入已构建，正在加载模型并预测 {len(packaged_df)} 支股票",
            }
        )
    predicted_df, bundle = predict_packaged_dataframe(
        packaged_df,
        checkpoint_path,
        progress=(
            (
                lambda current, total: progress(
                    {
                        "stage": "model_predict",
                        "current": int(current),
                        "total": int(total) if int(total) > 0 else int(len(packaged_df)),
                        "message": f"正在对全市场样本批量预测，进度 {current}/{total}",
                    }
                )
            )
            if progress
            else None
        ),
    )
    merged = _merge_prediction_frames(raw_df, predicted_df, packaged_df)
    coverage = load_training_coverage(
        symbols=[str(symbol) for symbol in merged["symbol"].astype(str).tolist()],
        industries=[str(industry) for industry in merged["industry_sw"].dropna().astype(str).unique().tolist()],
    )
    records = [
        _augment_decision_record(
            row,
            model_descriptor=bundle.descriptor,
            market_symbol_count=market_total_candidates,
            coverage=coverage,
        )
        for row in merged.to_dict(orient="records")
    ]
    ranking = _simple_rank_candidates(
        records,
        top_n=top_n,
        risk_preference=risk_preference,
        metadata=_decision_metadata_from_bundle(
            bundle,
            merged["feature_version"].iloc[0] if "feature_version" in merged.columns and not merged.empty else None,
        ),
        holding_days=holding_days,
    )
    return {
        "effective_trade_date": effective_trade_date,
        "holding_days": int(holding_days),
        "sample_size": int(len(raw_df)),
        "market_total_candidates": market_total_candidates,
        "total_candidates": int(len(merged)),
        "pool_size": int(ranking["pool_size"]),
        "selected_count": int(ranking["selected_count"]),
        "market_regime_counts": ranking["market_regime_counts"],
        "candidates": ranking["candidates"],
    }
def build_market_scan_v2_final(
    top_n: int,
    target_date: str | None,
    risk_preference: str = "balanced",
    scan_mode: str = "market",
    holding_days: int = 10,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    checkpoint_path = get_latest_checkpoint_path()
    config = get_app_config()
    analysis_date = _resolve_analysis_target_date(target_date)
    normalized_scan_mode = _normalize_scan_mode(scan_mode)
    selected_holding_days = _normalize_market_scan_holding_days(holding_days) or 10

    if normalized_scan_mode == "quick":
        ranking = _build_market_scan_quick_incremental(
            top_n=int(top_n),
            analysis_date=analysis_date,
            checkpoint_path=checkpoint_path,
            config=config,
            holding_days=selected_holding_days,
            progress=progress,
        )
    else:
        ranking = _build_market_scan_market_full(
            top_n=int(top_n),
            analysis_date=analysis_date,
            checkpoint_path=checkpoint_path,
            config=config,
            risk_preference=risk_preference,
            holding_days=selected_holding_days,
            progress=progress,
        )

    return {
        "analysis_date": analysis_date,
        "effective_trade_date": ranking["effective_trade_date"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "engine_version": ENGINE_VERSION,
        "scan_mode": normalized_scan_mode,
        "holding_days": selected_holding_days,
        "sample_size": int(ranking["sample_size"]),
        "market_total_candidates": int(ranking["market_total_candidates"]),
        "total_candidates": int(ranking["total_candidates"]),
        "top_n": int(top_n),
        "pool_size": int(ranking["pool_size"]),
        "selected_count": int(ranking["selected_count"]),
        "market_regime_counts": ranking["market_regime_counts"],
        "candidates": ranking["candidates"],
    }

def build_market_scan_v2(
    top_n: int,
    target_date: str | None,
    risk_preference: str = "balanced",
    scan_mode: str = "market",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    checkpoint_path = get_latest_checkpoint_path()
    config = get_app_config()
    analysis_date = _resolve_analysis_target_date(target_date)
    normalized_scan_mode = _normalize_scan_mode(scan_mode)

    if normalized_scan_mode == "quick":
        ranking = _build_market_scan_quick_incremental(
            top_n=int(top_n),
            analysis_date=analysis_date,
            checkpoint_path=checkpoint_path,
            config=config,
            progress=progress,
        )
    else:
        ranking = _build_market_scan_market_full(
            top_n=int(top_n),
            analysis_date=analysis_date,
            checkpoint_path=checkpoint_path,
            config=config,
            risk_preference=risk_preference,
            progress=progress,
        )

    return {
        "analysis_date": analysis_date,
        "effective_trade_date": ranking["effective_trade_date"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "engine_version": ENGINE_VERSION,
        "scan_mode": normalized_scan_mode,
        "sample_size": int(ranking["sample_size"]),
        "market_total_candidates": int(ranking["market_total_candidates"]),
        "total_candidates": int(ranking["total_candidates"]),
        "top_n": int(top_n),
        "pool_size": int(ranking["pool_size"]),
        "selected_count": int(ranking["selected_count"]),
        "market_regime_counts": ranking["market_regime_counts"],
        "candidates": ranking["candidates"],
    }


def build_market_scan_v2(
    top_n: int,
    target_date: str | None,
    risk_preference: str = "balanced",
    scan_mode: str = "market",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    checkpoint_path = get_latest_checkpoint_path()
    config = get_app_config()
    analysis_date = _resolve_analysis_target_date(target_date)
    normalized_scan_mode = _normalize_scan_mode(scan_mode)

    if normalized_scan_mode == "quick":
        ranking = _build_market_scan_quick_incremental(
            top_n=int(top_n),
            analysis_date=analysis_date,
            checkpoint_path=checkpoint_path,
            config=config,
            progress=progress,
        )
    else:
        ranking = _build_market_scan_market_full(
            top_n=int(top_n),
            analysis_date=analysis_date,
            checkpoint_path=checkpoint_path,
            config=config,
            risk_preference=risk_preference,
            progress=progress,
        )

    return {
        "analysis_date": analysis_date,
        "effective_trade_date": ranking["effective_trade_date"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "engine_version": ENGINE_VERSION,
        "scan_mode": normalized_scan_mode,
        "sample_size": int(ranking["sample_size"]),
        "market_total_candidates": int(ranking["market_total_candidates"]),
        "total_candidates": int(ranking["total_candidates"]),
        "top_n": int(top_n),
        "pool_size": int(ranking["pool_size"]),
        "selected_count": int(ranking["selected_count"]),
        "market_regime_counts": ranking["market_regime_counts"],
        "candidates": ranking["candidates"],
    }


def _rank_market_scan_quick_threshold(
    records: list[Mapping[str, Any]],
    *,
    top_n: int,
    holding_days: int,
) -> dict[str, Any]:
    selected_holding_days = _normalize_market_scan_holding_days(holding_days) or 10
    if not records:
        return {
            "effective_trade_date": None,
            "holding_days": selected_holding_days,
            "top_n": int(top_n),
            "pool_size": 0,
            "selected_count": 0,
            "market_regime_counts": {},
            "candidates": [],
        }

    selected_rows: list[dict[str, Any]] = []
    for record in records:
        best_horizon = int(record.get("quick_best_horizon") or 0)
        best_p_win = _safe_float(record.get("quick_best_raw_p_win"), -1.0)
        best_ret_mu = _safe_float(record.get("quick_best_ret_mu"))
        if best_horizon <= 0 or best_p_win < 0:
            best_horizon = selected_holding_days
            raw_p_win = _safe_float(record.get(f"p_win_prob_{best_horizon}"), 0.0)
            best_p_win = raw_p_win
            best_ret_mu = _safe_float(record.get(f"ret_mu_pred_{best_horizon}"))
        if best_p_win < MARKET_SCAN_QUICK_MIN_WIN_RATE or best_ret_mu <= MARKET_SCAN_QUICK_MIN_RET_MU:
            continue
        selected_rows.append(
            {
                "record": record,
                "best_horizon": int(best_horizon),
                "best_p_win": float(best_p_win),
                "best_ret_mu": float(best_ret_mu),
                "signal_score": _safe_float(record.get("signal_score")),
                "rank_score": _safe_float(record.get("rank_score_pred")),
            }
        )

    selected_rows.sort(
        key=lambda item: (
            item["best_p_win"],
            item["best_ret_mu"],
            item["signal_score"],
            item["rank_score"],
        ),
        reverse=True,
    )
    selected = selected_rows[: int(top_n)]

    candidates: list[dict[str, Any]] = []
    for rank_index, item in enumerate(selected, start=1):
        record = item["record"]
        best_horizon = int(item["best_horizon"])
        best_p_win = float(item["best_p_win"])
        decision_result = {
            "symbol": str(record.get("symbol") or ""),
            "symbol_name": str(record.get("name") or ""),
            "trade_date": str(record.get("trade_date") or ""),
            "decision": {
                "action": "buy",
                "action_cn": "快速推荐",
                "confidence": best_p_win,
                "suggested_hold_days": best_horizon,
                "best_horizon": str(best_horizon),
                "path": "market_scan_quick_threshold",
                "priority": 1,
            },
            "scores": {
                "S_final": _safe_float(record.get("signal_score")),
                "S_3": _safe_float(record.get("p_win_prob_3")),
                "S_5": _safe_float(record.get("p_win_prob_5")),
                "S_10": _safe_float(record.get("p_win_prob_10")),
                "S_20": _safe_float(record.get("p_win_prob_20")),
                "S_40": _safe_float(record.get("p_win_prob_40")),
                "consistency": 0.0,
                "R_score": best_p_win,
            },
            "market_regime": "neutral",
            "risk_flags": list(record.get("risk_flags", [])),
            "risk_review": {
                "passed": True,
                "original_action": "buy",
                "final_action": "buy",
                "downgraded": False,
                "risk_flags": list(record.get("risk_flags", [])),
                "risk_warnings": [],
                "blocked_rules": [],
            },
                "reasons": [
                f"{best_horizon}日模型原始胜率最高，为 {best_p_win * 100:.1f}%",
                f"预测收益 {item['best_ret_mu'] * 100:.2f}%",
            ],
        }
        candidates.append(
            {
                "rank": rank_index,
                "symbol": str(record.get("symbol") or ""),
                "name": str(record.get("name") or ""),
                "industry_sw": str(record.get("industry_sw") or ""),
                "board": str(record.get("board") or ""),
                "close": _safe_float(record.get("close")),
                "pct_chg": _safe_float(record.get("pct_chg")),
                "avg_amount_5": _safe_float(record.get("avg_amount_5")),
                "recommended_hold_days": best_horizon,
                "recommended_hold_label": f"{best_horizon}d",
                "predicted_win_rate": best_p_win,
                "raw_predicted_win_rate": best_p_win,
                "win_rate_source": "raw_model_output",
                "signal_score": _safe_float(record.get("signal_score")),
                "rank_score_pred": _safe_float(record.get("rank_score_pred")),
                "ret_mu_pred": item["best_ret_mu"],
                "risk_dd_pred": _safe_float(record.get(f"risk_dd_pred_{best_horizon}")),
                "bigloss_prob": _safe_float(record.get(f"bigloss_prob_{best_horizon}")),
                "market_regime_prob": _safe_float(record.get("market_regime_prob"), 0.5),
                "feature_missing_rate": _safe_float(record.get("feature_missing_rate")),
                "decision_result": decision_result,
                "risk_flags": list(record.get("risk_flags", [])),
                "reasons": decision_result["reasons"],
                "prediction": {
                    "signal_score": _safe_float(record.get("signal_score")),
                    "rank_score_pred": _safe_float(record.get("rank_score_pred")),
                    "p_win_prob_3": _safe_float(record.get("p_win_prob_3")),
                    "p_win_prob_5": _safe_float(record.get("p_win_prob_5")),
                    "p_win_prob_10": _safe_float(record.get("p_win_prob_10")),
                    "p_win_prob_20": _safe_float(record.get("p_win_prob_20")),
                    "p_win_prob_40": _safe_float(record.get("p_win_prob_40")),
                    "ret_mu_pred_3": _safe_float(record.get("ret_mu_pred_3")),
                    "ret_mu_pred_5": _safe_float(record.get("ret_mu_pred_5")),
                    "ret_mu_pred_10": _safe_float(record.get("ret_mu_pred_10")),
                    "ret_mu_pred_20": _safe_float(record.get("ret_mu_pred_20")),
                    "ret_mu_pred_40": _safe_float(record.get("ret_mu_pred_40")),
                    "risk_dd_pred_3": _safe_float(record.get("risk_dd_pred_3")),
                    "risk_dd_pred_5": _safe_float(record.get("risk_dd_pred_5")),
                    "risk_dd_pred_10": _safe_float(record.get("risk_dd_pred_10")),
                    "risk_dd_pred_20": _safe_float(record.get("risk_dd_pred_20")),
                    "risk_dd_pred_40": _safe_float(record.get("risk_dd_pred_40")),
                    "market_regime_prob": _safe_float(record.get("market_regime_prob"), 0.5),
                },
                "market_snapshot": {
                    "close": _safe_float(record.get("close")),
                    "pct_chg": _safe_float(record.get("pct_chg")),
                    "turnover_rate": _safe_float(record.get("turnover_rate")),
                    "vol_ratio_5": _safe_float(record.get("vol_ratio_5")),
                    "ret_20": _safe_float(record.get("ret_20")),
                    "ret_5": _safe_float(record.get("ret_5")),
                    "industry_rank_20d": _safe_float(record.get("industry_rank_20d")),
                    "market_volatility_5": _safe_float(record.get("market_volatility_5")),
                    "up_limit_count": _safe_float(record.get("up_limit_count")),
                    "down_limit_count": _safe_float(record.get("down_limit_count")),
                    "avg_amount_5": _safe_float(record.get("avg_amount_5")),
                },
            }
        )

    return {
        "effective_trade_date": str(records[0].get("trade_date") or ""),
        "holding_days": selected_holding_days,
        "top_n": int(top_n),
        "pool_size": len(selected_rows),
        "selected_count": len(candidates),
        "market_regime_counts": {"neutral": len(selected_rows)},
        "candidates": candidates,
    }


def build_market_scan_v2(
    top_n: int,
    target_date: str | None,
    risk_preference: str = "balanced",
    scan_mode: str = "market",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    checkpoint_path = get_latest_checkpoint_path()
    config = get_app_config()
    analysis_date = _resolve_analysis_target_date(target_date)
    normalized_scan_mode = _normalize_scan_mode(scan_mode)

    if normalized_scan_mode == "quick":
        ranking = _build_market_scan_quick_incremental(
            top_n=int(top_n),
            analysis_date=analysis_date,
            checkpoint_path=checkpoint_path,
            config=config,
            progress=progress,
        )
    else:
        ranking = _build_market_scan_market_full(
            top_n=int(top_n),
            analysis_date=analysis_date,
            checkpoint_path=checkpoint_path,
            config=config,
            risk_preference=risk_preference,
            progress=progress,
        )

    return {
        "analysis_date": analysis_date,
        "effective_trade_date": ranking["effective_trade_date"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "engine_version": ENGINE_VERSION,
        "scan_mode": normalized_scan_mode,
        "sample_size": int(ranking["sample_size"]),
        "market_total_candidates": int(ranking["market_total_candidates"]),
        "total_candidates": int(ranking["total_candidates"]),
        "top_n": int(top_n),
        "pool_size": int(ranking["pool_size"]),
        "selected_count": int(ranking["selected_count"]),
        "market_regime_counts": ranking["market_regime_counts"],
        "candidates": ranking["candidates"],
    }


def _build_market_scan_quick_incremental(
    *,
    top_n: int,
    analysis_date: str,
    checkpoint_path: Path | None,
    config: AppConfig,
    holding_days: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    selected_holding_days = _normalize_market_scan_holding_days(holding_days) or 10
    active_symbols = _list_active_market_symbols()
    if not active_symbols:
        raise ValueError("quick market scan has no active symbols to scan")
    rng = np.random.default_rng(seed=int(pd.Timestamp(analysis_date).strftime("%Y%m%d")))
    active_symbols = [str(symbol) for symbol in rng.permutation(np.asarray(active_symbols, dtype=object)).tolist()]

    total_symbols = int(len(active_symbols))
    target_count = max(1, int(top_n))
    effective_trade_date: str | None = analysis_date
    qualified_records: list[dict[str, Any]] = []
    scanned_records = 0
    _ensure_quick_scan_shared_inputs(analysis_date)

    if progress:
        progress(
            {
                "stage": "candidate_select",
                "current": 0,
                "total": target_count,
                "message": f"从 {total_symbols} 支股票中随机扫描 {analysis_date} 数据",
            }
        )

    for start in range(0, total_symbols, MARKET_SCAN_QUICK_BATCH_SIZE):
        batch_symbols = active_symbols[start : start + MARKET_SCAN_QUICK_BATCH_SIZE]
        if not _has_quick_scan_batch_daily_rows(analysis_date, batch_symbols):
            _refresh_quick_scan_batch_symbol_inputs(analysis_date, batch_symbols, progress=progress)
        raw_df, context_df, batch_trade_date = build_prediction_frame(
            str(CONFIG_PATH),
            target_date=analysis_date,
            target_symbol=None,
            target_symbols=batch_symbols,
            strict_target_date=True,
        )
        if raw_df.empty or context_df.empty or not batch_trade_date:
            _refresh_quick_scan_batch_symbol_inputs(analysis_date, batch_symbols, progress=progress)
            raw_df, context_df, batch_trade_date = build_prediction_frame(
                str(CONFIG_PATH),
                target_date=analysis_date,
                target_symbol=None,
                target_symbols=batch_symbols,
                strict_target_date=True,
            )
        if raw_df.empty or context_df.empty or not batch_trade_date:
            if progress and batch_symbols:
                progress(
                    {
                        "stage": "candidate_select",
                        "current": min(len(qualified_records), target_count),
                        "total": target_count,
                        "message": f"已扫描到 {batch_symbols[-1]}，当前命中 {len(qualified_records)}/{target_count}",
                    }
                )
            continue
        if batch_trade_date != analysis_date:
            continue
        effective_trade_date = batch_trade_date
        packaged_df = _build_prediction_export_frame(
            context_prediction_df=context_df,
            effective_trade_date=batch_trade_date,
            seq_length=int(config.project.get("seq_length", 20)),
        )
        if packaged_df.empty:
            continue
        predicted_df, bundle = predict_packaged_dataframe(packaged_df, checkpoint_path)
        merged = _merge_prediction_frames(raw_df, predicted_df, packaged_df)
        if merged.empty:
            continue
        scanned_records += int(len(merged))
        coverage = load_training_coverage(
            symbols=[str(item) for item in merged["symbol"].astype(str).tolist()],
            industries=[str(industry) for industry in merged["industry_sw"].dropna().astype(str).unique().tolist()],
        )
        batch_records = [
            _augment_decision_record(
                row,
                model_descriptor=bundle.descriptor,
                market_symbol_count=total_symbols,
                coverage=coverage,
            )
            for row in merged.to_dict(orient="records")
        ]
        for record in batch_records:
            best_horizon = selected_holding_days
            best_raw_p_win = _safe_float(record.get(f"p_win_prob_{best_horizon}"), 0.0)
            best_ret_mu = _safe_float(record.get(f"ret_mu_pred_{best_horizon}"))
            record["quick_best_horizon"] = best_horizon
            record["quick_best_raw_p_win"] = best_raw_p_win
            record["quick_best_ret_mu"] = best_ret_mu
            if best_raw_p_win >= MARKET_SCAN_QUICK_MIN_WIN_RATE and best_ret_mu > MARKET_SCAN_QUICK_MIN_RET_MU:
                qualified_records.append(record)
                if progress:
                    progress(
                        {
                            "stage": "candidate_select",
                            "current": min(len(qualified_records), target_count),
                            "total": target_count,
                            "message": (
                                f"已命中 {len(qualified_records)}/{target_count}，"
                                f"最新命中 {record.get('symbol')} "
                                f"({best_raw_p_win * 100:.1f}%, 收益 {best_ret_mu * 100:.2f}%)"
                            ),
                        }
                    )
                if len(qualified_records) >= target_count:
                    break
            elif progress:
                progress(
                    {
                        "stage": "candidate_select",
                        "current": min(len(qualified_records), target_count),
                        "total": target_count,
                        "message": f"已扫描 {record.get('symbol')}，当前命中 {len(qualified_records)}/{target_count}",
                    }
                )
        if len(qualified_records) >= target_count:
            break

    if not qualified_records:
        return {
            "effective_trade_date": effective_trade_date or analysis_date,
            "holding_days": selected_holding_days,
            "sample_size": total_symbols,
            "market_total_candidates": total_symbols,
            "total_candidates": scanned_records,
            "pool_size": 0,
            "selected_count": 0,
            "market_regime_counts": {},
            "candidates": [],
        }

    ranking = _rank_market_scan_quick_threshold(
        qualified_records,
        top_n=min(target_count, len(qualified_records)),
        holding_days=selected_holding_days,
    )
    return {
        "effective_trade_date": effective_trade_date or ranking.get("effective_trade_date") or analysis_date,
        "holding_days": selected_holding_days,
        "sample_size": total_symbols,
        "market_total_candidates": total_symbols,
        "total_candidates": scanned_records,
        "pool_size": int(len(qualified_records)),
        "selected_count": int(ranking["selected_count"]),
        "market_regime_counts": ranking["market_regime_counts"],
        "candidates": ranking["candidates"],
    }


def find_user_by_account(session: Session, account: str) -> UserAccount | None:
    value = account.strip()
    stmt = select(UserAccount).where(or_(UserAccount.username == value, UserAccount.email == value))
    return session.execute(stmt).scalar_one_or_none()


def build_market_scan_v2(
    top_n: int,
    target_date: str | None,
    risk_preference: str = "balanced",
    scan_mode: str = "market",
    holding_days: int = 10,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return build_market_scan_v2_final(
        top_n=top_n,
        target_date=target_date,
        risk_preference=risk_preference,
        scan_mode=scan_mode,
        holding_days=holding_days,
        progress=progress,
    )
    if progress:
        progress(
            {
                "stage": "candidate_select",
                "current": 0,
                "message": "正在准备推荐股票样本",
            }
        )
    _ensure_exact_market_trade_date_inputs(
        analysis_date,
        target_symbols=selected_symbols,
        min_context_rows=(min(int(top_n), len(selected_symbols)) if selected_symbols else 1),
        progress=progress,
    )
    raw_df, context_df, effective_trade_date = build_prediction_frame(
        str(CONFIG_PATH),
        target_date=analysis_date,
        target_symbol=None,
        target_symbols=selected_symbols,
        strict_target_date=True,
    )
    if effective_trade_date != analysis_date:
        raise ValueError(f"未能构建 {analysis_date} 的当日全市场样本，系统当前拿到的是 {effective_trade_date}。")
    if raw_df.empty or context_df.empty:
        raise ValueError("当前数据库中没有可用于市场扫描的预测样本。")

    market_total_candidates = int(get_active_security_count())
    available_candidate_count = int(len(raw_df))
    quick_packaged_df = pd.DataFrame()
    if normalized_scan_mode == "quick":
        quick_packaged_df = _build_prediction_export_frame(
            context_prediction_df=context_df,
            effective_trade_date=effective_trade_date,
            seq_length=int(config.project.get("seq_length", 20)),
        )
        available_candidate_count = int(len(quick_packaged_df))
        if available_candidate_count < int(top_n):
            raise ValueError(
                f"快速推荐需要 {int(top_n)} 支有效候选股票，"
                f"但 {analysis_date} 当前只有 {available_candidate_count} 支。"
            )
    candidate_count_for_prediction = int(len(raw_df))
    if progress:
        progress(
            {
                "stage": "candidate_select",
                "current": int(planned_total),
                "message": f"股票样本准备完成，有效样本 {candidate_count_for_prediction}/{planned_total}",
            }
        )

    merged = pd.DataFrame()
    if checkpoint_path is not None and normalized_scan_mode == "market":
        cache_path = _build_cache_path("market_scan", checkpoint_path, effective_trade_date)
        required_columns = {"signal_score", "avg_amount_5", "market_regime_prob", "neighbor_symbol_ids", "bigloss_prob_5"}
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            merged = cached if required_columns.issubset(set(cached.columns)) else pd.DataFrame()

    if progress:
        progress(
            {
                "stage": "model_predict",
                "current": 0,
                "message": "正在批量推理股票预测结果",
            }
        )
    if merged.empty:
        packaged_df = _build_prediction_export_frame(
            context_prediction_df=context_df,
            effective_trade_date=effective_trade_date,
            seq_length=int(config.project.get("seq_length", 20)),
        )
        if packaged_df.empty:
            raise ValueError("当前交易日未能构建市场推荐样本")
        predicted_df, bundle = predict_packaged_dataframe(
            packaged_df,
            checkpoint_path,
            progress=(
                (
                    lambda current, total: progress(
                        {
                            "stage": "model_predict",
                            "current": int(
                                planned_total
                                if int(total) <= 0
                                else min(
                                    planned_total,
                                    max(0, int(round((float(current) / max(float(total), 1.0)) * planned_total))),
                                )
                            ),
                            "message": f"正在批量推理股票预测结果，有效样本 {current}/{planned_total}",
                        }
                    )
                )
                if progress
                else None
            ),
        )
        merged = _merge_prediction_frames(raw_df, predicted_df, packaged_df)
        if checkpoint_path is not None and normalized_scan_mode == "market":
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(cache_path, index=False)
    else:
        bundle = load_prediction_bundle(str(checkpoint_path.resolve()))
        if selected_symbols:
            merged = _filter_prediction_frame_by_symbols(merged, selected_symbols)
        if progress:
            progress(
                {
                    "stage": "model_predict",
                    "current": int(planned_total),
                    "total": int(planned_total),
                    "message": f"已复用缓存预测结果，有效样本 {len(merged)}/{planned_total}",
                }
            )

    coverage = load_training_coverage(
        symbols=[str(symbol) for symbol in merged["symbol"].astype(str).tolist()],
        industries=[str(industry) for industry in merged["industry_sw"].dropna().astype(str).unique().tolist()],
    )
    market_symbol_count = get_active_security_count()
    records = [
        _augment_decision_record(
            row,
            model_descriptor=bundle.descriptor,
            market_symbol_count=market_symbol_count,
            coverage=coverage,
        )
        for row in merged.to_dict(orient="records")
    ]
    if normalized_scan_mode == "quick":
        if len(records) < int(top_n):
            raise ValueError(
                f"快速推荐必须返回 {int(top_n)} 支股票，"
                f"但模型输出只有 {len(records)} 条有效结果。"
            )
        ranking = rank_market_candidates_quick(records, top_n=top_n)
    else:
        ranking = _simple_rank_candidates(
            records,
            top_n=top_n,
            risk_preference=risk_preference,
            metadata=_decision_metadata_from_bundle(
                bundle,
                merged["feature_version"].iloc[0] if "feature_version" in merged.columns and not merged.empty else None,
            ),
        )
    
    return {
        "analysis_date": analysis_date,
        "effective_trade_date": effective_trade_date,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "engine_version": ENGINE_VERSION,
        "scan_mode": normalized_scan_mode,
        "sample_size": int(planned_sample_size),
        "market_total_candidates": int(market_total_candidates),
        "total_candidates": int(len(merged)),
        "top_n": int(top_n),
        "pool_size": int(ranking["pool_size"]),
        "selected_count": int(ranking["selected_count"]),
        "market_regime_counts": ranking["market_regime_counts"],
        "candidates": ranking["candidates"],
    }

