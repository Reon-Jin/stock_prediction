from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pytorch_dataset import (
    DEFAULT_COMPANY_ID_COLUMNS,
    DEFAULT_COMPANY_PROFILE_COLUMNS,
    DEFAULT_EVENT_COLUMNS,
    DEFAULT_LABEL_GROUPS,
    DEFAULT_MKT_COLUMNS,
    DEFAULT_SEQ_COLUMNS,
    DEFAULT_TAB_COLUMNS,
    EVENT_EMBEDDING_COLUMN,
    DATA_AUGMENTATION_SOURCE_COLUMNS,
    augment_aligned_features,
)

CACHE_VERSION = 7
EPS = 1e-6
BIGLOSS_DRAWDOWN_THRESHOLD = -0.05
DERIVED_FEATURE_SOURCE_COLUMNS = [
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover_rate",
    "ret_5",
    "ret_20",
    "volatility_5",
    "volatility_20",
    "up_limit_count",
    "down_limit_count",
    "sector_hotness_top1",
    "sector_hotness_top3_mean",
    "risk_on_flag",
    "risk_off_flag",
]

EVENT_SIDECAR_SUFFIX = "_events.parquet"


def _source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_s": int(stat.st_mtime),  # second-level granularity avoids cache invalidation on file touch
    }


def _schema_columns(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def _default_event_source_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}{EVENT_SIDECAR_SUFFIX}")


def _merged_validation_source_path(valid_path: Path, test_path: Path, cache_dir: Path) -> Path:
    signatures = [_source_signature(valid_path), _source_signature(test_path)]
    digest = hashlib.sha1(json.dumps(signatures, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"valid_merged_{digest}.parquet"


def materialize_merged_validation_split(valid_path: Path, test_path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    merged_path = _merged_validation_source_path(valid_path, test_path, cache_dir)
    merged_event_path = _default_event_source_path(merged_path)
    if merged_path.exists():
        return merged_path

    frames = [pd.read_parquet(path) for path in (valid_path, test_path)]
    merged = pd.concat(frames, ignore_index=True)
    if "trade_date" in merged.columns:
        merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
        sort_columns = [column for column in ("symbol", "trade_date") if column in merged.columns]
        if sort_columns:
            merged = merged.sort_values(sort_columns).reset_index(drop=True)
    merged.to_parquet(merged_path, index=False)

    event_frames: list[pd.DataFrame] = []
    for path in (valid_path, test_path):
        event_path = _default_event_source_path(path)
        if event_path.exists():
            event_frames.append(pd.read_parquet(event_path))
    if event_frames:
        merged_events = pd.concat(event_frames, ignore_index=True)
        if "trade_date" in merged_events.columns:
            merged_events["trade_date"] = pd.to_datetime(merged_events["trade_date"], errors="coerce")
            merged_events = (
                merged_events.dropna(subset=["trade_date"])
                .drop_duplicates(subset=["trade_date"], keep="last")
                .sort_values("trade_date")
                .reset_index(drop=True)
            )
        merged_events.to_parquet(merged_event_path, index=False)

    return merged_path


def _resolve_read_columns(schema_columns: set[str]) -> list[str]:
    columns: list[str] = []

    def extend(existing: list[str]) -> None:
        for column in existing:
            if column in schema_columns and column not in columns:
                columns.append(column)

    extend(DERIVED_FEATURE_SOURCE_COLUMNS)
    extend(DEFAULT_SEQ_COLUMNS)
    extend(DEFAULT_TAB_COLUMNS)
    extend(DEFAULT_MKT_COLUMNS)
    extend(DEFAULT_COMPANY_ID_COLUMNS)
    extend(DEFAULT_COMPANY_PROFILE_COLUMNS)
    for label_columns in DEFAULT_LABEL_GROUPS.values():
        extend(label_columns)
    if EVENT_EMBEDDING_COLUMN in schema_columns:
        extend([EVENT_EMBEDDING_COLUMN])
    else:
        extend(list(DEFAULT_EVENT_COLUMNS))
    extend(["neighbor_symbol_ids", "neighbor_scores"])
    return columns


def _format_seconds(duration: float) -> str:
    if duration < 1:
        return f"{duration * 1000:.0f}ms"
    return f"{duration:.1f}s"


def _safe_float_matrix(frame: pd.DataFrame, columns: list[str], rows: int) -> torch.Tensor:
    if not columns:
        return torch.empty((rows, 0), dtype=torch.float32)
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    values = numeric.to_numpy(dtype=np.float32)
    values[~np.isfinite(values)] = 0.0
    return torch.tensor(values, dtype=torch.float32)


def _safe_long_matrix(frame: pd.DataFrame, columns: list[str], rows: int) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    for column in columns:
        if column not in frame.columns:
            outputs[column] = torch.zeros(rows, dtype=torch.long)
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(np.int64)
        outputs[column] = torch.tensor(values.to_numpy(dtype=np.int64), dtype=torch.long)
    return outputs


def _build_profile_scaler_stats(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(dtype=float)
        clean = series.dropna()
        median = float(clean.median()) if not clean.empty else 0.0
        q1 = float(clean.quantile(0.25)) if not clean.empty else 0.0
        q3 = float(clean.quantile(0.75)) if not clean.empty else 1.0
        iqr = q3 - q1
        stats[column] = {
            "median": median,
            "iqr": iqr if np.isfinite(iqr) and iqr != 0 else 1.0,
        }
    return stats


def _scale_company_profile_frame(
    frame: pd.DataFrame,
    columns: list[str],
    stats: dict[str, dict[str, float]],
) -> torch.Tensor:
    if not columns:
        return torch.empty((len(frame), 0), dtype=torch.float32)
    values: list[np.ndarray] = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce") if column in frame.columns else pd.Series(np.nan, index=frame.index)
        column_stats = stats.get(column, {"median": 0.0, "iqr": 1.0})
        filled = series.fillna(column_stats["median"]).astype(np.float32)
        scaled = (filled.to_numpy(dtype=np.float32) - float(column_stats["median"])) / float(column_stats["iqr"])
        scaled[~np.isfinite(scaled)] = 0.0
        values.append(scaled)
    matrix = np.stack(values, axis=1)
    return torch.tensor(matrix, dtype=torch.float32)


def _parse_json_array(raw: Any) -> list[Any]:
    if isinstance(raw, np.ndarray):
        return raw.tolist()
    if isinstance(raw, list):
        return raw
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_fixed_float_array(raw: Any, length: int) -> np.ndarray:
    output = np.zeros(length, dtype=np.float32)
    values = _parse_json_array(raw)
    if not values:
        return output
    clipped = np.asarray(values[:length], dtype=np.float32)
    output[: len(clipped)] = clipped
    output[~np.isfinite(output)] = 0.0
    return output


def _parse_fixed_int_array(raw: Any, length: int) -> np.ndarray:
    output = np.zeros(length, dtype=np.int64)
    values = _parse_json_array(raw)
    if not values:
        return output
    for idx, value in enumerate(values[:length]):
        numeric = pd.to_numeric(value, errors="coerce")
        output[idx] = 0 if pd.isna(numeric) else int(numeric)
    return output


def _build_event_matrix(date_frame: pd.DataFrame, event_columns: list[str]) -> torch.Tensor:
    rows = len(date_frame)
    if not rows:
        return torch.empty((0, len(event_columns)), dtype=torch.float32)
    if all(column in date_frame.columns for column in event_columns):
        return _safe_float_matrix(date_frame, event_columns, rows)

    vectors: list[np.ndarray] = []
    iterator = tqdm(
        date_frame.get(EVENT_EMBEDDING_COLUMN, pd.Series([None] * rows)).tolist(),
        desc="Parsing daily news embeddings",
        leave=False,
)
    for raw in iterator:
        vectors.append(_parse_fixed_float_array(raw, len(event_columns)))
    return torch.tensor(np.stack(vectors, axis=0), dtype=torch.float32)


def _load_event_by_date(
    frame: pd.DataFrame,
    event_source_path: Path | None,
    event_columns: list[str],
) -> tuple[torch.Tensor, torch.Tensor, str | None, dict[str, Any] | None]:
    unique_dates = pd.Index(frame["trade_date"].drop_duplicates().sort_values())
    if event_source_path is not None and event_source_path.exists():
        event_schema = _schema_columns(event_source_path)
        read_columns = ["trade_date"]
        if EVENT_EMBEDDING_COLUMN in event_schema:
            read_columns.append(EVENT_EMBEDDING_COLUMN)
        else:
            read_columns.extend([column for column in DEFAULT_EVENT_COLUMNS if column in event_schema])
        event_frame = pd.read_parquet(event_source_path, columns=read_columns)
        event_frame["trade_date"] = pd.to_datetime(event_frame["trade_date"], errors="coerce").dt.normalize()
        event_frame = event_frame.dropna(subset=["trade_date"]).sort_values("trade_date")
        duplicate_dates = event_frame["trade_date"].duplicated().sum()
        if duplicate_dates:
            raise ValueError(
                f"event sidecar has duplicate trade_date rows: path={event_source_path} duplicates={int(duplicate_dates)}"
            )
        available_dates = set(event_frame["trade_date"].dt.strftime("%Y-%m-%d"))
        required_dates = set(unique_dates.strftime("%Y-%m-%d"))
        missing_dates = sorted(required_dates - available_dates)
        if missing_dates:
            raise ValueError(
                f"event sidecar missing {len(missing_dates)} trade_date rows, examples={missing_dates[:5]}"
            )
        event_frame = event_frame.set_index("trade_date").loc[unique_dates].reset_index()
        event_by_date = _build_event_matrix(event_frame, event_columns)
        event_source_signature = _source_signature(event_source_path)
        resolved_event_path = str(event_source_path)
    else:
        date_frame = frame.groupby("trade_date", sort=True).first().reset_index()
        event_by_date = _build_event_matrix(date_frame, event_columns)
        event_source_signature = None
        resolved_event_path = None
    row_to_date_idx = torch.tensor(unique_dates.get_indexer(frame["trade_date"]), dtype=torch.long)
    return event_by_date, row_to_date_idx, resolved_event_path, event_source_signature


def _build_neighbor_tensors(frame: pd.DataFrame, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    total_rows = len(frame)
    ids = np.zeros((total_rows, topk), dtype=np.int64)
    scores = np.zeros((total_rows, topk), dtype=np.float32)

    if "neighbor_symbol_ids" not in frame.columns or "neighbor_scores" not in frame.columns:
        return torch.tensor(ids, dtype=torch.long), torch.tensor(scores, dtype=torch.float32)

    id_values = frame["neighbor_symbol_ids"].tolist()
    score_values = frame["neighbor_scores"].tolist()
    iterator = tqdm(range(total_rows), desc="Parsing neighbor lists", leave=False)
    for idx in iterator:
        ids[idx] = _parse_fixed_int_array(id_values[idx], topk)
        scores[idx] = _parse_fixed_float_array(score_values[idx], topk)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(scores, dtype=torch.float32)


def _ensure_derived_risk_labels(frame: pd.DataFrame) -> pd.DataFrame:
    augmented = frame.copy()
    derived_pairs = (
        ("label_bigloss_5", "label_maxdd_5"),
        ("label_bigloss_20", "label_maxdd_20"),
    )
    for target_column, source_column in derived_pairs:
        if target_column in augmented.columns or source_column not in augmented.columns:
            continue
        drawdown = pd.to_numeric(augmented[source_column], errors="coerce")
        derived = (drawdown <= BIGLOSS_DRAWDOWN_THRESHOLD).astype(float)
        derived = derived.where(drawdown.notna(), np.nan)
        augmented[target_column] = derived
    return augmented


def _build_p_win_rate_by_date(
    labels: dict[str, torch.Tensor],
    row_to_date_idx: torch.Tensor,
    sample_row_indices: torch.Tensor,
    date_count: int,
) -> torch.Tensor:
    p_win_labels = labels.get("p_win")
    if p_win_labels is None or p_win_labels.numel() == 0:
        return torch.empty((date_count, 0), dtype=torch.float32)
    if sample_row_indices.numel() == 0 or date_count <= 0:
        return torch.zeros((date_count, p_win_labels.shape[1]), dtype=torch.float32)

    sample_rows = sample_row_indices.long()
    sample_date_idx = row_to_date_idx[sample_rows].long()
    sample_p_win = p_win_labels[sample_rows].float()
    date_sums = torch.zeros((date_count, sample_p_win.shape[1]), dtype=torch.float32)
    date_counts = torch.zeros((date_count, 1), dtype=torch.float32)
    date_sums.index_add_(0, sample_date_idx, sample_p_win)
    date_counts.index_add_(
        0,
        sample_date_idx,
        torch.ones((sample_date_idx.numel(), 1), dtype=torch.float32),
    )
    return date_sums / date_counts.clamp_min(1.0)


@dataclass
class PreparedSplit:
    split_name: str
    source_path: str
    source_signature: dict[str, Any]
    event_source_path: str | None
    event_source_signature: dict[str, Any] | None
    seq_features: torch.Tensor
    tab_features: torch.Tensor
    company_profile_features: torch.Tensor
    company_ids: dict[str, torch.Tensor]
    event_by_date: torch.Tensor
    mkt_by_date: torch.Tensor
    p_win_rate_by_date: torch.Tensor
    row_to_date_idx: torch.Tensor
    neighbor_symbol_ids: torch.Tensor
    neighbor_scores: torch.Tensor
    labels: dict[str, torch.Tensor]
    sample_row_indices: torch.Tensor
    sample_start_indices: torch.Tensor
    input_dims: dict[str, int]
    vocab_sizes: dict[str, int]
    head_dims: dict[str, int]
    row_count: int
    sample_count: int

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "cache_version": CACHE_VERSION,
            "split_name": self.split_name,
            "source_path": self.source_path,
            "source_signature": self.source_signature,
            "event_source_path": self.event_source_path,
            "event_source_signature": self.event_source_signature,
            "seq_features": self.seq_features,
            "tab_features": self.tab_features,
            "company_profile_features": self.company_profile_features,
            "company_ids": self.company_ids,
            "event_by_date": self.event_by_date,
            "mkt_by_date": self.mkt_by_date,
            "p_win_rate_by_date": self.p_win_rate_by_date,
            "row_to_date_idx": self.row_to_date_idx,
            "neighbor_symbol_ids": self.neighbor_symbol_ids,
            "neighbor_scores": self.neighbor_scores,
            "labels": self.labels,
            "sample_row_indices": self.sample_row_indices,
            "sample_start_indices": self.sample_start_indices,
            "input_dims": self.input_dims,
            "vocab_sizes": self.vocab_sizes,
            "head_dims": self.head_dims,
            "row_count": self.row_count,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_cache_dict(cls, payload: dict[str, Any]) -> "PreparedSplit":
        return cls(
            split_name=payload["split_name"],
            source_path=payload["source_path"],
            source_signature=payload["source_signature"],
            event_source_path=payload.get("event_source_path"),
            event_source_signature=payload.get("event_source_signature"),
            seq_features=payload["seq_features"],
            tab_features=payload["tab_features"],
            company_profile_features=payload["company_profile_features"],
            company_ids=payload["company_ids"],
            event_by_date=payload["event_by_date"],
            mkt_by_date=payload["mkt_by_date"],
            p_win_rate_by_date=payload.get("p_win_rate_by_date", torch.empty((0, 0), dtype=torch.float32)),
            row_to_date_idx=payload["row_to_date_idx"],
            neighbor_symbol_ids=payload["neighbor_symbol_ids"],
            neighbor_scores=payload["neighbor_scores"],
            labels=payload["labels"],
            sample_row_indices=payload["sample_row_indices"],
            sample_start_indices=payload["sample_start_indices"],
            input_dims=payload["input_dims"],
            vocab_sizes=payload["vocab_sizes"],
            head_dims=payload["head_dims"],
            row_count=int(payload["row_count"]),
            sample_count=int(payload["sample_count"]),
        )


@dataclass
class FeatureNormalizer:
    seq_mean: torch.Tensor
    seq_std: torch.Tensor
    tab_mean: torch.Tensor
    tab_std: torch.Tensor
    event_mean: torch.Tensor
    event_std: torch.Tensor
    mkt_mean: torch.Tensor
    mkt_std: torch.Tensor
    profile_mean: torch.Tensor
    profile_std: torch.Tensor
    clip_value: float = 10.0

    @classmethod
    def fit(cls, prepared: PreparedSplit) -> "FeatureNormalizer":
        return cls(
            seq_mean=prepared.seq_features.mean(dim=0),
            seq_std=prepared.seq_features.std(dim=0, unbiased=False).clamp_min(EPS),
            tab_mean=prepared.tab_features.mean(dim=0),
            tab_std=prepared.tab_features.std(dim=0, unbiased=False).clamp_min(EPS),
            event_mean=prepared.event_by_date.mean(dim=0),
            event_std=prepared.event_by_date.std(dim=0, unbiased=False).clamp_min(EPS),
            mkt_mean=prepared.mkt_by_date.mean(dim=0),
            mkt_std=prepared.mkt_by_date.std(dim=0, unbiased=False).clamp_min(EPS),
            profile_mean=prepared.company_profile_features.mean(dim=0),
            profile_std=prepared.company_profile_features.std(dim=0, unbiased=False).clamp_min(EPS),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "seq_mean": self.seq_mean,
            "seq_std": self.seq_std,
            "tab_mean": self.tab_mean,
            "tab_std": self.tab_std,
            "event_mean": self.event_mean,
            "event_std": self.event_std,
            "mkt_mean": self.mkt_mean,
            "mkt_std": self.mkt_std,
            "profile_mean": self.profile_mean,
            "profile_std": self.profile_std,
            "clip_value": torch.tensor(float(self.clip_value)),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "FeatureNormalizer":
        payload = dict(state)
        clip_tensor = payload.pop("clip_value", torch.tensor(10.0))
        return cls(
            seq_mean=payload["seq_mean"],
            seq_std=payload["seq_std"],
            tab_mean=payload["tab_mean"],
            tab_std=payload["tab_std"],
            event_mean=payload.get("event_mean", torch.zeros(0)),
            event_std=payload.get("event_std", torch.ones(0)),
            mkt_mean=payload["mkt_mean"],
            mkt_std=payload["mkt_std"],
            profile_mean=payload["profile_mean"],
            profile_std=payload["profile_std"],
            clip_value=float(clip_tensor.item() if hasattr(clip_tensor, "item") else clip_tensor),
        )

    def normalize_seq(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.seq_mean) / self.seq_std
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.clip_value, self.clip_value)

    def normalize_tab(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.tab_mean) / self.tab_std
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.clip_value, self.clip_value)

    def normalize_event(self, x: torch.Tensor) -> torch.Tensor:
        if self.event_mean.numel() == 0 or self.event_std.numel() == 0:
            return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.clip_value, self.clip_value)
        normalized = (x - self.event_mean) / self.event_std
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.clip_value, self.clip_value)

    def normalize_mkt(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.mkt_mean) / self.mkt_std
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.clip_value, self.clip_value)

    def normalize_profile(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (x - self.profile_mean) / self.profile_std
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).clamp(-self.clip_value, self.clip_value)


class FastStockDataset(Dataset[dict[str, Any]]):
    def __init__(self, prepared: PreparedSplit, normalizer: FeatureNormalizer, normalize_on_the_fly: bool = True):
        self.prepared = prepared
        self.normalizer = normalizer
        self.normalize_on_the_fly = bool(normalize_on_the_fly)

    def _feature(self, name: str, value: torch.Tensor) -> torch.Tensor:
        if not self.normalize_on_the_fly:
            return value
        if name == "seq":
            return self.normalizer.normalize_seq(value)
        if name == "tab":
            return self.normalizer.normalize_tab(value)
        if name == "event":
            return self.normalizer.normalize_event(value)
        if name == "mkt":
            return self.normalizer.normalize_mkt(value)
        if name == "profile":
            return self.normalizer.normalize_profile(value)
        return value

    def __len__(self) -> int:
        return self.prepared.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_idx = int(self.prepared.sample_row_indices[index].item())
        start_idx = int(self.prepared.sample_start_indices[index].item())
        date_idx = int(self.prepared.row_to_date_idx[row_idx].item())
        return {
            "X_seq": self._feature("seq", self.prepared.seq_features[start_idx : row_idx + 1]),
            "X_tab": self._feature("tab", self.prepared.tab_features[row_idx]),
            "X_event": self._feature("event", self.prepared.event_by_date[date_idx]),
            "X_mkt": self._feature("mkt", self.prepared.mkt_by_date[date_idx]),
            "X_company_ids": {
                key: value[row_idx] for key, value in self.prepared.company_ids.items()
            },
            "X_company_profile": self._feature("profile", self.prepared.company_profile_features[row_idx]),
            "neighbors": {
                "neighbor_symbol_ids": self.prepared.neighbor_symbol_ids[row_idx],
                "neighbor_scores": self.prepared.neighbor_scores[row_idx],
            },
            "y": {
                head_name: values[row_idx]
                for head_name, values in self.prepared.labels.items()
            },
            "aux": {
                "p_win_rate_by_date": self.prepared.p_win_rate_by_date[date_idx],
                "date_idx": torch.tensor(date_idx, dtype=torch.long),
            },
        }


@dataclass
class DataBundle:
    train_split: PreparedSplit
    valid_split: PreparedSplit
    test_split: PreparedSplit | None
    normalizer: FeatureNormalizer
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader | None


def prepare_split(
    split_name: str,
    source_path: Path,
    seq_length: int,
    neighbor_topk: int = 10,
    event_source_path: Path | None = None,
) -> PreparedSplit:
    read_columns = _resolve_read_columns(_schema_columns(source_path))
    frame = pd.read_parquet(source_path, columns=read_columns)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.normalize()
    frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    frame = augment_aligned_features(frame)
    frame = _ensure_derived_risk_labels(frame)
    schema_columns = set(frame.columns)
    seq_columns = [column for column in DEFAULT_SEQ_COLUMNS if column in schema_columns]
    tab_columns = [column for column in DEFAULT_TAB_COLUMNS if column in schema_columns]
    mkt_columns = [column for column in DEFAULT_MKT_COLUMNS if column in schema_columns]
    company_id_columns = [column for column in DEFAULT_COMPANY_ID_COLUMNS if column in schema_columns]
    company_profile_columns = [column for column in DEFAULT_COMPANY_PROFILE_COLUMNS if column in schema_columns]
    wide_event_columns = [column for column in DEFAULT_EVENT_COLUMNS if column in schema_columns]
    event_columns = wide_event_columns or list(DEFAULT_EVENT_COLUMNS)
    label_groups = {
        head_name: [column for column in columns if column in schema_columns]
        for head_name, columns in DEFAULT_LABEL_GROUPS.items()
    }
    label_columns = [column for columns in label_groups.values() for column in columns]

    row_count = len(frame)
    seq_features = _safe_float_matrix(frame, seq_columns, row_count)
    tab_features = _safe_float_matrix(frame, tab_columns, row_count)
    profile_scaler_stats = _build_profile_scaler_stats(frame, company_profile_columns)
    company_profile_features = _scale_company_profile_frame(frame, company_profile_columns, profile_scaler_stats)
    company_ids = _safe_long_matrix(frame, company_id_columns, row_count)
    labels = {
        head_name: _safe_float_matrix(frame, columns, row_count)
        for head_name, columns in label_groups.items()
        if columns
    }
    head_dims = {head_name: int(values.shape[-1]) for head_name, values in labels.items()}

    date_frame = frame.groupby("trade_date", sort=True).first().reset_index()
    event_by_date, row_to_date_idx, resolved_event_path, event_source_signature = _load_event_by_date(
        frame=frame,
        event_source_path=event_source_path,
        event_columns=event_columns,
    )
    mkt_by_date = _safe_float_matrix(date_frame, mkt_columns, len(date_frame))
    neighbor_symbol_ids, neighbor_scores = _build_neighbor_tensors(frame, neighbor_topk)

    label_mask = np.ones(row_count, dtype=bool)
    for values in labels.values():
        label_mask &= torch.isfinite(values).all(dim=1).cpu().numpy()

    symbols = frame["symbol"].astype(str).to_numpy()
    boundaries = np.flatnonzero(symbols[1:] != symbols[:-1]) + 1 if row_count > 1 else np.array([], dtype=np.int64)
    boundaries = np.concatenate(([0], boundaries, [row_count]))

    sample_rows: list[np.ndarray] = []
    sample_starts: list[np.ndarray] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        group_rows = np.arange(start, end, dtype=np.int64)
        if len(group_rows) < seq_length:
            continue
        eligible_rows = group_rows[seq_length - 1 :]
        eligible_rows = eligible_rows[label_mask[eligible_rows]]
        if len(eligible_rows) == 0:
            continue
        sample_rows.append(eligible_rows)
        sample_starts.append(eligible_rows - seq_length + 1)

    if sample_rows:
        sample_row_indices = torch.tensor(np.concatenate(sample_rows), dtype=torch.long)
        sample_start_indices = torch.tensor(np.concatenate(sample_starts), dtype=torch.long)
    else:
        sample_row_indices = torch.empty(0, dtype=torch.long)
        sample_start_indices = torch.empty(0, dtype=torch.long)

    p_win_rate_by_date = _build_p_win_rate_by_date(
        labels=labels,
        row_to_date_idx=row_to_date_idx,
        sample_row_indices=sample_row_indices,
        date_count=len(date_frame),
    )

    vocab_sizes = {
        column: int(values.max().item()) + 1 if values.numel() else 1
        for column, values in company_ids.items()
    }
    input_dims = {
        "seq_length": int(seq_length),
        "f_seq": int(seq_features.shape[1]),
        "f_tab": int(tab_features.shape[1]),
        "f_event": int(event_by_date.shape[1]),
        "f_mkt": int(mkt_by_date.shape[1]),
        "f_company_profile": int(company_profile_features.shape[1]),
        "neighbor_topk": int(neighbor_topk),
    }

    del frame
    del date_frame
    gc.collect()

    return PreparedSplit(
        split_name=split_name,
        source_path=str(source_path),
        source_signature=_source_signature(source_path),
        event_source_path=resolved_event_path,
        event_source_signature=event_source_signature,
        seq_features=seq_features,
        tab_features=tab_features,
        company_profile_features=company_profile_features,
        company_ids=company_ids,
        event_by_date=event_by_date,
        mkt_by_date=mkt_by_date,
        p_win_rate_by_date=p_win_rate_by_date,
        row_to_date_idx=row_to_date_idx,
        neighbor_symbol_ids=neighbor_symbol_ids,
        neighbor_scores=neighbor_scores,
        labels=labels,
        sample_row_indices=sample_row_indices,
        sample_start_indices=sample_start_indices,
        input_dims=input_dims,
        vocab_sizes=vocab_sizes,
        head_dims=head_dims,
        row_count=row_count,
        sample_count=int(sample_row_indices.numel()),
    )


def build_or_load_split(
    split_name: str,
    source_path: Path,
    cache_dir: Path,
    seq_length: int,
    rebuild_cache: bool = False,
    neighbor_topk: int = 10,
    verbose: bool = True,
) -> PreparedSplit:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split_name}_seq{seq_length}_cache_v{CACHE_VERSION}.pt"
    source_signature = _source_signature(source_path)
    candidate_event_source_path = _default_event_source_path(source_path)
    event_source_path = candidate_event_source_path if candidate_event_source_path.exists() else None
    event_source_signature = _source_signature(event_source_path) if event_source_path is not None else None
    if cache_path.exists() and not rebuild_cache:
        started_at = time.perf_counter()
        if verbose:
            print(f"[data] loading cached {split_name} split from {cache_path.name}")
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            payload.get("cache_version") == CACHE_VERSION
            and payload.get("source_signature") == source_signature
            and payload.get("event_source_signature") == event_source_signature
        ):
            prepared = PreparedSplit.from_cache_dict(payload)
            if verbose:
                print(
                    f"[data] ready {split_name} split in {_format_seconds(time.perf_counter() - started_at)} "
                    f"(rows={prepared.row_count:,}, samples={prepared.sample_count:,})"
                )
            return prepared
        if verbose:
            print(f"[data] cache for {split_name} split is stale, rebuilding from parquet")

    started_at = time.perf_counter()
    if verbose:
        print(f"[data] preparing {split_name} split from {source_path.name}")
    prepared = prepare_split(
        split_name=split_name,
        source_path=source_path,
        seq_length=seq_length,
        neighbor_topk=neighbor_topk,
        event_source_path=event_source_path,
    )
    torch.save(prepared.to_cache_dict(), cache_path)
    if verbose:
        print(
            f"[data] cached {split_name} split in {_format_seconds(time.perf_counter() - started_at)} "
            f"(rows={prepared.row_count:,}, samples={prepared.sample_count:,})"
        )
    return prepared


def build_loader(
    prepared_split: PreparedSplit,
    normalizer: FeatureNormalizer,
    batch_size: int,
    num_workers: int | None,
    device_type: str = "cpu",
    shuffle: bool = False,
    sampler: WeightedRandomSampler | None = None,
    normalize_on_the_fly: bool = True,
) -> DataLoader:
    workers = resolve_num_workers(num_workers)
    pin_memory = device_type == "cuda"
    loader_kwargs: dict[str, Any] = {
        "batch_size": int(batch_size),
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    dataset = FastStockDataset(prepared_split, normalizer, normalize_on_the_fly=normalize_on_the_fly)
    return DataLoader(
        dataset,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=False,
        **loader_kwargs,
    )


def load_split_for_evaluation(
    split_name: str,
    source_path: Path,
    cache_dir: Path,
    seq_length: int,
    normalizer: FeatureNormalizer,
    batch_size: int,
    num_workers: int | None,
    rebuild_cache: bool = False,
    device_type: str = "cpu",
    pre_normalize_features: bool = False,
) -> tuple[PreparedSplit, DataLoader]:
    prepared_split = build_or_load_split(
        split_name=split_name,
        source_path=source_path,
        cache_dir=cache_dir,
        seq_length=seq_length,
        rebuild_cache=rebuild_cache,
    )
    if pre_normalize_features:
        _pre_normalize_split_features(prepared_split, normalizer)
    loader = build_loader(
        prepared_split=prepared_split,
        normalizer=normalizer,
        batch_size=batch_size,
        num_workers=num_workers,
        device_type=device_type,
        shuffle=False,
        normalize_on_the_fly=not pre_normalize_features,
    )
    return prepared_split, loader


def _pre_normalize_split_features(prepared_split: PreparedSplit, normalizer: FeatureNormalizer) -> None:
    prepared_split.seq_features = normalizer.normalize_seq(prepared_split.seq_features)
    prepared_split.tab_features = normalizer.normalize_tab(prepared_split.tab_features)
    prepared_split.event_by_date = normalizer.normalize_event(prepared_split.event_by_date)
    prepared_split.mkt_by_date = normalizer.normalize_mkt(prepared_split.mkt_by_date)
    prepared_split.company_profile_features = normalizer.normalize_profile(prepared_split.company_profile_features)


def resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is not None:
        return max(0, int(num_workers))
    if os.name == "nt":
        return 0
    cpu_count = os.cpu_count() or 2
    return max(0, min(4, cpu_count // 2))


def build_weighted_train_sampler(
    train_split: PreparedSplit,
    recency_power: float = 1.5,
    positive_balance_power: float = 0.5,
    balance_horizon_index: int = 1,
) -> WeightedRandomSampler | None:
    sample_count = int(train_split.sample_count)
    if sample_count <= 0:
        return None

    weights = torch.ones(sample_count, dtype=torch.double)
    sample_rows = train_split.sample_row_indices

    if positive_balance_power > 0 and "p_win" in train_split.labels and train_split.labels["p_win"].numel() > 0:
        p_win_labels = train_split.labels["p_win"][sample_rows]
        if p_win_labels.ndim == 2 and p_win_labels.size(1) > 0:
            horizon_index = min(max(int(balance_horizon_index), 0), p_win_labels.size(1) - 1)
            class_ids = (p_win_labels[:, horizon_index] >= 0.5).long()
            class_counts = torch.bincount(class_ids, minlength=2).clamp_min(1)
            class_weights = class_counts.sum().float() / class_counts.float()
            class_weights = class_weights / class_weights.mean()
            weights = weights * class_weights[class_ids].double().pow(float(positive_balance_power))

    if recency_power > 0 and train_split.row_to_date_idx.numel() > 0:
        date_positions = train_split.row_to_date_idx[sample_rows].float()
        max_position = float(date_positions.max().item()) if date_positions.numel() else 0.0
        if max_position > 0:
            normalized_positions = (date_positions / max_position).clamp(0.0, 1.0)
            # Keep some base probability for early samples while biasing toward later regimes.
            recency_weights = 0.35 + 0.65 * normalized_positions
            weights = weights * recency_weights.double().pow(float(recency_power))

    weights = (weights / weights.mean().clamp_min(1e-6)).clamp_min(1e-3)
    return WeightedRandomSampler(weights=weights, num_samples=sample_count, replacement=True)


def build_data_bundle(
    train_path: Path,
    valid_path: Path,
    test_path: Path,
    cache_dir: Path,
    seq_length: int,
    batch_size: int,
    num_workers: int | None,
    rebuild_cache: bool = False,
    device_type: str = "cpu",
    use_weighted_sampler: bool = False,
    sampler_recency_power: float = 1.5,
    sampler_positive_balance_power: float = 0.5,
    sampler_balance_horizon_index: int = 1,
    load_test_split: bool = True,
    merge_test_into_valid: bool = False,
    pre_normalize_features: bool = False,
) -> DataBundle:
    train_split = build_or_load_split("train", train_path, cache_dir, seq_length, rebuild_cache=rebuild_cache)
    normalizer = FeatureNormalizer.fit(train_split)
    valid_source_path = valid_path
    valid_split_name = "valid"
    if merge_test_into_valid and test_path.exists():
        valid_source_path = materialize_merged_validation_split(valid_path, test_path, cache_dir)
        valid_split_name = "valid_merged"
    valid_split = build_or_load_split(valid_split_name, valid_source_path, cache_dir, seq_length, rebuild_cache=rebuild_cache)
    test_split = None
    test_loader = None
    if load_test_split and not merge_test_into_valid:
        test_split = build_or_load_split("test", test_path, cache_dir, seq_length, rebuild_cache=rebuild_cache)

    if pre_normalize_features:
        _pre_normalize_split_features(train_split, normalizer)
        _pre_normalize_split_features(valid_split, normalizer)
        if test_split is not None:
            _pre_normalize_split_features(test_split, normalizer)

    train_sampler = None
    if use_weighted_sampler:
        train_sampler = build_weighted_train_sampler(
            train_split,
            recency_power=sampler_recency_power,
            positive_balance_power=sampler_positive_balance_power,
            balance_horizon_index=sampler_balance_horizon_index,
        )

    return DataBundle(
        train_split=train_split,
        valid_split=valid_split,
        test_split=test_split,
        normalizer=normalizer,
        train_loader=build_loader(
            prepared_split=train_split,
            normalizer=normalizer,
            batch_size=batch_size,
            num_workers=num_workers,
            device_type=device_type,
            shuffle=True,
            sampler=train_sampler,
            normalize_on_the_fly=not pre_normalize_features,
        ),
        valid_loader=build_loader(
            prepared_split=valid_split,
            normalizer=normalizer,
            batch_size=batch_size,
            num_workers=num_workers,
            device_type=device_type,
            shuffle=False,
            normalize_on_the_fly=not pre_normalize_features,
        ),
        test_loader=(
            build_loader(
                prepared_split=test_split,
                normalizer=normalizer,
                batch_size=batch_size,
                num_workers=num_workers,
                device_type=device_type,
                shuffle=False,
                normalize_on_the_fly=not pre_normalize_features,
            )
            if test_split is not None
            else None
        ),
    )
