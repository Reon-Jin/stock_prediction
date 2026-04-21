from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from datasets.pytorch_dataset import DEFAULT_EVENT_COLUMNS, EVENT_EMBEDDING_COLUMN
from utils.io import dump_json

UNSAFE_RANDOM_SPLIT_MESSAGE = (
    "random split is unsafe for time-series financial prediction and should not be used for formal evaluation"
)
SUPPORTED_SPLIT_METHOD = "time_purged"
DEFAULT_SPLIT_NAMES = ("train", "valid", "test")
EVENT_SIDECAR_SUFFIX = "_events"


@dataclass(frozen=True, slots=True)
class SplitWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp

    def to_summary(self) -> dict[str, str]:
        return {
            "start": self.start.strftime("%Y-%m-%d"),
            "end": self.end.strftime("%Y-%m-%d"),
        }


@dataclass(frozen=True, slots=True)
class PurgedTimeSeriesSplitConfig:
    split_method: str
    seq_length: int
    label_horizons: tuple[int, ...]
    max_horizon: int
    min_gap_days: int
    gap_days: int
    windows: dict[str, SplitWindow]

    @property
    def train(self) -> SplitWindow:
        return self.windows["train"]

    @property
    def valid(self) -> SplitWindow:
        return self.windows["valid"]

    @property
    def test(self) -> SplitWindow:
        return self.windows["test"]

    def to_summary(self) -> dict[str, Any]:
        return {
            "split_method": self.split_method,
            "seq_length": self.seq_length,
            "label_horizons": list(self.label_horizons),
            "max_horizon": self.max_horizon,
            "gap_days": self.gap_days,
            "minimum_safe_gap_days": self.min_gap_days,
            "windows": {name: window.to_summary() for name, window in self.windows.items()},
        }


@dataclass(slots=True)
class SplitExportStats:
    rows: int = 0
    min_trade_date: pd.Timestamp | None = None
    max_trade_date: pd.Timestamp | None = None
    symbol_counts: dict[str, int] = field(default_factory=dict, init=False)
    unique_dates: set[str] = field(default_factory=set, init=False)

    def update(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        chunk = frame.copy()
        chunk["trade_date"] = pd.to_datetime(chunk["trade_date"], errors="coerce").dt.normalize()
        chunk = chunk.dropna(subset=["trade_date"])
        if chunk.empty:
            return
        if "symbol" not in chunk.columns:
            raise ValueError("split export frame must contain 'symbol'")
        chunk = chunk.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
        self.rows += int(len(chunk))
        chunk_min = chunk["trade_date"].min()
        chunk_max = chunk["trade_date"].max()
        self.min_trade_date = chunk_min if self.min_trade_date is None else min(self.min_trade_date, chunk_min)
        self.max_trade_date = chunk_max if self.max_trade_date is None else max(self.max_trade_date, chunk_max)
        self.unique_dates.update(chunk["trade_date"].dt.strftime("%Y-%m-%d").tolist())
        for symbol, count in chunk["symbol"].astype(str).value_counts().items():
            self.symbol_counts[symbol] = self.symbol_counts.get(symbol, 0) + int(count)

    @property
    def unique_symbols(self) -> int:
        return len(self.symbol_counts)

    @property
    def unique_date_count(self) -> int:
        return len(self.unique_dates)

    def estimated_sample_count(self, seq_length: int) -> int:
        if seq_length <= 0:
            return 0
        return int(sum(max(0, count - seq_length + 1) for count in self.symbol_counts.values()))

    def to_summary(self, seq_length: int) -> dict[str, Any]:
        return {
            "rows": int(self.rows),
            "unique_symbols": int(self.unique_symbols),
            "unique_dates": int(self.unique_date_count),
            "min_trade_date": self.min_trade_date.strftime("%Y-%m-%d") if self.min_trade_date is not None else None,
            "max_trade_date": self.max_trade_date.strftime("%Y-%m-%d") if self.max_trade_date is not None else None,
            "estimated_samples": int(self.estimated_sample_count(seq_length)),
        }


def _section(config_or_mapping: Any, name: str) -> dict[str, Any]:
    if hasattr(config_or_mapping, name):
        value = getattr(config_or_mapping, name)
        return value if isinstance(value, dict) else {}
    if isinstance(config_or_mapping, dict):
        value = config_or_mapping.get(name, {})
        return value if isinstance(value, dict) else {}
    return {}


def _project_split_block(config_or_mapping: Any) -> dict[str, Any]:
    project = _section(config_or_mapping, "project")
    split_block = project.get("split", {})
    if isinstance(split_block, dict) and split_block:
        return split_block
    root_split = _section(config_or_mapping, "split")
    return root_split if isinstance(root_split, dict) else {}


def _parse_timestamp(raw: Any, field_name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(raw)
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must be a valid date, got {raw!r}")
    return timestamp.normalize()


def _format_date(value: pd.Timestamp | None) -> str | None:
    return value.strftime("%Y-%m-%d") if value is not None else None


def event_sidecar_name(split_name: str) -> str:
    return f"{split_name}{EVENT_SIDECAR_SUFFIX}.parquet"


def _normalize_trade_frame(frame: pd.DataFrame, required_columns: tuple[str, ...] = ("symbol", "trade_date")) -> pd.DataFrame:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"split input missing required columns: {missing}")
    normalized = frame.copy()
    normalized["trade_date"] = pd.to_datetime(normalized["trade_date"], errors="coerce").dt.normalize()
    normalized = normalized.dropna(subset=["trade_date"])
    return normalized.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _resolve_label_horizons(config_or_mapping: Any) -> tuple[int, ...]:
    labels = _section(config_or_mapping, "labels")
    raw_periods = labels.get("holding_periods", [])
    horizons = tuple(int(value) for value in raw_periods if int(value) > 0)
    if not horizons:
        raise ValueError("labels.holding_periods must contain at least one positive horizon")
    return horizons


def _resolve_seq_length(config_or_mapping: Any) -> int:
    project = _section(config_or_mapping, "project")
    seq_length = int(project.get("seq_length", 0))
    if seq_length <= 0:
        raise ValueError("project.seq_length must be a positive integer")
    return seq_length


def _resolve_split_method(config_or_mapping: Any) -> str:
    project = _section(config_or_mapping, "project")
    return str(project.get("split_method", "")).strip().lower()


def derive_gap_days(seq_length: int, label_horizons: list[int] | tuple[int, ...], configured_gap_days: Any = None) -> int:
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    horizons = [int(value) for value in label_horizons if int(value) > 0]
    if not horizons:
        raise ValueError("label_horizons must contain at least one positive integer")
    derived_gap = int(seq_length + max(horizons))
    if configured_gap_days is None:
        return derived_gap
    gap_days = int(configured_gap_days)
    if gap_days <= 0:
        raise ValueError("gap_days must be a positive integer")
    if gap_days < derived_gap:
        raise ValueError(
            f"gap_days={gap_days} is unsafe; it must be at least seq_length + max_horizon = {derived_gap}"
        )
    return gap_days


def build_time_purged_split_config(config_or_mapping: Any) -> PurgedTimeSeriesSplitConfig:
    split_method = _resolve_split_method(config_or_mapping)
    if split_method == "random":
        warnings.warn(UNSAFE_RANDOM_SPLIT_MESSAGE, category=UserWarning, stacklevel=2)
        raise ValueError(UNSAFE_RANDOM_SPLIT_MESSAGE)
    if split_method != SUPPORTED_SPLIT_METHOD:
        raise ValueError(f"unsupported split_method={split_method!r}, expected {SUPPORTED_SPLIT_METHOD!r}")

    seq_length = _resolve_seq_length(config_or_mapping)
    label_horizons = _resolve_label_horizons(config_or_mapping)
    max_horizon = int(max(label_horizons))
    min_gap_days = int(seq_length + max_horizon)
    split_block = _project_split_block(config_or_mapping)
    if not split_block:
        raise ValueError("project.split configuration is required for time_purged split")

    windows = {
        name: SplitWindow(
            name=name,
            start=_parse_timestamp(split_block.get(f"{name}_start"), f"split.{name}_start"),
            end=_parse_timestamp(split_block.get(f"{name}_end"), f"split.{name}_end"),
        )
        for name in DEFAULT_SPLIT_NAMES
    }

    for name, window in windows.items():
        if window.start > window.end:
            raise ValueError(f"{name} split is empty: start={window.start.date()} end={window.end.date()}")

    ordered = [windows[name] for name in DEFAULT_SPLIT_NAMES]
    if not (
        ordered[0].start < ordered[0].end < ordered[1].start < ordered[1].end < ordered[2].start < ordered[2].end
    ):
        raise ValueError(
            "split windows must satisfy train_start < train_end < valid_start < valid_end < test_start < test_end"
        )

    gap_days = derive_gap_days(seq_length, label_horizons, split_block.get("gap_days"))
    for left, right in zip(ordered[:-1], ordered[1:]):
        boundary_gap = int((right.start - left.end).days)
        if boundary_gap < gap_days:
            raise ValueError(
                f"{left.name}->{right.name} gap is {boundary_gap} days, smaller than configured gap_days={gap_days}"
            )
        label_reach = left.end + pd.Timedelta(days=max_horizon)
        right_input_window_start = right.start - pd.Timedelta(days=seq_length - 1)
        if label_reach >= right_input_window_start:
            raise ValueError(
                f"{left.name}->{right.name} boundary is still unsafe: left labels reach {_format_date(label_reach)} "
                f"while {right.name} input window starts {_format_date(right_input_window_start)}"
            )

    return PurgedTimeSeriesSplitConfig(
        split_method=split_method,
        seq_length=seq_length,
        label_horizons=label_horizons,
        max_horizon=max_horizon,
        min_gap_days=min_gap_days,
        gap_days=gap_days,
        windows=windows,
    )


def split_by_time_purged(
    frame: pd.DataFrame,
    split_config: PurgedTimeSeriesSplitConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    normalized = _normalize_trade_frame(frame)
    result: dict[str, pd.DataFrame] = {}
    for name, window in split_config.windows.items():
        mask = (normalized["trade_date"] >= window.start) & (normalized["trade_date"] <= window.end)
        result[name] = normalized.loc[mask].sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    validate_no_temporal_leakage(
        result["train"],
        result["valid"],
        result["test"],
        seq_length=split_config.seq_length,
        max_horizon=split_config.max_horizon,
        gap_days=split_config.gap_days,
        split_windows=split_config.windows,
    )
    return result["train"], result["valid"], result["test"]


def _load_split_frame(source: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return _normalize_trade_frame(source)
    path = Path(source)
    columns = ["symbol", "trade_date"]
    frame = pd.read_parquet(path, columns=columns)
    return _normalize_trade_frame(frame, required_columns=("symbol", "trade_date"))


def _validate_disjoint_dates(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> None:
    overlap = set(left["trade_date"].dt.strftime("%Y-%m-%d")) & set(right["trade_date"].dt.strftime("%Y-%m-%d"))
    if overlap:
        examples = sorted(overlap)[:5]
        raise ValueError(f"{left_name}/{right_name} date ranges overlap: {examples}")


def _validate_symbol_boundary(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
    seq_length: int,
    max_horizon: int,
) -> None:
    if left.empty or right.empty:
        raise ValueError(f"{left_name} or {right_name} split is empty after filtering")
    left_symbol_end = left.groupby("symbol", sort=False)["trade_date"].max()
    right_symbol_start = right.groupby("symbol", sort=False)["trade_date"].min()
    shared_symbols = left_symbol_end.index.intersection(right_symbol_start.index)
    if shared_symbols.empty:
        return
    diagnostics = pd.DataFrame(
        {
            "left_last_date": left_symbol_end.loc[shared_symbols],
            "right_first_date": right_symbol_start.loc[shared_symbols],
        }
    )
    diagnostics["label_reach"] = diagnostics["left_last_date"] + pd.Timedelta(days=max_horizon)
    diagnostics["input_window_start"] = diagnostics["right_first_date"] - pd.Timedelta(days=seq_length - 1)
    risky = diagnostics[diagnostics["label_reach"] >= diagnostics["input_window_start"]]
    if not risky.empty:
        sample = risky.head(5)
        examples = [
            (
                str(symbol),
                _format_date(row["left_last_date"]),
                _format_date(row["right_first_date"]),
            )
            for symbol, row in sample.iterrows()
        ]
        raise ValueError(
            f"{left_name}->{right_name} leakage risk for shared symbols near the split boundary: {examples}"
        )


def validate_no_temporal_leakage(
    train_df: pd.DataFrame | str | Path,
    valid_df: pd.DataFrame | str | Path,
    test_df: pd.DataFrame | str | Path,
    seq_length: int,
    max_horizon: int,
    gap_days: int | None = None,
    split_windows: dict[str, SplitWindow] | None = None,
) -> dict[str, Any]:
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive")

    required_gap = int(gap_days or (seq_length + max_horizon))
    frames = {
        "train": _load_split_frame(train_df),
        "valid": _load_split_frame(valid_df),
        "test": _load_split_frame(test_df),
    }

    for name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"{name} split is empty")

    _validate_disjoint_dates(frames["train"], frames["valid"], "train", "valid")
    _validate_disjoint_dates(frames["train"], frames["test"], "train", "test")
    _validate_disjoint_dates(frames["valid"], frames["test"], "valid", "test")

    summary: dict[str, Any] = {
        "required_gap_days": required_gap,
        "date_ranges": {},
        "boundary_checks": {},
    }
    for name, frame in frames.items():
        summary["date_ranges"][name] = {
            "min_trade_date": _format_date(frame["trade_date"].min()),
            "max_trade_date": _format_date(frame["trade_date"].max()),
            "rows": int(len(frame)),
            "unique_symbols": int(frame["symbol"].nunique()),
            "unique_dates": int(frame["trade_date"].nunique()),
        }

    for left_name, right_name in (("train", "valid"), ("valid", "test")):
        left_end = frames[left_name]["trade_date"].max()
        right_start = frames[right_name]["trade_date"].min()
        actual_gap = int((right_start - left_end).days)
        if actual_gap < required_gap:
            raise ValueError(
                f"{left_name}->{right_name} gap is {actual_gap} days, smaller than required gap_days={required_gap}"
            )
        _validate_symbol_boundary(frames[left_name], frames[right_name], left_name, right_name, seq_length, max_horizon)
        boundary_summary = {
            "left_end": _format_date(left_end),
            "right_start": _format_date(right_start),
            "actual_gap_days": actual_gap,
            "left_label_reach": _format_date(left_end + pd.Timedelta(days=max_horizon)),
            "right_input_window_start": _format_date(right_start - pd.Timedelta(days=seq_length - 1)),
        }
        if split_windows is not None:
            configured_left = split_windows[left_name]
            configured_right = split_windows[right_name]
            if left_end > configured_left.end or right_start < configured_right.start:
                raise ValueError(
                    f"{left_name}/{right_name} exported data is outside configured windows: "
                    f"{_format_date(left_end)} / {_format_date(right_start)}"
                )
        summary["boundary_checks"][f"{left_name}_to_{right_name}"] = boundary_summary

    summary["passed"] = True
    return summary


def _writer_for_split(target: Path, schema: pa.Schema | None) -> pq.ParquetWriter | None:
    if schema is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(target, schema)


def _iter_frame_chunks(frame_or_chunks: pd.DataFrame | Any) -> Any:
    if isinstance(frame_or_chunks, pd.DataFrame):
        yield frame_or_chunks
        return
    yield from frame_or_chunks


def _event_columns_in_frame(frame: pd.DataFrame) -> list[str]:
    if EVENT_EMBEDDING_COLUMN in frame.columns:
        return [EVENT_EMBEDDING_COLUMN]
    return [column for column in DEFAULT_EVENT_COLUMNS if column in frame.columns]


def _build_event_sidecar_frame(
    prepared: pd.DataFrame,
    event_records: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    event_columns = _event_columns_in_frame(prepared)
    if not event_columns:
        return pd.DataFrame(columns=["trade_date"])
    event_frame = prepared.loc[:, ["trade_date", *event_columns]].copy()
    event_frame["trade_date"] = pd.to_datetime(event_frame["trade_date"], errors="coerce").dt.normalize()
    event_frame = event_frame.dropna(subset=["trade_date"]).drop_duplicates(subset=["trade_date"], keep="last")
    for row in event_frame.to_dict(orient="records"):
        trade_date = pd.Timestamp(row["trade_date"]).strftime("%Y-%m-%d")
        event_records[trade_date] = row
    return event_frame


def _sorted_event_sidecar_frame(event_records: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if not event_records:
        return pd.DataFrame(columns=["trade_date", EVENT_EMBEDDING_COLUMN])
    frame = pd.DataFrame(list(event_records.values()))
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    return frame.sort_values("trade_date").reset_index(drop=True)


def export_time_purged_splits(
    split_frames: dict[str, pd.DataFrame | Any],
    export_dir: str | Path,
    split_config: PurgedTimeSeriesSplitConfig,
    progress_bar: Any | None = None,
    expected_rows: dict[str, int] | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    output = Path(export_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    per_split_summary: dict[str, Any] = {}
    event_paths: dict[str, Path] = {}
    event_summaries: dict[str, Any] = {}

    for split_name in DEFAULT_SPLIT_NAMES:
        window = split_config.windows[split_name]
        target = output / f"{split_name}.parquet"
        event_target = output / event_sidecar_name(split_name)
        stats = SplitExportStats()
        event_records: dict[str, dict[str, Any]] = {}
        split_expected_rows = int((expected_rows or {}).get(split_name, 0))
        chunk_index = 0
        schema: pa.Schema | None = None
        writer: pq.ParquetWriter | None = None
        try:
            for chunk in _iter_frame_chunks(split_frames[split_name]):
                if chunk is None or chunk.empty:
                    continue
                chunk_index += 1
                prepared = _normalize_trade_frame(chunk)
                prepared = prepared[
                    (prepared["trade_date"] >= window.start) & (prepared["trade_date"] <= window.end)
                ].sort_values(["symbol", "trade_date"]).reset_index(drop=True)
                if prepared.empty:
                    continue
                _build_event_sidecar_frame(prepared, event_records)
                event_columns = _event_columns_in_frame(prepared)
                if event_columns:
                    prepared = prepared.drop(columns=event_columns, errors="ignore")
                stats.update(prepared)
                table = pa.Table.from_pandas(prepared, preserve_index=False)
                if writer is None:
                    schema = table.schema
                    writer = _writer_for_split(target, schema)
                elif schema is not None and table.schema != schema:
                    table = table.cast(schema, safe=False)
                if writer is not None:
                    writer.write_table(table)
                if progress_bar is not None:
                    progress_bar.set_postfix_str(
                        f"{split_name} chunk={chunk_index} rows={stats.rows:,}/{split_expected_rows:,}"
                        if split_expected_rows > 0
                        else f"{split_name} chunk={chunk_index} rows={stats.rows:,}"
                    )
                    progress_bar.update(1)
            if writer is None:
                pd.DataFrame().to_parquet(target, index=False)
        finally:
            if writer is not None:
                writer.close()

        paths[split_name] = target
        event_paths[split_name] = event_target
        summary = stats.to_summary(split_config.seq_length)
        if summary["rows"] <= 0:
            raise ValueError(f"{split_name} split exported 0 rows for window {window.to_summary()}")
        event_sidecar = _sorted_event_sidecar_frame(event_records)
        if event_sidecar.empty:
            raise ValueError(f"{split_name} event sidecar is empty for window {window.to_summary()}")
        event_sidecar.to_parquet(event_target, index=False)
        event_summary = {
            "rows": int(len(event_sidecar)),
            "min_trade_date": _format_date(event_sidecar["trade_date"].min()),
            "max_trade_date": _format_date(event_sidecar["trade_date"].max()),
        }
        per_split_summary[split_name] = {
            **window.to_summary(),
            **summary,
            "event_path": str(event_target),
            "event_rows": event_summary["rows"],
        }
        event_summaries[split_name] = event_summary
        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"{split_name} rows={summary['rows']:,} symbols={summary['unique_symbols']:,} dates={summary['unique_dates']:,}"
            )

    leakage_summary = validate_no_temporal_leakage(
        paths["train"],
        paths["valid"],
        paths["test"],
        seq_length=split_config.seq_length,
        max_horizon=split_config.max_horizon,
        gap_days=split_config.gap_days,
        split_windows=split_config.windows,
    )
    summary_payload = {
        **split_config.to_summary(),
        "splits": per_split_summary,
        "event_splits": event_summaries,
        "leakage_check": leakage_summary,
        "paths": {
            **{name: str(path) for name, path in paths.items()},
            **{f"{name}_events": str(path) for name, path in event_paths.items()},
        },
    }
    dump_json(output / "split_summary.json", summary_payload)
    dump_json(output / "split_manifest.json", summary_payload["paths"])
    return paths, summary_payload
