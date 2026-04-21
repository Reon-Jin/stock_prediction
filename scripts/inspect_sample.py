from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pytorch_dataset import DEFAULT_LABEL_GROUPS, META_COLUMNS
from train.data import FastStockDataset, FeatureNormalizer, build_or_load_split


def _default_source(split: str) -> Path:
    return PROJECT_ROOT / "data" / "exports" / f"{split}.parquet"


def _resolve_path(raw_path: str | None, split: str) -> Path:
    path = Path(raw_path) if raw_path else _default_source(split)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect training samples using the real train.data assembly path.")
    parser.add_argument("--source", type=str, default=None, help="Path to train/valid/test parquet.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"])
    parser.add_argument("--index", type=int, default=0, help="Sample index in assembled dataset.")
    parser.add_argument("--count", type=int, default=1, help="Number of consecutive samples to print.")
    parser.add_argument("--symbol", type=str, default=None, help="Optional symbol lookup, e.g. 000001.SZ")
    parser.add_argument("--trade-date", type=str, default=None, help="Optional trade date lookup, YYYY-MM-DD")
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--cache-dir", type=str, default="train/cache")
    parser.add_argument("--view", type=str, default="dataset", choices=["dataset", "raw", "schema"])
    parser.add_argument("--raw-columns", type=int, default=30)
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")


def _read_meta_frame(source: Path) -> pd.DataFrame:
    requested = [*META_COLUMNS]
    for label_columns in DEFAULT_LABEL_GROUPS.values():
        requested.extend(label_columns)
    requested = list(dict.fromkeys(requested))
    full_frame = pd.read_parquet(source)
    frame = full_frame.loc[:, [column for column in requested if column in full_frame.columns]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    frame = frame.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return frame


def _build_normalizer(
    split_name: str,
    source_path: Path,
    cache_dir: Path,
    seq_length: int,
    rebuild_cache: bool,
) -> tuple[FeatureNormalizer, str]:
    if split_name == "train":
        prepared = build_or_load_split(
            split_name="train",
            source_path=source_path,
            cache_dir=cache_dir,
            seq_length=seq_length,
            rebuild_cache=rebuild_cache,
            verbose=False,
        )
        return FeatureNormalizer.fit(prepared), "train"

    train_source = _default_source("train").resolve()
    if train_source.exists():
        prepared = build_or_load_split(
            split_name="train",
            source_path=train_source,
            cache_dir=cache_dir,
            seq_length=seq_length,
            rebuild_cache=rebuild_cache,
            verbose=False,
        )
        return FeatureNormalizer.fit(prepared), "train"

    prepared = build_or_load_split(
        split_name=split_name,
        source_path=source_path,
        cache_dir=cache_dir,
        seq_length=seq_length,
        rebuild_cache=rebuild_cache,
        verbose=False,
    )
    return FeatureNormalizer.fit(prepared), split_name


def _tensor_preview(value: Any, max_items: int = 8) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return {"preview": value.item()}
        if value.ndim == 1:
            return {"preview": value[:max_items].tolist()}
        return {"preview_rows": value[:2, :max_items].tolist()}
    if isinstance(value, list):
        return {"preview": value[:max_items]}
    return {"preview": value}


def _sample_payload(
    source: Path,
    dataset: FastStockDataset,
    meta_frame: pd.DataFrame,
    sample_index: int,
    normalizer_source: str,
) -> dict[str, Any]:
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"Sample index {sample_index} out of range, dataset length is {len(dataset)}")

    prepared = dataset.prepared
    sample = dataset[sample_index]
    row_idx = int(prepared.sample_row_indices[sample_index].item())
    start_idx = int(prepared.sample_start_indices[sample_index].item())
    row = meta_frame.iloc[row_idx]

    return {
        "source": str(source),
        "sample_index": sample_index,
        "row_index": row_idx,
        "window_start_row": start_idx,
        "window_end_row": row_idx,
        "normalizer_source": normalizer_source,
        "meta": {
            "symbol": row.get("symbol"),
            "trade_date": str(pd.Timestamp(row.get("trade_date")).date()) if pd.notna(row.get("trade_date")) else None,
            "name": row.get("name"),
            "industry_sw": row.get("industry_sw"),
            "board": row.get("board"),
        },
        "shapes": {
            "X_seq": list(sample["X_seq"].shape),
            "X_tab": list(sample["X_tab"].shape),
            "X_event": list(sample["X_event"].shape),
            "X_mkt": list(sample["X_mkt"].shape),
            "X_company_profile": list(sample["X_company_profile"].shape),
            "neighbor_symbol_ids": list(sample["neighbors"]["neighbor_symbol_ids"].shape),
            "neighbor_scores": list(sample["neighbors"]["neighbor_scores"].shape),
        },
        "X_company_ids": {
            key: int(value.item()) if isinstance(value, torch.Tensor) else int(value)
            for key, value in sample["X_company_ids"].items()
        },
        "X_seq_preview": _tensor_preview(sample["X_seq"]),
        "X_tab_preview": _tensor_preview(sample["X_tab"]),
        "X_event_preview": _tensor_preview(sample["X_event"]),
        "X_mkt_preview": _tensor_preview(sample["X_mkt"]),
        "X_company_profile_preview": _tensor_preview(sample["X_company_profile"]),
        "neighbor_symbol_ids_preview": _tensor_preview(sample["neighbors"]["neighbor_symbol_ids"]),
        "neighbor_scores_preview": _tensor_preview(sample["neighbors"]["neighbor_scores"]),
        "labels": {
            key: value.detach().cpu().tolist()
            for key, value in sample["y"].items()
        },
    }


def _lookup_sample_index(meta_frame: pd.DataFrame, prepared_row_indices: torch.Tensor, symbol: str, trade_date: str) -> int:
    target_symbol = str(symbol).strip().upper()
    target_date = pd.Timestamp(trade_date).normalize()
    sample_rows = prepared_row_indices.detach().cpu().tolist()
    for sample_index, row_idx in enumerate(sample_rows):
        row = meta_frame.iloc[int(row_idx)]
        row_symbol = str(row.get("symbol", "")).strip().upper()
        row_date = pd.Timestamp(row.get("trade_date")).normalize()
        if row_symbol == target_symbol and row_date == target_date:
            return sample_index
    raise LookupError(f"No assembled sample found for symbol={target_symbol} trade_date={target_date.date()}")


def _build_dataset_view(args: argparse.Namespace, source: Path, cache_dir: Path) -> Any:
    prepared = build_or_load_split(
        split_name=args.split,
        source_path=source,
        cache_dir=cache_dir,
        seq_length=args.seq_length,
        rebuild_cache=args.rebuild_cache,
        verbose=False,
    )
    normalizer, normalizer_source = _build_normalizer(
        split_name=args.split,
        source_path=source,
        cache_dir=cache_dir,
        seq_length=args.seq_length,
        rebuild_cache=args.rebuild_cache,
    )
    dataset = FastStockDataset(prepared, normalizer)
    meta_frame = _read_meta_frame(source)

    if args.symbol and args.trade_date:
        sample_index = _lookup_sample_index(meta_frame, prepared.sample_row_indices, args.symbol, args.trade_date)
        return _sample_payload(source, dataset, meta_frame, sample_index, normalizer_source)

    if args.count == 1:
        return _sample_payload(source, dataset, meta_frame, args.index, normalizer_source)
    return [
        _sample_payload(source, dataset, meta_frame, sample_index, normalizer_source)
        for sample_index in range(args.index, args.index + args.count)
    ]


def _build_raw_view(args: argparse.Namespace, source: Path) -> Any:
    frame = pd.read_parquet(source)
    if frame.empty:
        raise ValueError("Parquet file is empty.")
    if "trade_date" in frame.columns:
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    if args.symbol and args.trade_date:
        matched = frame[
            (frame["symbol"].astype(str).str.upper() == str(args.symbol).strip().upper())
            & (frame["trade_date"] == pd.Timestamp(args.trade_date).normalize())
        ]
        if matched.empty:
            raise LookupError(f"No raw row found for symbol={args.symbol} trade_date={args.trade_date}")
        row = matched.iloc[0]
        preview = {column: row[column] for column in matched.columns[: args.raw_columns]}
        return {
            "source": str(source),
            "lookup": {"symbol": args.symbol, "trade_date": args.trade_date},
            "preview_column_count": min(args.raw_columns, len(matched.columns)),
            "preview": preview,
            "all_columns": matched.columns.tolist(),
        }
    if args.count == 1:
        row = frame.iloc[args.index]
        preview = {column: row[column] for column in frame.columns[: args.raw_columns]}
        return {
            "source": str(source),
            "row_index": args.index,
            "row_count": len(frame),
            "preview_column_count": min(args.raw_columns, len(frame.columns)),
            "preview": preview,
            "all_columns": frame.columns.tolist(),
        }
    payload: list[dict[str, Any]] = []
    for row_index in range(args.index, args.index + args.count):
        row = frame.iloc[row_index]
        preview = {column: row[column] for column in frame.columns[: args.raw_columns]}
        payload.append(
            {
                "source": str(source),
                "row_index": row_index,
                "preview": preview,
            }
        )
    return payload


def _build_schema_view(args: argparse.Namespace, source: Path, cache_dir: Path) -> dict[str, Any]:
    prepared = build_or_load_split(
        split_name=args.split,
        source_path=source,
        cache_dir=cache_dir,
        seq_length=args.seq_length,
        rebuild_cache=args.rebuild_cache,
        verbose=False,
    )
    return {
        "source": str(source),
        "row_count": prepared.row_count,
        "sample_count": prepared.sample_count,
        "input_dims": prepared.input_dims,
        "head_dims": prepared.head_dims,
        "vocab_sizes": prepared.vocab_sizes,
        "event_source_path": prepared.event_source_path,
    }


def main() -> None:
    args = _parse_args()
    if bool(args.symbol) ^ bool(args.trade_date):
        raise ValueError("--symbol and --trade-date must be provided together")
    if args.count <= 0:
        raise ValueError("--count must be positive")

    source = _resolve_path(args.source, args.split)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (PROJECT_ROOT / cache_dir).resolve()
    _ensure_exists(source)

    if args.view == "dataset":
        payload = _build_dataset_view(args, source, cache_dir)
    elif args.view == "raw":
        payload = _build_raw_view(args, source)
    else:
        payload = _build_schema_view(args, source, cache_dir)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
