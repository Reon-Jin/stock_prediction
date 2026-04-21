from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.pytorch_dataset import MultiInputInferenceDataset
from train.data import FeatureNormalizer
from train.model import TinyMultiInputModel

PACKAGED_REQUIRED_COLUMNS = {
    "x_seq",
    "x_tab",
    "x_event",
    "x_mkt",
    "x_company_profile",
}
P_WIN_HORIZONS = [3, 5, 10, 20, 40]


def resolve_device_name(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def _parse_array(raw: Any, dtype: np.dtype, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        if raw.dtype == object:
            arr = np.asarray(raw.tolist(), dtype=dtype)
        else:
            arr = raw.astype(dtype, copy=False)
    elif isinstance(raw, list):
        arr = np.asarray(raw, dtype=dtype)
    elif isinstance(raw, str):
        arr = np.asarray(json.loads(raw), dtype=dtype)
    else:
        arr = np.asarray(raw, dtype=dtype)
    if expected_shape is not None:
        arr = arr.reshape(expected_shape)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def collate_inference_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = {
        "X_seq": torch.stack([item["X_seq"] for item in items], dim=0),
        "X_tab": torch.stack([item["X_tab"] for item in items], dim=0),
        "X_event": torch.stack([item["X_event"] for item in items], dim=0),
        "X_mkt": torch.stack([item["X_mkt"] for item in items], dim=0),
        "X_company_profile": torch.stack([item["X_company_profile"] for item in items], dim=0),
        "X_company_ids": {
            "symbol_id": torch.stack([item["X_company_ids"]["symbol_id"] for item in items], dim=0),
            "industry_id": torch.stack([item["X_company_ids"]["industry_id"] for item in items], dim=0),
            "board_id": torch.stack([item["X_company_ids"]["board_id"] for item in items], dim=0),
        },
        "neighbors": {
            "neighbor_symbol_ids": torch.stack([item["neighbors"]["neighbor_symbol_ids"] for item in items], dim=0),
            "neighbor_scores": torch.stack([item["neighbors"]["neighbor_scores"] for item in items], dim=0),
        },
        "meta": [item["meta"] for item in items],
    }
    return batch


class PackagedInferenceDataset(Dataset[dict[str, Any]]):
    def __init__(self, source_path: Path, normalizer: FeatureNormalizer, limit: int | None = None):
        self.df = pd.read_parquet(source_path).reset_index(drop=True)
        if limit is not None:
            self.df = self.df.head(int(limit)).reset_index(drop=True)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        x_seq = torch.tensor(_parse_array(row["x_seq"], np.float32), dtype=torch.float32)
        x_tab = torch.tensor(_parse_array(row["x_tab"], np.float32), dtype=torch.float32)
        x_event = torch.tensor(_parse_array(row["x_event"], np.float32), dtype=torch.float32)
        x_mkt = torch.tensor(_parse_array(row["x_mkt"], np.float32), dtype=torch.float32)
        x_company_profile = torch.tensor(_parse_array(row["x_company_profile"], np.float32), dtype=torch.float32)
        neighbor_symbol_ids = torch.tensor(_parse_array(row.get("neighbor_symbol_ids", []), np.int64), dtype=torch.long)
        neighbor_scores = torch.tensor(_parse_array(row.get("neighbor_scores", []), np.float32), dtype=torch.float32)
        return {
            "X_seq": self.normalizer.normalize_seq(x_seq),
            "X_tab": self.normalizer.normalize_tab(x_tab),
            "X_event": self.normalizer.normalize_event(x_event),
            "X_mkt": self.normalizer.normalize_mkt(x_mkt),
            "X_company_profile": self.normalizer.normalize_profile(x_company_profile),
            "X_company_ids": {
                "symbol_id": torch.tensor(int(pd.to_numeric(row.get("symbol_id"), errors="coerce") or 0), dtype=torch.long),
                "industry_id": torch.tensor(int(pd.to_numeric(row.get("industry_id"), errors="coerce") or 0), dtype=torch.long),
                "board_id": torch.tensor(int(pd.to_numeric(row.get("board_id"), errors="coerce") or 0), dtype=torch.long),
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


class RawInferenceDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        source_path: Path,
        normalizer: FeatureNormalizer,
        seq_length: int,
        latest_date_only: bool = True,
        limit: int | None = None,
    ):
        frame = pd.read_parquet(source_path)
        self.dataset = MultiInputInferenceDataset(frame, seq_length=seq_length)
        self.normalizer = normalizer
        self.indices = list(range(len(self.dataset)))
        if latest_date_only and len(self.dataset) > 0:
            latest_date = pd.to_datetime(self.dataset.df["trade_date"], errors="coerce").max()
            if pd.notna(latest_date):
                latest_norm = pd.Timestamp(latest_date).normalize()
                self.indices = [
                    idx
                    for idx, (_, _, row_idx) in enumerate(self.dataset._row_positions)
                    if pd.Timestamp(self.dataset.df.loc[row_idx, "trade_date"]).normalize() == latest_norm
                ]
        if limit is not None:
            self.indices = self.indices[: int(limit)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[self.indices[index]]
        return {
            "X_seq": self.normalizer.normalize_seq(sample.x_seq),
            "X_tab": self.normalizer.normalize_tab(sample.x_tab),
            "X_event": self.normalizer.normalize_event(sample.x_event),
            "X_mkt": self.normalizer.normalize_mkt(sample.x_mkt),
            "X_company_profile": self.normalizer.normalize_profile(sample.x_company_profile),
            "X_company_ids": sample.x_company_ids,
            "neighbors": {
                "neighbor_symbol_ids": sample.neighbor_symbol_ids,
                "neighbor_scores": sample.neighbor_scores,
            },
            "meta": sample.meta,
        }


@dataclass
class PredictConfig:
    checkpoint_path: str
    input_path: str
    output_path: str | None
    batch_size: int
    device: str
    limit: int | None
    all_dates: bool


def parse_args() -> PredictConfig:
    parser = argparse.ArgumentParser(description="Run inference with a trained tiny multi-input stock model.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True, help="today_infer parquet or raw parquet path")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=None, help="Only predict the first N selected samples")
    parser.add_argument("--all-dates", action="store_true", help="For raw parquet, predict every eligible sample instead of latest trade_date only")
    args = parser.parse_args()

    device = resolve_device_name(args.device)
    batch_size = args.batch_size or (2048 if device == "cuda" else 512)
    return PredictConfig(
        checkpoint_path=args.checkpoint_path,
        input_path=args.input_path,
        output_path=args.output_path,
        batch_size=batch_size,
        device=device,
        limit=args.limit,
        all_dates=args.all_dates,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            output[key] = {
                sub_key: sub_value.to(device, non_blocking=True)
                if isinstance(sub_value, torch.Tensor)
                else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            output[key] = value
    return output


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[TinyMultiInputModel, FeatureNormalizer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config", {})
    model = TinyMultiInputModel(
        input_dims=checkpoint["input_dims"],
        vocab_sizes=checkpoint["vocab_sizes"],
        head_dims=checkpoint["head_dims"],
        **model_config,
    ).to(device)
    load_result = model.load_state_dict(checkpoint["model_state"], strict=False)
    missing_keys = list(getattr(load_result, "missing_keys", []))
    unexpected_keys = list(getattr(load_result, "unexpected_keys", []))
    allowed_missing_prefixes = (
        "p_win_regime_head.",
        "ret_sigma_head.",
        "upside_head.",
        "decision_score_head.",
        "event_encoder.confidence_gate.",
        "neighbor_encoder.",
    )
    allowed_unexpected_prefixes = (
        "p_win_regime_scale",
        "ret_mu_to_p_win.",
        "risk_dd_to_penalty.",
        "rank_to_p_win.",
        "bigloss_to_penalty.",
        "ret_mu_scale",
        "risk_dd_scale",
        "bigloss_scale",
        "rank_score_scale",
        "neighbor_proj.",
    )
    blocking_missing = [key for key in missing_keys if not key.startswith(allowed_missing_prefixes)]
    blocking_unexpected = [key for key in unexpected_keys if not key.startswith(allowed_unexpected_prefixes)]
    if blocking_missing or blocking_unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing_keys={blocking_missing or missing_keys} unexpected_keys={blocking_unexpected or unexpected_keys}"
        )
    model.eval()
    normalizer_state = {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in checkpoint["normalizer_state"].items()
    }
    normalizer = FeatureNormalizer.from_state_dict(normalizer_state)
    return model, normalizer, checkpoint


def resolve_p_win_thresholds(checkpoint: dict[str, Any]) -> np.ndarray:
    raw_thresholds = checkpoint.get("p_win_thresholds")
    if raw_thresholds is None:
        return np.full(len(P_WIN_HORIZONS), 0.5, dtype=np.float32)
    if isinstance(raw_thresholds, torch.Tensor):
        values = raw_thresholds.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    else:
        values = np.asarray(raw_thresholds, dtype=np.float32).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, len(P_WIN_HORIZONS))
    if values.size < len(P_WIN_HORIZONS):
        padded = np.full(len(P_WIN_HORIZONS), 0.5, dtype=np.float32)
        padded[: values.size] = values
        return padded
    return values[: len(P_WIN_HORIZONS)]


def build_dataset(
    input_path: Path,
    normalizer: FeatureNormalizer,
    seq_length: int,
    limit: int | None,
    all_dates: bool,
) -> Dataset[dict[str, Any]]:
    columns = set(pq.read_schema(input_path).names)
    if PACKAGED_REQUIRED_COLUMNS.issubset(columns):
        return PackagedInferenceDataset(input_path, normalizer, limit=limit)
    return RawInferenceDataset(
        input_path,
        normalizer,
        seq_length=seq_length,
        latest_date_only=not all_dates,
        limit=limit,
    )


def default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    if stem.endswith(".parquet"):
        stem = stem[:-8]
    return input_path.with_name(f"{stem}_pred.parquet")


@torch.no_grad()
def run_prediction(
    model: TinyMultiInputModel,
    loader: DataLoader,
    device: torch.device,
    p_win_thresholds: np.ndarray | None = None,
    progress_callback: Any | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    autocast_enabled = device.type == "cuda"
    thresholds = (
        np.asarray(p_win_thresholds, dtype=np.float32).reshape(-1)
        if p_win_thresholds is not None
        else np.full(len(P_WIN_HORIZONS), 0.5, dtype=np.float32)
    )
    processed = 0
    total = len(loader.dataset) if hasattr(loader, "dataset") else 0
    for batch in tqdm(loader, desc="Predict", leave=False):
        meta = batch["meta"]
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            outputs = model(batch)
        p_win_prob = torch.sigmoid(outputs["p_win"]).detach().cpu().numpy()
        ret_mu_pred = outputs["ret_mu"].detach().cpu().numpy()
        risk_dd_pred = outputs["risk_dd"].detach().cpu().numpy()
        rank_score_pred = torch.sigmoid(outputs["rank_score"]).detach().cpu().numpy()
        decision_score_pred = outputs["decision_score"].detach().cpu().numpy()
        upside_pred = outputs["upside"].detach().cpu().numpy() if "upside" in outputs else None
        ret_sigma_pred = outputs["ret_sigma"].detach().cpu().numpy() if "ret_sigma" in outputs else None
        p_win_base_prob = (
            torch.sigmoid(outputs["p_win_base"]).detach().cpu().numpy()
            if "p_win_base" in outputs
            else None
        )
        p_win_regime_prob = (
            torch.sigmoid(outputs["p_win_regime"]).detach().cpu().numpy()
            if "p_win_regime" in outputs
            else None
        )
        bigloss_prob = (
            outputs["bigloss_prob"].detach().cpu().numpy()
            if "bigloss_prob" in outputs
            else (torch.sigmoid(outputs["bigloss"]).detach().cpu().numpy() if "bigloss" in outputs else None)
        )

        for idx, item_meta in enumerate(meta):
            row: dict[str, Any] = {
                "symbol": item_meta.get("symbol"),
                "trade_date": item_meta.get("trade_date"),
                "name": item_meta.get("name"),
                "industry_sw": item_meta.get("industry_sw"),
                "board": item_meta.get("board"),
                "rank_score_pred": float(rank_score_pred[idx][0]),
                "decision_score": float(decision_score_pred[idx][min(2, decision_score_pred.shape[1] - 1)]),
            }
            for horizon_idx, horizon in enumerate(P_WIN_HORIZONS):
                row[f"p_win_prob_{horizon}"] = float(p_win_prob[idx][horizon_idx])
                row[f"p_win_pred_{horizon}"] = int(p_win_prob[idx][horizon_idx] >= thresholds[horizon_idx])
                row[f"ret_mu_pred_{horizon}"] = float(ret_mu_pred[idx][horizon_idx])
                row[f"risk_dd_pred_{horizon}"] = float(risk_dd_pred[idx][horizon_idx])
                row[f"decision_score_{horizon}"] = float(decision_score_pred[idx][horizon_idx])
                if upside_pred is not None:
                    row[f"upside_pred_{horizon}"] = float(upside_pred[idx][horizon_idx])
                if ret_sigma_pred is not None:
                    row[f"ret_sigma_pred_{horizon}"] = float(ret_sigma_pred[idx][horizon_idx])
                if p_win_base_prob is not None:
                    row[f"p_win_base_prob_{horizon}"] = float(p_win_base_prob[idx][horizon_idx])
                if p_win_regime_prob is not None:
                    row[f"p_win_regime_prob_{horizon}"] = float(p_win_regime_prob[idx][horizon_idx])
            if p_win_regime_prob is not None:
                row["market_regime_prob"] = float(np.mean(p_win_regime_prob[idx]))
            if bigloss_prob is not None:
                for loss_idx, horizon in enumerate([5, 20][: bigloss_prob.shape[1]]):
                    row[f"bigloss_prob_{horizon}"] = float(bigloss_prob[idx][loss_idx])
            row["signal_score"] = float(row.get("decision_score_10", row["decision_score"]))
            rows.append(row)
        processed += len(meta)
        if callable(progress_callback):
            progress_callback(min(processed, total), total)

    output = pd.DataFrame(rows)
    if not output.empty:
        output = output.sort_values(["trade_date", "signal_score"], ascending=[True, False]).reset_index(drop=True)
    return output


def main() -> None:
    config = parse_args()
    checkpoint_path = PROJECT_ROOT / config.checkpoint_path if not Path(config.checkpoint_path).is_absolute() else Path(config.checkpoint_path)
    input_path = PROJECT_ROOT / config.input_path if not Path(config.input_path).is_absolute() else Path(config.input_path)
    if config.output_path:
        output_candidate = Path(config.output_path)
        output_path = PROJECT_ROOT / output_candidate if not output_candidate.is_absolute() else output_candidate
    else:
        output_path = default_output_path(input_path)

    device = torch.device(config.device)
    model, normalizer, checkpoint = load_model(checkpoint_path, device)
    p_win_thresholds = resolve_p_win_thresholds(checkpoint)
    dataset = build_dataset(
        input_path,
        normalizer,
        seq_length=int(checkpoint["input_dims"]["seq_length"]),
        limit=config.limit,
        all_dates=config.all_dates,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_inference_batch,
    )

    predictions = run_prediction(model, loader, device, p_win_thresholds=p_win_thresholds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path, index=False)
    csv_path = output_path.with_suffix(".csv")
    predictions.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Predictions saved to: {output_path}")
    print(f"CSV copy saved to: {csv_path}")
    print(f"Predicted rows: {len(predictions)}")
    if not predictions.empty:
        print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
