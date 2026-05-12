from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.data import DataBundle, build_data_bundle
from train.model import BIGLOSS_HORIZONS, BIGLOSS_PROXY_BY_HORIZON, HORIZONS, TinyMultiInputModel
from train.visualization import plot_business_history, plot_topk_metrics, plot_training_history

DEFAULT_P_WIN_HORIZONS = (3, 5, 10, 20, 40)
LOSS_PART_NAMES = (
    "loss_p_win",
    "loss_p_win_brier",
    "loss_p_win_logit_l2",
    "loss_ret_mu",
    "loss_ret_sigma",
    "loss_risk_dd",
    "loss_bigloss",
    "loss_upside",
    "loss_rank_pairwise",
    "loss_rank_score",
    "loss_bigloss_margin",
    "loss_anti_conservative",
    "loss_calibration",
    "loss_p_win_regime",
    "loss_prediction_entropy",
    "loss_expert_gate_entropy",
    "loss_expert_gate_prior",
)


@dataclass
class TrainConfig:
    train_path: str
    valid_path: str
    output_dir: str
    cache_dir: str
    seq_length: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    dropout: float
    seq_model_dim: int
    seq_output_dim: int
    seq_attn_heads: int
    branch_hidden_dim: int
    context_dim: int
    company_dim: int
    fusion_dim: int
    task_hidden_dim: int
    horizon_expert_dim: int
    grad_clip: float
    early_stop_patience: int
    num_workers: int | None
    device: str
    rebuild_cache: bool
    seed: int
    label_smoothing: float
    warmup_epochs: int
    min_lr_ratio: float
    ema_decay: float
    use_amp: bool
    focal_gamma: float
    p_win_brier_weight: float
    p_win_logit_l2_weight: float
    decision_detach_aux: bool
    horizon_source_gate: bool
    diagnostic_eval_batches: int
    use_weighted_sampler: bool
    sampler_recency_power: float
    sampler_positive_balance_power: float
    sampler_balance_horizon_index: int
    p_win_weight: float
    ret_mu_weight: float
    ret_sigma_weight: float
    risk_dd_weight: float
    bigloss_weight: float
    upside_weight: float
    rank_pairwise_weight: float
    rank_score_weight: float
    bigloss_margin_weight: float
    anti_conservative_weight: float
    calibration_weight: float
    p_win_regime_weight: float
    p_win_negative_weight: float
    p_win_downside_weight: float
    prediction_entropy_weight: float
    prediction_entropy_target_std: float
    expert_gate_entropy_weight: float
    expert_gate_min_entropy_ratio: float
    expert_gate_max_entropy_ratio: float
    expert_gate_prior_weight: float
    primary_horizon_index: int
    topk: int
    best_model_metric: str
    early_stop_min_delta: float
    max_train_batches: int | None
    max_eval_batches: int | None
    amp_dtype: str
    pre_normalize_features: bool
    disable_tqdm: bool


@dataclass
class LossContext:
    p_win_pos_weight: torch.Tensor
    p_win_horizon_weight: torch.Tensor
    ret_mu_horizon_weight: torch.Tensor
    risk_dd_horizon_weight: torch.Tensor


def resolve_device_name(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested


def recommended_batch_size(device_name: str) -> int:
    if device_name != "cuda":
        print("Using CPU")
        return 128
    print("Using GPU")
    return 4096


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a professional multi-input financial prediction model.")
    parser.add_argument("--train-path", type=str, default="data/exports/train.parquet")
    parser.add_argument("--valid-path", type=str, default="data/exports/valid.parquet")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="train/cache")
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--seq-model-dim", type=int, default=88)
    parser.add_argument("--seq-output-dim", type=int, default=120)
    parser.add_argument("--seq-attn-heads", type=int, default=4)
    parser.add_argument("--branch-hidden-dim", type=int, default=120)
    parser.add_argument("--context-dim", type=int, default=80)
    parser.add_argument("--company-dim", type=int, default=44)
    parser.add_argument("--fusion-dim", type=int, default=184)
    parser.add_argument("--task-hidden-dim", type=int, default=96)
    parser.add_argument("--horizon-expert-dim", type=int, default=120)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--use-amp", action="store_true", help="Enable CUDA autocast; float32 is the default.")
    parser.add_argument("--focal-gamma", type=float, default=0.25)
    parser.add_argument("--p-win-brier-weight", type=float, default=0.05)
    parser.add_argument("--p-win-logit-l2-weight", type=float, default=3e-4)
    parser.add_argument("--no-decision-detach-aux", action="store_true")
    parser.add_argument("--disable-horizon-source-gate", action="store_true")
    parser.add_argument("--diagnostic-eval-batches", type=int, default=2)
    parser.add_argument("--enable-weighted-sampler", action="store_true")
    parser.add_argument("--disable-weighted-sampler", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--sampler-recency-power", type=float, default=0.75)
    parser.add_argument("--sampler-positive-balance-power", type=float, default=0.25)
    parser.add_argument("--sampler-balance-horizon-index", type=int, default=1)
    parser.add_argument("--p-win-weight", type=float, default=0.9)
    parser.add_argument("--ret-mu-weight", type=float, default=1.0)
    parser.add_argument("--ret-sigma-weight", type=float, default=0.05)
    parser.add_argument("--risk-dd-weight", type=float, default=0.1)
    parser.add_argument("--bigloss-weight", type=float, default=0.05)
    parser.add_argument("--upside-weight", type=float, default=0.05)
    parser.add_argument("--rank-pairwise-weight", type=float, default=0.5)
    parser.add_argument("--rank-score-weight", type=float, default=0.0)
    parser.add_argument("--bigloss-margin-weight", type=float, default=0.0)
    parser.add_argument("--anti-conservative-weight", type=float, default=0.0)
    parser.add_argument("--calibration-weight", type=float, default=0.0)
    parser.add_argument("--p-win-regime-weight", type=float, default=0.0)
    parser.add_argument("--p-win-negative-weight", type=float, default=1.18)
    parser.add_argument("--p-win-downside-weight", type=float, default=0.75)
    parser.add_argument("--prediction-entropy-weight", type=float, default=0.02)
    parser.add_argument("--prediction-entropy-target-std", type=float, default=0.06)
    parser.add_argument("--expert-gate-entropy-weight", type=float, default=0.02)
    parser.add_argument("--expert-gate-min-entropy-ratio", type=float, default=0.45)
    parser.add_argument("--expert-gate-max-entropy-ratio", type=float, default=0.92)
    parser.add_argument("--expert-gate-prior-weight", type=float, default=0.005)
    parser.add_argument("--primary-horizon-index", type=int, default=2, help="0-based main horizon; default 2 means 10d.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--best-model-metric",
        type=str,
        default="valid_loss",
        choices=[
            "valid_loss",
            "valid_p_win_acc",
            "valid_p_win_bal_acc",
            "valid_business_score",
            "valid_topk_utility",
        ],
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-4,
        help="Minimum absolute validation improvement required to reset early stopping patience.",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="auto",
        choices=["auto", "bfloat16", "float16"],
        help="CUDA autocast dtype when --use-amp is enabled. auto prefers bfloat16 when available.",
    )
    parser.add_argument(
        "--pre-normalize-features",
        action="store_true",
        help="Normalize cached split tensors once at load time instead of per sample.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    device = resolve_device_name(args.device)
    batch_size = args.batch_size or recommended_batch_size(device)
    output_dir = args.output_dir or f"train/artifacts/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return TrainConfig(
        train_path=args.train_path,
        valid_path=args.valid_path,
        output_dir=output_dir,
        cache_dir=args.cache_dir,
        seq_length=args.seq_length,
        batch_size=batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seq_model_dim=args.seq_model_dim,
        seq_output_dim=args.seq_output_dim,
        seq_attn_heads=args.seq_attn_heads,
        branch_hidden_dim=args.branch_hidden_dim,
        context_dim=args.context_dim,
        company_dim=args.company_dim,
        fusion_dim=args.fusion_dim,
        task_hidden_dim=args.task_hidden_dim,
        horizon_expert_dim=args.horizon_expert_dim,
        grad_clip=args.grad_clip,
        early_stop_patience=args.early_stop_patience,
        num_workers=args.num_workers,
        device=device,
        rebuild_cache=args.rebuild_cache,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
        ema_decay=args.ema_decay,
        use_amp=args.use_amp,
        focal_gamma=args.focal_gamma,
        p_win_brier_weight=args.p_win_brier_weight,
        p_win_logit_l2_weight=args.p_win_logit_l2_weight,
        decision_detach_aux=not args.no_decision_detach_aux,
        horizon_source_gate=not args.disable_horizon_source_gate,
        diagnostic_eval_batches=args.diagnostic_eval_batches,
        use_weighted_sampler=bool(args.enable_weighted_sampler and not args.disable_weighted_sampler),
        sampler_recency_power=args.sampler_recency_power,
        sampler_positive_balance_power=args.sampler_positive_balance_power,
        sampler_balance_horizon_index=args.sampler_balance_horizon_index,
        p_win_weight=args.p_win_weight,
        ret_mu_weight=args.ret_mu_weight,
        ret_sigma_weight=args.ret_sigma_weight,
        risk_dd_weight=args.risk_dd_weight,
        bigloss_weight=args.bigloss_weight,
        upside_weight=args.upside_weight,
        rank_pairwise_weight=args.rank_pairwise_weight,
        rank_score_weight=args.rank_score_weight,
        bigloss_margin_weight=args.bigloss_margin_weight,
        anti_conservative_weight=args.anti_conservative_weight,
        calibration_weight=args.calibration_weight,
        p_win_regime_weight=args.p_win_regime_weight,
        p_win_negative_weight=args.p_win_negative_weight,
        p_win_downside_weight=args.p_win_downside_weight,
        prediction_entropy_weight=args.prediction_entropy_weight,
        prediction_entropy_target_std=args.prediction_entropy_target_std,
        expert_gate_entropy_weight=args.expert_gate_entropy_weight,
        expert_gate_min_entropy_ratio=args.expert_gate_min_entropy_ratio,
        expert_gate_max_entropy_ratio=args.expert_gate_max_entropy_ratio,
        expert_gate_prior_weight=args.expert_gate_prior_weight,
        primary_horizon_index=args.primary_horizon_index,
        topk=args.topk,
        best_model_metric=args.best_model_metric,
        early_stop_min_delta=args.early_stop_min_delta,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        amp_dtype=args.amp_dtype,
        pre_normalize_features=args.pre_normalize_features,
        disable_tqdm=args.disable_tqdm,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(device: str) -> None:
    torch.set_float32_matmul_precision("high")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True


def autocast_config(device: torch.device, config: TrainConfig) -> tuple[bool, torch.dtype]:
    enabled = bool(config.use_amp and device.type == "cuda")
    if not enabled:
        return False, torch.float32
    requested_dtype = str(_config_value(config, "amp_dtype", "auto"))
    if requested_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif requested_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return True, dtype


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            moved[key] = {
                sub_key: sub_value.to(device, non_blocking=True)
                if isinstance(sub_value, torch.Tensor)
                else sub_value
                for sub_key, sub_value in value.items()
            }
        else:
            moved[key] = value
    return moved


def decision_aux_features_from_config(config: TrainConfig) -> list[str]:
    feature_weights = {
        "ret_mu": config.ret_mu_weight,
        "ret_sigma": config.ret_sigma_weight,
        "p_win": config.p_win_weight,
        "risk_dd": config.risk_dd_weight,
        "bigloss": config.bigloss_weight,
        "upside": config.upside_weight,
    }
    return [feature_name for feature_name, weight in feature_weights.items() if float(weight) > 0.0]


def build_model_kwargs(config: TrainConfig) -> dict[str, Any]:
    return {
        "seq_model_dim": config.seq_model_dim,
        "seq_output_dim": config.seq_output_dim,
        "seq_attn_heads": config.seq_attn_heads,
        "branch_hidden_dim": config.branch_hidden_dim,
        "context_dim": config.context_dim,
        "company_dim": config.company_dim,
        "fusion_dim": config.fusion_dim,
        "task_hidden_dim": config.task_hidden_dim,
        "horizon_expert_dim": config.horizon_expert_dim,
        "horizon_source_gate": config.horizon_source_gate,
        "decision_aux_features": decision_aux_features_from_config(config),
        "decision_detach_aux": config.decision_detach_aux,
        "dropout": config.dropout,
    }


def build_loss_context(bundle: DataBundle, device: torch.device) -> LossContext:
    indices = bundle.train_split.sample_row_indices
    p_win = bundle.train_split.labels["p_win"][indices]
    positive_rate = p_win.mean(dim=0).clamp(0.05, 0.95)
    pos_weight = ((1.0 - positive_rate) / positive_rate).clamp(0.5, 3.0)

    num_h = int(p_win.shape[1])
    if num_h == 5:
        p_win_hw = torch.tensor([1.0, 1.0, 1.2, 0.7, 0.3], dtype=torch.float32, device=device)
        ret_mu_hw = torch.tensor([0.6, 1.0, 1.2, 1.1, 0.8], dtype=torch.float32, device=device)
        risk_dd_hw = torch.tensor([0.7, 1.0, 1.2, 1.2, 0.9], dtype=torch.float32, device=device)
    else:
        p_win_hw = torch.ones(num_h, dtype=torch.float32, device=device)
        ret_mu_hw = torch.ones(num_h, dtype=torch.float32, device=device)
        risk_dd_hw = torch.ones(num_h, dtype=torch.float32, device=device)

    return LossContext(
        p_win_pos_weight=pos_weight.to(device),
        p_win_horizon_weight=p_win_hw,
        ret_mu_horizon_weight=ret_mu_hw,
        risk_dd_horizon_weight=risk_dd_hw,
    )


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = copy.deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_params = dict(self.shadow.named_parameters())
        model_params = dict(model.named_parameters())
        for name, parameter in model_params.items():
            ema_params[name].lerp_(parameter.detach(), 1.0 - self.decay)

        ema_buffers = dict(self.shadow.named_buffers())
        model_buffers = dict(model.named_buffers())
        for name, buffer in model_buffers.items():
            ema_buffers[name].copy_(buffer.detach())


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    warmup_epochs = max(0, min(int(config.warmup_epochs), max(0, config.epochs - 1)))
    cosine_epochs = max(1, config.epochs - warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=config.lr * max(0.0, min(config.min_lr_ratio, 1.0)),
    )
    if warmup_epochs == 0:
        return cosine
    warmup = LinearLR(optimizer, start_factor=0.35, end_factor=1.0, total_iters=warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def resolve_p_win_thresholds(
    device: torch.device,
    head_dim: int,
    thresholds: torch.Tensor | None = None,
) -> torch.Tensor:
    if thresholds is None:
        return torch.full((head_dim,), 0.5, dtype=torch.float32, device=device)
    resolved = thresholds.to(device=device, dtype=torch.float32)
    if resolved.ndim == 0:
        resolved = resolved.unsqueeze(0)
    if resolved.numel() == 1:
        resolved = resolved.repeat(head_dim)
    return resolved[:head_dim]


def resolve_horizon_labels(head_dim: int) -> list[str]:
    labels: list[str] = []
    for idx in range(max(0, int(head_dim))):
        if idx < len(DEFAULT_P_WIN_HORIZONS):
            labels.append(f"{DEFAULT_P_WIN_HORIZONS[idx]}d")
        else:
            labels.append(f"h{idx + 1}")
    return labels


def _expand_bigloss_targets(bigloss_targets: torch.Tensor, target_dim: int) -> torch.Tensor:
    if bigloss_targets.ndim != 2 or target_dim <= 0:
        return torch.zeros((bigloss_targets.size(0), target_dim), dtype=bigloss_targets.dtype, device=bigloss_targets.device)
    if bigloss_targets.size(1) == target_dim:
        return bigloss_targets
    expanded = torch.zeros((bigloss_targets.size(0), target_dim), dtype=bigloss_targets.dtype, device=bigloss_targets.device)
    if bigloss_targets.size(1) == len(BIGLOSS_HORIZONS) and target_dim <= len(HORIZONS):
        source_column_by_horizon = {
            source_horizon: column_idx for column_idx, source_horizon in enumerate(BIGLOSS_HORIZONS)
        }
        for target_column, horizon in enumerate(HORIZONS[:target_dim]):
            source_horizon = BIGLOSS_PROXY_BY_HORIZON[horizon]
            expanded[:, target_column] = bigloss_targets[:, source_column_by_horizon[source_horizon]]
        return expanded
    copy_dim = min(bigloss_targets.size(1), target_dim)
    expanded[:, :copy_dim] = bigloss_targets[:, :copy_dim]
    return expanded


def _downside_weight_matrix(
    targets: dict[str, torch.Tensor],
    config: TrainConfig,
    head_dim: int,
) -> torch.Tensor:
    device = targets["p_win"].device
    batch_size = targets["p_win"].size(0)
    weights = torch.ones((batch_size, head_dim), dtype=torch.float32, device=device)
    negative_mask = (targets["p_win"] < 0.5).float()
    negative_weight = float(_config_value(config, "p_win_negative_weight", 1.0))
    weights = weights * (1.0 + (max(0.0, negative_weight) - 1.0) * negative_mask)

    if "risk_dd" in targets and targets["risk_dd"].numel() > 0:
        drawdown_severity = torch.relu((-targets["risk_dd"].float()) - 0.02)
        downside_weight = float(_config_value(config, "p_win_downside_weight", 0.0))
        drawdown_boost = 1.0 + max(0.0, downside_weight) * drawdown_severity
        weights = weights * drawdown_boost[:, :head_dim]

    if "bigloss" in targets and targets["bigloss"].numel() > 0:
        expanded_bigloss = _expand_bigloss_targets(targets["bigloss"].float(), head_dim)
        weights = weights * (1.0 + 1.5 * expanded_bigloss)

    return weights.clamp(1.0, 6.0)


def _primary_index(config: TrainConfig, width: int) -> int:
    if width <= 0:
        return 0
    return min(max(int(_config_value(config, "primary_horizon_index", 0)), 0), width - 1)


def _config_value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    gamma: float = 0.75,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal = (1.0 - pt).clamp(0.0, 1.0).pow(float(gamma))
    return bce * focal


def _pairwise_rank_loss(
    decision_scores: torch.Tensor,
    returns: torch.Tensor,
    date_idx: torch.Tensor | None,
    horizon_idx: int,
    min_return_gap: float = 0.003,
    max_pairs_per_date: int = 65536,
    max_matrix_items: int = 2048,
) -> torch.Tensor:
    if date_idx is None or decision_scores.numel() == 0 or returns.numel() == 0:
        return decision_scores.new_tensor(0.0)
    if decision_scores.ndim == 2:
        score = decision_scores[:, _primary_index_for_width(horizon_idx, decision_scores.size(1))]
    else:
        score = decision_scores.reshape(-1)
    ret = returns[:, _primary_index_for_width(horizon_idx, returns.size(1))] if returns.ndim == 2 else returns.reshape(-1)
    dates = date_idx.reshape(-1).long()

    valid = torch.isfinite(score) & torch.isfinite(ret)
    score = score[valid]
    ret = ret[valid]
    dates = dates[valid]
    if score.numel() < 2:
        return decision_scores.new_tensor(0.0)
    if score.numel() > max_matrix_items:
        selection = torch.randperm(score.numel(), device=score.device)[:max_matrix_items]
        score = score[selection]
        ret = ret[selection]
        dates = dates[selection]

    same_date = dates[:, None].eq(dates[None, :])
    ret_gap = ret[:, None] - ret[None, :]
    pair_mask = same_date & (ret_gap > float(min_return_gap))
    if not pair_mask.any():
        return decision_scores.new_tensor(0.0)

    score_gap = score[:, None] - score[None, :]
    pair_losses = F.softplus(-score_gap[pair_mask])
    pair_weights = ret_gap[pair_mask].clamp_min(float(min_return_gap))
    if pair_losses.numel() > max_pairs_per_date:
        selection = torch.randperm(pair_losses.numel(), device=pair_losses.device)[:max_pairs_per_date]
        pair_losses = pair_losses[selection]
        pair_weights = pair_weights[selection]
    return (pair_losses * pair_weights).mean()


def _risk_adjusted_utility_targets(targets: dict[str, torch.Tensor], target_dim: int) -> torch.Tensor:
    ret = targets["ret_mu"].float()
    width = min(target_dim, ret.size(1)) if ret.ndim == 2 else target_dim
    ret = ret[:, :width] if ret.ndim == 2 else ret.reshape(-1, 1)
    device = ret.device
    dtype = ret.dtype

    risk_dd = targets.get("risk_dd")
    drawdown = torch.zeros_like(ret)
    if risk_dd is not None and risk_dd.numel() > 0:
        risk_values = risk_dd.float()
        risk_values = risk_values[:, :width] if risk_values.ndim == 2 else risk_values.reshape(-1, 1)
        drawdown = torch.relu(-risk_values)

    bigloss = torch.zeros_like(ret)
    if "bigloss" in targets and targets["bigloss"].numel() > 0:
        bigloss = _expand_bigloss_targets(targets["bigloss"].float(), width)

    if width == len(HORIZONS):
        drawdown_weight = torch.tensor([0.6, 0.6, 0.6, 0.8, 0.8], dtype=dtype, device=device)
        bigloss_weight = torch.tensor([0.8, 0.8, 0.8, 0.6, 0.6], dtype=dtype, device=device)
        upside_weight = torch.tensor([0.2, 0.2, 0.2, 0.3, 0.3], dtype=dtype, device=device)
    else:
        drawdown_weight = torch.full((width,), 0.7, dtype=dtype, device=device)
        bigloss_weight = torch.full((width,), 0.7, dtype=dtype, device=device)
        upside_weight = torch.full((width,), 0.2, dtype=dtype, device=device)

    upside = torch.relu(ret)
    return ret - drawdown_weight.unsqueeze(0) * drawdown - bigloss_weight.unsqueeze(0) * bigloss + upside_weight.unsqueeze(0) * upside


def _multi_horizon_pairwise_rank_loss(
    decision_scores: torch.Tensor,
    utility_targets: torch.Tensor,
    date_idx: torch.Tensor | None,
) -> torch.Tensor:
    if date_idx is None or decision_scores.numel() == 0 or utility_targets.numel() == 0:
        return decision_scores.new_tensor(0.0)
    width = min(decision_scores.size(1), utility_targets.size(1))
    if width <= 0:
        return decision_scores.new_tensor(0.0)
    if width == len(HORIZONS):
        horizon_weights = torch.tensor([1.0, 1.0, 1.2, 0.7, 0.3], dtype=decision_scores.dtype, device=decision_scores.device)
    else:
        horizon_weights = torch.ones(width, dtype=decision_scores.dtype, device=decision_scores.device)

    losses = []
    weights = []
    for horizon_idx in range(width):
        loss = _pairwise_rank_loss(
            decision_scores[:, horizon_idx : horizon_idx + 1],
            utility_targets[:, horizon_idx : horizon_idx + 1],
            date_idx,
            horizon_idx=0,
        )
        losses.append(loss)
        weights.append(horizon_weights[horizon_idx])
    stacked_losses = torch.stack(losses)
    stacked_weights = torch.stack(weights)
    return (stacked_losses * stacked_weights).sum() / stacked_weights.sum().clamp_min(1e-6)


def _global_pairwise_rank_loss(
    decision_scores: torch.Tensor,
    rank_targets: torch.Tensor,
    horizon_idx: int,
    min_rank_gap: float = 0.05,
    max_pairs: int = 8192,
) -> torch.Tensor:
    if decision_scores.numel() == 0 or rank_targets.numel() == 0 or decision_scores.size(0) < 2:
        return decision_scores.new_tensor(0.0)
    score = decision_scores[:, _primary_index_for_width(horizon_idx, decision_scores.size(1))]
    rank = rank_targets.reshape(rank_targets.size(0), -1)[:, 0]
    rank_gap = rank.unsqueeze(1) - rank.unsqueeze(0)
    pair_mask = rank_gap > float(min_rank_gap)
    if not pair_mask.any():
        return decision_scores.new_tensor(0.0)
    score_gap = score.unsqueeze(1) - score.unsqueeze(0)
    pair_losses = F.softplus(-score_gap[pair_mask])
    pair_weights = rank_gap[pair_mask].abs().clamp_min(float(min_rank_gap))
    if pair_losses.numel() > max_pairs:
        selection = torch.randperm(pair_losses.numel(), device=pair_losses.device)[:max_pairs]
        pair_losses = pair_losses[selection]
        pair_weights = pair_weights[selection]
    return (pair_losses * pair_weights).mean()


def _std_floor_loss(values: torch.Tensor, target_std: float) -> torch.Tensor:
    if values.ndim < 2 or values.size(0) < 2 or values.numel() == 0:
        return values.new_tensor(0.0)
    finite_values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    std = finite_values.std(dim=0, unbiased=False)
    floor = max(float(target_std), 1e-6)
    return F.relu(floor - std).pow(2).mean() / (floor * floor)


def _prediction_entropy_loss(
    outputs: dict[str, torch.Tensor],
    decision_scores: torch.Tensor | None,
    target_std: float,
) -> torch.Tensor:
    base = outputs["p_win"].new_tensor(0.0)
    losses: list[torch.Tensor] = []
    if "p_win" in outputs:
        losses.append(_std_floor_loss(torch.sigmoid(outputs["p_win"]), target_std))
    if "ret_mu" in outputs:
        losses.append(_std_floor_loss(torch.tanh(outputs["ret_mu"].float() / 0.10), target_std))
    if decision_scores is not None:
        losses.append(_std_floor_loss(torch.sigmoid(decision_scores.float()), target_std))
    if not losses:
        return base
    return torch.stack(losses).mean()


def _expert_gate_entropy_loss(
    outputs: dict[str, torch.Tensor],
    min_entropy_ratio: float,
    max_entropy_ratio: float,
) -> torch.Tensor:
    weights = outputs.get("expert_source_weights")
    if weights is None or weights.numel() == 0:
        return outputs["p_win"].new_tensor(0.0)
    weights = weights.float().clamp_min(1e-8)
    entropy = -(weights * weights.log()).sum(dim=-1)
    max_entropy = torch.log(weights.new_tensor(float(weights.size(-1))))
    min_entropy = max_entropy * max(0.0, min(float(min_entropy_ratio), 1.0))
    max_allowed_entropy = max_entropy * max(0.0, min(float(max_entropy_ratio), 1.0))
    low_entropy_penalty = F.relu(min_entropy - entropy).pow(2)
    uniform_penalty = F.relu(entropy - max_allowed_entropy).pow(2)
    return (low_entropy_penalty + uniform_penalty).mean() / max_entropy.pow(2).clamp_min(1e-6)


def _expert_gate_prior_loss(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = outputs.get("expert_source_weights")
    prior = outputs.get("expert_source_prior")
    if weights is None or prior is None or weights.numel() == 0 or prior.numel() == 0:
        return outputs["p_win"].new_tensor(0.0)
    width = min(weights.size(0), prior.size(0))
    source_width = min(weights.size(1), prior.size(1))
    if width <= 0 or source_width <= 0:
        return outputs["p_win"].new_tensor(0.0)
    weights = weights[:width, :source_width].float().clamp_min(1e-8)
    prior = prior[:width, :source_width].to(device=weights.device, dtype=weights.dtype).clamp_min(1e-8)
    return F.kl_div(weights.log(), prior, reduction="batchmean")


def _primary_index_for_width(index: int, width: int) -> int:
    return min(max(int(index), 0), max(0, int(width) - 1))


def compute_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    config: TrainConfig,
    loss_context: LossContext,
    aux_targets: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    p_win_binary_targets = targets["p_win"].float()
    p_win_targets = p_win_binary_targets
    p_win_head_dim = int(p_win_targets.shape[1]) if p_win_targets.ndim == 2 else 1
    horizon_idx = _primary_index(config, p_win_head_dim)
    decision_scores = outputs.get("decision_score", outputs.get("rank_score"))
    label_smoothing = float(_config_value(config, "label_smoothing", 0.0))
    if label_smoothing > 0:
        smoothing = max(0.0, min(label_smoothing, 0.2))
        p_win_targets = p_win_targets * (1.0 - smoothing) + 0.5 * smoothing
    loss_p_win = _focal_bce_with_logits(
        outputs["p_win"],
        p_win_targets,
        pos_weight=loss_context.p_win_pos_weight,
        gamma=float(_config_value(config, "focal_gamma", 0.75)),
    )
    p_win_sample_weight = _downside_weight_matrix(targets, config, p_win_head_dim)
    loss_p_win = (
        loss_p_win
        * loss_context.p_win_horizon_weight.unsqueeze(0)
        * p_win_sample_weight
    ).mean()
    loss_p_win_brier = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "p_win_brier_weight", 0.0)) > 0:
        p_win_prob = torch.sigmoid(outputs["p_win"])
        loss_p_win_brier = F.mse_loss(p_win_prob, p_win_binary_targets, reduction="none")
        loss_p_win_brier = (loss_p_win_brier * loss_context.p_win_horizon_weight.unsqueeze(0)).mean()
    loss_p_win_logit_l2 = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "p_win_logit_l2_weight", 0.0)) > 0:
        loss_p_win_logit_l2 = (outputs["p_win"].float().pow(2) * loss_context.p_win_horizon_weight.unsqueeze(0)).mean()
    loss_ret_mu = F.smooth_l1_loss(outputs["ret_mu"], targets["ret_mu"], reduction="none")
    loss_ret_mu = (loss_ret_mu * loss_context.ret_mu_horizon_weight.unsqueeze(0)).mean()
    loss_ret_sigma = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "ret_sigma_weight", 0.0)) > 0 and "ret_sigma" in outputs:
        abs_error_target = torch.abs(outputs["ret_mu"].detach() - targets["ret_mu"])
        loss_ret_sigma = F.smooth_l1_loss(outputs["ret_sigma"], abs_error_target)
    loss_risk_dd = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "risk_dd_weight", 0.0)) > 0 and "risk_dd" in outputs and "risk_dd" in targets:
        loss_risk_dd = F.smooth_l1_loss(outputs["risk_dd"], targets["risk_dd"], reduction="none")
        deep_dd_weight = 1.0 + 2.0 * (targets["risk_dd"].float() < -0.05).float()
        loss_risk_dd = loss_risk_dd * deep_dd_weight
        loss_risk_dd = (loss_risk_dd * loss_context.risk_dd_horizon_weight.unsqueeze(0)).mean()
    loss_bigloss = outputs["p_win"].new_tensor(0.0)
    if (
        float(_config_value(config, "bigloss_weight", 0.0)) > 0
        and "bigloss" in outputs
        and "bigloss" in targets
        and targets["bigloss"].numel() > 0
    ):
        bigloss_targets = targets["bigloss"].float()
        loss_bigloss = _focal_bce_with_logits(
            outputs["bigloss"],
            bigloss_targets,
            gamma=float(_config_value(config, "focal_gamma", 0.75)),
        )
        severity_weight = 1.0 + 2.0 * bigloss_targets
        loss_bigloss = (loss_bigloss * severity_weight).mean()
    loss_upside = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "upside_weight", 0.0)) > 0 and "upside" in outputs:
        upside_targets = torch.relu(targets["ret_mu"].float())
        strong_up_weight = 1.0 + 2.0 * (upside_targets > 0.03).float()
        loss_upside = (F.smooth_l1_loss(outputs["upside"], upside_targets, reduction="none") * strong_up_weight).mean()
    loss_rank_score = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "rank_score_weight", 0.0)) > 0 and "rank_score" in outputs and "rank_score" in targets:
        rank_score_pred = torch.sigmoid(outputs["rank_score"])
        loss_rank_score = F.smooth_l1_loss(rank_score_pred, targets["rank_score"])
    date_idx = None
    if aux_targets and "date_idx" in aux_targets:
        date_idx = aux_targets["date_idx"]
    loss_rank_pairwise = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "rank_pairwise_weight", 0.0)) > 0 and decision_scores is not None:
        utility_targets = _risk_adjusted_utility_targets(targets, decision_scores.size(1))
        loss_rank_pairwise = _multi_horizon_pairwise_rank_loss(decision_scores, utility_targets, date_idx)
    if (
        float(_config_value(config, "rank_pairwise_weight", 0.0)) > 0
        and decision_scores is not None
        and "rank_score" in targets
        and float(loss_rank_pairwise.detach().abs().item()) < 1e-12
    ):
        loss_rank_pairwise = _global_pairwise_rank_loss(decision_scores, targets["rank_score"], horizon_idx)
    loss_bigloss_margin = outputs["p_win"].new_tensor(0.0)
    if (
        float(_config_value(config, "bigloss_margin_weight", 0.0)) > 0
        and decision_scores is not None
        and "bigloss" in targets
        and targets["bigloss"].numel() > 0
    ):
        expanded_bigloss = _expand_bigloss_targets(targets["bigloss"].float(), decision_scores.size(1))
        decision_primary = decision_scores[:, _primary_index_for_width(horizon_idx, decision_scores.size(1))]
        bigloss_primary = expanded_bigloss[:, _primary_index_for_width(horizon_idx, expanded_bigloss.size(1))]
        if bool((bigloss_primary > 0.5).any().item()):
            loss_bigloss_margin = F.relu(decision_primary[bigloss_primary > 0.5]).mean()
    loss_anti_conservative = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "anti_conservative_weight", 0.0)) > 0 and decision_scores is not None:
        decision_primary = decision_scores[:, _primary_index_for_width(horizon_idx, decision_scores.size(1))]
        ret_primary = targets["ret_mu"][:, _primary_index_for_width(horizon_idx, targets["ret_mu"].size(1))]
        strong_up = ret_primary > 0.03
        if bool(strong_up.any().item()):
            loss_anti_conservative = F.relu(0.2 - decision_primary[strong_up]).mean()
    loss_calibration = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "calibration_weight", 0.0)) > 0:
        loss_calibration = torch.abs(
            torch.sigmoid(outputs["p_win"]).mean(dim=0) - targets["p_win"].float().mean(dim=0)
        ).mean()
    loss_p_win_regime = outputs["p_win"].new_tensor(0.0)
    if (
        aux_targets
        and float(_config_value(config, "p_win_regime_weight", 0.0)) > 0
        and "p_win_regime" in outputs
        and "p_win_rate_by_date" in aux_targets
        and aux_targets["p_win_rate_by_date"].numel() > 0
    ):
        regime_targets = aux_targets["p_win_rate_by_date"].float().clamp(0.0, 1.0)
        loss_p_win_regime = F.binary_cross_entropy_with_logits(
            outputs["p_win_regime"],
            regime_targets,
        )
    loss_prediction_entropy = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "prediction_entropy_weight", 0.0)) > 0:
        loss_prediction_entropy = _prediction_entropy_loss(
            outputs,
            decision_scores,
            target_std=float(_config_value(config, "prediction_entropy_target_std", 0.06)),
        )
    loss_expert_gate_entropy = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "expert_gate_entropy_weight", 0.0)) > 0:
        loss_expert_gate_entropy = _expert_gate_entropy_loss(
            outputs,
            min_entropy_ratio=float(_config_value(config, "expert_gate_min_entropy_ratio", 0.45)),
            max_entropy_ratio=float(_config_value(config, "expert_gate_max_entropy_ratio", 0.92)),
        )
    loss_expert_gate_prior = outputs["p_win"].new_tensor(0.0)
    if float(_config_value(config, "expert_gate_prior_weight", 0.0)) > 0:
        loss_expert_gate_prior = _expert_gate_prior_loss(outputs)
    total = (
        float(_config_value(config, "p_win_weight", 1.0)) * loss_p_win
        + float(_config_value(config, "p_win_brier_weight", 0.0)) * loss_p_win_brier
        + float(_config_value(config, "p_win_logit_l2_weight", 0.0)) * loss_p_win_logit_l2
        + float(_config_value(config, "ret_mu_weight", 1.0)) * loss_ret_mu
        + float(_config_value(config, "ret_sigma_weight", 0.0)) * loss_ret_sigma
        + float(_config_value(config, "risk_dd_weight", 0.0)) * loss_risk_dd
        + float(_config_value(config, "bigloss_weight", 0.0)) * loss_bigloss
        + float(_config_value(config, "upside_weight", 0.0)) * loss_upside
        + float(_config_value(config, "rank_pairwise_weight", 0.0)) * loss_rank_pairwise
        + float(_config_value(config, "rank_score_weight", 0.0)) * loss_rank_score
        + float(_config_value(config, "bigloss_margin_weight", 0.0)) * loss_bigloss_margin
        + float(_config_value(config, "anti_conservative_weight", 0.0)) * loss_anti_conservative
        + float(_config_value(config, "calibration_weight", 0.0)) * loss_calibration
        + float(_config_value(config, "p_win_regime_weight", 0.0)) * loss_p_win_regime
        + float(_config_value(config, "prediction_entropy_weight", 0.0)) * loss_prediction_entropy
        + float(_config_value(config, "expert_gate_entropy_weight", 0.0)) * loss_expert_gate_entropy
        + float(_config_value(config, "expert_gate_prior_weight", 0.0)) * loss_expert_gate_prior
    )
    return total, {
        "loss_p_win": loss_p_win,
        "loss_p_win_brier": loss_p_win_brier,
        "loss_p_win_logit_l2": loss_p_win_logit_l2,
        "loss_ret_mu": loss_ret_mu,
        "loss_ret_sigma": loss_ret_sigma,
        "loss_risk_dd": loss_risk_dd,
        "loss_bigloss": loss_bigloss,
        "loss_upside": loss_upside,
        "loss_rank_pairwise": loss_rank_pairwise,
        "loss_rank_score": loss_rank_score,
        "loss_bigloss_margin": loss_bigloss_margin,
        "loss_anti_conservative": loss_anti_conservative,
        "loss_calibration": loss_calibration,
        "loss_p_win_regime": loss_p_win_regime,
        "loss_prediction_entropy": loss_prediction_entropy,
        "loss_expert_gate_entropy": loss_expert_gate_entropy,
        "loss_expert_gate_prior": loss_expert_gate_prior,
    }


def _float_loss_tensors(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: value.float() if torch.is_tensor(value) and value.is_floating_point() else value
        for name, value in values.items()
    }


def _finite_tensor_dict(values: dict[str, torch.Tensor], names: tuple[str, ...]) -> tuple[bool, str]:
    for name in names:
        value = values.get(name)
        if value is not None and torch.is_tensor(value) and value.is_floating_point():
            if not torch.isfinite(value).all():
                return False, name
    return True, ""


def _first_nonfinite_parameter(model: nn.Module) -> str | None:
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and not torch.isfinite(parameter).all():
            return name
    return None


def _optimizer_step_with_finite_grads(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    grad_clip: float,
) -> tuple[bool, float]:
    try:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
            error_if_nonfinite=True,
        )
    except RuntimeError as exc:
        if "non-finite" not in str(exc).lower():
            raise
        optimizer.zero_grad(set_to_none=True)
        scaler.update()
        return False, float("nan")

    grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
    if not np.isfinite(grad_norm_value):
        optimizer.zero_grad(set_to_none=True)
        scaler.update()
        return False, grad_norm_value

    scaler.step(optimizer)
    scaler.update()
    return True, grad_norm_value


def _safe_tensor_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    safe_denominator = denominator.clamp_min(1.0)
    ratio = numerator / safe_denominator
    ratio = torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
    zero_mask = denominator <= 0
    if zero_mask.any():
        ratio = torch.where(zero_mask, torch.zeros_like(ratio), ratio)
    return ratio


def _compute_topk_metrics_from_chunks(running: dict[str, Any]) -> dict[str, Any]:
    if not running["decision_score_chunks"] or not running["date_idx_chunks"]:
        return {
            "topk_avg_ret_by_head": [],
            "topk_median_ret_by_head": [],
            "topk_win_rate_by_head": [],
            "topk_bigloss_rate_by_head": [],
            "topk_avg_drawdown_by_head": [],
            "topk_sharpe_like_by_head": [],
            "topk_utility_by_head": [],
            "topk_utility": 0.0,
            "business_score": 0.0,
        }

    decision = torch.cat(running["decision_score_chunks"], dim=0).float()
    ret = torch.cat(running["ret_mu_target_chunks"], dim=0).float()
    risk_dd = torch.cat(running["risk_dd_target_chunks"], dim=0).float()
    dates = torch.cat(running["date_idx_chunks"], dim=0).long().reshape(-1)
    if running["bigloss_target_chunks"]:
        bigloss = torch.cat(running["bigloss_target_chunks"], dim=0).float()
        bigloss = _expand_bigloss_targets(bigloss, decision.size(1))
    else:
        bigloss = torch.zeros((decision.size(0), decision.size(1)), dtype=torch.float32)

    head_count = min(decision.size(1), ret.size(1), risk_dd.size(1))
    if head_count <= 0:
        return {
            "topk_avg_ret_by_head": [],
            "topk_median_ret_by_head": [],
            "topk_win_rate_by_head": [],
            "topk_bigloss_rate_by_head": [],
            "topk_avg_drawdown_by_head": [],
            "topk_sharpe_like_by_head": [],
            "topk_utility_by_head": [],
            "topk_utility": 0.0,
            "business_score": 0.0,
        }

    avg_ret: list[float] = []
    median_ret: list[float] = []
    win_rate: list[float] = []
    bigloss_rate: list[float] = []
    avg_drawdown: list[float] = []
    sharpe_like: list[float] = []
    utility: list[float] = []
    k = max(1, int(running.get("topk", 10)))
    unique_dates = torch.unique(dates)
    for head_idx in range(head_count):
        selected_returns: list[torch.Tensor] = []
        selected_drawdowns: list[torch.Tensor] = []
        selected_bigloss: list[torch.Tensor] = []
        for date in unique_dates:
            mask = dates == date
            if not bool(mask.any().item()):
                continue
            date_scores = decision[mask, head_idx]
            selected_count = min(k, int(date_scores.numel()))
            top_indices = torch.topk(date_scores, k=selected_count).indices
            selected_returns.append(ret[mask, head_idx][top_indices])
            selected_drawdowns.append(risk_dd[mask, head_idx][top_indices])
            selected_bigloss.append(bigloss[mask, head_idx][top_indices])
        if not selected_returns:
            avg_ret.append(0.0)
            median_ret.append(0.0)
            win_rate.append(0.0)
            bigloss_rate.append(0.0)
            avg_drawdown.append(0.0)
            sharpe_like.append(0.0)
            utility.append(0.0)
            continue
        ret_values = torch.cat(selected_returns)
        dd_values = torch.cat(selected_drawdowns)
        bigloss_values = torch.cat(selected_bigloss)
        ret_mean = float(ret_values.mean().item())
        ret_std = float(ret_values.std(unbiased=False).clamp_min(1e-6).item())
        dd_mean = float(dd_values.mean().item())
        bigloss_mean = float(bigloss_values.mean().item())
        avg_ret.append(ret_mean)
        median_ret.append(float(ret_values.median().item()))
        win_rate.append(float((ret_values > 0).float().mean().item()))
        bigloss_rate.append(bigloss_mean)
        avg_drawdown.append(dd_mean)
        sharpe_like.append(ret_mean / ret_std)
        utility.append(ret_mean - 0.6 * bigloss_mean - 0.4 * abs(dd_mean))

    primary_idx = _primary_index_for_width(int(running.get("primary_horizon_index", 2)), len(utility))
    return {
        "topk_avg_ret_by_head": avg_ret,
        "topk_median_ret_by_head": median_ret,
        "topk_win_rate_by_head": win_rate,
        "topk_bigloss_rate_by_head": bigloss_rate,
        "topk_avg_drawdown_by_head": avg_drawdown,
        "topk_sharpe_like_by_head": sharpe_like,
        "topk_utility_by_head": utility,
        "topk_utility": float(utility[primary_idx]) if utility else 0.0,
        "business_score": float(utility[primary_idx]) if utility else 0.0,
    }


def _init_running_totals(
    p_win_head_dim: int,
    ret_mu_head_dim: int,
    risk_dd_head_dim: int,
    topk: int = 10,
    primary_horizon_index: int = 2,
) -> dict[str, Any]:
    return {
        "loss_sum": 0.0,
        "sample_count": 0.0,
        "skipped_batches": 0.0,
        "grad_norm_sum": 0.0,
        "grad_norm_count": 0.0,
        "topk": int(topk),
        "primary_horizon_index": int(primary_horizon_index),
        "p_win_correct": 0.0,
        "p_win_total": 0.0,
        "ret_mu_abs": 0.0,
        "ret_mu_total": 0.0,
        "risk_dd_abs": 0.0,
        "risk_dd_total": 0.0,
        "rank_score_abs": 0.0,
        "rank_score_total": 0.0,
        **{f"{name}_sum": 0.0 for name in LOSS_PART_NAMES},
        "p_win_correct_by_head": torch.zeros(p_win_head_dim, dtype=torch.float64),
        "p_win_total_by_head": torch.zeros(p_win_head_dim, dtype=torch.float64),
        "p_win_true_pos_by_head": torch.zeros(p_win_head_dim, dtype=torch.float64),
        "p_win_true_neg_by_head": torch.zeros(p_win_head_dim, dtype=torch.float64),
        "p_win_pred_pos_by_head": torch.zeros(p_win_head_dim, dtype=torch.float64),
        "p_win_true_positive_by_head": torch.zeros(p_win_head_dim, dtype=torch.float64),
        "p_win_true_negative_by_head": torch.zeros(p_win_head_dim, dtype=torch.float64),
        "p_win_prob_sum_by_head": torch.zeros(p_win_head_dim, dtype=torch.float64),
        "p_win_logit_sum_by_head": torch.zeros(p_win_head_dim, dtype=torch.float64),
        "p_win_logit_sq_sum_by_head": torch.zeros(p_win_head_dim, dtype=torch.float64),
        "ret_mu_abs_by_head": torch.zeros(ret_mu_head_dim, dtype=torch.float64),
        "ret_mu_total_by_head": torch.zeros(ret_mu_head_dim, dtype=torch.float64),
        "risk_dd_abs_by_head": torch.zeros(risk_dd_head_dim, dtype=torch.float64),
        "risk_dd_total_by_head": torch.zeros(risk_dd_head_dim, dtype=torch.float64),
        "decision_score_chunks": [],
        "ret_mu_target_chunks": [],
        "risk_dd_target_chunks": [],
        "bigloss_target_chunks": [],
        "date_idx_chunks": [],
    }


def _finalize_metrics(running: dict[str, Any]) -> dict[str, Any]:
    p_win_acc_by_head = _safe_tensor_ratio(running["p_win_correct_by_head"], running["p_win_total_by_head"])
    p_win_target_pos_rate_by_head = _safe_tensor_ratio(
        running["p_win_true_pos_by_head"],
        running["p_win_total_by_head"],
    )
    p_win_pred_pos_rate_by_head = _safe_tensor_ratio(
        running["p_win_pred_pos_by_head"],
        running["p_win_total_by_head"],
    )
    p_win_prob_mean_by_head = _safe_tensor_ratio(
        running["p_win_prob_sum_by_head"],
        running["p_win_total_by_head"],
    )
    p_win_recall_pos_by_head = _safe_tensor_ratio(
        running["p_win_true_positive_by_head"],
        running["p_win_true_pos_by_head"],
    )
    p_win_recall_neg_by_head = _safe_tensor_ratio(
        running["p_win_true_negative_by_head"],
        running["p_win_true_neg_by_head"],
    )
    p_win_bal_acc_by_head = (p_win_recall_pos_by_head + p_win_recall_neg_by_head) / 2.0
    p_win_majority_baseline_by_head = torch.maximum(
        p_win_target_pos_rate_by_head,
        1.0 - p_win_target_pos_rate_by_head,
    )
    p_win_logit_mean_by_head = _safe_tensor_ratio(
        running["p_win_logit_sum_by_head"],
        running["p_win_total_by_head"],
    )
    p_win_logit_sq_mean_by_head = _safe_tensor_ratio(
        running["p_win_logit_sq_sum_by_head"],
        running["p_win_total_by_head"],
    )
    p_win_logit_std_by_head = (p_win_logit_sq_mean_by_head - p_win_logit_mean_by_head.pow(2)).clamp_min(0.0).sqrt()
    ret_mu_mae_by_head = _safe_tensor_ratio(running["ret_mu_abs_by_head"], running["ret_mu_total_by_head"])
    risk_dd_mae_by_head = _safe_tensor_ratio(running["risk_dd_abs_by_head"], running["risk_dd_total_by_head"])
    topk_metrics = _compute_topk_metrics_from_chunks(running)
    primary_idx = _primary_index_for_width(int(running.get("primary_horizon_index", 2)), len(p_win_bal_acc_by_head))
    primary_bal_acc = float(p_win_bal_acc_by_head[primary_idx].item()) if len(p_win_bal_acc_by_head) else 0.0
    primary_majority_baseline = (
        float(p_win_majority_baseline_by_head[primary_idx].item()) if len(p_win_majority_baseline_by_head) else 0.0
    )
    topk_metrics["business_score"] = float(topk_metrics.get("topk_utility", 0.0)) + 0.1 * primary_bal_acc
    metrics = {
        "loss": running["loss_sum"] / max(1.0, running["sample_count"]),
        "p_win_acc": running["p_win_correct"] / max(1.0, running["p_win_total"]),
        "p_win_bal_acc": primary_bal_acc,
        "p_win_majority_baseline": primary_majority_baseline,
        "ret_mu_mae": running["ret_mu_abs"] / max(1.0, running["ret_mu_total"]),
        "risk_dd_mae": running["risk_dd_abs"] / max(1.0, running["risk_dd_total"]),
        "rank_score_mae": running["rank_score_abs"] / max(1.0, running["rank_score_total"]),
        "skipped_batches": running.get("skipped_batches", 0.0),
        "grad_norm": running.get("grad_norm_sum", 0.0) / max(1.0, running.get("grad_norm_count", 0.0)),
        "p_win_acc_by_head": [float(value) for value in p_win_acc_by_head.tolist()],
        "p_win_bal_acc_by_head": [float(value) for value in p_win_bal_acc_by_head.tolist()],
        "p_win_target_pos_rate_by_head": [float(value) for value in p_win_target_pos_rate_by_head.tolist()],
        "p_win_pred_pos_rate_by_head": [float(value) for value in p_win_pred_pos_rate_by_head.tolist()],
        "p_win_prob_mean_by_head": [float(value) for value in p_win_prob_mean_by_head.tolist()],
        "p_win_logit_mean_by_head": [float(value) for value in p_win_logit_mean_by_head.tolist()],
        "p_win_logit_std_by_head": [float(value) for value in p_win_logit_std_by_head.tolist()],
        "p_win_majority_baseline_by_head": [float(value) for value in p_win_majority_baseline_by_head.tolist()],
        "ret_mu_mae_by_head": [float(value) for value in ret_mu_mae_by_head.tolist()],
        "risk_dd_mae_by_head": [float(value) for value in risk_dd_mae_by_head.tolist()],
    }
    metrics.update(
        {
            name: running[f"{name}_sum"] / max(1.0, running["sample_count"])
            for name in LOSS_PART_NAMES
        }
    )
    metrics.update(topk_metrics)
    return metrics


def flatten_metrics_for_history(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    scalar_keys = (
        "loss",
        "p_win_acc",
        "p_win_bal_acc",
        "p_win_majority_baseline",
        "ret_mu_mae",
        "risk_dd_mae",
        "rank_score_mae",
        "skipped_batches",
        "grad_norm",
        "topk_utility",
        "business_score",
        *LOSS_PART_NAMES,
    )
    for key in scalar_keys:
        if key in metrics:
            flattened[f"{prefix}_{key}"] = float(metrics[key])

    p_win_labels = resolve_horizon_labels(len(metrics.get("p_win_acc_by_head", [])))
    ret_labels = resolve_horizon_labels(len(metrics.get("ret_mu_mae_by_head", [])))
    risk_labels = resolve_horizon_labels(len(metrics.get("risk_dd_mae_by_head", [])))

    for label, value in zip(p_win_labels, metrics.get("p_win_acc_by_head", [])):
        flattened[f"{prefix}_p_win_acc_{label}"] = float(value)
    for label, value in zip(p_win_labels, metrics.get("p_win_bal_acc_by_head", [])):
        flattened[f"{prefix}_p_win_bal_acc_{label}"] = float(value)
    for label, value in zip(p_win_labels, metrics.get("p_win_target_pos_rate_by_head", [])):
        flattened[f"{prefix}_p_win_target_pos_rate_{label}"] = float(value)
    for label, value in zip(p_win_labels, metrics.get("p_win_pred_pos_rate_by_head", [])):
        flattened[f"{prefix}_p_win_pred_pos_rate_{label}"] = float(value)
    for label, value in zip(p_win_labels, metrics.get("p_win_prob_mean_by_head", [])):
        flattened[f"{prefix}_p_win_prob_mean_{label}"] = float(value)
    for label, value in zip(p_win_labels, metrics.get("p_win_logit_mean_by_head", [])):
        flattened[f"{prefix}_p_win_logit_mean_{label}"] = float(value)
    for label, value in zip(p_win_labels, metrics.get("p_win_logit_std_by_head", [])):
        flattened[f"{prefix}_p_win_logit_std_{label}"] = float(value)
    for label, value in zip(p_win_labels, metrics.get("p_win_majority_baseline_by_head", [])):
        flattened[f"{prefix}_p_win_majority_baseline_{label}"] = float(value)
    for label, value in zip(ret_labels, metrics.get("ret_mu_mae_by_head", [])):
        flattened[f"{prefix}_ret_mu_mae_{label}"] = float(value)
    for label, value in zip(risk_labels, metrics.get("risk_dd_mae_by_head", [])):
        flattened[f"{prefix}_risk_dd_mae_{label}"] = float(value)
    topk_labels = resolve_horizon_labels(len(metrics.get("topk_utility_by_head", [])))
    for label, value in zip(topk_labels, metrics.get("topk_avg_ret_by_head", [])):
        flattened[f"{prefix}_topk_avg_ret_{label}"] = float(value)
    for label, value in zip(topk_labels, metrics.get("topk_bigloss_rate_by_head", [])):
        flattened[f"{prefix}_topk_bigloss_rate_{label}"] = float(value)
    for label, value in zip(topk_labels, metrics.get("topk_avg_drawdown_by_head", [])):
        flattened[f"{prefix}_topk_avg_drawdown_{label}"] = float(value)
    for label, value in zip(topk_labels, metrics.get("topk_utility_by_head", [])):
        flattened[f"{prefix}_topk_utility_{label}"] = float(value)
    return flattened


def print_metric_block(split_name: str, metrics: dict[str, Any]) -> None:
    p_win_labels = resolve_horizon_labels(len(metrics.get("p_win_acc_by_head", [])))
    if p_win_labels:
        acc_parts = " ".join(
            f"{label}:{value:.3f}" for label, value in zip(p_win_labels, metrics["p_win_acc_by_head"])
        )
        bal_parts = " ".join(
            f"{label}:{value:.3f}" for label, value in zip(p_win_labels, metrics["p_win_bal_acc_by_head"])
        )
        pos_parts = " ".join(
            (
                f"{label}:tgt={target:.3f}/pred={pred:.3f}/base={baseline:.3f}"
            )
            for label, target, pred, baseline in zip(
                p_win_labels,
                metrics["p_win_target_pos_rate_by_head"],
                metrics["p_win_pred_pos_rate_by_head"],
                metrics["p_win_majority_baseline_by_head"],
            )
        )
        print(f"  {split_name} p_win_acc_by_horizon: {acc_parts}")
        print(f"  {split_name} p_win_bal_acc_by_horizon: {bal_parts}")
        print(f"  {split_name} p_win_pos_rate_by_horizon: {pos_parts}")

    ret_labels = resolve_horizon_labels(len(metrics.get("ret_mu_mae_by_head", [])))
    if ret_labels:
        ret_parts = " ".join(
            f"{label}:{value:.4f}" for label, value in zip(ret_labels, metrics["ret_mu_mae_by_head"])
        )
        print(f"  {split_name} ret_mu_mae_by_horizon: {ret_parts}")

    risk_labels = resolve_horizon_labels(len(metrics.get("risk_dd_mae_by_head", [])))
    if risk_labels:
        risk_parts = " ".join(
            f"{label}:{value:.4f}" for label, value in zip(risk_labels, metrics["risk_dd_mae_by_head"])
        )
        print(f"  {split_name} risk_dd_mae_by_horizon: {risk_parts}")

    topk_labels = resolve_horizon_labels(len(metrics.get("topk_utility_by_head", [])))
    if topk_labels:
        topk_parts = " ".join(
            (
                f"{label}:ret={ret:.4f}/big={big:.3f}/dd={dd:.4f}/util={util:.4f}"
            )
            for label, ret, big, dd, util in zip(
                topk_labels,
                metrics["topk_avg_ret_by_head"],
                metrics["topk_bigloss_rate_by_head"],
                metrics["topk_avg_drawdown_by_head"],
                metrics["topk_utility_by_head"],
            )
        )
        print(f"  {split_name} topk_business_by_horizon: {topk_parts}")


def _update_metrics_from_batch(
    running: dict[str, Any],
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    loss: torch.Tensor,
    parts: dict[str, torch.Tensor],
    p_win_thresholds: torch.Tensor,
) -> None:
    """Update running metric accumulators with results from a single batch.

    Shared between train_one_epoch and evaluate to eliminate duplicated metric tracking code.
    """
    batch_size = float(batch["X_seq"].size(0))
    running["loss_sum"] += float(loss.detach().item()) * batch_size
    running["sample_count"] += batch_size
    for name in LOSS_PART_NAMES:
        running[f"{name}_sum"] += float(parts[name].detach().item()) * batch_size

    p_win_logits = outputs["p_win"].detach().float()
    p_win_prob = torch.sigmoid(p_win_logits)
    p_win_pred = (p_win_prob >= p_win_thresholds.unsqueeze(0)).float()
    p_win_true = batch["y"]["p_win"]
    p_win_true_binary = (p_win_true >= 0.5).float()
    p_win_pred_binary = (p_win_pred >= 0.5).float()
    running["p_win_correct"] += float((p_win_pred == p_win_true).sum().item())
    running["p_win_total"] += float(p_win_true.numel())
    running["p_win_correct_by_head"] += (p_win_pred_binary == p_win_true_binary).sum(dim=0).double().cpu()
    running["p_win_total_by_head"] += torch.full(
        (p_win_true.size(1),),
        float(p_win_true.size(0)),
        dtype=torch.float64,
    )
    running["p_win_true_pos_by_head"] += p_win_true_binary.sum(dim=0).double().cpu()
    running["p_win_true_neg_by_head"] += (1.0 - p_win_true_binary).sum(dim=0).double().cpu()
    running["p_win_pred_pos_by_head"] += p_win_pred_binary.sum(dim=0).double().cpu()
    running["p_win_true_positive_by_head"] += (
        (p_win_pred_binary == 1.0) & (p_win_true_binary == 1.0)
    ).sum(dim=0).double().cpu()
    running["p_win_true_negative_by_head"] += (
        (p_win_pred_binary == 0.0) & (p_win_true_binary == 0.0)
    ).sum(dim=0).double().cpu()
    running["p_win_prob_sum_by_head"] += p_win_prob.sum(dim=0).double().cpu()
    running["p_win_logit_sum_by_head"] += p_win_logits.sum(dim=0).double().cpu()
    running["p_win_logit_sq_sum_by_head"] += p_win_logits.pow(2).sum(dim=0).double().cpu()
    running["ret_mu_abs"] += float(torch.abs(outputs["ret_mu"].detach() - batch["y"]["ret_mu"]).sum().item())
    running["ret_mu_total"] += float(batch["y"]["ret_mu"].numel())
    running["ret_mu_abs_by_head"] += torch.abs(
        outputs["ret_mu"].detach() - batch["y"]["ret_mu"]
    ).sum(dim=0).double().cpu()
    running["ret_mu_total_by_head"] += torch.full(
        (batch["y"]["ret_mu"].size(1),),
        float(batch["y"]["ret_mu"].size(0)),
        dtype=torch.float64,
    )
    running["risk_dd_abs"] += float(torch.abs(outputs["risk_dd"].detach() - batch["y"]["risk_dd"]).sum().item())
    running["risk_dd_total"] += float(batch["y"]["risk_dd"].numel())
    running["risk_dd_abs_by_head"] += torch.abs(
        outputs["risk_dd"].detach() - batch["y"]["risk_dd"]
    ).sum(dim=0).double().cpu()
    running["risk_dd_total_by_head"] += torch.full(
        (batch["y"]["risk_dd"].size(1),),
        float(batch["y"]["risk_dd"].size(0)),
        dtype=torch.float64,
    )
    rank_score_pred = torch.sigmoid(outputs["rank_score"].detach())
    running["rank_score_abs"] += float(torch.abs(rank_score_pred - batch["y"]["rank_score"]).sum().item())
    running["rank_score_total"] += float(batch["y"]["rank_score"].numel())
    running["decision_score_chunks"].append(outputs["decision_score"].detach().float().cpu())
    running["ret_mu_target_chunks"].append(batch["y"]["ret_mu"].detach().float().cpu())
    running["risk_dd_target_chunks"].append(batch["y"]["risk_dd"].detach().float().cpu())
    if "bigloss" in batch["y"]:
        running["bigloss_target_chunks"].append(batch["y"]["bigloss"].detach().float().cpu())
    if "aux" in batch and "date_idx" in batch["aux"]:
        running["date_idx_chunks"].append(batch["aux"]["date_idx"].detach().long().cpu())


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
    loss_context: LossContext,
    ema: ModelEMA | None = None,
) -> dict[str, Any]:
    model.train()
    running = _init_running_totals(
        p_win_head_dim=int(loss_context.p_win_pos_weight.numel()),
        ret_mu_head_dim=int(loss_context.ret_mu_horizon_weight.numel()),
        risk_dd_head_dim=int(loss_context.risk_dd_horizon_weight.numel()),
        topk=config.topk,
        primary_horizon_index=config.primary_horizon_index,
    )
    progress = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False, disable=config.disable_tqdm)
    autocast_enabled, autocast_dtype = autocast_config(device, config)
    p_win_thresholds = resolve_p_win_thresholds(device, loss_context.p_win_pos_weight.numel())

    for step, batch in enumerate(progress, start=1):
        if config.max_train_batches is not None and step > config.max_train_batches:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            outputs = model(batch)
        outputs_for_loss = _float_loss_tensors(outputs)
        finite_outputs, bad_output = _finite_tensor_dict(
            outputs_for_loss,
            ("p_win", "ret_mu", "risk_dd", "bigloss", "rank_score", "decision_score"),
        )
        if not finite_outputs:
            running["skipped_batches"] += 1.0
            progress.set_postfix(skip=f"nonfinite:{bad_output}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            continue

        loss, parts = compute_losses(
            outputs_for_loss,
            _float_loss_tensors(batch["y"]),
            config,
            loss_context,
            aux_targets=batch.get("aux"),
        )
        if not torch.isfinite(loss):
            running["skipped_batches"] += 1.0
            optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(skip="nonfinite:loss", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            continue

        scaler.scale(loss).backward()
        step_applied, grad_norm = _optimizer_step_with_finite_grads(model, optimizer, scaler, config.grad_clip)
        if not step_applied:
            running["skipped_batches"] += 1.0
            progress.set_postfix(skip="nonfinite:grad", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            continue
        running["grad_norm_sum"] += grad_norm
        running["grad_norm_count"] += 1.0

        bad_parameter = _first_nonfinite_parameter(model)
        if bad_parameter is not None:
            raise RuntimeError(f"Non-finite model parameter after optimizer step: {bad_parameter}")

        if ema is not None:
            ema.update(model)

        _update_metrics_from_batch(running, batch, outputs_for_loss, loss, parts, p_win_thresholds)

        metrics = _finalize_metrics(running)
        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            pwin=f"{metrics['loss_p_win']:.3f}",
            ret=f"{metrics['loss_ret_mu']:.3f}",
            dd=f"{metrics['loss_risk_dd']:.3f}",
            big=f"{metrics['loss_bigloss']:.3f}",
            rank=f"{metrics['loss_rank_pairwise']:.3f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
    return _finalize_metrics(running)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainConfig,
    loss_context: LossContext,
    desc: str,
    p_win_thresholds: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    running = _init_running_totals(
        p_win_head_dim=int(loss_context.p_win_pos_weight.numel()),
        ret_mu_head_dim=int(loss_context.ret_mu_horizon_weight.numel()),
        risk_dd_head_dim=int(loss_context.risk_dd_horizon_weight.numel()),
        topk=config.topk,
        primary_horizon_index=config.primary_horizon_index,
    )
    progress = tqdm(loader, desc=desc, leave=False, disable=config.disable_tqdm)
    autocast_enabled, autocast_dtype = autocast_config(device, config)
    resolved_thresholds = resolve_p_win_thresholds(device, loss_context.p_win_pos_weight.numel(), p_win_thresholds)

    for step, batch in enumerate(progress, start=1):
        if config.max_eval_batches is not None and step > config.max_eval_batches:
            break
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            outputs = model(batch)
            loss, parts = compute_losses(outputs, batch["y"], config, loss_context, aux_targets=batch.get("aux"))

        _update_metrics_from_batch(running, batch, outputs, loss, parts, resolved_thresholds)

        metrics = _finalize_metrics(running)
        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            util=f"{metrics['topk_utility']:.4f}",
            acc=f"{metrics['p_win_acc']:.3f}",
        )
    return _finalize_metrics(running)


def evaluate_limited(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainConfig,
    loss_context: LossContext,
    desc: str,
    max_batches: int,
) -> dict[str, Any]:
    original_max_eval_batches = config.max_eval_batches
    config.max_eval_batches = max(1, int(max_batches))
    try:
        return evaluate(model, loader, device, config, loss_context, desc=desc)
    finally:
        config.max_eval_batches = original_max_eval_batches


def checkpoint_payload(
    model: nn.Module,
    bundle: DataBundle,
    config: TrainConfig,
    epoch: int,
    best_valid_loss: float,
    best_metric_name: str,
    best_metric_value: float,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    ema: ModelEMA | None = None,
    validation_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model_state": model.state_dict(),
        "input_dims": bundle.train_split.input_dims,
        "vocab_sizes": bundle.train_split.vocab_sizes,
        "head_dims": bundle.train_split.head_dims,
        "model_config": build_model_kwargs(config),
        "normalizer_state": bundle.normalizer.state_dict(),
        "config": asdict(config),
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "validation_summary": validation_summary or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if ema is not None:
        payload["ema_state"] = ema.shadow.state_dict()
    return payload


@torch.no_grad()
def collect_p_win_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: TrainConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    autocast_enabled, autocast_dtype = autocast_config(device, config)
    for step, batch in enumerate(loader, start=1):
        if config.max_eval_batches is not None and step > config.max_eval_batches:
            break
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_enabled):
            outputs = model(batch)
        probs.append(torch.sigmoid(outputs["p_win"].detach()).cpu())
        targets.append(batch["y"]["p_win"].detach().cpu())
    if not probs:
        return torch.empty((0, 0)), torch.empty((0, 0))
    return torch.cat(probs, dim=0), torch.cat(targets, dim=0)


def calibrate_p_win_thresholds(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    min_threshold: float = 0.2,
    max_threshold: float = 0.8,
    steps: int = 25,
) -> torch.Tensor:
    if probabilities.numel() == 0 or targets.numel() == 0:
        head_dim = int(targets.shape[1]) if targets.ndim == 2 else 1
        return torch.full((head_dim,), 0.5, dtype=torch.float32)

    candidates = torch.linspace(min_threshold, max_threshold, steps=steps, dtype=torch.float32)
    best_thresholds: list[torch.Tensor] = []
    for column_idx in range(probabilities.shape[1]):
        column_probs = probabilities[:, column_idx].unsqueeze(1)
        column_targets = targets[:, column_idx].unsqueeze(1)
        candidate_acc = ((column_probs >= candidates.unsqueeze(0)) == column_targets).float().mean(dim=0)
        best_thresholds.append(candidates[int(candidate_acc.argmax().item())])
    return torch.stack(best_thresholds)


def build_model(bundle: DataBundle, config: TrainConfig, device: torch.device) -> TinyMultiInputModel:
    model = TinyMultiInputModel(
        input_dims=bundle.train_split.input_dims,
        vocab_sizes=bundle.train_split.vocab_sizes,
        head_dims=bundle.train_split.head_dims,
        **build_model_kwargs(config),
    )
    return model.to(device)


def print_split_summary(bundle: DataBundle, config: TrainConfig) -> None:
    print("Prepared datasets:")
    print(
        f"  train rows={bundle.train_split.row_count:,} samples={bundle.train_split.sample_count:,} "
        f"valid rows={bundle.valid_split.row_count:,} samples={bundle.valid_split.sample_count:,}"
    )
    print(
        f"  dims: seq={bundle.train_split.input_dims['f_seq']} "
        f"tab={bundle.train_split.input_dims['f_tab']} "
        f"event={bundle.train_split.input_dims['f_event']} "
        f"mkt={bundle.train_split.input_dims['f_mkt']} "
        f"profile={bundle.train_split.input_dims['f_company_profile']}"
    )
    print(f"  device={config.device} batch_size={config.batch_size} epochs={config.epochs}")
    print(
        f"  model: seq_model_dim={config.seq_model_dim} seq_output_dim={config.seq_output_dim} "
        f"fusion_dim={config.fusion_dim} task_hidden_dim={config.task_hidden_dim} "
        f"horizon_expert_dim={config.horizon_expert_dim}"
    )


def print_data_contract_warnings(bundle: DataBundle, config: TrainConfig) -> None:
    summary_path = (PROJECT_ROOT / config.valid_path).parent / "split_summary.json"
    if summary_path.exists():
        try:
            split_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            split_summary = {}
        export_seq_length = split_summary.get("seq_length")
        if export_seq_length is not None and int(export_seq_length) != int(config.seq_length):
            print(
                "WARNING: training --seq-length does not match exported split_summary seq_length "
                f"({config.seq_length} vs {export_seq_length}). Re-export the split or use --seq-length {export_seq_length} "
                "for a cleaner validation window."
            )
    if bundle.valid_split.sample_count <= 0:
        raise ValueError("validation split produced zero samples; reduce --seq-length or rebuild exports.")
    sample_ratio = bundle.valid_split.sample_count / max(1, bundle.valid_split.row_count)
    if sample_ratio < 0.10:
        print(
            "WARNING: validation sample_count is very small relative to rows "
            f"({bundle.valid_split.sample_count:,}/{bundle.valid_split.row_count:,}). "
            "Validation metrics may be dominated by only the last few dates."
        )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    configure_runtime(config.device)

    output_dir = ensure_dir(PROJECT_ROOT / config.output_dir)
    cache_dir = ensure_dir(PROJECT_ROOT / config.cache_dir)
    save_json(output_dir / "config.json", asdict(config))

    device = torch.device(config.device)
    print("Preparing train/valid splits and loaders...")
    bundle = build_data_bundle(
        train_path=PROJECT_ROOT / config.train_path,
        valid_path=PROJECT_ROOT / config.valid_path,
        cache_dir=cache_dir,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        rebuild_cache=config.rebuild_cache,
        device_type=device.type,
        use_weighted_sampler=config.use_weighted_sampler,
        sampler_recency_power=config.sampler_recency_power,
        sampler_positive_balance_power=config.sampler_positive_balance_power,
        sampler_balance_horizon_index=config.sampler_balance_horizon_index,
        pre_normalize_features=config.pre_normalize_features,
    )
    print_split_summary(bundle, config)
    print_data_contract_warnings(bundle, config)
    loss_context = build_loss_context(bundle, device)

    model = build_model(bundle, config, device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(device.type, enabled=bool(config.use_amp and device.type == "cuda"))
    ema = ModelEMA(model, config.ema_decay) if 0.0 < config.ema_decay < 1.0 else None

    history: list[dict[str, float]] = []
    best_valid_loss = float("inf")
    best_metric_name = str(config.best_model_metric)
    max_metric_names = {"valid_p_win_acc", "valid_p_win_bal_acc", "valid_business_score", "valid_topk_utility"}
    if best_metric_name in max_metric_names:
        best_metric_value = float("-inf")
    else:
        best_metric_value = float("inf")
    best_epoch = 0
    patience = 0
    checkpoint_path = output_dir / "best_model.pt"

    training_started_at = time.time()
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            bundle.train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            config,
            loss_context,
            ema=ema,
        )
        valid_live_metrics = evaluate(
            model,
            bundle.valid_loader,
            device,
            config,
            loss_context,
            desc=f"Epoch {epoch} [valid live]",
        )
        if ema is not None:
            valid_ema_metrics = evaluate(
                ema.shadow,
                bundle.valid_loader,
                device,
                config,
                loss_context,
                desc=f"Epoch {epoch} [valid EMA]",
            )
        else:
            valid_ema_metrics = valid_live_metrics
        valid_metrics = valid_ema_metrics
        eval_model = ema.shadow if ema is not None else model
        train_live_eval_metrics: dict[str, Any] = {}
        if config.diagnostic_eval_batches > 0:
            train_live_eval_metrics = evaluate_limited(
                model,
                bundle.train_loader,
                device,
                config,
                loss_context,
                desc=f"Epoch {epoch} [train live eval]",
                max_batches=config.diagnostic_eval_batches,
            )
        scheduler.step()

        record = {
            "epoch": float(epoch),
            "lr": optimizer.param_groups[0]["lr"],
            **flatten_metrics_for_history("train", train_metrics),
            **flatten_metrics_for_history("train_live_eval", train_live_eval_metrics),
            **flatten_metrics_for_history("valid_live", valid_live_metrics),
            **flatten_metrics_for_history("valid_ema", valid_ema_metrics),
            **flatten_metrics_for_history("valid", valid_metrics),
        }
        record["generalization_loss_gap"] = float(valid_metrics["loss"] - train_metrics["loss"])
        record["generalization_p_win_acc_gap"] = float(train_metrics["p_win_acc"] - valid_metrics["p_win_acc"])
        record["generalization_p_win_bal_acc_gap"] = float(
            train_metrics["p_win_bal_acc"] - valid_metrics["p_win_bal_acc"]
        )
        history.append(record)
        save_history_csv(output_dir / "history.csv", history)
        save_json(output_dir / "history.json", {"epochs": history})
        plot_training_history(history, output_dir / "training_curves.png")
        plot_business_history(history, output_dir / "business_curves.png")
        plot_topk_metrics(history, output_dir / "topk_metrics.png")

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} valid_loss={valid_metrics['loss']:.4f} "
            f"valid_live_loss={valid_live_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['p_win_acc']:.4f} "
            f"valid_bal_acc={valid_metrics['p_win_bal_acc']:.4f} "
            f"valid_majority={valid_metrics['p_win_majority_baseline']:.4f} "
            f"valid_business={valid_metrics['business_score']:.4f} "
            f"topk_util={valid_metrics['topk_utility']:.4f} "
            f"ret_mae={valid_metrics['ret_mu_mae']:.4f} "
            f"dd_mae={valid_metrics['risk_dd_mae']:.4f} "
            f"bigloss_loss={valid_metrics['loss_bigloss']:.4f} "
            f"rank_pair={valid_metrics['loss_rank_pairwise']:.4f} "
            f"regime_loss={valid_metrics['loss_p_win_regime']:.4f} "
            f"entropy={valid_metrics['loss_prediction_entropy']:.4f} "
            f"gate={valid_metrics['loss_expert_gate_entropy']:.4f} "
            f"skipped={train_metrics['skipped_batches']:.0f}"
        )
        print_metric_block("train", train_metrics)
        print_metric_block("valid", valid_metrics)

        epoch_checkpoint_path = output_dir / f"epoch_{epoch:02d}.pt"
        torch.save(
            checkpoint_payload(
                eval_model,
                bundle,
                config,
                epoch,
                best_valid_loss,
                best_metric_name,
                best_metric_value,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                validation_summary=valid_metrics,
            ),
            epoch_checkpoint_path,
        )

        if best_metric_name == "valid_p_win_acc":
            current_metric_value = float(valid_metrics["p_win_acc"])
        elif best_metric_name == "valid_p_win_bal_acc":
            current_metric_value = float(valid_metrics["p_win_bal_acc"])
        elif best_metric_name == "valid_business_score":
            current_metric_value = float(valid_metrics["business_score"])
        elif best_metric_name == "valid_topk_utility":
            current_metric_value = float(valid_metrics["topk_utility"])
        else:
            current_metric_value = float(valid_metrics["loss"])
        min_delta = max(0.0, float(config.early_stop_min_delta))
        is_improved = (
            current_metric_value > best_metric_value + min_delta
            if best_metric_name in max_metric_names
            else current_metric_value < best_metric_value - min_delta
        )

        if is_improved:
            best_valid_loss = valid_metrics["loss"]
            best_metric_value = current_metric_value
            best_epoch = epoch
            patience = 0
            torch.save(
                checkpoint_payload(
                    eval_model,
                    bundle,
                    config,
                    epoch,
                    best_valid_loss,
                    best_metric_name,
                    best_metric_value,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    ema=ema,
                    validation_summary=valid_metrics,
                ),
                checkpoint_path,
            )
            print(f"  saved new best_model.pt by {best_metric_name}={best_metric_value:.4f}")
        else:
            patience += 1
            if patience >= config.early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    best_valid_metrics_raw = evaluate(model, bundle.valid_loader, device, config, loss_context, desc="Best [valid]")
    valid_probabilities, valid_targets = collect_p_win_predictions(model, bundle.valid_loader, device, config)
    calibrated_thresholds = calibrate_p_win_thresholds(valid_probabilities, valid_targets)
    best_valid_metrics_calibrated = evaluate(
        model,
        bundle.valid_loader,
        device,
        config,
        loss_context,
        desc="Best [valid calibrated]",
        p_win_thresholds=calibrated_thresholds,
    )
    gc.collect()
    total_minutes = (time.time() - training_started_at) / 60.0

    checkpoint["p_win_thresholds"] = calibrated_thresholds.cpu()
    checkpoint["valid_metrics_raw"] = best_valid_metrics_raw
    checkpoint["valid_metrics_calibrated"] = best_valid_metrics_calibrated
    torch.save(checkpoint, checkpoint_path)

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "best_metric_name": best_metric_name,
        "best_metric_value": best_metric_value,
        "valid_metrics_raw": best_valid_metrics_raw,
        "valid_metrics_calibrated": best_valid_metrics_calibrated,
        "p_win_thresholds": [float(value) for value in calibrated_thresholds.cpu().tolist()],
        "train_minutes": total_minutes,
    }
    save_json(output_dir / "validation_metrics.json", metrics_payload)
    print(
        f"Best epoch={best_epoch} {best_metric_name}={best_metric_value:.4f} "
        f"valid_loss={best_valid_loss:.4f} | "
        f"valid_acc={best_valid_metrics_calibrated['p_win_acc']:.4f} "
        f"valid_ret_mae={best_valid_metrics_calibrated['ret_mu_mae']:.4f} "
        f"valid_dd_mae={best_valid_metrics_calibrated['risk_dd_mae']:.4f}"
    )
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
