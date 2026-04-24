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

from train.data import DataBundle, build_data_bundle, load_split_for_evaluation
from train.model import TinyMultiInputModel
from train.visualization import plot_business_history, plot_topk_metrics, plot_training_history

DEFAULT_P_WIN_HORIZONS = (3, 5, 10, 20, 40)
LOSS_PART_NAMES = (
    "loss_p_win",
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
)


@dataclass
class TrainConfig:
    train_path: str
    valid_path: str
    test_path: str
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
    primary_horizon_index: int
    topk: int
    best_model_metric: str
    max_train_batches: int | None
    max_eval_batches: int | None
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
    return 8192


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a professional multi-input financial prediction model.")
    parser.add_argument("--train-path", type=str, default="data/exports/train.parquet")
    parser.add_argument("--valid-path", type=str, default="data/exports/valid.parquet")
    parser.add_argument("--test-path", type=str, default="data/exports/test.parquet")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="train/cache")
    parser.add_argument("--seq-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seq-model-dim", type=int, default=96)
    parser.add_argument("--seq-output-dim", type=int, default=128)
    parser.add_argument("--seq-attn-heads", type=int, default=4)
    parser.add_argument("--branch-hidden-dim", type=int, default=192)
    parser.add_argument("--context-dim", type=int, default=96)
    parser.add_argument("--company-dim", type=int, default=48)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--task-hidden-dim", type=int, default=128)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--min-lr-ratio", type=float, default=0.15)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--disable-weighted-sampler", action="store_true")
    parser.add_argument("--sampler-recency-power", type=float, default=1.5)
    parser.add_argument("--sampler-positive-balance-power", type=float, default=0.5)
    parser.add_argument("--sampler-balance-horizon-index", type=int, default=1)
    parser.add_argument("--p-win-weight", type=float, default=0.9)
    parser.add_argument("--ret-mu-weight", type=float, default=1.0)
    parser.add_argument("--ret-sigma-weight", type=float, default=0.15)
    parser.add_argument("--risk-dd-weight", type=float, default=1.1)
    parser.add_argument("--bigloss-weight", type=float, default=1.2)
    parser.add_argument("--upside-weight", type=float, default=0.6)
    parser.add_argument("--rank-pairwise-weight", type=float, default=1.0)
    parser.add_argument("--rank-score-weight", type=float, default=0.15)
    parser.add_argument("--bigloss-margin-weight", type=float, default=0.6)
    parser.add_argument("--anti-conservative-weight", type=float, default=0.5)
    parser.add_argument("--calibration-weight", type=float, default=0.15)
    parser.add_argument("--p-win-regime-weight", type=float, default=0.2)
    parser.add_argument("--p-win-negative-weight", type=float, default=1.18)
    parser.add_argument("--p-win-downside-weight", type=float, default=0.75)
    parser.add_argument("--primary-horizon-index", type=int, default=2, help="0-based main horizon; default 2 means 10d.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--best-model-metric",
        type=str,
        default="valid_business_score",
        choices=["valid_loss", "valid_p_win_acc", "valid_business_score", "valid_topk_utility"],
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    device = resolve_device_name(args.device)
    batch_size = args.batch_size or recommended_batch_size(device)
    output_dir = args.output_dir or f"train/artifacts/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return TrainConfig(
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
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
        use_weighted_sampler=not args.disable_weighted_sampler,
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
        primary_horizon_index=args.primary_horizon_index,
        topk=args.topk,
        best_model_metric=args.best_model_metric,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
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
        "dropout": config.dropout,
    }


def build_loss_context(bundle: DataBundle, device: torch.device) -> LossContext:
    indices = bundle.train_split.sample_row_indices
    p_win = bundle.train_split.labels["p_win"][indices]
    positive_rate = p_win.mean(dim=0).clamp(0.05, 0.95)
    pos_weight = ((1.0 - positive_rate) / positive_rate).clamp(0.5, 3.0)
    return LossContext(
        p_win_pos_weight=pos_weight.to(device),
        p_win_horizon_weight=torch.tensor([0.8, 1.0, 1.2, 1.1, 0.9], dtype=torch.float32, device=device),
        ret_mu_horizon_weight=torch.tensor([0.6, 1.0, 1.2, 1.1, 0.8], dtype=torch.float32, device=device),
        risk_dd_horizon_weight=torch.tensor([0.7, 1.0, 1.2, 1.2, 0.9], dtype=torch.float32, device=device),
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
    if bigloss_targets.size(1) >= 1 and target_dim >= 1:
        expanded[:, 0] = bigloss_targets[:, 0]
    if bigloss_targets.size(1) >= 1 and target_dim >= 2:
        expanded[:, 1] = bigloss_targets[:, 0]
    if bigloss_targets.size(1) >= 1 and target_dim >= 3:
        expanded[:, 2] = bigloss_targets[:, 0]
    if bigloss_targets.size(1) >= 2 and target_dim >= 4:
        expanded[:, 3] = bigloss_targets[:, 1]
    if bigloss_targets.size(1) >= 2 and target_dim >= 5:
        expanded[:, 4] = bigloss_targets[:, 1]
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
    weights = weights * (1.0 + (max(0.0, float(config.p_win_negative_weight)) - 1.0) * negative_mask)

    if "risk_dd" in targets and targets["risk_dd"].numel() > 0:
        drawdown_severity = torch.relu((-targets["risk_dd"].float()) - 0.02)
        drawdown_boost = 1.0 + max(0.0, float(config.p_win_downside_weight)) * drawdown_severity
        weights = weights * drawdown_boost[:, :head_dim]

    if "bigloss" in targets and targets["bigloss"].numel() > 0:
        expanded_bigloss = _expand_bigloss_targets(targets["bigloss"].float(), head_dim)
        weights = weights * (1.0 + 1.5 * expanded_bigloss)

    return weights.clamp(1.0, 6.0)


def _primary_index(config: TrainConfig, width: int) -> int:
    if width <= 0:
        return 0
    return min(max(int(config.primary_horizon_index), 0), width - 1)


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
    gamma: float = 1.5,
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
    max_pairs_per_date: int = 4096,
) -> torch.Tensor:
    if date_idx is None or decision_scores.numel() == 0 or returns.numel() == 0:
        return decision_scores.new_tensor(0.0)
    if decision_scores.ndim == 2:
        score = decision_scores[:, _primary_index_for_width(horizon_idx, decision_scores.size(1))]
    else:
        score = decision_scores.reshape(-1)
    ret = returns[:, _primary_index_for_width(horizon_idx, returns.size(1))] if returns.ndim == 2 else returns.reshape(-1)
    dates = date_idx.reshape(-1).long()
    losses: list[torch.Tensor] = []
    for date in torch.unique(dates):
        mask = dates == date
        if int(mask.sum().item()) < 2:
            continue
        date_score = score[mask]
        date_ret = ret[mask]
        ret_gap = date_ret.unsqueeze(1) - date_ret.unsqueeze(0)
        pair_mask = ret_gap > float(min_return_gap)
        if not pair_mask.any():
            continue
        score_gap = date_score.unsqueeze(1) - date_score.unsqueeze(0)
        pair_losses = F.softplus(-score_gap[pair_mask])
        pair_weights = ret_gap[pair_mask].abs().clamp_min(float(min_return_gap))
        if pair_losses.numel() > max_pairs_per_date:
            selection = torch.randperm(pair_losses.numel(), device=pair_losses.device)[:max_pairs_per_date]
            pair_losses = pair_losses[selection]
            pair_weights = pair_weights[selection]
        losses.append((pair_losses * pair_weights).mean())
    if not losses:
        return decision_scores.new_tensor(0.0)
    return torch.stack(losses).mean()


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


def _primary_index_for_width(index: int, width: int) -> int:
    return min(max(int(index), 0), max(0, int(width) - 1))


def compute_losses(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    config: TrainConfig,
    loss_context: LossContext,
    aux_targets: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    p_win_targets = targets["p_win"]
    p_win_head_dim = int(p_win_targets.shape[1]) if p_win_targets.ndim == 2 else 1
    horizon_idx = _primary_index(config, p_win_head_dim)
    decision_scores = outputs.get("decision_score", outputs.get("rank_score"))
    if config.label_smoothing > 0:
        smoothing = max(0.0, min(float(config.label_smoothing), 0.2))
        p_win_targets = p_win_targets * (1.0 - smoothing) + 0.5 * smoothing
    loss_p_win = _focal_bce_with_logits(
        outputs["p_win"],
        p_win_targets,
        pos_weight=loss_context.p_win_pos_weight,
    )
    p_win_sample_weight = _downside_weight_matrix(targets, config, p_win_head_dim)
    loss_p_win = (
        loss_p_win
        * loss_context.p_win_horizon_weight.unsqueeze(0)
        * p_win_sample_weight
    ).mean()
    loss_ret_mu = F.smooth_l1_loss(outputs["ret_mu"], targets["ret_mu"], reduction="none")
    loss_ret_mu = (loss_ret_mu * loss_context.ret_mu_horizon_weight.unsqueeze(0)).mean()
    loss_ret_sigma = outputs["p_win"].new_tensor(0.0)
    if "ret_sigma" in outputs:
        abs_error_target = torch.abs(outputs["ret_mu"].detach() - targets["ret_mu"])
        loss_ret_sigma = F.smooth_l1_loss(outputs["ret_sigma"], abs_error_target)
    loss_risk_dd = F.smooth_l1_loss(outputs["risk_dd"], targets["risk_dd"], reduction="none")
    deep_dd_weight = 1.0 + 2.0 * (targets["risk_dd"].float() < -0.05).float()
    loss_risk_dd = loss_risk_dd * deep_dd_weight
    loss_risk_dd = (loss_risk_dd * loss_context.risk_dd_horizon_weight.unsqueeze(0)).mean()
    loss_bigloss = outputs["p_win"].new_tensor(0.0)
    if "bigloss" in outputs and "bigloss" in targets and targets["bigloss"].numel() > 0:
        bigloss_targets = targets["bigloss"].float()
        loss_bigloss = _focal_bce_with_logits(
            outputs["bigloss"],
            bigloss_targets,
        )
        severity_weight = 1.0 + 2.0 * bigloss_targets
        loss_bigloss = (loss_bigloss * severity_weight).mean()
    loss_upside = outputs["p_win"].new_tensor(0.0)
    if "upside" in outputs:
        upside_targets = torch.relu(targets["ret_mu"].float())
        strong_up_weight = 1.0 + 2.0 * (upside_targets > 0.03).float()
        loss_upside = (F.smooth_l1_loss(outputs["upside"], upside_targets, reduction="none") * strong_up_weight).mean()
    loss_rank_score = outputs["p_win"].new_tensor(0.0)
    if "rank_score" in outputs and "rank_score" in targets:
        rank_score_pred = torch.sigmoid(outputs["rank_score"])
        loss_rank_score = F.smooth_l1_loss(rank_score_pred, targets["rank_score"])
    date_idx = None
    if aux_targets and "date_idx" in aux_targets:
        date_idx = aux_targets["date_idx"]
    loss_rank_pairwise = (
        _pairwise_rank_loss(
            decision_scores,
            targets["ret_mu"],
            date_idx,
            horizon_idx=horizon_idx,
        )
        if decision_scores is not None
        else outputs["p_win"].new_tensor(0.0)
    )
    if (
        decision_scores is not None
        and "rank_score" in targets
        and float(loss_rank_pairwise.detach().abs().item()) < 1e-12
    ):
        loss_rank_pairwise = _global_pairwise_rank_loss(decision_scores, targets["rank_score"], horizon_idx)
    loss_bigloss_margin = outputs["p_win"].new_tensor(0.0)
    if decision_scores is not None and "bigloss" in targets and targets["bigloss"].numel() > 0:
        expanded_bigloss = _expand_bigloss_targets(targets["bigloss"].float(), decision_scores.size(1))
        decision_primary = decision_scores[:, _primary_index_for_width(horizon_idx, decision_scores.size(1))]
        bigloss_primary = expanded_bigloss[:, _primary_index_for_width(horizon_idx, expanded_bigloss.size(1))]
        if bool((bigloss_primary > 0.5).any().item()):
            loss_bigloss_margin = F.relu(decision_primary[bigloss_primary > 0.5]).mean()
    loss_anti_conservative = outputs["p_win"].new_tensor(0.0)
    if decision_scores is not None:
        decision_primary = decision_scores[:, _primary_index_for_width(horizon_idx, decision_scores.size(1))]
        ret_primary = targets["ret_mu"][:, _primary_index_for_width(horizon_idx, targets["ret_mu"].size(1))]
        strong_up = ret_primary > 0.03
        if bool(strong_up.any().item()):
            loss_anti_conservative = F.relu(0.2 - decision_primary[strong_up]).mean()
    loss_calibration = torch.abs(torch.sigmoid(outputs["p_win"]).mean(dim=0) - targets["p_win"].float().mean(dim=0)).mean()
    loss_p_win_regime = outputs["p_win"].new_tensor(0.0)
    if (
        aux_targets
        and config.p_win_regime_weight > 0
        and "p_win_regime" in outputs
        and "p_win_rate_by_date" in aux_targets
        and aux_targets["p_win_rate_by_date"].numel() > 0
    ):
        regime_targets = aux_targets["p_win_rate_by_date"].float().clamp(0.0, 1.0)
        loss_p_win_regime = F.binary_cross_entropy_with_logits(
            outputs["p_win_regime"],
            regime_targets,
        )
    total = (
        config.p_win_weight * loss_p_win
        + config.ret_mu_weight * loss_ret_mu
        + config.ret_sigma_weight * loss_ret_sigma
        + config.risk_dd_weight * loss_risk_dd
        + config.bigloss_weight * loss_bigloss
        + config.upside_weight * loss_upside
        + config.rank_pairwise_weight * loss_rank_pairwise
        + config.rank_score_weight * loss_rank_score
        + config.bigloss_margin_weight * loss_bigloss_margin
        + config.anti_conservative_weight * loss_anti_conservative
        + config.calibration_weight * loss_calibration
        + config.p_win_regime_weight * loss_p_win_regime
    )
    return total, {
        "loss_p_win": loss_p_win,
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
    }


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
    ret_mu_mae_by_head = _safe_tensor_ratio(running["ret_mu_abs_by_head"], running["ret_mu_total_by_head"])
    risk_dd_mae_by_head = _safe_tensor_ratio(running["risk_dd_abs_by_head"], running["risk_dd_total_by_head"])
    topk_metrics = _compute_topk_metrics_from_chunks(running)
    primary_idx = _primary_index_for_width(int(running.get("primary_horizon_index", 2)), len(p_win_bal_acc_by_head))
    primary_bal_acc = float(p_win_bal_acc_by_head[primary_idx].item()) if len(p_win_bal_acc_by_head) else 0.0
    topk_metrics["business_score"] = float(topk_metrics.get("topk_utility", 0.0)) + 0.1 * primary_bal_acc
    metrics = {
        "loss": running["loss_sum"] / max(1.0, running["sample_count"]),
        "p_win_acc": running["p_win_correct"] / max(1.0, running["p_win_total"]),
        "ret_mu_mae": running["ret_mu_abs"] / max(1.0, running["ret_mu_total"]),
        "risk_dd_mae": running["risk_dd_abs"] / max(1.0, running["risk_dd_total"]),
        "rank_score_mae": running["rank_score_abs"] / max(1.0, running["rank_score_total"]),
        "p_win_acc_by_head": [float(value) for value in p_win_acc_by_head.tolist()],
        "p_win_bal_acc_by_head": [float(value) for value in p_win_bal_acc_by_head.tolist()],
        "p_win_target_pos_rate_by_head": [float(value) for value in p_win_target_pos_rate_by_head.tolist()],
        "p_win_pred_pos_rate_by_head": [float(value) for value in p_win_pred_pos_rate_by_head.tolist()],
        "p_win_prob_mean_by_head": [float(value) for value in p_win_prob_mean_by_head.tolist()],
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
        "ret_mu_mae",
        "risk_dd_mae",
        "rank_score_mae",
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
    autocast_enabled = device.type == "cuda"
    p_win_thresholds = resolve_p_win_thresholds(device, loss_context.p_win_pos_weight.numel())

    for step, batch in enumerate(progress, start=1):
        if config.max_train_batches is not None and step > config.max_train_batches:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            outputs = model(batch)
            loss, parts = compute_losses(outputs, batch["y"], config, loss_context, aux_targets=batch.get("aux"))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        batch_size = float(batch["X_seq"].size(0))
        running["loss_sum"] += float(loss.detach().item()) * batch_size
        running["sample_count"] += batch_size
        for name in LOSS_PART_NAMES:
            running[f"{name}_sum"] += float(parts[name].detach().item()) * batch_size

        p_win_prob = torch.sigmoid(outputs["p_win"].detach())
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
    autocast_enabled = device.type == "cuda"
    resolved_thresholds = resolve_p_win_thresholds(device, loss_context.p_win_pos_weight.numel(), p_win_thresholds)

    for step, batch in enumerate(progress, start=1):
        if config.max_eval_batches is not None and step > config.max_eval_batches:
            break
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
            outputs = model(batch)
            loss, parts = compute_losses(outputs, batch["y"], config, loss_context, aux_targets=batch.get("aux"))

        batch_size = float(batch["X_seq"].size(0))
        running["loss_sum"] += float(loss.detach().item()) * batch_size
        running["sample_count"] += batch_size
        for name in LOSS_PART_NAMES:
            running[f"{name}_sum"] += float(parts[name].detach().item()) * batch_size

        p_win_prob = torch.sigmoid(outputs["p_win"].detach())
        p_win_pred = (p_win_prob >= resolved_thresholds.unsqueeze(0)).float()
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

        metrics = _finalize_metrics(running)
        progress.set_postfix(
            loss=f"{metrics['loss']:.4f}",
            util=f"{metrics['topk_utility']:.4f}",
            acc=f"{metrics['p_win_acc']:.3f}",
        )
    return _finalize_metrics(running)


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
    autocast_enabled = device.type == "cuda"
    for step, batch in enumerate(loader, start=1):
        if config.max_eval_batches is not None and step > config.max_eval_batches:
            break
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
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
    test_summary = "test split deferred until final evaluation"
    if bundle.test_split is not None:
        test_summary = f"test rows={bundle.test_split.row_count:,} samples={bundle.test_split.sample_count:,}"
    print("Prepared datasets:")
    print(
        f"  train rows={bundle.train_split.row_count:,} samples={bundle.train_split.sample_count:,} "
        f"valid rows={bundle.valid_split.row_count:,} samples={bundle.valid_split.sample_count:,} "
        f"{test_summary}"
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
        f"fusion_dim={config.fusion_dim} task_hidden_dim={config.task_hidden_dim}"
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
        test_path=PROJECT_ROOT / config.test_path,
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
        load_test_split=False,
    )
    print_split_summary(bundle, config)
    loss_context = build_loss_context(bundle, device)

    model = build_model(bundle, config, device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    ema = ModelEMA(model, config.ema_decay) if 0.0 < config.ema_decay < 1.0 else None

    history: list[dict[str, float]] = []
    best_valid_loss = float("inf")
    best_metric_name = str(config.best_model_metric)
    if best_metric_name in {"valid_p_win_acc", "valid_business_score", "valid_topk_utility"}:
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
        eval_model = ema.shadow if ema is not None else model
        valid_metrics = evaluate(
            eval_model,
            bundle.valid_loader,
            device,
            config,
            loss_context,
            desc=f"Epoch {epoch} [valid]",
        )
        scheduler.step()

        record = {
            "epoch": float(epoch),
            "lr": optimizer.param_groups[0]["lr"],
            **flatten_metrics_for_history("train", train_metrics),
            **flatten_metrics_for_history("valid", valid_metrics),
        }
        history.append(record)
        save_history_csv(output_dir / "history.csv", history)
        save_json(output_dir / "history.json", {"epochs": history})
        plot_training_history(history, output_dir / "training_curves.png")
        plot_business_history(history, output_dir / "business_curves.png")
        plot_topk_metrics(history, output_dir / "topk_metrics.png")

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_metrics['loss']:.4f} valid_loss={valid_metrics['loss']:.4f} "
            f"valid_acc={valid_metrics['p_win_acc']:.4f} "
            f"valid_business={valid_metrics['business_score']:.4f} "
            f"topk_util={valid_metrics['topk_utility']:.4f} "
            f"ret_mae={valid_metrics['ret_mu_mae']:.4f} "
            f"dd_mae={valid_metrics['risk_dd_mae']:.4f} "
            f"bigloss_loss={valid_metrics['loss_bigloss']:.4f} "
            f"rank_pair={valid_metrics['loss_rank_pairwise']:.4f} "
            f"regime_loss={valid_metrics['loss_p_win_regime']:.4f}"
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
        elif best_metric_name == "valid_business_score":
            current_metric_value = float(valid_metrics["business_score"])
        elif best_metric_name == "valid_topk_utility":
            current_metric_value = float(valid_metrics["topk_utility"])
        else:
            current_metric_value = float(valid_metrics["loss"])
        is_improved = (
            current_metric_value > best_metric_value
            if best_metric_name in {"valid_p_win_acc", "valid_business_score", "valid_topk_utility"}
            else current_metric_value < best_metric_value
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
    print("Loading deferred test split for final evaluation...")
    test_split, test_loader = load_split_for_evaluation(
        split_name="test",
        source_path=PROJECT_ROOT / config.test_path,
        cache_dir=cache_dir,
        seq_length=config.seq_length,
        normalizer=bundle.normalizer,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        rebuild_cache=config.rebuild_cache,
        device_type=device.type,
    )
    print(f"Test split ready: rows={test_split.row_count:,} samples={test_split.sample_count:,}")
    test_metrics_raw = evaluate(model, test_loader, device, config, loss_context, desc="Test")
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        config,
        loss_context,
        desc="Test calibrated",
        p_win_thresholds=calibrated_thresholds,
    )
    total_minutes = (time.time() - training_started_at) / 60.0

    checkpoint["p_win_thresholds"] = calibrated_thresholds.cpu()
    checkpoint["valid_metrics_raw"] = best_valid_metrics_raw
    checkpoint["valid_metrics_calibrated"] = best_valid_metrics_calibrated
    checkpoint["test_metrics_raw"] = test_metrics_raw
    checkpoint["test_metrics_calibrated"] = test_metrics
    torch.save(checkpoint, checkpoint_path)

    save_json(
        output_dir / "test_metrics.json",
        {
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "valid_metrics_raw": best_valid_metrics_raw,
            "valid_metrics_calibrated": best_valid_metrics_calibrated,
            "test_metrics_raw": test_metrics_raw,
            "test_metrics_calibrated": test_metrics,
            "p_win_thresholds": [float(value) for value in calibrated_thresholds.cpu().tolist()],
            "train_minutes": total_minutes,
        },
    )
    print(
        f"Best epoch={best_epoch} {best_metric_name}={best_metric_value:.4f} valid_loss={best_valid_loss:.4f} | "
        f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['p_win_acc']:.4f} "
        f"(raw_acc={test_metrics_raw['p_win_acc']:.4f}) "
        f"test_ret_mae={test_metrics['ret_mu_mae']:.4f} "
        f"test_dd_mae={test_metrics['risk_dd_mae']:.4f} "
        f"test_rank_mae={test_metrics['rank_score_mae']:.4f}"
    )
    print(f"Artifacts saved to: {output_dir}")


if __name__ == "__main__":
    main()
