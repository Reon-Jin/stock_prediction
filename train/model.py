from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.company_encoder import CompanyEncoder


HORIZONS: tuple[int, ...] = (3, 5, 10, 20, 40)
BIGLOSS_HORIZONS: tuple[int, ...] = (5, 20)
DECISION_AUX_FEATURES: tuple[str, ...] = ("ret_mu", "ret_sigma", "p_win", "risk_dd", "bigloss", "upside")
# Bigloss labels only exist for 5d and 20d; nearby horizons use the closest available risk proxy.
BIGLOSS_PROXY_BY_HORIZON: dict[int, int] = {3: 5, 5: 5, 10: 5, 20: 20, 40: 20}
ALIBI_INITIAL_SLOPE = 0.1


def _inverse_softplus_scalar(value: float) -> torch.Tensor:
    return torch.log(torch.expm1(torch.tensor(float(value))))


class GatedResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        norm_x = self.norm(x)
        gated = torch.sigmoid(self.gate(norm_x))
        hidden = self.fc1(norm_x)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        return residual + gated * hidden


class DeepMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(hidden_dim, hidden_dim * 2, dropout=dropout) for _ in range(max(1, depth))]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.left_padding = max(0, kernel_size - 1)
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=0, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.norm(x).transpose(1, 2)
        hidden = F.pad(hidden, (self.left_padding, 0))
        hidden = self.depthwise(hidden)
        hidden = self.activation(hidden)
        hidden = self.pointwise(hidden).transpose(1, 2)
        hidden = self.dropout(hidden)
        return residual + hidden


class TemporalMixerBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.left_padding = max(0, kernel_size - 1)
        self.temporal_norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=0, groups=dim)
        self.channel_norm = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.temporal_norm(x).transpose(1, 2)
        hidden = F.pad(hidden, (self.left_padding, 0))
        hidden = self.depthwise(hidden).transpose(1, 2)
        x = x + self.dropout(self.activation(hidden))
        return x + self.channel_mlp(self.channel_norm(x))


class AttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x).squeeze(-1), dim=-1)
        return torch.einsum("bs,bsd->bd", weights, x)


class TemporalFinancialEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_length: int,
        model_dim: int = 96,
        output_dim: int = 128,
        attn_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.input_norm = nn.LayerNorm(input_dim)
        self.feature_gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.input_dropout = nn.Dropout1d(dropout)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.relative_age_proj = nn.Sequential(
            nn.Linear(1, model_dim),
            nn.Tanh(),
        )
        self.alibi_slope = nn.Parameter(_inverse_softplus_scalar(ALIBI_INITIAL_SLOPE))
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalMixerBlock(model_dim, kernel_size=3, dropout=dropout),
                TemporalMixerBlock(model_dim, kernel_size=5, dropout=dropout),
                TemporalMixerBlock(model_dim, kernel_size=3, dropout=dropout),
            ]
        )
        self.gru = nn.GRU(
            input_size=model_dim,
            hidden_size=model_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )
        self.self_attention = nn.MultiheadAttention(model_dim, num_heads=attn_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn_pool = AttentionPooling(model_dim)
        self.output = nn.Sequential(
            nn.Linear(model_dim * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _causal_alibi_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        distance = (positions[:, None] - positions[None, :]).clamp_min(0).to(dtype=dtype)
        future_mask = positions[None, :] > positions[:, None]
        slope = F.softplus(self.alibi_slope).to(dtype=dtype)
        mask = -slope * distance
        return mask.masked_fill(future_mask, torch.finfo(dtype).min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = x * self.feature_gate(x)
        x = self.input_dropout(x.transpose(1, 2)).transpose(1, 2)
        hidden = self.input_proj(x)
        relative_age = torch.linspace(0.0, 1.0, hidden.size(1), device=hidden.device, dtype=hidden.dtype).view(1, -1, 1)
        hidden = hidden + self.relative_age_proj(relative_age)
        for block in self.temporal_blocks:
            hidden = block(hidden)
        hidden, _ = self.gru(hidden)
        attn_mask = self._causal_alibi_mask(hidden.size(1), hidden.device, hidden.dtype)
        attn_hidden, _ = self.self_attention(hidden, hidden, hidden, attn_mask=attn_mask, need_weights=False)
        hidden = self.attn_norm(hidden + attn_hidden)
        pooled = self.attn_pool(hidden)
        recent = hidden[:, -min(5, hidden.size(1)) :, :].mean(dim=1)
        last = hidden[:, -1, :]
        combined = torch.cat([pooled, recent, last], dim=-1)
        return self.output(combined)


class CrossGate(nn.Module):
    def __init__(self, dim: int, context_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim + context_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.context_proj = nn.Linear(context_dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([x, context], dim=-1))
        context_term = self.context_proj(context)
        return x + gate * context_term


class HorizonProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShortTermHorizonExpert(nn.Module):
    """Expert for 3/5/10 day targets: event-sensitive, faster-changing signals."""

    def __init__(self, input_dim: int, expert_dim: int, horizon: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.horizon_scale = nn.Parameter(torch.tensor(float(horizon) / 10.0))
        self.input = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fast_path = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, expert_dim),
        )
        self.shock_gate = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, expert_dim),
            nn.Sigmoid(),
        )
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(expert_dim, expert_dim * 2, dropout=dropout) for _ in range(2)]
        )
        self.output = nn.Sequential(nn.LayerNorm(expert_dim), nn.Linear(expert_dim, expert_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x)
        fast_signal = self.fast_path(x)
        shock_gate = self.shock_gate(x)
        hidden = hidden + shock_gate * fast_signal * torch.tanh(self.horizon_scale)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class LongTermHorizonExpert(nn.Module):
    """Expert for 20/40 day targets: smoother trend and risk-cycle representation."""

    def __init__(self, input_dim: int, expert_dim: int, horizon: int, dropout: float = 0.1) -> None:
        super().__init__()
        stable_dropout = max(0.05, dropout * 0.75)
        self.horizon_scale = nn.Parameter(torch.tensor(float(horizon) / 40.0))
        self.trend_path = nn.Sequential(
            nn.Linear(input_dim, expert_dim * 2),
            nn.LayerNorm(expert_dim * 2),
            nn.GELU(),
            nn.Dropout(stable_dropout),
            nn.Linear(expert_dim * 2, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.GELU(),
        )
        self.cycle_path = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.LayerNorm(expert_dim),
            nn.SiLU(),
            nn.Dropout(stable_dropout),
            nn.Linear(expert_dim, expert_dim),
        )
        self.risk_gate = nn.Sequential(
            nn.Linear(input_dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, expert_dim),
            nn.Sigmoid(),
        )
        self.blocks = nn.ModuleList(
            [GatedResidualBlock(expert_dim, expert_dim * 2, dropout=stable_dropout) for _ in range(3)]
        )
        self.output = nn.Sequential(nn.LayerNorm(expert_dim), nn.Linear(expert_dim, expert_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.trend_path(x)
        cycle = self.cycle_path(x)
        hidden = trend + self.risk_gate(x) * cycle * torch.tanh(self.horizon_scale)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


class HorizonExpertBank(nn.Module):
    def __init__(self, input_dim: int, expert_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.horizons = HORIZONS
        self.experts = nn.ModuleDict(
            {
                str(horizon): (
                    ShortTermHorizonExpert(input_dim, expert_dim, horizon=horizon, dropout=dropout)
                    if horizon <= 10
                    else LongTermHorizonExpert(input_dim, expert_dim, horizon=horizon, dropout=dropout)
                )
                for horizon in self.horizons
            }
        )

    def forward(
        self,
        fused: torch.Tensor,
        seq_repr: torch.Tensor,
        context_repr: torch.Tensor,
        tab_repr: torch.Tensor,
        company_repr: torch.Tensor,
        neighbor_repr: torch.Tensor,
    ) -> dict[int, torch.Tensor]:
        expert_input = torch.cat([fused, seq_repr, context_repr, tab_repr, company_repr, neighbor_repr], dim=-1)
        return {horizon: self.experts[str(horizon)](expert_input) for horizon in self.horizons}


class HorizonMoEHead(nn.Module):
    def __init__(
        self,
        expert_dim: int,
        output_horizons: tuple[int, ...],
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_horizons = tuple(output_horizons)
        self.heads = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.Linear(expert_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for horizon in self.output_horizons
            }
        )

    def forward(self, expert_reprs: dict[int, torch.Tensor]) -> torch.Tensor:
        return torch.cat([self.heads[str(horizon)](expert_reprs[horizon]) for horizon in self.output_horizons], dim=-1)


class HorizonDecisionHead(nn.Module):
    def __init__(
        self,
        expert_dim: int,
        output_horizons: tuple[int, ...],
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.output_horizons = tuple(output_horizons)
        self.aux_feature_count = len(DECISION_AUX_FEATURES)
        self.heads = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.Linear(expert_dim + self.aux_feature_count, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for horizon in self.output_horizons
            }
        )

    @staticmethod
    def _column_or_zeros(
        values: torch.Tensor | None,
        column_idx: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if values is None or values.numel() == 0 or column_idx >= values.size(1):
            return torch.zeros(batch_size, 1, dtype=dtype, device=device)
        return values[:, column_idx : column_idx + 1]

    def forward(
        self,
        expert_reprs: dict[int, torch.Tensor],
        outputs: dict[str, torch.Tensor],
        bigloss_by_horizon: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = next(iter(expert_reprs.values())).size(0)
        device = next(iter(expert_reprs.values())).device
        dtype = next(iter(expert_reprs.values())).dtype
        columns: list[torch.Tensor] = []
        for column_idx, horizon in enumerate(self.output_horizons):
            features = [
                expert_reprs[horizon],
                self._column_or_zeros(outputs.get("ret_mu"), column_idx, batch_size, device, dtype),
                self._column_or_zeros(outputs.get("ret_sigma"), column_idx, batch_size, device, dtype),
                torch.sigmoid(self._column_or_zeros(outputs.get("p_win"), column_idx, batch_size, device, dtype)),
                self._column_or_zeros(outputs.get("risk_dd"), column_idx, batch_size, device, dtype),
                self._column_or_zeros(bigloss_by_horizon, column_idx, batch_size, device, dtype),
                self._column_or_zeros(outputs.get("upside"), column_idx, batch_size, device, dtype),
            ]
            columns.append(self.heads[str(horizon)](torch.cat(features, dim=-1)))
        return torch.cat(columns, dim=-1)


class EventEncoderWithGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = DeepMLPEncoder(input_dim, hidden_dim, output_dim, depth=depth, dropout=dropout)
        self.confidence_gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        strength = self.confidence_gate(x)
        return self.encoder(x) * strength, strength


class NeighborAttentionEncoder(nn.Module):
    def __init__(
        self,
        symbol_embedding: nn.Embedding,
        company_dim: int,
        symbol_emb_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.symbol_embedding = symbol_embedding
        self.neighbor_proj = nn.Sequential(
            nn.Linear(symbol_emb_dim, company_dim),
            nn.LayerNorm(company_dim),
            nn.GELU(),
        )
        self.attn = nn.Sequential(
            nn.Linear(company_dim * 2 + 1, company_dim),
            nn.LayerNorm(company_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(company_dim, 1),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(company_dim),
            nn.Linear(company_dim, company_dim),
            nn.GELU(),
        )

    def forward(
        self,
        company_repr: torch.Tensor,
        neighbor_symbol_ids: torch.Tensor,
        neighbor_scores: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_emb = self.symbol_embedding(neighbor_symbol_ids.long())
        neighbor_repr = self.neighbor_proj(neighbor_emb)
        scores = neighbor_scores.float().unsqueeze(-1)
        company_context = company_repr.unsqueeze(1).expand(-1, neighbor_repr.size(1), -1)
        attn_logits = self.attn(torch.cat([neighbor_repr, company_context, scores], dim=-1)).squeeze(-1)
        valid_mask = (neighbor_symbol_ids.long() > 0) & (neighbor_scores.float() > 0)
        attn_logits = attn_logits.masked_fill(~valid_mask, -1e4)
        weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        weights = torch.where(valid_mask.unsqueeze(-1), weights, torch.zeros_like(weights))
        pooled = (neighbor_repr * weights).sum(dim=1)
        return self.output(pooled)


class ProfessionalFinancialModel(nn.Module):
    def __init__(
        self,
        input_dims: dict[str, int],
        vocab_sizes: dict[str, int],
        head_dims: dict[str, int],
        seq_model_dim: int = 96,
        seq_output_dim: int = 128,
        seq_attn_heads: int = 4,
        branch_hidden_dim: int = 192,
        context_dim: int = 96,
        company_dim: int = 48,
        fusion_dim: int = 256,
        task_hidden_dim: int = 128,
        horizon_expert_dim: int = 128,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.temporal_encoder = TemporalFinancialEncoder(
            input_dim=input_dims["f_seq"],
            seq_length=input_dims["seq_length"],
            model_dim=seq_model_dim,
            output_dim=seq_output_dim,
            attn_heads=seq_attn_heads,
            dropout=dropout,
        )
        self.tab_encoder = DeepMLPEncoder(input_dims["f_tab"], branch_hidden_dim, 64, depth=2, dropout=dropout)
        self.event_encoder = EventEncoderWithGate(input_dims["f_event"], branch_hidden_dim, 96, depth=2, dropout=dropout)
        self.mkt_encoder = DeepMLPEncoder(input_dims["f_mkt"], branch_hidden_dim // 2, 48, depth=2, dropout=dropout)
        self.context_fusion = nn.Sequential(
            nn.Linear(96 + 48, context_dim),
            nn.LayerNorm(context_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.company_encoder = CompanyEncoder(
            num_symbols=vocab_sizes.get("symbol_id", 1),
            num_industries=vocab_sizes.get("industry_id", 1),
            num_boards=vocab_sizes.get("board_id", 1),
            profile_input_dim=input_dims["f_company_profile"],
            embedding_dim=company_dim,
            symbol_emb_dim=24,
            industry_emb_dim=12,
            board_emb_dim=8,
            profile_hidden_dims=(96, company_dim),
            dropout=dropout,
        )
        symbol_emb_dim = self.company_encoder.symbol_embedding.embedding_dim
        self.neighbor_encoder = NeighborAttentionEncoder(
            symbol_embedding=self.company_encoder.symbol_embedding,
            company_dim=company_dim,
            symbol_emb_dim=symbol_emb_dim,
            dropout=dropout,
        )

        self.seq_gate = CrossGate(seq_output_dim, context_dim, dropout=dropout)
        self.tab_gate = CrossGate(64, context_dim, dropout=dropout)
        self.company_gate = CrossGate(company_dim, context_dim, dropout=dropout)
        self.neighbor_gate = CrossGate(company_dim, context_dim, dropout=dropout)

        fusion_input_dim = seq_output_dim + 64 + context_dim + company_dim + company_dim
        self.fusion_in = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_blocks = nn.ModuleList(
            [GatedResidualBlock(fusion_dim, fusion_dim * 2, dropout=dropout) for _ in range(3)]
        )
        self.shared_out = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
        )
        horizon_expert_input_dim = fusion_dim + seq_output_dim + context_dim + 64 + company_dim + company_dim
        self.horizon_experts = HorizonExpertBank(
            input_dim=horizon_expert_input_dim,
            expert_dim=horizon_expert_dim,
            dropout=dropout,
        )
        self.horizon_heads = nn.ModuleDict()
        self.shared_heads = nn.ModuleDict()
        self.head_horizons: dict[str, tuple[int, ...]] = {}
        for head_name, head_dim in head_dims.items():
            output_horizons = self._resolve_output_horizons(head_name, int(head_dim))
            if output_horizons:
                self.horizon_heads[head_name] = HorizonMoEHead(
                    expert_dim=horizon_expert_dim,
                    output_horizons=output_horizons,
                    hidden_dim=task_hidden_dim,
                    dropout=dropout,
                )
                self.head_horizons[head_name] = output_horizons
            else:
                self.shared_heads[head_name] = nn.Sequential(
                    nn.Linear(fusion_dim, task_hidden_dim),
                    nn.LayerNorm(task_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(task_hidden_dim, int(head_dim)),
                )
        p_win_dim = int(head_dims.get("p_win", 0))
        ret_mu_dim = int(head_dims.get("ret_mu", p_win_dim))
        ret_sigma_horizons = self._resolve_output_horizons("ret_mu", ret_mu_dim)
        if ret_sigma_horizons:
            self.ret_sigma_head: nn.Module = HorizonMoEHead(
                expert_dim=horizon_expert_dim,
                output_horizons=ret_sigma_horizons,
                hidden_dim=task_hidden_dim,
                dropout=dropout,
            )
            self.upside_head: nn.Module = HorizonMoEHead(
                expert_dim=horizon_expert_dim,
                output_horizons=ret_sigma_horizons,
                hidden_dim=task_hidden_dim,
                dropout=dropout,
            )
        else:
            self.ret_sigma_head = nn.Sequential(
                nn.Linear(fusion_dim, task_hidden_dim),
                nn.LayerNorm(task_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(task_hidden_dim, ret_mu_dim),
            )
            self.upside_head = nn.Sequential(
                nn.Linear(fusion_dim, task_hidden_dim),
                nn.LayerNorm(task_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(task_hidden_dim, ret_mu_dim),
            )
        decision_horizons = self._resolve_output_horizons("p_win", p_win_dim)
        self.decision_score_head: nn.Module = HorizonDecisionHead(
            expert_dim=horizon_expert_dim,
            output_horizons=decision_horizons or HORIZONS[: max(1, p_win_dim)],
            hidden_dim=task_hidden_dim,
            dropout=dropout,
        )
        self.p_win_regime_head: nn.Module | None = None
        if p_win_dim > 0:
            self.p_win_regime_head = nn.Sequential(
                nn.Linear(context_dim, task_hidden_dim),
                nn.LayerNorm(task_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(task_hidden_dim, p_win_dim),
            )

    @staticmethod
    def _resolve_output_horizons(head_name: str, head_dim: int) -> tuple[int, ...]:
        if head_dim <= 0:
            return ()
        if head_dim == len(HORIZONS):
            return HORIZONS
        if head_name == "bigloss" and head_dim == len(BIGLOSS_HORIZONS):
            return BIGLOSS_HORIZONS
        return ()

    @staticmethod
    def _expand_bigloss_to_horizons(
        bigloss_logits: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if bigloss_logits is None or bigloss_logits.numel() == 0:
            return None
        bigloss_prob = torch.sigmoid(bigloss_logits)
        if bigloss_prob.size(1) == len(HORIZONS):
            return bigloss_prob
        if bigloss_prob.size(1) == len(BIGLOSS_HORIZONS):
            expanded = torch.zeros(batch_size, len(HORIZONS), dtype=bigloss_prob.dtype, device=device)
            source_column_by_horizon = {
                source_horizon: column_idx for column_idx, source_horizon in enumerate(BIGLOSS_HORIZONS)
            }
            for target_column, horizon in enumerate(HORIZONS):
                source_horizon = BIGLOSS_PROXY_BY_HORIZON[horizon]
                expanded[:, target_column] = bigloss_prob[:, source_column_by_horizon[source_horizon]]
            return expanded
        return bigloss_prob

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        seq_repr = self.temporal_encoder(batch["X_seq"].float())
        tab_repr = self.tab_encoder(batch["X_tab"].float())
        event_repr, event_strength = self.event_encoder(batch["X_event"].float())
        mkt_repr = self.mkt_encoder(batch["X_mkt"].float())
        context_repr = self.context_fusion(torch.cat([event_repr, mkt_repr], dim=-1))

        company_repr = self.company_encoder(
            symbol_id=batch["X_company_ids"]["symbol_id"],
            industry_id=batch["X_company_ids"]["industry_id"],
            board_id=batch["X_company_ids"]["board_id"],
            company_profile_features=batch["X_company_profile"].float(),
        )
        neighbor_repr = self.neighbor_encoder(
            company_repr,
            batch["neighbors"]["neighbor_symbol_ids"],
            batch["neighbors"]["neighbor_scores"].float(),
        )

        seq_repr = self.seq_gate(seq_repr, context_repr)
        tab_repr = self.tab_gate(tab_repr, context_repr)
        company_repr = self.company_gate(company_repr, context_repr)
        neighbor_repr = self.neighbor_gate(neighbor_repr, context_repr)

        fused = self.fusion_in(torch.cat([seq_repr, tab_repr, context_repr, company_repr, neighbor_repr], dim=-1))
        for block in self.fusion_blocks:
            fused = block(fused)
        fused = self.shared_out(fused)
        expert_reprs = self.horizon_experts(
            fused=fused,
            seq_repr=seq_repr,
            context_repr=context_repr,
            tab_repr=tab_repr,
            company_repr=company_repr,
            neighbor_repr=neighbor_repr,
        )
        outputs = {head_name: head(fused) for head_name, head in self.shared_heads.items()}
        outputs.update({head_name: head(expert_reprs) for head_name, head in self.horizon_heads.items()})
        outputs["event_strength"] = event_strength
        ret_sigma_raw = (
            self.ret_sigma_head(expert_reprs)
            if isinstance(self.ret_sigma_head, HorizonMoEHead)
            else self.ret_sigma_head(fused)
        )
        upside_raw = (
            self.upside_head(expert_reprs)
            if isinstance(self.upside_head, HorizonMoEHead)
            else self.upside_head(fused)
        )
        outputs["ret_sigma"] = F.softplus(ret_sigma_raw) + 1e-4
        outputs["upside"] = F.softplus(upside_raw)
        if "bigloss" in outputs:
            outputs["bigloss_prob"] = torch.sigmoid(outputs["bigloss"])
        bigloss_by_horizon = self._expand_bigloss_to_horizons(outputs.get("bigloss"), fused.size(0), fused.device)
        outputs["decision_score"] = self.decision_score_head(expert_reprs, outputs, bigloss_by_horizon)
        if self.p_win_regime_head is not None and "p_win" in outputs:
            regime_logits = self.p_win_regime_head(context_repr)
            outputs["p_win_base"] = outputs["p_win"]
            outputs["p_win_regime"] = regime_logits
        return outputs


TinyMultiInputModel = ProfessionalFinancialModel
