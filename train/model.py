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
        gated = torch.sigmoid(self.gate(self.norm(x)))
        hidden = self.fc1(self.norm(x))
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
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        hidden = self.norm(x).transpose(1, 2)
        hidden = self.depthwise(hidden)
        hidden = self.activation(hidden)
        hidden = self.pointwise(hidden).transpose(1, 2)
        hidden = self.dropout(hidden)
        return residual + hidden


class TemporalMixerBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.temporal_norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
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
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, seq_length, model_dim))
        self.temporal_blocks = nn.ModuleList(
            [
                TemporalMixerBlock(model_dim, kernel_size=3, dropout=dropout),
                TemporalMixerBlock(model_dim, kernel_size=5, dropout=dropout),
                TemporalMixerBlock(model_dim, kernel_size=3, dropout=dropout),
            ]
        )
        self.gru = nn.GRU(
            input_size=model_dim,
            hidden_size=model_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = x * self.feature_gate(x)
        hidden = self.input_proj(x)
        hidden = hidden + self.position_embedding[:, : hidden.size(1)]
        for block in self.temporal_blocks:
            hidden = block(hidden)
        hidden, _ = self.gru(hidden)
        attn_hidden, _ = self.self_attention(hidden, hidden, hidden, need_weights=False)
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
        self.heads = nn.ModuleDict(
            {
                head_name: nn.Sequential(
                    nn.Linear(fusion_dim, task_hidden_dim),
                    nn.LayerNorm(task_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(task_hidden_dim, int(head_dim)),
                )
                for head_name, head_dim in head_dims.items()
            }
        )
        p_win_dim = int(head_dims.get("p_win", 0))
        ret_mu_dim = int(head_dims.get("ret_mu", p_win_dim))
        risk_dd_dim = int(head_dims.get("risk_dd", p_win_dim))
        bigloss_dim = int(head_dims.get("bigloss", 0))
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
        decision_feature_dim = fusion_dim + ret_mu_dim + ret_mu_dim + p_win_dim + risk_dd_dim + bigloss_dim + ret_mu_dim
        self.decision_score_head = nn.Sequential(
            nn.Linear(decision_feature_dim, task_hidden_dim),
            nn.LayerNorm(task_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(task_hidden_dim, max(1, p_win_dim)),
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
        outputs = {head_name: head(fused) for head_name, head in self.heads.items()}
        outputs["event_strength"] = event_strength
        outputs["ret_sigma"] = F.softplus(self.ret_sigma_head(fused)) + 1e-4
        outputs["upside"] = F.softplus(self.upside_head(fused))
        if "bigloss" in outputs:
            outputs["bigloss_prob"] = torch.sigmoid(outputs["bigloss"])
        decision_features = [
            fused,
            outputs.get("ret_mu", fused.new_zeros(fused.size(0), 0)),
            outputs["ret_sigma"],
            torch.sigmoid(outputs["p_win"]) if "p_win" in outputs else fused.new_zeros(fused.size(0), 0),
            outputs.get("risk_dd", fused.new_zeros(fused.size(0), 0)),
            torch.sigmoid(outputs["bigloss"]) if "bigloss" in outputs else fused.new_zeros(fused.size(0), 0),
            outputs["upside"],
        ]
        outputs["decision_score"] = self.decision_score_head(torch.cat(decision_features, dim=-1))
        if self.p_win_regime_head is not None and "p_win" in outputs:
            regime_logits = self.p_win_regime_head(context_repr)
            outputs["p_win_base"] = outputs["p_win"]
            outputs["p_win_regime"] = regime_logits
        return outputs


TinyMultiInputModel = ProfessionalFinancialModel
