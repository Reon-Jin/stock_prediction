from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class _MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        dims = [int(input_dim), *(int(dim) for dim in hidden_dims), int(output_dim)]
        layers: list[nn.Module] = []
        for idx in range(len(dims) - 1):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[idx + 1]))
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompanyEncoder(nn.Module):
    def __init__(
        self,
        num_symbols: int,
        num_industries: int,
        num_boards: int,
        profile_input_dim: int,
        embedding_dim: int = 32,
        symbol_emb_dim: int = 16,
        industry_emb_dim: int = 8,
        board_emb_dim: int = 4,
        profile_hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.symbol_embedding = nn.Embedding(max(1, int(num_symbols)), int(symbol_emb_dim), padding_idx=0)
        self.industry_embedding = nn.Embedding(max(1, int(num_industries)), int(industry_emb_dim), padding_idx=0)
        self.board_embedding = nn.Embedding(max(1, int(num_boards)), int(board_emb_dim), padding_idx=0)
        self.symbol_dropout = nn.Dropout(dropout)

        profile_output_dim = profile_hidden_dims[-1] if profile_hidden_dims else int(embedding_dim)
        self.profile_mlp = _MLP(
            input_dim=max(1, int(profile_input_dim)),
            hidden_dims=profile_hidden_dims[:-1] if profile_hidden_dims else (),
            output_dim=int(profile_output_dim),
            dropout=dropout,
        )
        self.profile_norm = nn.LayerNorm(int(profile_output_dim))
        self.industry_profile_gate = nn.Sequential(
            nn.Linear(int(industry_emb_dim) + int(profile_output_dim), int(profile_output_dim)),
            nn.LayerNorm(int(profile_output_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(profile_output_dim), int(profile_output_dim)),
            nn.Sigmoid(),
        )
        self.board_profile_gate = nn.Sequential(
            nn.Linear(int(board_emb_dim) + int(profile_output_dim), int(profile_output_dim)),
            nn.LayerNorm(int(profile_output_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(profile_output_dim), int(profile_output_dim)),
            nn.Sigmoid(),
        )
        concat_dim = int(symbol_emb_dim) + int(industry_emb_dim) + int(board_emb_dim) + int(profile_output_dim)
        self.layer_norm = nn.LayerNorm(concat_dim)
        self.output_mlp = _MLP(
            input_dim=concat_dim,
            hidden_dims=(max(int(embedding_dim), concat_dim // 2),),
            output_dim=int(embedding_dim),
            dropout=dropout,
        )
        self.residual_proj = nn.Linear(concat_dim, int(embedding_dim))
        self.output_norm = nn.LayerNorm(int(embedding_dim))

    def forward(
        self,
        symbol_id: torch.Tensor,
        industry_id: torch.Tensor,
        board_id: torch.Tensor,
        company_profile_features: torch.Tensor,
    ) -> torch.Tensor:
        symbol_emb = self.symbol_dropout(self.symbol_embedding(symbol_id.long()))
        industry_emb = self.industry_embedding(industry_id.long())
        board_emb = self.board_embedding(board_id.long())
        profile_emb = self.profile_norm(self.profile_mlp(company_profile_features.float()))
        industry_gate = self.industry_profile_gate(torch.cat([industry_emb, profile_emb], dim=-1))
        board_gate = self.board_profile_gate(torch.cat([board_emb, profile_emb], dim=-1))
        profile_emb = profile_emb * (1.0 + 0.5 * industry_gate + 0.5 * board_gate)
        fused = torch.cat([symbol_emb, industry_emb, board_emb, profile_emb], dim=-1)
        fused = self.layer_norm(fused)
        output = self.output_mlp(fused)
        residual = self.residual_proj(fused)
        return self.output_norm(output + residual)


def compute_similarity_regularization(
    company_embeddings: torch.Tensor,
    symbol_ids: torch.Tensor,
    neighbor_symbol_ids: torch.Tensor | None = None,
    neighbor_scores: torch.Tensor | None = None,
    lambda_sim: float = 0.05,
) -> torch.Tensor:
    if neighbor_symbol_ids is None or neighbor_scores is None:
        return company_embeddings.new_tensor(0.0)
    if company_embeddings.ndim != 2 or symbol_ids.ndim != 1:
        raise ValueError("company_embeddings must be [B, D] and symbol_ids must be [B]")

    batch_lookup = {int(symbol_id.item()): idx for idx, symbol_id in enumerate(symbol_ids)}
    losses: list[torch.Tensor] = []
    for row_idx in range(company_embeddings.size(0)):
        for col_idx in range(neighbor_symbol_ids.size(1)):
            neighbor_id = int(neighbor_symbol_ids[row_idx, col_idx].item())
            weight = float(neighbor_scores[row_idx, col_idx].item())
            if neighbor_id <= 0 or weight <= 0:
                continue
            neighbor_pos = batch_lookup.get(neighbor_id)
            if neighbor_pos is None:
                continue
            diff = company_embeddings[row_idx] - company_embeddings[neighbor_pos]
            losses.append(diff.pow(2).sum() * weight)
    if not losses:
        return company_embeddings.new_tensor(0.0)
    sim_loss = torch.stack(losses).mean()
    return sim_loss * float(lambda_sim)
