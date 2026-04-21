from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.company_profile_builder import COMPANY_PROFILE_COLUMNS, build_company_id_maps
from models.company_encoder import CompanyEncoder
from utils.config import load_config
from utils.torch_runtime import resolve_torch_device
from warehouse.repository import WarehouseRepository


def _latest_profile(profiles: pd.DataFrame, symbol: str) -> pd.Series | None:
    if profiles.empty:
        return None
    df = profiles[profiles["symbol"] == symbol].copy()
    if df.empty:
        return None
    df["asof_date"] = pd.to_datetime(df["asof_date"])
    return df.sort_values("asof_date").iloc[-1]


def _build_encoder(config, samples: pd.DataFrame | None, securities: pd.DataFrame) -> CompanyEncoder:
    identity_map = build_company_id_maps(securities)
    num_symbols = int(identity_map["symbol_id"].max()) + 1 if not identity_map.empty else 1
    num_industries = int(identity_map["industry_id"].max()) + 1 if not identity_map.empty else 1
    num_boards = int(identity_map["board_id"].max()) + 1 if not identity_map.empty else 1
    profile_dim = len([column for column in COMPANY_PROFILE_COLUMNS if samples is None or column in samples.columns])
    company_conf = config.company_encoding
    return CompanyEncoder(
        num_symbols=num_symbols,
        num_industries=num_industries,
        num_boards=num_boards,
        profile_input_dim=max(1, profile_dim),
        embedding_dim=int(company_conf.get("embedding_dim", 32)),
        symbol_emb_dim=int(company_conf.get("symbol_emb_dim", 16)),
        industry_emb_dim=int(company_conf.get("industry_emb_dim", 8)),
        board_emb_dim=int(company_conf.get("board_emb_dim", 4)),
        profile_hidden_dims=tuple(company_conf.get("profile_hidden_dims", [64, 32])),
    )


def _encode_company(
    encoder: CompanyEncoder,
    identity_map: pd.DataFrame,
    profile_row: pd.Series | None,
    symbol: str,
    device: torch.device,
) -> torch.Tensor | None:
    if profile_row is None or identity_map.empty:
        return None
    row = identity_map[identity_map["symbol"] == symbol]
    if row.empty:
        return None
    identity = row.iloc[0]
    profile_values = []
    for column in COMPANY_PROFILE_COLUMNS:
        value = pd.to_numeric(profile_row.get(column), errors="coerce")
        profile_values.append(float(0.0 if pd.isna(value) else value))
    with torch.inference_mode():
        return encoder(
            symbol_id=torch.tensor([int(identity.get("symbol_id", 0))], dtype=torch.long, device=device),
            industry_id=torch.tensor([int(identity.get("industry_id", 0))], dtype=torch.long, device=device),
            board_id=torch.tensor([int(identity.get("board_id", 0))], dtype=torch.long, device=device),
            company_profile_features=torch.tensor([profile_values], dtype=torch.float32, device=device),
        ).squeeze(0).detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect top-k similar companies for one symbol")
    parser.add_argument("symbol", help="Stock symbol, e.g. 601857.SH")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config yaml")
    parser.add_argument("--topk", type=int, default=10, help="Number of neighbors to print")
    parser.add_argument("--checkpoint", default=None, help="Optional CompanyEncoder checkpoint path")
    args = parser.parse_args()

    config = load_config(args.config)
    repo = WarehouseRepository.from_config_path(args.config)
    device = resolve_torch_device(config.runtime.get("torch_device", "auto"))
    if bool(config.runtime.get("require_gpu", False)) and device.type != "cuda":
        raise RuntimeError("runtime.require_gpu=true but CUDA is not available")
    securities = repo.get_securities(active_only=False)
    profiles = repo.fetch_table("company_profiles")
    similarity = repo.fetch_table("company_similarity")
    identity_map = build_company_id_maps(securities)

    encoder = _build_encoder(config, profiles, securities).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        encoder.load_state_dict(state_dict, strict=False)
        print(f"loaded encoder checkpoint: {args.checkpoint}")
    else:
        print("no checkpoint provided, embedding norm uses current encoder initialization")
    encoder.eval()
    print(f"torch_device: {device}")

    symbol = args.symbol.upper()
    source_profile = _latest_profile(profiles, symbol)
    source_embedding = _encode_company(encoder, identity_map, source_profile, symbol, device)
    if source_profile is None:
        print(f"symbol {symbol} has no company profile snapshot")
        return

    sec_row = securities[securities["symbol"] == symbol]
    industry = sec_row["industry"].iloc[0] if not sec_row.empty else None
    board = sec_row["board"].iloc[0] if not sec_row.empty else None
    emb_norm = float(source_embedding.norm(p=2).item()) if source_embedding is not None else float("nan")

    print(f"symbol: {symbol}")
    print(f"industry: {industry}")
    print(f"board: {board}")
    print(f"embedding_l2_norm: {emb_norm:.6f}")
    print("profile_summary:")
    for column in COMPANY_PROFILE_COLUMNS:
        value = source_profile.get(column)
        print(f"  {column}: {value}")

    sim_df = similarity[similarity["symbol"] == symbol].copy()
    if sim_df.empty:
        print("no company similarity rows found")
        return

    sim_df = sim_df.sort_values("sim_rank").head(args.topk)
    print("top_neighbors:")
    for _, row in sim_df.iterrows():
        neighbor_symbol = str(row["neighbor_symbol"])
        neighbor_profile = _latest_profile(profiles, neighbor_symbol)
        neighbor_embedding = _encode_company(encoder, identity_map, neighbor_profile, neighbor_symbol, device)
        neighbor_norm = float(neighbor_embedding.norm(p=2).item()) if neighbor_embedding is not None else float("nan")
        neighbor_sec = securities[securities["symbol"] == neighbor_symbol]
        neighbor_industry = neighbor_sec["industry"].iloc[0] if not neighbor_sec.empty else None
        profile_excerpt = {}
        if neighbor_profile is not None:
            for column in ["market_cap_log", "roe", "profit_yoy", "volatility_120"]:
                profile_excerpt[column] = neighbor_profile.get(column)
        print(
            f"  rank={int(row['sim_rank'])} symbol={neighbor_symbol} sim_score={float(row['sim_score']):.4f} "
            f"industry={neighbor_industry} emb_norm={neighbor_norm:.6f} profile={profile_excerpt}"
        )


if __name__ == "__main__":
    main()
