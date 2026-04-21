"""Lightweight model components for A-share training."""

from .company_encoder import CompanyEncoder, compute_similarity_regularization

__all__ = ["CompanyEncoder", "compute_similarity_regularization"]
