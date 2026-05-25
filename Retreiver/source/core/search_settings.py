# source/core/search_settings.py
# type: ignore
from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

from pydantic import BaseModel, Field

from source.config import load_search_settings


# ----- small coercers -----
def _i(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default

def _f(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _fo(v: Any, default: Optional[float]) -> Optional[float]:
    if v is None:
        return default
    try:
        val = float(v)
        # clamp to 0..1 because this is a similarity threshold
        if val < 0.0: val = 0.0
        if val > 1.0: val = 1.0
        return val
    except Exception:
        return default


class SearchSettings(BaseModel):
    # Embedding/model defaults used by search (e.g., zero-vector fallback)
    default_embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    embed_dimension_fallback: int = Field(default=1024, ge=1)

    # Top-k & thresholds
    top_k_default: int = Field(default=5, ge=1)
    top_k_cap: int = Field(default=100, ge=1)
    simple_limit_cap: int = Field(default=50, ge=1)  # GET /simple limit max
    score_threshold_default: float | None = Field(default=None, ge=0.0, le=1.0)

    # Document content paging
    content_page_size_default: int = Field(default=20, ge=1, le=200)
    content_fetch_hard_cap: int = Field(default=2000, ge=1)  # max chunks fetched before pagination

    # Stats & estimation
    doc_chunks_per_document_estimate: int = Field(default=10, ge=1)

    # Concurrency for stats calls
    stats_concurrency: int = Field(default=8, ge=1)


@lru_cache(maxsize=1)
def get_search_settings() -> SearchSettings:
    cfg = load_search_settings() or {}

    # Build with safe coercion + sane defaults
    top_k_cap = _i(cfg.get("top_k_cap", 100), 100)
    top_k_default = _i(cfg.get("top_k_default", 5), 5)
    if top_k_default < 1:
        top_k_default = 1
    if top_k_default > top_k_cap:
        top_k_default = top_k_cap

    content_page_size_default = _i(cfg.get("content_page_size_default", 20), 20)
    # Keep this bounded so you don't accidentally return thousands per page
    if content_page_size_default < 1:
        content_page_size_default = 1
    if content_page_size_default > 200:
        content_page_size_default = 200

    s = SearchSettings(
        default_embedding_model=str(cfg.get("default_embedding_model") or "nvidia/nv-embedqa-e5-v5"),
        embed_dimension_fallback=_i(cfg.get("embed_dimension_fallback", 1024), 1024),

        top_k_default=top_k_default,
        top_k_cap=top_k_cap,
        simple_limit_cap=_i(cfg.get("simple_limit_cap", 50), 50),
        score_threshold_default=_fo(cfg.get("score_threshold_default", None), None),

        content_page_size_default=content_page_size_default,
        content_fetch_hard_cap=_i(cfg.get("content_fetch_hard_cap", 2000), 2000),

        doc_chunks_per_document_estimate=_i(cfg.get("doc_chunks_per_document_estimate", 10), 10),
        stats_concurrency=_i(cfg.get("stats_concurrency", 8), 8),
    )
    return s
