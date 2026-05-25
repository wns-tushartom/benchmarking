# source/core/query_analyzer_settings.py
# type: ignore
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from source.config import load_query_settings


# ---------- coercers ----------
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

def _b(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "0", "no", "n", "off"}:
            return False
    return default


class QueryAnalyzerSettings(BaseModel):
    # Estimation / complexity
    chars_per_token: int = Field(default=4, ge=1)
    complex_threshold_tokens: int = Field(default=3000, ge=1)
    research_threshold_tokens: int = Field(default=6000, ge=1)

    # Output shaping limits
    max_key_concepts: int = Field(default=8, ge=1)
    max_sub_questions: int = Field(default=6, ge=1)
    max_search_terms: int = Field(default=8, ge=1)

    # Heuristics
    ambiguity_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    multi_step_min_complexity: str = Field(default="moderate")  # "simple"|"moderate"|"complex"|"research_intensive"

    # Retrieval/search preferences
    enable_web_research: bool = True
    default_search_scope: str = Field(default="focused")  # "focused"|"broad"
    semantic_first: bool = True
    allowed_sources: List[str] = Field(default_factory=lambda: ["vector_store"])

    # Domain hints: {"finance":["earnings","pe ratio"], ...}
    domain_hints: Dict[str, List[str]] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def get_query_analyzer_settings() -> QueryAnalyzerSettings:
    cfg = load_query_settings() or {}

    return QueryAnalyzerSettings(
        chars_per_token=_i(cfg.get("chars_per_token", 4), 4),
        complex_threshold_tokens=_i(cfg.get("complex_threshold_tokens", 3000), 3000),
        research_threshold_tokens=_i(cfg.get("research_threshold_tokens", 6000), 6000),

        max_key_concepts=_i(cfg.get("max_key_concepts", 8), 8),
        max_sub_questions=_i(cfg.get("max_sub_questions", 6), 6),
        max_search_terms=_i(cfg.get("max_search_terms", 8), 8),

        ambiguity_threshold=_f(cfg.get("ambiguity_threshold", 0.4), 0.4),
        multi_step_min_complexity=str(cfg.get("multi_step_min_complexity", "moderate")),

        enable_web_research=_b(cfg.get("enable_web_research", True), True),
        default_search_scope=str(cfg.get("default_search_scope", "focused")),
        semantic_first=_b(cfg.get("semantic_first", True), True),
        allowed_sources=list(cfg.get("allowed_sources") or ["vector_store"]),

        domain_hints=dict(cfg.get("domain_hints") or {}),
    )
