# source/core/llm_settings.py
# type: ignore
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from source.config import load_llm_settings


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


def _normalize_available_models(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Supports:
      1) llm_settings.available_models: { model: {description,max_tokens,...}, ... }
      2) numbered keys (your current YAML):
         available_models_one/two/three + *_description + *_max_tokens
    """
    out: Dict[str, Dict[str, Any]] = {}

    # Preferred mapping shape
    if isinstance(cfg.get("available_models"), dict) and cfg["available_models"]:
        for name, meta in cfg["available_models"].items():
            meta = meta or {}
            if not isinstance(meta, dict):
                meta = {"description": str(meta)}
            out[str(name)] = {
                "description": str(meta.get("description", "")),
                "max_tokens": _i(meta.get("max_tokens", 128_000), 128_000),
                "in_consideration": _b(meta.get("in_consideration", False), False),
            }

    # Legacy numbered keys
    slots = ["one", "two", "three", "four"]
    for s in slots:
        name = cfg.get(f"available_models_{s}")
        if not name:
            continue
        out[str(name)] = {
            "description": str(cfg.get(f"available_models_{s}_description", cfg.get(f"description_{s}", ""))),
            "max_tokens": _i(cfg.get(f"available_models_{s}_max_tokens", cfg.get(f"max_tokens_{s}", 128_000)), 128_000),
            "in_consideration": _b(cfg.get(f"available_models_{s}_in_consideration", cfg.get(f"in_consideration_{s}", False)), False),
        }

    return out


class LLMSettings(BaseModel):
    # --- Models / Routing ---
    default_model: str = "gpt-4o"
    fast_model: str = "gpt-4o-mini"  # note: your YAML uses "gpt-40-mini"; we'll accept whatever is set there
    enable_azure_resolver: bool = Field(default=True, description="If client exposes _resolve_model, use it")

    # --- Tokens / Temperature / Context ---
    max_tokens_default: int = Field(default=2000, ge=1)
    max_tokens_cap: int = Field(default=128_000, ge=1)
    temperature_default: float = Field(default=0.7, ge=0.0, le=2.0)
    context_k_default: int = Field(default=5, ge=1)  # chunks for single-doc chat
    context_k_per_doc_default: int = Field(default=3, ge=1)  # chunks per doc (multi-doc)
    followup_top_k: int = Field(default=5, ge=1)

    # --- Request limits ---
    max_documents_per_request: int = Field(default=5, ge=1)

    # --- SSE / Streaming ---
    sse_heartbeat_ms: int = Field(default=15_000, ge=0, description="0 disables heartbeats")
    sse_headers_cache_control: str = "no-cache"
    sse_headers_connection: str = "keep-alive"
    sse_headers_accel_buffering: str = "no"

    # --- Summary types ---
    summary_types: List[str] = Field(default_factory=lambda: ["comprehensive", "key_points", "abstract", "brief"])
    summary_default: str = "comprehensive"
    # NEW: how many chunks to fetch for summarization context (top-K retrieval)
    summary_context_cap: int = Field(default=16, ge=1)

    # --- Catalog of models (for /models endpoint) ---
    available_models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    recommended_for_rag: str = "gpt-4o"


@lru_cache(maxsize=1)
def get_llm_settings() -> LLMSettings:
    cfg = load_llm_settings() or {}

    models = _normalize_available_models(cfg)

    default_model = str(cfg.get("default_model") or "gpt-4o")
    fast_model = str(
        cfg.get("default_fast_model")
        or cfg.get("fast_model")
        or "gpt-4o-mini"  # fallback
    )

    # Allow misspelled description key in your YAML without breaking anything
    _ = cfg.get("default_model_description") or cfg.get("default_model_descritpion")

    # Read summary_context_cap from YAML or ENV (ENV wins if present and valid)
    cap_src: Any = cfg.get("summary_context_cap", None)
    env_cap = os.getenv("LLM_SUMMARY_CONTEXT_CAP", "").strip()
    if env_cap:
        cap_src = env_cap  # env override
    summary_context_cap = _i(cap_src if cap_src is not None else 16, 16)

    s = LLMSettings(
        default_model=default_model,
        fast_model=fast_model,
        enable_azure_resolver=_b(cfg.get("enable_azure_resolver", True), True),

        max_tokens_default=_i(cfg.get("max_tokens_default", 2000), 2000),
        max_tokens_cap=_i(cfg.get("max_tokens_cap", 128_000), 128_000),
        temperature_default=_f(cfg.get("temperature_default", 0.7), 0.7),
        context_k_default=_i(cfg.get("context_k_default", 5), 5),
        context_k_per_doc_default=_i(cfg.get("context_k_per_doc_default", 3), 3),
        followup_top_k=_i(cfg.get("followup_top_k", 5), 5),

        max_documents_per_request=_i(cfg.get("max_documents_per_request", 5), 5),

        sse_heartbeat_ms=_i(cfg.get("sse_heartbeat_ms", 15_000), 15_000),
        sse_headers_cache_control=str(cfg.get("sse_headers_cache_control", "no-cache")),
        sse_headers_connection=str(cfg.get("sse_headers_connection", "keep-alive")),
        sse_headers_accel_buffering=str(cfg.get("sse_headers_accel_buffering", "no")),

        summary_types=list(cfg.get("summary_types") or ["comprehensive", "key_points", "abstract", "brief"]),
        summary_default=str(cfg.get("summary_default") or "comprehensive"),
        summary_context_cap=summary_context_cap,

        available_models=models,
        recommended_for_rag=str(cfg.get("recommended_model_for_rag") or default_model),
    )
    return s
