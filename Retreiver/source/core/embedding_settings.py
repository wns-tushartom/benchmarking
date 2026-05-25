# source/core/embedding_settings.py
# type: ignore
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from source.config import load_embedding_settings, load_cloud_services


# ---------- helpers ----------
def _coerce_bool(v: Any, default: bool) -> bool:
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
        # If it's a descriptive string (like your current YAML), assume feature ON
        return default
    return default


def _coerce_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _normalize_available_models(cfg: Dict[str, Any], cloud: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Accepts either:
      - proper mapping: embedding_settings.available_models: { model: {meta...}, ... }
      - numbered keys: available_models_one/two + description_one/two + dimension_one/two + recommended_one/two
    Also folds in cloud_services' NVIDIA model hints if present.
    """
    out: Dict[str, Dict[str, Any]] = {}

    # Preferred shape
    if isinstance(cfg.get("available_models"), dict) and cfg["available_models"]:
        # ensure minimal normalization
        for name, meta in cfg["available_models"].items():
            if not isinstance(meta, dict):
                meta = {"description": str(meta)}
            out[str(name)] = {
                "description": str(meta.get("description", "")),
                "dimension": _coerce_int(meta.get("dimension", 1024), 1024),
                "recommended": bool(meta.get("recommended", False)),
            }

    # Legacy numbered shape
    numbered = [
        ("one", cfg.get("available_models_one")),
        ("two", cfg.get("available_models_two")),
        ("three", cfg.get("available_models_three")),
    ]
    for suffix, model_name in numbered:
        if not model_name:
            continue
        desc = cfg.get(f"description_{suffix}", "")
        dim = _coerce_int(cfg.get(f"dimension_{suffix}", 1024), 1024)
        rec = bool(cfg.get(f"recommended_{suffix}", False))
        out[str(model_name)] = {"description": str(desc), "dimension": dim, "recommended": rec}

    # Fold in cloud NVIDIA hints if useful
    nv_main = cloud.get("nvidia_embedding_model")
    if nv_main and nv_main not in out:
        out[str(nv_main)] = {
            "description": "NVIDIA embedding (cloud hint)",
            "dimension": _coerce_int(cfg.get("embed_dimension_fallback", 1024), 1024),
            "recommended": True if not out else False,
        }
    nv_fallback = cloud.get("nvidia_fallback_embedding_model")
    if nv_fallback and nv_fallback not in out:
        out[str(nv_fallback)] = {
            "description": "NVIDIA embedding fallback (cloud hint)",
            "dimension": _coerce_int(cfg.get("embed_dimension_fallback", 1024), 1024),
            "recommended": False,
        }

    return out


# ---------- model ----------
class EmbeddingSettings(BaseModel):
    # Models
    default_model: str = Field(default="nvidia/nv-embedqa-e5-v5")
    enable_model_resolver: bool = Field(default=True, description="Use embedding_service._resolve_model if available")

    # Batch limits / sizes
    single_batch_max_texts: int = Field(default=100, ge=1)
    compare_max_texts: int = Field(default=50, ge=1)
    cluster_max_texts: int = Field(default=20, ge=2)
    embedding_batch_size: int = Field(default=64, ge=1)

    # Similarity / clustering
    top_k_default: int = Field(default=10, ge=1)
    similarity_threshold_default: float = Field(default=0.8, ge=0.0, le=1.0)
    similarity_matrix_limit: int = Field(default=50, ge=1)

    # Models catalog (used by /models fallback)
    available_models: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def get_embedding_settings() -> EmbeddingSettings:
    cfg = load_embedding_settings() or {}
    cloud = load_cloud_services() or {}

    # Build available_models first (needed to pick default sensibly)
    models = _normalize_available_models(cfg, cloud)

    # default_model preference: explicit → cfg.default_model → first available → NVIDIA hint → fallback
    default_model = str(
        cfg.get("default_model")
        or (next(iter(models.keys())) if models else None)
        or cloud.get("nvidia_embedding_model")
        or "NV-Embed-QA"
    )

    settings = EmbeddingSettings(
        default_model=default_model,
        enable_model_resolver=_coerce_bool(cfg.get("enable_model_resolver", True), True),
        single_batch_max_texts=_coerce_int(cfg.get("single_batch_max_texts", 100), 100),
        compare_max_texts=_coerce_int(cfg.get("compare_max_texts", 50), 50),
        cluster_max_texts=_coerce_int(cfg.get("cluster_max_texts", 20), 20),
        embedding_batch_size=_coerce_int(cfg.get("embedding_batch_size", 64), 64),
        # If duplicated in YAML (you had two top_k_default), the last assignment wins; we force a single value:
        top_k_default=_coerce_int(cfg.get("top_k_default", 10), 10),
        similarity_threshold_default=float(cfg.get("similarity_threshold_default", 0.8)),
        similarity_matrix_limit=_coerce_int(cfg.get("similarity_matrix_limit", 50), 50),
        available_models=models,
    )
    return settings
