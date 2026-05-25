# source/config.py
# type: ignore

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

# --------------------------------------------------------------------------------------
# Paths
#   Default layout: <repo_root>/config/QA_Text/config.yaml
# --------------------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parents[1]  # .../source -> repo root
CONFIG_PATH = _BASE / "config" / "QA_Text" / "config.yaml"

# In-process cache
__CONFIG_CACHE: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _as_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return default

def _as_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return default

def _as_bool(v: Any, default: Optional[bool] = None) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
    return default


# --------------------------------------------------------------------------------------
# Normalizers (mutate dict in-place; tolerant to missing keys)
# --------------------------------------------------------------------------------------
def _normalize_llm(cfg: Dict[str, Any]) -> None:
    sec = cfg.get("llm_settings") or {}
    if not isinstance(sec, dict):
        cfg["llm_settings"] = {}
        return

    # fix common typo/value
    if sec.get("default_fast_model") == "gpt-40-mini":
        sec["default_fast_model"] = "gpt-4o-mini"

    # ensure boolean for resolver
    if "enable_azure_resolver" not in sec:
        # fall back to the description key or default True
        sec["enable_azure_resolver"] = True if sec.get("enable_azure_resolver_description") else True

    # consolidate available models (old *_one/_two to a list)
    if "available_models" not in sec:
        models: List[Dict[str, Any]] = []
        if "available_models_one" in sec:
            models.append({
                "model": sec.get("available_models_one"),
                "description": sec.get("available_models_one_description", ""),
                "max_tokens": _as_int(sec.get("available_models_one_max_tokens"), None),
            })
        if "available_models_two" in sec:
            models.append({
                "model": sec.get("available_models_two"),
                "description": sec.get("available_models_two_description", ""),
                "max_tokens": _as_int(sec.get("available_models_two_max_tokens"), None),
            })
        if models:
            sec["available_models"] = models

    # Safe numeric coercions
    for k in (
        "max_tokens_default", "max_tokens_cap",
        "context_k_default", "context_k_per_doc_default",
        "followup_top_k", "max_documents_per_request",
        "sse_heartbeat_ms",
    ):
        if k in sec:
            sec[k] = _as_int(sec[k], sec[k])

    # keep section
    cfg["llm_settings"] = sec


def _normalize_embeddings(cfg: Dict[str, Any]) -> None:
    sec = cfg.get("embedding_settings") or {}
    if not isinstance(sec, dict):
        cfg["embedding_settings"] = {}
        return

    # boolean for resolver
    if "enable_model_resolver" in sec:
        sec["enable_model_resolver"] = _as_bool(sec["enable_model_resolver"], True)

    # de-dup & coerce numbers
    for k in (
        "single_batch_max_texts", "compare_max_texts", "cluster_max_texts",
        "similarity_matrix_limit", "top_k_default",
    ):
        if k in sec:
            sec[k] = _as_int(sec[k], sec[k])

    if "similarity_threshold_default" in sec:
        sec["similarity_threshold_default"] = _as_float(sec["similarity_threshold_default"], 0.8)

    # build normalized available_models list if missing
    if "available_models" not in sec:
        models: List[Dict[str, Any]] = []
        if sec.get("available_models_one"):
            models.append({
                "model": sec.get("available_models_one"),
                "description": sec.get("description_one", ""),
                "dimension": _as_int(sec.get("dimension_one"), None),
                "recommended": bool(sec.get("recommended_one", False)),
            })
        if sec.get("available_models_two"):
            models.append({
                "model": sec.get("available_models_two"),
                "description": sec.get("description_two", ""),
                "dimension": _as_int(sec.get("dimension_two"), None),
                "recommended": bool(sec.get("recommended_two", False)),
            })
        if models:
            sec["available_models"] = models

    cfg["embedding_settings"] = sec


def _normalize_documents(cfg: Dict[str, Any]) -> None:
    sec = cfg.get("document_settings") or {}
    if not isinstance(sec, dict):
        cfg["document_settings"] = {}
        return

    # coerce optional numeric fields if present
    for k in ("max_file_size", "upload_chunk_size", "batch_upload_concurrency"):
        if k in sec:
            sec[k] = _as_int(sec[k], sec[k])

    cfg["document_settings"] = sec


def _normalize_search(cfg: Dict[str, Any]) -> None:
    sec = cfg.get("search_settings") or {}
    if not isinstance(sec, dict):
        cfg["search_settings"] = {}
        return

    for k in (
        "embed_dimension_fallback", "top_k_default", "top_k_cap", "simple_limit_cap",
        "content_page_size_default", "content_fetch_hard_cap",
        "doc_chunks_per_document_estimate", "stats_concurrency",
    ):
        if k in sec:
            sec[k] = _as_int(sec[k], sec[k])

    cfg["search_settings"] = sec


def _normalize_cloud_services(cfg: Dict[str, Any]) -> None:
    sec = cfg.get("cloud_services") or {}
    if not isinstance(sec, dict):
        cfg["cloud_services"] = {}
        return

    # numeric coercions
    for k in ("milvus_port", "api_port", "embedding_dim", "max_file_size", "chunk_size", "chunk_overlap"):
        if k in sec:
            sec[k] = _as_int(sec[k], sec[k])

    cfg["cloud_services"] = sec


def _normalize_in_place(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply all normalizers safely; return cfg for convenience."""
    _normalize_llm(cfg)
    _normalize_embeddings(cfg)
    _normalize_documents(cfg)
    _normalize_search(cfg)
    _normalize_cloud_services(cfg)
    return cfg


# --------------------------------------------------------------------------------------
# Core I/O
# --------------------------------------------------------------------------------------
def _read_config() -> Dict[str, Any]:
    """Read and cache the YAML config once; normalize for callers."""
    global __CONFIG_CACHE
    if __CONFIG_CACHE is not None:
        return __CONFIG_CACHE

    if not CONFIG_PATH.exists():
        logger.warning("Config file not found: %s", CONFIG_PATH)
        __CONFIG_CACHE = {}
        return __CONFIG_CACHE

    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(text) or {}
        if not isinstance(data, dict):
            logger.warning("Top-level YAML is not a mapping; got %s", type(data))
            data = {}
        __CONFIG_CACHE = _normalize_in_place(data)
        return __CONFIG_CACHE
    except Exception as e:
        logger.error("Failed to load YAML config %s: %s", CONFIG_PATH, e)
        __CONFIG_CACHE = {}
        return __CONFIG_CACHE


def reload_config() -> Dict[str, Any]:
    """Force re-read of the YAML file (clears cache)."""
    global __CONFIG_CACHE
    __CONFIG_CACHE = None
    return _read_config()


def get_config_path() -> Path:
    """Return the resolved config file path."""
    return CONFIG_PATH


def _get(section: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Safe getter for a mapping section."""
    cfg = _read_config()
    val = cfg.get(section, default or {})
    if not isinstance(val, dict):
        logger.warning("Section '%s' is not a mapping; returning empty dict", section)
        return {}
    return val


# --------------------------------------------------------------------------------------
# Public API - compatible with existing callers
# --------------------------------------------------------------------------------------
def load_config() -> Dict[str, Any]:
    """Load the entire configuration."""
    return _read_config()

def load_llm_settings() -> Dict[str, Any]:
    return _get("llm_settings", {})

def load_document_settings() -> Dict[str, Any]:
    return _get("document_settings", {})

def load_embedding_settings() -> Dict[str, Any]:
    return _get("embedding_settings", {})

def load_search_settings() -> Dict[str, Any]:
    return _get("search_settings", {})

def load_llm_service() -> Dict[str, Any]:
    # Some codepaths expect 'llm_service' (older naming). Keep it.
    return _get("llm_service", {})

def load_document_parser() -> Dict[str, Any]:
    return _get("document_parser", {})

def load_cloud_services() -> Dict[str, Any]:
    return _get("cloud_services", {})


# --------------------------------------------------------------------------------------
# New / adjusted helpers
# --------------------------------------------------------------------------------------
def load_llm_client() -> Dict[str, Any]:
    """
    Config for the centralized LLM client service.

    Resolution order (most specific first):
      1) llm_client
      2) llm_service (back-compat)
      3) llm_settings (back-compat)
    """
    cfg = _read_config()
    for key in ("llm_client", "llm_service", "llm_settings"):
        sec = cfg.get(key)
        if isinstance(sec, dict) and sec:
            return sec
    return {}

def load_query_settings() -> Dict[str, Any]:
    """
    Preferred by the new Query Analyzer. Falls back to search_settings if
    query_settings is not present to avoid breaking older configs.
    """
    cfg = _read_config()
    qs = cfg.get("query_settings")
    if isinstance(qs, dict) and qs:
        return qs
    return load_search_settings()

def load_milvus_settings() -> Dict[str, Any]:
    """
    Milvus-only helper. We aggregate from any of your current locations:
    - vector_store.milvus (future)
    - cloud_services (current)
    - llm_service (legacy)
    Returns a single dict with host/port/uri + collection defaults.
    """
    cfg = _read_config()

    # 1) vector_store.milvus (if you add this later)
    vs = cfg.get("vector_store") or {}
    if isinstance(vs, dict):
        milvus = vs.get("milvus")
        if isinstance(milvus, dict) and milvus:
            return milvus

    # 2) cloud_services (your current primary store)
    cs = cfg.get("cloud_services") or {}
    if isinstance(cs, dict) and (cs.get("milvus_uri") or cs.get("milvus_host")):
        return {
            "uri": cs.get("milvus_uri") or f"http://{cs.get('milvus_host','localhost')}:{cs.get('milvus_port',19530)}",
            "host": cs.get("milvus_host", "localhost"),
            "port": _as_int(cs.get("milvus_port"), 19530),
            "default_collection": cs.get("default_collection") or cs.get("default_collections") or "documents",
            "metric_type": (cs.get("metric_type") or "COSINE").upper(),
            "embedding_dim": _as_int(cs.get("embedding_dim"), None),
        }

    # 3) llm_service (legacy backup)
    ls = cfg.get("llm_service") or {}
    if isinstance(ls, dict) and (ls.get("milvus_host") or ls.get("milvus_port")):
        host = ls.get("milvus_host", "localhost")
        port = _as_int(ls.get("milvus_port"), 19530)
        return {
            "uri": f"http://{host}:{port}",
            "host": host,
            "port": port,
            "default_collection": "documents",
            "metric_type": (ls.get("metric") or "COSINE").upper(),
        }

    # Fallback: empty dict
    return {}
