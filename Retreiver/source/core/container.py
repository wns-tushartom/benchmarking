# source/core/container.py
from __future__ import annotations

"""
Bootstrapper that:
- Loads YAML via source.config
- Registers core providers in the global registry
- Exposes convenience getters

Safe to import multiple times; bootstrap() is idempotent per-process.
"""

import logging
import threading
from typing import Any

from source.core.registry import register, set_default, resolve, registry as _registry
from source.config import (
    load_llm_settings,
    load_embedding_settings,
    load_search_settings,
    load_document_settings,
    load_cloud_services,
    load_llm_service,
)

# Optional settings modules for seeding/config
try:
    from source.core.prompt_manager_settings import get_prompt_manager_settings
except Exception:  # pragma: no cover
    get_prompt_manager_settings = None  # type: ignore

try:
    from source.core.query_analyzer_settings import get_query_analyzer_settings
except Exception:  # pragma: no cover
    get_query_analyzer_settings = None  # type: ignore

# Avoid configuring the root logger here; main/app should own logging.
log = logging.getLogger(__name__)
if not log.handlers:
    # Use a NullHandler so importing this module never configures global logging.
    log.addHandler(logging.NullHandler())

_BOOTSTRAPPED = False
_LOCK = threading.RLock()


# ---------- small helpers ----------

def _maybe_call(obj: Any, method: str, *args, **kwargs):
    fn = getattr(obj, method, None)
    if callable(fn):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            log.debug("Call %s.%s failed: %s", type(obj).__name__, method, e, exc_info=True)
    return None


def _detect_vector_provider() -> str:
    cs = load_cloud_services() or {}
    # explicit override
    prov = (cs.get("vector_provider") or "").strip().lower()
    if prov:
        return prov
    # heuristics
    if cs.get("weaviate_url"):
        return "weaviate"
    if cs.get("pinecone_api_key") or cs.get("pinecone_environment"):
        return "pinecone"
    if cs.get("qdrant_url"):
        return "qdrant"
    if cs.get("chroma_path"):
        return "chroma"
    if cs.get("pgvector_dsn") or cs.get("postgres_url"):
        return "pgvector"
    if cs.get("milvus_uri") or cs.get("milvus_host"):
        return "milvus"
    # default
    return "milvus"


def _detect_llm_provider() -> str:
    llm = load_llm_settings() or {}
    prov = str(llm.get("provider") or "").strip().lower()
    if prov:
        return prov
    cs = load_cloud_services() or {}
    if cs.get("azure_endpoint_url") or cs.get("azure_api_key"):
        return "azure"
    if cs.get("openai_api_key"):
        return "openai"
    if cs.get("nvidia_api_key") or cs.get("nvidia_base_url"):
        # If you're using NVIDIA for chat too
        return "nvidia"
    return "openai"


def _detect_embedding_provider() -> str:
    emb = load_embedding_settings() or {}
    prov = str(emb.get("provider") or "").strip().lower()
    if prov:
        return prov
    cs = load_cloud_services() or {}
    if cs.get("nvidia_api_key") or cs.get("nvidia_base_url"):
        return "nvidia"
    if cs.get("openai_api_key"):
        return "openai"
    # Default to NVIDIA if your YAML points to NVIDIA models, else OpenAI
    if cs.get("nvidia_embedding_model") or cs.get("nvidia_fallback_embedding_model"):
        return "nvidia"
    return "openai"


# ---------- provider factories (reuse your existing singletons) ----------

def _vector_factory_milvus() -> Any:
    from source.services.vector_service import get_vector_service
    return get_vector_service()

# Additional vector factories could be added here for weaviate/qdrant/pinecone/etc.

def _llm_client_factory_generic() -> Any:
    from source.services.llm_client import llm_client
    return llm_client

def _llm_service_factory_generic() -> Any:
    from source.services.llm_service import llm_service
    return llm_service

def _embedding_service_factory_generic() -> Any:
    from source.services.embedding_service import embedding_service
    return embedding_service

def _prompt_manager_factory_generic() -> Any:
    from source.services.prompt_manager import prompt_manager
    return prompt_manager

def _query_analyzer_factory_generic() -> Any:
    from source.services.query_analyzer import query_analyzer
    return query_analyzer


# Map detected vector provider -> factory
_VECTOR_FACTORIES = {
    "milvus": _vector_factory_milvus,
    # "weaviate": _vector_factory_weaviate,
    # "qdrant": _vector_factory_qdrant,
    # "pinecone": _vector_factory_pinecone,
    # "chroma": _vector_factory_chroma,
    # "pgvector": _vector_factory_pgvector,
}


# ---------- bootstrap ----------

def bootstrap(force: bool = False) -> None:
    """
    Idempotent per-process bootstrap. Registers providers and seeds services
    based on current YAML config.
    """
    global _BOOTSTRAPPED
    with _LOCK:
        if _BOOTSTRAPPED and not force:
            return

        # Load once (config module caches internally)
        llm_cfg = load_llm_settings() or {}
        emb_cfg = load_embedding_settings() or {}
        search_cfg = load_search_settings() or {}
        docs_cfg = load_document_settings() or {}
        cloud_cfg = load_cloud_services() or {}
        _ = load_llm_service()  # legacy; optional

        prov_vector = _detect_vector_provider()
        prov_llm = _detect_llm_provider()
        prov_emb = _detect_embedding_provider()

        # Keep the boot log concise (only container-level; registry logs are DEBUG in registry)
        log.info(
            "Bootstrapping: providers llm=%s embeddings=%s vector=%s ",
            prov_llm, prov_emb, prov_vector
        )

        # ----- Vector provider -----
        vec_factory = _VECTOR_FACTORIES.get(prov_vector)
        if vec_factory is None:
            # default fallback
            vec_factory = _VECTOR_FACTORIES["milvus"]
            prov_vector = "milvus"

        if not _registry.has("vector", prov_vector):
            register("vector", prov_vector, vec_factory, singleton=True)
        set_default("vector", prov_vector)

        # ----- LLM client + service -----
        if not _registry.has("llm", prov_llm):
            register("llm", prov_llm, _llm_client_factory_generic, singleton=True)
        if not _registry.has("llm_service", prov_llm):
            register("llm_service", prov_llm, _llm_service_factory_generic, singleton=True)
        set_default("llm", prov_llm)
        set_default("llm_service", prov_llm)

        # ----- Embeddings -----
        if not _registry.has("embeddings", prov_emb):
            register("embeddings", prov_emb, _embedding_service_factory_generic, singleton=True)
        set_default("embeddings", prov_emb)

        # ----- Managers -----
        if not _registry.has("prompt_manager", "default"):
            register("prompt_manager", "default", _prompt_manager_factory_generic, singleton=True)
        if not _registry.has("query_analyzer", "default"):
            register("query_analyzer", "default", _query_analyzer_factory_generic, singleton=True)
        set_default("prompt_manager", "default")
        set_default("query_analyzer", "default")

        # ----- Configure services (best-effort/no-op safe) -----
        try:
            llm_client = resolve("llm")
            _maybe_call(llm_client, "init_from_settings", llm_cfg, cloud_cfg)
            _maybe_call(llm_client, "configure", llm_cfg)
        except Exception:
            pass  # already logged in _maybe_call

        try:
            llm_service = resolve("llm_service")
            _maybe_call(llm_service, "init_from_settings", llm_cfg, cloud_cfg, search_cfg)
            _maybe_call(llm_service, "configure", llm_cfg)
        except Exception:
            pass

        try:
            embedding = resolve("embeddings")
            _maybe_call(embedding, "init_from_settings", emb_cfg, cloud_cfg)
            _maybe_call(embedding, "configure", emb_cfg)
        except Exception:
            pass

        try:
            vector = resolve("vector")
            _maybe_call(vector, "init_from_settings", cloud_cfg, search_cfg)
            _maybe_call(vector, "configure", {"cloud_services": cloud_cfg, "search_settings": search_cfg})
        except Exception:
            pass

        # ----- Seed Prompt Manager (system prompts + templates from YAML) -----
        try:
            pm = resolve("prompt_manager")
            if get_prompt_manager_settings:
                pms = get_prompt_manager_settings()
                # system prompts
                if pms.system_prompts:
                    _maybe_call(pm, "add_system_prompt", "__noop__", "")  # probe
                    for name, text in pms.system_prompts.items():
                        try:
                            pm.add_system_prompt(name, text, overwrite=False)  # type: ignore[attr-defined]
                        except Exception:
                            log.debug("system_prompt upsert failed for %s", name, exc_info=True)
                # templates
                if pms.templates:
                    try:
                        from source.services.prompt_manager import PromptTemplate  # dataclass
                    except Exception:
                        PromptTemplate = None  # type: ignore
                    for name, spec in pms.templates.items():
                        try:
                            if PromptTemplate:
                                t = PromptTemplate(
                                    name=spec.name,
                                    template=spec.template,
                                    required_fields=spec.required_fields,
                                    optional_fields=spec.optional_fields,
                                    description=spec.description,
                                    version=spec.version,
                                    category=spec.category,
                                    max_length=spec.max_length,
                                    tags=spec.tags,
                                )
                                pm.add_template(t, overwrite=False)  # type: ignore[attr-defined]
                        except Exception:
                            log.debug("template upsert failed for %s", name, exc_info=True)

            # pass common settings
            _maybe_call(pm, "configure", {"document_settings": docs_cfg, "search_settings": search_cfg})
        except Exception:
            pass

        # ----- Configure Query Analyzer -----
        try:
            qa = resolve("query_analyzer")
            if get_query_analyzer_settings:
                qa_cfg = get_query_analyzer_settings()
                _maybe_call(qa, "configure", {"query_analyzer_settings": qa_cfg.dict(), "search_settings": search_cfg})
            else:
                _maybe_call(qa, "configure", {"search_settings": search_cfg})
        except Exception:
            pass

        _BOOTSTRAPPED = True
        log.info("Container bootstrap complete.")


# ---------- convenience getters ----------

def get_vector_service() -> Any:
    bootstrap()
    return resolve("vector")

def get_llm_client() -> Any:
    bootstrap()
    return resolve("llm")

def get_llm_service() -> Any:
    bootstrap()
    return resolve("llm_service")

def get_embedding_service() -> Any:
    bootstrap()
    return resolve("embeddings")

def get_prompt_manager() -> Any:
    bootstrap()
    return resolve("prompt_manager")

def get_query_analyzer() -> Any:
    bootstrap()
    return resolve("query_analyzer")
