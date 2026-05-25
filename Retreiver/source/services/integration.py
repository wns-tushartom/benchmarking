# source/services/integration.py
# type: ignore
from __future__ import annotations

"""
Integration Service
-------------------
Unified façade over Vector Service, Embedding Service, LLM Service, and Prompt Manager.

- Config-driven (no env / no CORS here)
- Non-RAG path goes through llm_client (single call-site)
- RAG path goes through llm_service (so retrieval + prompting is centralized there)
- Prompt building via prompt_manager when available
- Embedding + vector helpers with adapter-first API and tolerant fallbacks

This module intentionally keeps I/O shapes minimal and tolerant so routers/controllers
can rely on it without worrying about provider/client specifics.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Iterable, Union

# --- Service imports (adapter-first; tolerant fallbacks) ---
try:
    from source.services.vector_service import get_vector_service
    from source.services.embedding_service import get_embeddings
    from source.services.llm_service import llm_service
    from source.services.llm_client import llm_client
    from source.services.prompt_manager import prompt_manager
    from source.config import load_config, load_llm_service, load_query_settings
except Exception:  # pragma: no cover - dev/test fallback
    logging.warning("Using fallback imports for services (testing/dev).")
    try:
        from services.vector_service import get_vector_service  # type: ignore
        from services.embedding_service import get_embeddings  # type: ignore
        from services.llm_service import llm_service  # type: ignore
        from services.llm_client import llm_client  # type: ignore
        from services.prompt_manager import prompt_manager  # type: ignore
        from config import load_config, load_llm_service, load_query_settings  # type: ignore
    except Exception:
        # Minimal mocks if even that fails
        get_vector_service = lambda: None  # type: ignore
        get_embeddings = lambda: None  # type: ignore
        llm_service = None  # type: ignore
        llm_client = None  # type: ignore
        prompt_manager = None  # type: ignore
        load_config = lambda: {}  # type: ignore
        load_llm_service = lambda: {}  # type: ignore
        load_query_settings = lambda: {}  # type: ignore


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


class IntegrationService:
    """
    Unified interface for:
      - LLM (direct chat or RAG)
      - Embeddings (batch)
      - Vector store (upsert/search)
      - Prompt resolution (system+template)
    """

    def __init__(self) -> None:
        # Config blocks
        self.config: Dict[str, Any] = load_config() or {}
        self.llm_config: Dict[str, Any] = load_llm_service() or {}
        self.query_config: Dict[str, Any] = load_query_settings() or {}

        # Service adapters
        self.vector_service = get_vector_service()
        self.embedding_service = get_embeddings()
        self.llm_service = llm_service
        self.llm_client = llm_client

        # Defaults (query settings take precedence for generation params)
        self.default_model: str = (
            self.llm_config.get("default_model")
            or getattr(self.llm_client, "default_model", None)
            or "gpt-4o"
        )
        self.default_temperature: float = float(self.query_config.get("temperature_default", 0.7))
        self.default_max_tokens: int = int(self.query_config.get("max_tokens_default", 1000))

        # Optional fine-tuned model
        self.finetuning_model: str = self.llm_config.get("finetuning_model", "llama3.3-70b-wns-airesearch-domain")

        # Basic template mapping hint (prompt_manager has richer catalog)
        self.prompt_templates = self.llm_config.get(
            "prompt_templates",
            {
                "query_analysis": "query_analysis_comprehensive",
                "document_summary": "document_summary_structured",
                "comparison_analysis": "comparison_analysis",
            },
        )

        logger.info(
            "IntegrationService initialized: default_model=%s, finetuning_model=%s",
            self.default_model,
            self.finetuning_model,
        )

    # ---------------------------------------------------------------------
    # Core LLM entry point
    # ---------------------------------------------------------------------
    async def call_llm(
        self,
        prompt_type: str,
        content: str,
        model: Optional[str] = None,
        system_prompt_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        use_finetuned_model: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Unified LLM call.

        Args:
            prompt_type: A hint/template key (e.g., "query_analysis", "document_summary", "chat_with_context", etc.)
            content: Primary user content (or the query)
            model: Optional model override
            system_prompt_key: Optional system prompt key (falls back to prompt_type or sensible default)
            temperature: Override generation temperature
            max_tokens: Override token cap
            stream: Whether to request streaming results
            use_finetuned_model: If True, try the configured fine-tuned/custom model
            response_format: Provider-specific response_format (passed through to client)
            **kwargs: Template fields / extra LLM kwargs (e.g., use_rag=True, context_k=5, etc.)

        Returns:
            Non-stream: provider ChatCompletion-like object (OpenAI-style)
            Stream: an iterator yielding OpenAI-like stream chunks or dicts {"content": "...", "finish_reason": ...}
        """
        if not content or not content.strip():
            raise ValueError("content cannot be empty")

        # Generation params
        temp = float(self._coalesce(temperature, self.default_temperature))
        tokens = int(self._coalesce(max_tokens, self.default_max_tokens))

        # Choose model
        selected_model = (
            self.finetuning_model if use_finetuned_model else (model or self.default_model)
        )

        # RAG toggle: accept explicit kwarg; also treat certain prompt_type as RAG-ish
        use_rag = bool(kwargs.pop("use_rag", False) or prompt_type in {"rag", "chat_with_rag"})

        # System prompt via prompt_manager (optional)
        system_prompt = ""
        if prompt_manager:
            key = system_prompt_key or self._default_system_key(prompt_type, use_rag=use_rag, stream=stream)
            try:
                system_prompt = prompt_manager.get_system_prompt(key) or ""
            except Exception as e:
                logger.debug("prompt_manager.get_system_prompt failed (%s) -> empty system.", e)

        # Build user prompt for non-RAG (RAG path handled by llm_service)
        user_prompt = content
        if prompt_manager and not use_rag:
            try:
                user_prompt = self._build_user_prompt(prompt_type, content, **kwargs)
            except Exception as e:
                logger.debug("Prompt build failed for '%s': %s. Falling back to raw content.", prompt_type, e)
                user_prompt = content

        # Log (avoid huge payloads)
        logger.info(
            "LLM call: model=%s, rag=%s, stream=%s, prompt_type=%s, temp=%.2f, max_tokens=%d",
            selected_model, use_rag, stream, prompt_type, temp, tokens
        )

        start = time.time()
        try:
            # ---- RAG path -> go through llm_service so retrieval + prompting stay centralized ----
            if use_rag:
                if not self.llm_service:
                    raise RuntimeError("RAG requested but llm_service is unavailable")
                context_k = int(kwargs.pop("context_k", self.query_config.get("context_k_default", 5)))
                if stream:
                    # Return the provider stream/iterable; caller will consume it
                    return self.llm_service.generate_streaming_response(  # type: ignore
                        message=content,
                        use_rag=True,
                        model=selected_model,
                        max_tokens=tokens,
                        temperature=temp,
                        context_k=context_k,
                    )
                # Non-streaming RAG
                return await self.llm_service.generate_response(  # type: ignore
                    message=content,
                    use_rag=True,
                    model=selected_model,
                    max_tokens=tokens,
                    temperature=temp,
                    context_k=context_k,
                )

            # ---- Non-RAG path -> go through llm_client (single call-site) ----
            if not self.llm_client:
                raise RuntimeError("llm_client is unavailable")

            messages = [
                {"role": "system", "content": system_prompt} if system_prompt else None,
                {"role": "user", "content": user_prompt},
            ]
            messages = [m for m in messages if m]  # prune None

            # llm_client is sync; avoid blocking event loop for non-stream via to_thread
            if stream:
                # Streaming: return iterator/async-iterator directly
                return self.llm_client.stream_chat(messages=messages, model=selected_model, max_tokens=tokens,
                                                   temperature=temp, response_format=response_format or {}, **kwargs)
            else:
                resp = await asyncio.to_thread(
                    self.llm_client.create_completion,
                    messages=messages,
                    model=selected_model,
                    max_tokens=tokens,
                    temperature=temp,
                    stream=False,
                    response_format=response_format or {},
                    **kwargs,
                )
                elapsed = time.time() - start
                logger.info("LLM (non-RAG) completed in %.2fs", elapsed)
                return resp

        except Exception as e:
            elapsed = time.time() - start
            logger.error("LLM call failed after %.2fs: %s", elapsed, e)
            raise

    # ---------------------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------------------
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of texts.
        Tries embedding_service.create_embeddings_batch and normalizes outputs to List[List[float]].
        """
        if not self.embedding_service:
            logger.warning("Embedding service not available")
            return [[] for _ in texts]

        try:
            out = await self.embedding_service.create_embeddings_batch(texts)  # type: ignore
        except Exception as e:
            logger.error("create_embeddings_batch failed: %s", e)
            raise

        # Normalize to List[List[float]]
        embeddings: List[List[float]] = []
        for item in out or []:
            if isinstance(item, dict) and "embedding" in item:
                embeddings.append([float(x) for x in item["embedding"]])
            elif hasattr(item, "embedding"):
                embeddings.append([float(x) for x in getattr(item, "embedding")])
            elif isinstance(item, (list, tuple)):
                embeddings.append([float(x) for x in item])  # assume it is already a vector
            else:
                embeddings.append([])
        return embeddings

    # ---------------------------------------------------------------------
    # Vector store helpers
    # ---------------------------------------------------------------------
    async def store_document_embeddings(
        self,
        document_id: str,
        embeddings: List[List[float]],
        texts: List[str],
        metadata: List[Dict[str, Any]],
        collection_name: str,
    ) -> bool:
        """
        Upsert/store embeddings into the vector DB via adapter-first API.
        Tries common method names to stay compatible with different adapters.
        """
        if not self.vector_service:
            logger.warning("Vector service not available")
            return False

        # Prefer store_document_embeddings; fall back to upsert_embeddings
        for meth_name in ("store_document_embeddings", "upsert_embeddings"):
            fn = getattr(self.vector_service, meth_name, None)
            if callable(fn):
                try:
                    ok = await fn(  # type: ignore
                        document_id=document_id,
                        embeddings=embeddings,
                        texts=texts,
                        metadata=metadata,
                        collection_name=collection_name,
                    )
                    return bool(ok if ok is not None else True)
                except Exception as e:
                    logger.error("%s failed: %s", meth_name, e)
                    # try next method name
        return False

    async def search_documents(
        self,
        query_embedding: List[float],
        collection_name: Optional[str] = None,
        limit: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors in the vector DB.

        Prefers adapter method names:
          - search_similar(query_embedding=..., top_k=..., filter_expr=...)
          - search(collection_name=..., query_embedding=..., limit=...)
        Returns a list of dict-like results (duck-typed to your SearchResult needs).
        """
        if not self.vector_service:
            logger.warning("Vector service not available")
            return []

        # 1) Preferred: search_similar
        fn = getattr(self.vector_service, "search_similar", None)
        if callable(fn):
            try:
                res = await fn(query_embedding=query_embedding, top_k=limit, **kwargs)  # type: ignore
                return self._normalize_results(res)
            except Exception as e:
                logger.warning("vector_service.search_similar failed: %s", e)

        # 2) Fallback: search(collection_name=..., limit=...)
        fn = getattr(self.vector_service, "search", None)
        if callable(fn):
            try:
                res = await fn(  # type: ignore
                    collection_name=collection_name or getattr(self.vector_service, "default_collection", "documents"),
                    query_embedding=query_embedding,
                    limit=limit,
                    **kwargs,
                )
                return self._normalize_results(res)
            except Exception as e:
                logger.warning("vector_service.search failed: %s", e)

        return []

    # ---------------------------------------------------------------------
    # Health / utilities
    # ---------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """Lightweight health snapshot of underlying services."""
        llm = "ok" if self.llm_client else "missing"
        rag = "ok" if self.llm_service else "missing"
        vec = "ok" if self.vector_service else "missing"
        emb = "ok" if self.embedding_service else "missing"
        sys_prompt = "ok" if prompt_manager else "missing"
        return {
            "status": "ok",
            "llm_client": llm,
            "llm_service": rag,
            "vector_service": vec,
            "embedding_service": emb,
            "prompt_manager": sys_prompt,
            "default_model": self.default_model,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _default_system_key(self, prompt_type: str, *, use_rag: bool, stream: bool) -> str:
        if prompt_manager is None:
            return ""
        if use_rag:
            return "chat_with_rag"
        if stream:
            return "chat_streaming"
        # generic fallback
        mapping = {
            "query_analysis": "query_analysis",
            "document_summary": "document_summarization",
            "comparison_analysis": "technical_expert",
        }
        return mapping.get(prompt_type, "chat_without_rag")

    def _build_user_prompt(self, prompt_type: str, content: str, **kwargs: Any) -> str:
        """
        Use prompt_manager to build the user prompt for known types.
        Falls back to returning content if template is missing or fails.
        """
        # Dedicated builders first
        if prompt_type == "query_analysis":
            return prompt_manager.build_analysis_prompt(content, context=kwargs.get("context", ""))  # type: ignore
        if prompt_type == "comparison_analysis":
            topic = kwargs.get("topic", "")
            items = kwargs.get("items", []) or []
            context = kwargs.get("context", "")
            criteria = kwargs.get("criteria") or []
            return prompt_manager.build_comparison_prompt(topic=topic or "Comparison", items=items, context=context, criteria=criteria)  # type: ignore
        if prompt_type == "document_summary":
            return prompt_manager.build_document_summary_prompt(  # type: ignore
                document_id=str(kwargs.get("document_id", "doc")),
                context=kwargs.get("context", content),
                summary_type=kwargs.get("summary_type", "comprehensive"),
                target_length=kwargs.get("target_length", "medium"),
                include_key_points=bool(kwargs.get("include_key_points", True)),
            )

        # Try treating prompt_type as a template name
        try:
            return prompt_manager.build_prompt(prompt_type, content=content, **kwargs)  # type: ignore
        except Exception:
            return content

    def _normalize_results(self, res: Any) -> List[Dict[str, Any]]:
        """
        Normalize vector results to a list of dicts with common keys.
        Accepts list of objects or dicts. Non-destructive (best-effort).
        """
        out: List[Dict[str, Any]] = []
        if not res:
            return out

        for r in res:
            if isinstance(r, dict):
                out.append(dict(r))
            else:
                item: Dict[str, Any] = {}
                for key in ("document_id", "page_number", "path", "title", "url"):
                    if hasattr(r, key):
                        item[key] = getattr(r, key)
                # unify text/content
                txt = getattr(r, "content", None) or getattr(r, "text", None)
                if txt is not None:
                    item["content"] = txt
                # similarity
                sim = getattr(r, "similarity_score", None) or getattr(r, "score", None)
                if sim is not None:
                    try:
                        item["similarity_score"] = float(sim)
                    except Exception:
                        pass
                out.append(item)
        return out

    @staticmethod
    def _coalesce(*vals: Any) -> Any:
        for v in vals:
            if v is not None:
                return v
        return None


# Global instance for convenience
integration_service = IntegrationService()


# Standalone quick test
if __name__ == "__main__":
    import json

    async def _smoke():
        print("IntegrationService smoke test\n" + "=" * 40)

        print("\nHealth:", json.dumps(integration_service.health(), indent=2))

        try:
            print("\nNon-RAG LLM test:")
            resp = await integration_service.call_llm(
                prompt_type="chat_without_rag",
                content="Say hello in three words.",
                max_tokens=8,
                temperature=0.3,
            )
            text = ""
            try:
                text = resp.choices[0].message.content  # type: ignore[attr-defined]
            except Exception:
                text = str(resp)
            print(" ->", text)
        except Exception as e:
            print("Non-RAG test failed:", e)

        try:
            print("\nRAG LLM test (will no-op if vector/emb not configured):")
            resp_rag = await integration_service.call_llm(
                prompt_type="rag",
                content="What is Retrieval Augmented Generation?",
                use_finetuned_model=False,
                max_tokens=128,
                temperature=0.2,
                use_rag=True,
            )
            # resp_rag is an LLMResponse dataclass from llm_service
            print(" ->", getattr(resp_rag, "response", str(resp_rag))[:120], "...")
        except Exception as e:
            print("RAG test failed:", e)

        try:
            print("\nEmbeddings test:")
            embs = await integration_service.create_embeddings(["test embedding"])
            print(f" -> dims: {len(embs[0]) if embs and embs[0] else 0}")
        except Exception as e:
            print("Embeddings failed:", e)

    asyncio.run(_smoke())
