# source/services/llm_service.py
# type: ignore
from __future__ import annotations
"""
CLI:
  python -m source.services.llm_service --ingest --root data/parsed_output [--collection documents]
  python -m source.services.llm_service --ask "Question..." --rag
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Union

from source.config import (
    load_llm_service,
    load_cloud_services,
    load_milvus_settings,
    load_search_settings,
    load_embedding_settings,
    load_document_settings,
)

from source.models.schemas import LLMResponse, SearchResult  # type: ignore
from source.services.embedding_service import embedding_service  # type: ignore
from source.services.llm_client import llm_client  # centralized LLM client (single call site)
from source.services.prompt_manager import prompt_manager  # centralized prompts/templates
from source.services.query_analyzer import query_analyzer

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

# ---------------------- Config (no env reads here) ---------------------- #
_llm_cfg: Dict[str, Any] = load_llm_service() or {}
_cloud: Dict[str, Any] = load_cloud_services() or {}
_milvus: Dict[str, Any] = load_milvus_settings() or {}
_search_cfg: Dict[str, Any] = load_search_settings() or {}
_embed_cfg: Dict[str, Any] = load_embedding_settings() or {}
_doc_cfg: Dict[str, Any] = load_document_settings() or {}

# Optional vector adapter (preferred)
try:
    # Use new adapter name if available; keep legacy finder name if not.
    from source.services.vector_service import get_vector as _get_vector  # type: ignore
    def get_vector_service():
        return _get_vector()
    _HAS_VECTOR = True
except Exception:
    try:
        from source.services.vector_service import get_vector_service as _get_vector_service  # type: ignore
        def get_vector_service():
            return _get_vector_service()
        _HAS_VECTOR = True
    except Exception:
        get_vector_service = None  # type: ignore
        _HAS_VECTOR = False

# Optional direct pymilvus fallback (kept lightweight)
try:
    from pymilvus import (
        connections,
        utility,
        FieldSchema,
        CollectionSchema,
        DataType,
        Collection,
    )
    _HAS_PYMILVUS = True
except Exception:
    _HAS_PYMILVUS = False


# ---------------------- Prompt key resolution ---------------------- #
# Prompt key resolution (supports either root.prompt_keys or llm_settings.prompt_keys)
_PROMPT_KEYS = (
    _llm_cfg.get("prompt_keys")
    or (
        _llm_cfg.get("llm_settings", {}).get("prompt_keys")
        if isinstance(_llm_cfg.get("llm_settings"), dict)
        else None
    )
    or {}
)

prompt_defaults = {
    "rag_system": "chat_with_rag",
    "rag_template": "chat_with_context",
    "nonrag_system": "chat_without_rag",
    "stream_system": "chat_streaming",
}

RAG_SYSTEM_KEY = _PROMPT_KEYS.get("rag_system", prompt_defaults["rag_system"])
RAG_TEMPLATE_KEY = _PROMPT_KEYS.get("rag_template", prompt_defaults["rag_template"])
NONRAG_SYSTEM_KEY = _PROMPT_KEYS.get("nonrag_system", prompt_defaults["nonrag_system"])
STREAM_SYSTEM_KEY = _PROMPT_KEYS.get("stream_system", prompt_defaults["stream_system"])


def _now_iso() -> str:
    return datetime.now().isoformat()


def _default_collection() -> str:
    """
    Determine the default collection name (search_settings > milvus > 'documents').
    """
    return (
        (_search_cfg.get("default_collection") or "").strip()
        or (_milvus.get("default_collection") or "").strip()
        or "documents"
    )


def _embedding_dim_default() -> int:
    """
    Prefer milvus embedding_dim (since that drives the store schema),
    fallback to 1024 if not present.
    """
    dim = _milvus.get("embedding_dim")
    if isinstance(dim, int) and dim > 0:
        return dim
    return 1024


# -------------------------- Milvus fallback -------------------------- #
@dataclass
class _MilvusCfg:
    host: str
    port: str
    collection: str
    dim: int
    metric: str = "COSINE"


def _milvus_host_port() -> tuple[str, str]:
    uri = (_milvus.get("uri") or "http://127.0.0.1:19530").replace("http://", "").replace("https://", "")
    parts = uri.split(":")
    host = parts[0].strip()
    port = parts[1].strip() if len(parts) > 1 else "19530"
    return host, port


def _connect_milvus() -> None:
    if not _HAS_PYMILVUS:
        raise RuntimeError("pymilvus not installed")
    if connections.has_connection("default"):
        try:
            connections.disconnect("default")
        except Exception:
            pass
    host, port = _milvus_host_port()
    connections.connect(alias="default", host=host, port=port)


def _ensure_collection(cfg: _MilvusCfg) -> Collection:
    if not _HAS_PYMILVUS:
        raise RuntimeError("pymilvus not installed")
    if utility.has_collection(cfg.collection):
        return Collection(cfg.collection)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16384),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=cfg.dim),
    ]
    schema = CollectionSchema(fields, description="RAG documents (markdown chunks)")
    col = Collection(name=cfg.collection, schema=schema, consistency_level="Strong")
    col.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": cfg.metric, "params": {"nlist": 1024}},
    )
    return col


# -------------------------- Normalizers -------------------------- #
def _normalize_completion_content(resp: Any) -> tuple[str, int]:
    """
    Extract (content, tokens_used) from provider-agnostic completion object.
    """
    content = ""
    tokens = 0

    if isinstance(resp, dict):
        content = resp.get("content") or ""
        usage = resp.get("usage") or {}
        tokens = int(usage.get("total_tokens") or usage.get("tokens") or 0)
        # OpenAI-like dict
        if not content and "choices" in resp and resp["choices"]:
            ch0 = resp["choices"][0]
            content = (
                (ch0.get("message") or {}).get("content")
                or ch0.get("text")
                or ch0.get("content")
                or ""
            )
        return content or "", int(tokens)

    # OpenAI-like object
    try:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            tokens = int(getattr(usage, "total_tokens", 0) or getattr(usage, "tokens", 0) or 0)
        choices = getattr(resp, "choices", None)
        if choices:
            ch0 = choices[0]
            msg = getattr(ch0, "message", None)
            content = (getattr(msg, "content", None) if msg else None) or getattr(ch0, "text", None) or getattr(ch0, "content", None) or ""
            return content or "", int(tokens)
    except Exception:
        pass

    # last resort
    try:
        content = getattr(resp, "content", "") or str(resp)
    except Exception:
        content = str(resp)
    return content or "", int(tokens or 0)


async def _aiter(obj: Any) -> AsyncGenerator[Any, None]:
    """
    Async-iterate over sync or async iterables uniformly.
    """
    if hasattr(obj, "__aiter__"):
        async for item in obj:
            yield item
    else:
        for item in (obj or []):
            yield item


def _extract_stream_delta(chunk: Any) -> tuple[str, Optional[str]]:
    """
    Extract streaming delta content and finish_reason from provider-agnostic chunk.
    """
    text = ""
    finish: Optional[str] = None

    if isinstance(chunk, dict):
        text = chunk.get("content") or chunk.get("delta") or chunk.get("text") or ""
        finish = chunk.get("finish_reason")
        return text or "", finish

    try:
        # OpenAI-like stream chunk
        choices = getattr(chunk, "choices", None)
        if choices:
            ch0 = choices[0]
            delta = getattr(ch0, "delta", None)
            text = (getattr(delta, "content", None) if delta else None) or getattr(ch0, "text", None) or ""
            finish = getattr(ch0, "finish_reason", None)
            return text or "", finish
    except Exception:
        pass

    # last resort
    try:
        text = getattr(chunk, "content", "") or ""
    except Exception:
        text = ""
    return text or "", None


# ------------------------------ Service ------------------------------ #
class LLMService:
    """
    Thin façade that composes:
    - llm_client (provider-agnostic client + model resolver)  <-- single call site
    - prompt_manager (central prompts/templates)
    - vector_service (DB adapter)
    """

    def __init__(self) -> None:
        # no direct SDK client here; we rely solely on llm_client
        try:
            self.client_type = llm_client.get_client_type()
        except Exception:
            self.client_type = "generic"
        self.vector = get_vector_service() if _HAS_VECTOR else None

        logger.info(
            "LLMService up: client_type=%s, vector_adapter=%s",
            self.client_type,
            "yes" if self.vector else ("pymilvus" if _HAS_PYMILVUS else "none"),
        )

    def minimize_token(self, text):
        # Import AutoTokenizer from transformers
        from transformers import AutoTokenizer
        # Load the tokenizer for roberta-large model
        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        # Tokenize with truncation, then decode only the truncated tokens to a minimized string
        minimized_string = tokenizer.decode(
            tokenizer(text, max_length=1200, truncation=True)["input_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return minimized_string

    async def generate_response(
        self,
        message: str,
        use_rag: bool = True,
        model: Optional[str] = None,
        max_tokens: int = int(_llm_cfg.get("max_tokens", 1000)),
        temperature: float = float(_llm_cfg.get("temperature", 0.7)),
        context_k: int = int(_llm_cfg.get("context_k", _search_cfg.get("top_k_default", 5))),
    ) -> LLMResponse:
        if not message or not message.strip():
            raise ValueError("Message cannot be empty")

        t0 = time.time()
        sources: List[SearchResult] = []
        system_prompt: str
        user_prompt: str
        messages = []
        if use_rag:
            sources = await self._retrieve_sources(message, top_k=context_k)
            #for source in sources:
            #    print(source)
            analysis = await query_analyzer.analyze_query(request.message)
            is_complex = str(getattr(analysis, "complexity", "")).lower() in {"complex", "research_intensive"}
            if is_complex and model in {"llama3.3-70b-wns-airesearch-domain", "meta/llama-3.3-70b-instruct"}:
                #h = max(sources, key=lambda x: x.similarity_score) if sources else None
                result_lines = []
                for i, source in enumerate(sources, start=1):
                    result_lines.append(f"source{i}:  Document Reference: {source.metadata['filename']} , {source.content}")
                #result_lines = (max(sources, key=lambda x: x.similarity_score) if sources else None).content
                result_text = "\n".join(result_lines)
                #print("Result Text: " + result_text)
                result_text = self.minimize_token(result_text)
                #print("Reduce Result Text: " + result_text)
                messages=[ {"query": message, "content": result_text}] #h.content}]
            else:
                system_prompt, user_prompt = prompt_manager.build_rag_prompt(
                    message,
                    sources,
                    system_prompt_type=RAG_SYSTEM_KEY,
                    template_name=RAG_TEMPLATE_KEY,
                )
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
        else:
            system_prompt = prompt_manager.get_system_prompt(NONRAG_SYSTEM_KEY)
            user_prompt = message
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        resolved_model = llm_client.resolve_model(model)
        # SINGLE CALL SITE
        resp = llm_client.create_completion(
            model=resolved_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        content, tokens_used = _normalize_completion_content(resp)
        processing_time = time.time() - t0

        return LLMResponse(
            response=content,
            sources=sources,
            model=resolved_model,
            tokens_used=tokens_used,
            processing_time=processing_time,
            context_used=bool(sources),
        )

    async def generate_streaming_response(
        self,
        message: str,
        use_rag: bool = True,
        model: Optional[str] = None,
        max_tokens: int = int(_llm_cfg.get("max_tokens", 1000)),
        temperature: float = float(_llm_cfg.get("temperature", 0.7)),
        context_k: int = int(_llm_cfg.get("context_k", _search_cfg.get("top_k_default", 5))),
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if not message or not message.strip():
            raise ValueError("Message cannot be empty")

        if use_rag:
            sources = await self._retrieve_sources(message, top_k=context_k)
            system_prompt, user_prompt = prompt_manager.build_rag_prompt(
                message,
                sources,
                system_prompt_type=RAG_SYSTEM_KEY,
                template_name=RAG_TEMPLATE_KEY,
            )
        else:
            sources = []
            system_prompt = prompt_manager.get_system_prompt(STREAM_SYSTEM_KEY)
            user_prompt = message

        resolved_model = llm_client.resolve_model(model)

        # SINGLE CALL SITE (stream=True)
        stream_obj = llm_client.create_completion(
            model=resolved_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        async for chunk in _aiter(stream_obj):
            piece, finish = _extract_stream_delta(chunk)
            if piece:
                yield {"content": piece, "sources": sources, "finished": bool(finish)}
        # ensure a final frame
        yield {"content": "", "sources": sources, "finished": True}

    async def ingest_markdown_directory(
        self,
        root_dir: str = "data/parsed_output",
        collection: Optional[str] = None,
        chunk_chars: int = int(_doc_cfg.get("chunk_chars", 1500)),
        chunk_overlap: int = int(_doc_cfg.get("chunk_overlap", 200)),
        min_section_chars: int = int(_doc_cfg.get("min_section_chars", 50)),
        batch_size: int = int(_doc_cfg.get("batch_size", 10)),
    ) -> Dict[str, Any]:
        """
        Parse markdown dir -> embed via EmbeddingService -> store via vector adapter (preferred),
        else minimal pymilvus fallback.
        """
        rows = await embedding_service.embed_markdown_directory(
            root_dir=root_dir,
            chunk_chars=chunk_chars,
            chunk_overlap=chunk_overlap,
            min_section_chars=min_section_chars,
            batch_size=batch_size,
            save_jsonl_path=None,
        )
        if not rows:
            return {"inserted": 0, "message": "No chunks produced from markdown"}

        coll = (collection or _default_collection()).strip()
        # Determine vector dimension from data or fallback
        first = rows[0] if isinstance(rows, list) and rows else {}
        dim = int(first.get("dimension") or (len(first.get("embedding") or []) if first else 0) or _embedding_dim_default())

        # Preferred path: use vector adapter if it exposes a compatible method
        if self.vector:
            upsert = getattr(self.vector, "store_document_embeddings", None) or getattr(self.vector, "upsert_embeddings", None)
            if callable(upsert):
                doc_id = "md_ingest:" + datetime.now().strftime("%Y%m%d%H%M%S")
                texts = [r.get("text") or r.get("content", "") for r in rows]
                embeddings = [r["embedding"] for r in rows if r.get("embedding") is not None]  # type: ignore
                metadatas = []
                for r in rows:
                    metadatas.append(
                        {
                            "chunk_index": r.get("chunk_index"),
                            "path": r.get("path"),
                            "page_number": r.get("page_number"),
                            "dimension": r.get("dimension"),
                        }
                    )
                try:
                    ok = await upsert(  # type: ignore
                        document_id=doc_id,
                        embeddings=embeddings,
                        texts=texts,
                        metadata=metadatas,
                        collection_name=coll,
                    )
                    return {
                        "inserted": len(embeddings) if ok else 0,
                        "collection": coll,
                        "via": "vector_service",
                    }
                except Exception as e:
                    logger.warning("vector adapter upsert failed; will try pymilvus fallback: %s", e)

        # Fallback: minimal pymilvus direct insert
        if not _HAS_PYMILVUS:
            raise RuntimeError("No vector adapter available and pymilvus not installed")

        _connect_milvus()
        host, port = _milvus_host_port()
        col = _ensure_collection(_MilvusCfg(host, port, coll, dim))
        BATCH = 100
        total = 0
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            # Build aligned columns (excluding auto_id)
            document_id = [str(r.get("document_id") or "md_ingest") for r in batch]
            chunk_index = [int(r.get("chunk_index", 0)) for r in batch]
            path = [str(r.get("path", ""))[:1024] for r in batch]
            page_number = [int(r.get("page_number", 0)) for r in batch]
            text = [str(r.get("text") or r.get("content", ""))[:16384] for r in batch]
            embeddings = [r["embedding"] for r in batch]  # type: ignore

            col.insert([document_id, chunk_index, path, page_number, text, embeddings])
            total += len(batch)

        col.flush()
        return {"inserted": total, "collection": coll, "via": "pymilvus"}

    # -------------------------- Internals -------------------------- #
    async def _retrieve_sources(self, query: str, top_k: int = int(_search_cfg.get("top_k_default", 5))) -> List[SearchResult]:
        """
        Adapter-first retrieval; pymilvus fallback if no adapter.
        """
        q = (query or "").strip()
        q_emb = await embedding_service.create_embedding(q)
        emb = list(q_emb.embedding or [])
        dim = _embedding_dim_default()
        if not emb or len(emb) != dim:
            # normalize to configured dimension if embedding service differs
            emb = (emb + [0.0] * dim)[:dim]

        # Preferred: adapter search
        if self.vector and hasattr(self.vector, "search_similar"):
            try:
                return await self.vector.search_similar(query_embedding=emb, top_k=top_k)  # type: ignore
            except Exception as e:
                logger.warning("vector adapter search failed: %s", e)

        # Fallback: direct pymilvus search (minimal)
        if not _HAS_PYMILVUS:
            return []
        try:
            _connect_milvus()
            coll_name = _default_collection()
            if coll_name not in utility.list_collections():
                return []
            col = Collection(coll_name)
            col.load()
            res = col.search(
                data=[emb],
                anns_field="embedding",
                param={"nprobe": int(_milvus.get("nprobe", 16))},
                limit=int(top_k),
                output_fields=_milvus.get(
                    "output_fields",
                    ["document_id", "chunk_index", "path", "text", "page_number"],
                ),
            )
            hits = res[0] if res else []
            results: List[SearchResult] = []  # type: ignore

            for h in hits:
                # Milvus hit object may vary; extract fields defensively
                try:
                    ent = h.entity if hasattr(h, "entity") else h
                    doc_id = getattr(ent, "get", lambda k, d=None: d)("document_id", None) if hasattr(ent, "get") else getattr(ent, "document_id", None)
                    page_no = getattr(ent, "get", lambda k, d=None: d)("page_number", None) if hasattr(ent, "get") else getattr(ent, "page_number", None)
                    txt = getattr(ent, "get", lambda k, d=None: d)("text", None) if hasattr(ent, "get") else getattr(ent, "text", "")
                except Exception:
                    doc_id, page_no, txt = None, None, ""

                class _SR:
                    pass

                sr = _SR()
                setattr(sr, "document_id", doc_id or "")
                setattr(sr, "page_number", page_no)
                setattr(sr, "text", txt or "")
                setattr(sr, "content", txt or "")
                try:
                    dist = float(getattr(h, "distance", 0.0))
                    score = max(0.0, 1.0 - dist)
                except Exception:
                    score = 0.0
                setattr(sr, "similarity_score", score)
                results.append(sr)  # type: ignore

            return results
        except Exception as e:
            logger.warning("pymilvus fallback search failed: %s", e)
            return []


# -------------------------- Global instance -------------------------- #
llm_service = LLMService()


# ------------------------------- CLI -------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM + RAG Service (adapter-first; single llm_client call site)")
    parser.add_argument("--ingest", action="store_true", help="Embed + ingest markdown into Milvus")
    parser.add_argument("--root", default="data/parsed_output", help="Markdown root for ingestion")
    parser.add_argument("--collection", default=None, help="Milvus collection name")
    parser.add_argument("--ask", default=None, help="Ask a question (non-streaming)")
    parser.add_argument("--rag", action="store_true", help="Use RAG for --ask")
    args = parser.parse_args()

    async def _main():
        if args.ingest:
            res = await llm_service.ingest_markdown_directory(root_dir=args.root, collection=args.collection)
            print(json.dumps(res, indent=2))

        if args.ask:
            resp = await llm_service.generate_response(args.ask, use_rag=args.rag)
            print("\n=== ANSWER ===\n", resp.response)
            if resp.sources:
                print("\n=== SOURCES ===")
                for i, s in enumerate(resp.sources, 1):
                    doc = getattr(s, "document_id", "")
                    page = getattr(s, "page_number", "")
                    print(f"[{i}] doc={doc} page={page}")

    asyncio.run(_main())
