# source/routers/search.py
# type: ignore
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Callable, Union

from fastapi import APIRouter, HTTPException, Query

from source.core.search_settings import get_search_settings
from source.models.schemas import (
    SearchRequest,
    SearchResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["search"])
S = get_search_settings()

# -------------------- prefer new factories; fall back to legacy --------------------
_use_new_embeddings = False
_use_new_vector = False

# Embeddings client (new -> legacy)
try:
    from source.services.embedding_service import get_embeddings  # provider-agnostic
    _emb_client = get_embeddings()
    _use_new_embeddings = True
except Exception:
    from source.services.embedding_service import embedding_service  # legacy singleton
    _emb_client = embedding_service
    _use_new_embeddings = False

# Vector client (new -> legacy)
try:
    from source.services.vector_service import get_vector  # provider-agnostic
    _vec_client_fn = get_vector
    _use_new_vector = True
except Exception:
    from source.services.vector_service import get_vector_service  # legacy accessor
    _vec_client_fn = get_vector_service
    _use_new_vector = False


# --------------------------------- helpers ---------------------------------
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_400(msg: str) -> HTTPException:
    return HTTPException(status_code=400, detail=msg)


def _http_500(msg: str = "Internal server error") -> HTTPException:
    return HTTPException(status_code=500, detail=msg)


def _sanitize_top_k(k: Optional[int], cap: int) -> int:
    if k is None:
        return min(S.top_k_default, cap)
    if k < 1:
        raise _http_400("top_k must be >= 1")
    return min(k, cap)


def _sanitize_threshold(th: Optional[float]) -> Optional[float]:
    if th is None:
        return S.score_threshold_default
    if th < 0.0 or th > 1.0:
        raise _http_400("score_threshold must be between 0.0 and 1.0")
    return th


def _maybe_async(fn: Callable, *args, **kwargs):
    """Call a function that might be sync or async."""
    if asyncio.iscoroutinefunction(fn):
        return fn(*args, **kwargs)
    res = fn(*args, **kwargs)
    if asyncio.iscoroutine(res):
        return res
    return res


def _coerce_vec(v: Union[List[float], EmbeddingResponse, Dict[str, Any]]) -> List[float]:
    """
    Normalize possible provider return shapes into a plain float list.
    """
    if isinstance(v, EmbeddingResponse):
        return list(map(float, v.embedding))
    if isinstance(v, dict):
        # Common shapes: {"embedding": [...]}, {"data":[...]} (first), {"vector":[...]}
        if "embedding" in v:
            return list(map(float, v["embedding"]))
        if "vector" in v:
            return list(map(float, v["vector"]))
        if "data" in v and isinstance(v["data"], list) and v["data"]:
            first = v["data"][0]
            if isinstance(first, dict):
                if "embedding" in first:
                    return list(map(float, first["embedding"]))
                if "vector" in first:
                    return list(map(float, first["vector"]))
    # Assume it's already a list-like of floats
    return list(map(float, v))  # type: ignore[arg-type]


async def _embed_text(text: str, model: Optional[str] = None) -> EmbeddingResponse:
    """
    Provider-agnostic embedding creation that ALWAYS returns an EmbeddingResponse.

    Tries in order:
      1) create_embedding(...),
      2) embed_query(...),
      3) embed_texts([text], ...),
      4) legacy embedding_service.create_embedding(...)
    """
    def _make_response(vec: List[float]) -> EmbeddingResponse:
        return EmbeddingResponse(
            embedding=list(map(float, vec)),
            dimension=len(vec),
            model=(model or getattr(S, "default_embedding_model", "")),
            text_length=len(text or ""),
        )

    # 1) create_embedding(...):
    fn = getattr(_emb_client, "create_embedding", None)
    if callable(fn):
        try:
            out = await _maybe_async(fn, text=text, model=model)
        except TypeError:
            try:
                out = await _maybe_async(fn, text)
            except TypeError:
                out = await _maybe_async(fn, text=text)
        vec = _coerce_vec(out)
        return _make_response(vec)

    # 2) embed_query(text, model?):
    fn = getattr(_emb_client, "embed_query", None)
    if callable(fn):
        try:
            out = await _maybe_async(fn, text=text, model=model)
        except TypeError:
            out = await _maybe_async(fn, text)
        vec = _coerce_vec(out)
        return _make_response(vec)

    # 3) embed_texts([text], model?):
    fn = getattr(_emb_client, "embed_texts", None)
    if callable(fn):
        try:
            out = await _maybe_async(fn, [text], model=model)
        except TypeError:
            out = await _maybe_async(fn, [text])
        first = out[0] if isinstance(out, (list, tuple)) else out
        vec = _coerce_vec(first)
        return _make_response(vec)

    # 4) legacy path:
    if hasattr(_emb_client, "create_embedding"):
        legacy = await _emb_client.create_embedding(text=text, model=model)  # type: ignore
        vec = _coerce_vec(legacy)
        return _make_response(vec)

    raise RuntimeError("No embedding creation method available on provider")


# ---- Filter helpers (safe) ----
def _build_doc_filter_expr(document_id: Optional[str]) -> Optional[str]:
    """
    Safe equality filter on scalar field document_id.
    Uses json.dumps to quote properly (avoids manual escaping).
    """
    if not document_id:
        return None
    return f'document_id == {json.dumps(str(document_id))}'


def _normalize_metadata_filter(md: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Clean up incoming metadata filters from Swagger:
      - None or {} -> None (omit)
      - Keep values as-is; VectorService will build a safe Milvus JSON expression:
          * str/bool/number -> equality
          * list/tuple -> ArrayContainsAny
          * dict non-empty -> JSONContains
          * dict empty -> skipped
          * None -> (skipped; or service may map to NOT EXISTS if desired)
    """
    if not md:
        return None
    # Strip keys that are empty dicts to avoid == {} downstream
    clean = {}
    for k, v in md.items():
        if isinstance(v, dict) and not v:
            # skip empty object entirely
            continue
        clean[k] = v
    return clean or None


# ------------------------------- routes ----------------------------------
@router.post("/", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Semantic search over indexed content (provider-agnostic)."""
    if not request.query or not request.query.strip():
        raise _http_400("Search query cannot be empty")

    top_k = _sanitize_top_k(request.top_k, S.top_k_cap)
    score_threshold = _sanitize_threshold(request.score_threshold)

    start_time = time.monotonic()
    try:
        logger.info(
            "Semantic search | q=%r top_k=%d collection=%s threshold=%s",
            request.query, top_k, request.collection_name, score_threshold
        )

        vs = _vec_client_fn()

        # Create query embedding (model selection inside embedding client/config)
        emb: EmbeddingResponse = await _embed_text(request.query)

        # Structured metadata filtering (let VectorService build safe Milvus JSON expr)
        md_filter = _normalize_metadata_filter(request.filter_metadata)

        # Vector search (provider-agnostic)
        results: List[SearchResult] = await vs.search_similar(
            query_embedding=emb.embedding,
            top_k=top_k,
            collection_name=request.collection_name,
            score_threshold=score_threshold,
            metadata_filter=md_filter,     # <-- pass dict (safe)
            filter_expr=None,              # <-- do not build raw JSON expr here
        )

        processing_time = time.monotonic() - start_time
        logger.info("Search done | results=%d time=%.3fs", len(results), processing_time)

        return SearchResponse(
            results=results,
            total_results=len(results),
            query=request.query,
            processing_time=processing_time,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Search failed | q=%r", request.query)
        raise _http_500()


@router.get("/simple")
async def simple_search(
    q: str = Query(..., description="Search query"),
    limit: int = Query(5, ge=1, le=S.simple_limit_cap, description="Number of results to return"),
    collection: Optional[str] = Query(None, description="Collection name"),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum similarity score"),
):
    """Search documents using query parameters."""
    req = SearchRequest(
        query=q,
        top_k=limit,
        collection_name=collection,
        score_threshold=threshold,
    )
    return await search_documents(req)


@router.post("/by-document")
async def search_within_document(
    document_id: str,
    query: str,
    top_k: int = Query(S.top_k_default, ge=1, le=S.top_k_cap),
    collection_name: Optional[str] = None,
    score_threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
):
    """Semantic search constrained to a single document."""
    if not query or not query.strip():
        raise _http_400("Search query cannot be empty")

    try:
        th = _sanitize_threshold(score_threshold)
        logger.info("Doc search | doc=%s q='%.80s' top_k=%d threshold=%s", document_id, query, top_k, th)
        vs = _vec_client_fn()

        emb = await _embed_text(query)
        # Safe equality filter for document_id
        filter_expr = _build_doc_filter_expr(document_id=document_id)

        results = await vs.search_similar(
            query_embedding=emb.embedding,
            top_k=top_k,
            collection_name=collection_name,
            score_threshold=th,
            filter_expr=filter_expr,       # doc_id is a scalar field, keep as expr
            metadata_filter=None,          # IMPORTANT: don't pass {} from Swagger
        )

        return {
            "document_id": document_id,
            "query": query,
            "results": results,
            "total_results": len(results),
            "timestamp": _utcnow_iso(),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Document search failed | doc=%s", document_id)
        raise _http_500()


@router.get("/collections")
async def list_collections():
    """Enumerate available collections with stats (concurrent, resilient)."""
    try:
        vs = _vec_client_fn()
        collections = await vs.list_collections()

        sem = asyncio.Semaphore(S.stats_concurrency)

        async def _fetch(name: str):
            async with sem:
                try:
                    return await vs.get_collection_stats(name)
                except Exception as e:
                    logger.warning("Collection stats failed | %s: %s", name, e)
                    return {"collection_name": name, "error": str(e)}

        stats = await asyncio.gather(*[_fetch(c) for c in collections])

        return {
            "collections": list(stats),
            "total_collections": len(collections),
            "timestamp": _utcnow_iso(),
        }
    except Exception:
        logger.exception("List collections failed")
        raise _http_500()


@router.post("/similar-to-text")
async def find_similar_to_text(
    text: str,
    top_k: int = Query(S.top_k_default, ge=1, le=S.top_k_cap),
    collection_name: Optional[str] = None,
    score_threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
):
    """Find content semantically similar to the provided raw text."""
    if not text or not text.strip():
        raise _http_400("Text cannot be empty")

    try:
        th = _sanitize_threshold(score_threshold)
        logger.info("Similar-to-text | len=%d top_k=%d threshold=%s", len(text), top_k, th)
        vs = _vec_client_fn()

        emb = await _embed_text(text)

        results = await vs.search_similar(
            query_embedding=emb.embedding,
            top_k=top_k,
            collection_name=collection_name,
            score_threshold=th,
            metadata_filter=None,
            filter_expr=None,
        )

        return {
            "input_text_length": len(text),
            "results": results,
            "total_results": len(results),
            "timestamp": _utcnow_iso(),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Similar-to-text failed")
        raise _http_500()


# Cache for embedding dimensions (per model)
_emb_dim_cache: Dict[str, int] = {}

async def _get_embedding_dimension_for(model: str) -> int:
    dim = _emb_dim_cache.get(model)
    if dim:
        return dim
    # Prefer provider method, then settings fallback, last resort probe
    try:
        fn = getattr(_emb_client, "get_embedding_dimension", None)
        if callable(fn):
            dim = int(await _maybe_async(fn, model))
            _emb_dim_cache[model] = dim
            return dim
    except Exception:
        logger.debug("Provider get_embedding_dimension failed for %s", model, exc_info=True)

    # Try to read from settings (support dict or list shapes)
    models_meta = getattr(S, "available_models", None)
    if isinstance(models_meta, dict) and model in models_meta and "dimension" in models_meta[model]:
        _emb_dim_cache[model] = int(models_meta[model]["dimension"])
        return _emb_dim_cache[model]
    if isinstance(models_meta, (list, tuple)):
        for m in models_meta:
            if isinstance(m, dict) and m.get("model") == model and "dimension" in m:
                _emb_dim_cache[model] = int(m["dimension"])
                return _emb_dim_cache[model]

    # Probe: embed tiny string and infer length
    try:
        probe = await _embed_text("dimension probe", model)
        dim = len(probe.embedding)
        _emb_dim_cache[model] = dim
        return dim
    except Exception:
        logger.warning("Falling back to configured embed dimension for model=%s", model, exc_info=True)
        return S.embed_dimension_fallback


@router.get("/document/{document_id}/content")
async def get_document_content_chunks(
    document_id: str,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(S.content_page_size_default, ge=1, le=S.content_page_size_default, description="Items per page"),
):
    """
    Return all content chunks for a document with server-side pagination.
    Tries provider native listing by filter; falls back to zero-vector similarity search.
    """
    try:
        vs = _vec_client_fn()
        filter_expr = _build_doc_filter_expr(document_id=document_id)

        # Preferred: provider-side filtered listing, if available
        list_by_filter = getattr(vs, "list_by_filter", None)
        if callable(list_by_filter):
            all_results: List[SearchResult] = await list_by_filter(
                filter_expr=filter_expr,
                limit=S.content_fetch_hard_cap,
            )
        else:
            # Fallback: use a zero-vector query with a document filter
            dim = await _get_embedding_dimension_for(S.default_embedding_model)
            zero_vec = [0.0] * dim
            all_results = await vs.search_similar(
                query_embedding=zero_vec,
                top_k=S.content_fetch_hard_cap,
                collection_name=None,
                filter_expr=filter_expr,
                metadata_filter=None,
            )

        if not all_results:
            raise _http_400("Document not found or has no content")

        # Sort chunks by page number if present
        def _page_key(r: SearchResult):
            try:
                # try attribute; fall back to metadata field
                return (getattr(r, "page_number", None)
                        or int(getattr(r, "metadata", {}).get("page_number", 0)))
            except Exception:
                return 0

        all_results.sort(key=_page_key)

        # Paginate
        total = len(all_results)
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total)
        page_items = all_results[start_idx:end_idx]

        return {
            "document_id": document_id,
            "page": page,
            "page_size": page_size,
            "total_chunks": total,
            "total_pages": (total + page_size - 1) // page_size,
            "chunks": page_items,
            "timestamp": _utcnow_iso(),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Get document content failed | doc=%s", document_id)
        raise _http_500()


@router.get("/stats")
async def get_search_stats():
    """Aggregate high-level search system stats."""
    try:
        vs = _vec_client_fn()
        collections = await vs.list_collections()

        sem = asyncio.Semaphore(S.stats_concurrency)

        async def _stats(name: str):
            async with sem:
                try:
                    return await vs.get_collection_stats(name)
                except Exception as e:
                    logger.warning("Stats error | %s: %s", name, e)
                    return None

        raw = await asyncio.gather(*[_stats(c) for c in collections])
        details = [r for r in raw if r]

        total_embeddings = sum(int(d.get("row_count", 0)) for d in details)
        est_docs = (total_embeddings // S.doc_chunks_per_document_estimate
                    if S.doc_chunks_per_document_estimate else 0)

        return {
            "total_collections": len(collections),
            "total_embeddings": total_embeddings,
            "estimated_documents": est_docs,
            "collections": details,
            "timestamp": _utcnow_iso(),
        }
    except Exception:
        logger.exception("Get search stats failed")
        raise _http_500()


@router.post("/test-embedding")
async def test_embedding_search(request: EmbeddingRequest):
    """
    Debug helper: returns an embedding for a given text.
    Beware of large responses; consider gateway limits.
    """
    try:
        emb = await _embed_text(request.text, model=request.model)
        return {
            "text": request.text,
            "text_length": len(request.text or ""),
            "model": emb.model,
            "embedding_dimension": emb.dimension,
            "embedding_preview": emb.embedding[:10],   # first 10 dims
            "full_embedding": emb.embedding,           # potentially large
            "timestamp": _utcnow_iso(),
        }
    except Exception:
        logger.exception("Test embedding failed")
        raise _http_500()
