# source/routers/embeddings.py
# type: ignore
from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Callable, Union

from fastapi import APIRouter, HTTPException, Query, Request

from source.core.embedding_settings import get_embedding_settings
from source.models.schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingBatchRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/embeddings", tags=["embeddings"])

S = get_embedding_settings()

# ---- Prefer new provider-agnostic factory; fallback to legacy singleton ----
_emb_client: Any = None
try:
    # If available in your service module
    from source.services.embedding_service import get_embeddings  # type: ignore
    _emb_client = get_embeddings()  # should return a provider-agnostic client/service
except Exception:
    # Fallback to the legacy global instance defined in embedding_service.py
    from source.services.embedding_service import embedding_service  # type: ignore
    _emb_client = embedding_service


# ------------------------------- helpers ---------------------------------

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _http_400(msg: str) -> HTTPException:
    return HTTPException(status_code=400, detail=msg)

def _http_500(msg: str = "Internal server error") -> HTTPException:
    return HTTPException(status_code=500, detail=msg)

def _coerce_float_list(v: Any) -> List[float]:
    return [float(x) for x in list(v)]

def _wrap_embedding(vec: Union[List[float], Any], model: str) -> EmbeddingResponse:
    """
    Build EmbeddingResponse whether we got:
      - a raw vector list
      - an object with `.embedding`
      - a dict with ['embedding']
    """
    if isinstance(vec, dict) and "embedding" in vec:
        arr = _coerce_float_list(vec["embedding"])
        # Try to get text_length from the dict if available
        text_length = vec.get("text_length", 0)
    elif hasattr(vec, "embedding"):
        arr = _coerce_float_list(getattr(vec, "embedding"))
        # Try to get text_length from the object if available
        text_length = getattr(vec, "text_length", 0)
    else:
        arr = _coerce_float_list(vec)
        text_length = 0
    
    return EmbeddingResponse(
        embedding=arr,
        dimension=len(arr),
        model=model,
        text_length=text_length
    )

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    num = 0.0
    da = 0.0
    db = 0.0
    for x, y in zip(a, b):
        num += x * y
        da += x * x
        db += y * y
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(num / (math.sqrt(da) * math.sqrt(db)))

async def _maybe_async(callable_obj: Callable, *args, **kwargs):
    """
    Run a callable that might be sync or async, returning the awaited result if needed.
    """
    res = callable_obj(*args, **kwargs)
    if asyncio.iscoroutine(res):
        return await res
    return res

async def _call_with_optional_model(fn: Callable, *args, model: Optional[str] = None, **kwargs):
    """
    Try calling fn(..., model=model) first; if it raises TypeError for unexpected
    kwargs, retry without the model kwarg. This avoids brittle signature introspection.
    """
    if model is None:
        return await _maybe_async(fn, *args, **kwargs)
    try:
        return await _maybe_async(fn, *args, model=model, **kwargs)
    except TypeError as e:
        # Retry without model kw only if it's about unexpected keyword
        if "unexpected keyword argument 'model'" in str(e) or "got an unexpected keyword argument 'model'" in str(e):
            return await _maybe_async(fn, *args, **kwargs)
        raise

def _resolve_model_name(requested: Optional[str]) -> str:
    desired = (requested or S.default_model).strip()
    if not getattr(S, "enable_model_resolver", False):
        return desired
    try:
        resolver = getattr(_emb_client, "_resolve_model", None)
        return resolver(desired) if callable(resolver) else desired
    except Exception:
        return desired


# ------------------------------- routes ---------------------------------

@router.post("/", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """Create embedding for a single text (provider-agnostic)."""
    if not request.text or not request.text.strip():
        raise _http_400("Text cannot be empty")
    try:
        model = _resolve_model_name(request.model)
        logger.info("Creating embedding | model=%s len=%d", model, len(request.text))

        # Prefer provider method order: create_embedding → embed_query → embed_texts([t])
        if callable(getattr(_emb_client, "create_embedding", None)):
            out = await _call_with_optional_model(_emb_client.create_embedding, request.text, model=model)
            return _wrap_embedding(out, model)

        if callable(getattr(_emb_client, "embed_query", None)):
            out = await _call_with_optional_model(_emb_client.embed_query, request.text, model=model)
            return _wrap_embedding(out, model)

        if callable(getattr(_emb_client, "embed_texts", None)):
            out = await _call_with_optional_model(_emb_client.embed_texts, [request.text], model=model)
            vec = out[0] if isinstance(out, list) else out
            return _wrap_embedding(vec, model)

        # Legacy path (already returns EmbeddingResponse)
        if hasattr(_emb_client, "create_embedding"):
            return await _emb_client.create_embedding(text=request.text, model=model)  # type: ignore

        raise RuntimeError("No embedding creation method available on provider")

    except HTTPException:
        raise
    except Exception:
        logger.exception("Embedding creation failed")
        raise _http_500()


@router.post("/batch", response_model=List[EmbeddingResponse])
async def create_embeddings_batch(
    request: EmbeddingBatchRequest,
    max_items: int = Query(default=None, ge=1),
):
    """Create embeddings for multiple texts (deduped, trimmed)."""
    if not request.texts:
        raise _http_400("Texts list cannot be empty")

    # Clean & dedupe
    seen = set()
    valid_texts: List[str] = []
    for t in request.texts:
        s = (t or "").strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        valid_texts.append(s)

    if not valid_texts:
        raise _http_400("No valid texts provided")

    limit = min(max_items or S.single_batch_max_texts, S.single_batch_max_texts)
    if len(valid_texts) > limit:
        raise _http_400(f"Maximum {limit} texts allowed per batch")

    try:
        model = _resolve_model_name(request.model)
        logger.info("Batch embeddings | model=%s count=%d", model, len(valid_texts))

        # Try provider-agnostic methods first
        if callable(getattr(_emb_client, "create_embeddings_batch", None)):
            out = await _call_with_optional_model(_emb_client.create_embeddings_batch, valid_texts, model=model)
        elif callable(getattr(_emb_client, "embed_texts", None)):
            out = await _call_with_optional_model(_emb_client.embed_texts, valid_texts, model=model)
        else:
            # Legacy path (returns List[EmbeddingResponse])
            if hasattr(_emb_client, "create_embeddings_batch"):
                return await _emb_client.create_embeddings_batch(texts=valid_texts, model=model)  # type: ignore
            raise RuntimeError("No batch embedding method available on provider")

        # Normalize to list of vectors / objects with 'embedding'
        if out and isinstance(out, list) and not isinstance(out[0], (dict, EmbeddingResponse)):
            return [_wrap_embedding(v, model) for v in out]
        return [_wrap_embedding(v, model) for v in out]

    except HTTPException:
        raise
    except Exception:
        logger.exception("Batch embedding creation failed")
        raise _http_500()


@router.post("/similarity")
async def calculate_similarity(
    text1: str,
    text2: str,
    model: Optional[str] = Query(default=None),
):
    """Calculate similarity between two texts (cosine fallback)."""
    if not (text1 and text1.strip()) or not (text2 and text2.strip()):
        raise _http_400("Both texts must be non-empty")

    try:
        resolved = _resolve_model_name(model)
        logger.info("Similarity | model=%s len1=%d len2=%d", resolved, len(text1), len(text2))

        # Embed both (provider-agnostic). Prefer batch if present.
        if callable(getattr(_emb_client, "create_embeddings_batch", None)):
            outs = await _call_with_optional_model(_emb_client.create_embeddings_batch, [text1, text2], model=resolved)
            e1 = _wrap_embedding(outs[0], resolved).embedding
            e2 = _wrap_embedding(outs[1], resolved).embedding
        elif callable(getattr(_emb_client, "embed_texts", None)):
            outs = await _call_with_optional_model(_emb_client.embed_texts, [text1, text2], model=resolved)
            e1 = _wrap_embedding(outs[0], resolved).embedding
            e2 = _wrap_embedding(outs[1], resolved).embedding
        else:
            single = getattr(_emb_client, "create_embedding", None) or getattr(_emb_client, "embed_query", None)
            if not callable(single):
                # legacy explicit
                emb1, emb2 = await asyncio.gather(
                    _emb_client.create_embedding(text1, resolved),  # type: ignore
                    _emb_client.create_embedding(text2, resolved),  # type: ignore
                )
                e1, e2 = emb1.embedding, emb2.embedding
            else:
                o1 = await _call_with_optional_model(single, text1, model=resolved)
                o2 = await _call_with_optional_model(single, text2, model=resolved)
                e1 = _wrap_embedding(o1, resolved).embedding
                e2 = _wrap_embedding(o2, resolved).embedding

        # Try provider similarity if available; else cosine
        sim_fn = getattr(_emb_client, "calculate_similarity", None)
        if callable(sim_fn):
            score = float(await _maybe_async(sim_fn, e1, e2))
        else:
            score = _cosine(e1, e2)

        # Try to infer dimension; fallback to len(e1)
        try:
            dim_fn = getattr(_emb_client, "get_embedding_dimension", None)
            if callable(dim_fn):
                dim = int(await _call_with_optional_model(dim_fn, model=resolved))
            else:
                dim = len(e1)
        except Exception:
            dim = len(e1)

        return {
            "text1": text1,
            "text2": text2,
            "text1_length": len(text1),
            "text2_length": len(text2),
            "model": resolved,
            "similarity_score": score,
            "embedding_dimension": dim,
            "timestamp": _utcnow_iso(),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Similarity calculation failed")
        raise _http_500()


@router.get("/models")
async def list_embedding_models():
    """List available embedding models (provider first, settings fallback)."""
    try:
        lister = getattr(_emb_client, "list_models", None)
        if callable(lister):
            models = await _maybe_async(lister)
            return {"models": models, "default_model": S.default_model}
    except Exception:
        logger.warning("embedding client list_models failed; falling back to settings", exc_info=True)

    models = [{"model": name, **meta} for name, meta in S.available_models.items()]
    return {"models": models, "default_model": S.default_model}


@router.get("/dimension/{model}")
async def get_embedding_dimension(model: str):
    """Get embedding dimension (provider or settings or probe)."""
    try:
        resolved = _resolve_model_name(model)
        dim_fn = getattr(_emb_client, "get_embedding_dimension", None)
        if callable(dim_fn):
            dim = await _call_with_optional_model(dim_fn, model=resolved)
            return {"model": resolved, "dimension": int(dim), "timestamp": _utcnow_iso()}

        # Fallback: settings metadata
        meta = S.available_models.get(resolved) or S.available_models.get(model)
        if meta and "dimension" in meta:
            return {"model": resolved, "dimension": int(meta["dimension"]), "timestamp": _utcnow_iso()}

        # Last resort: embed a tiny probe
        probe_text = "dimension probe"
        if hasattr(_emb_client, "embed_query"):
            v = await _call_with_optional_model(getattr(_emb_client, "embed_query"), probe_text, model=resolved)
        elif hasattr(_emb_client, "embed_texts"):
            v = (await _call_with_optional_model(getattr(_emb_client, "embed_texts"), [probe_text], model=resolved))[0]
        else:
            # legacy singleton
            v = (await _emb_client.create_embeddings_batch([probe_text], resolved))[0].embedding  # type: ignore
        dim = len(_wrap_embedding(v, resolved).embedding)
        return {"model": resolved, "dimension": int(dim), "timestamp": _utcnow_iso()}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to get embedding dimension for %s", model)
        raise _http_500()


@router.get("/compare-batch")
async def compare_text_with_batch_get(
    query_text: str = Query(..., description="The query text to compare against batch"),
    model: Optional[str] = Query(default=None, description="Embedding model to use"),
    top_k: Optional[int] = Query(default=None, ge=1, description="Number of top results to return"),
):
    # For GET requests, we need to provide at least one comparison text
    # We'll use the query text itself as a comparison if none provided
    comparison_texts = [query_text]  # Default to comparing with itself
    return await _compare_batch_impl(query_text, comparison_texts, model, top_k)








async def _compare_batch_impl(
    query_text: str,
    comparison_texts: List[str],
    model: Optional[str] = None,
    top_k: Optional[int] = None,
):
    """Compare a query text against many texts and return top matches."""
    if not query_text or not query_text.strip():
        raise _http_400("Query text cannot be empty")
    if not comparison_texts:
        raise _http_400("Comparison texts cannot be empty")

    # Clean & dedupe
    cleaned: List[str] = []
    seen = set()
    for t in comparison_texts:
        s = (t or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)

    if not cleaned:
        raise _http_400("No valid comparison texts provided")
    if len(cleaned) > S.compare_max_texts:
        raise _http_400(f"Maximum {S.compare_max_texts} comparison texts allowed")

    try:
        resolved = _resolve_model_name(model)
        logger.info("Compare-batch | model=%s comps=%d", resolved, len(cleaned))

        # Embed query + batch via best available methods
        batch_fn = getattr(_emb_client, "create_embeddings_batch", None) or getattr(_emb_client, "embed_texts", None)
        single_fn = getattr(_emb_client, "create_embedding", None) or getattr(_emb_client, "embed_query", None)

        if callable(batch_fn) and callable(single_fn):
            query_emb = await _call_with_optional_model(single_fn, query_text, model=resolved)
            query_vec = _wrap_embedding(query_emb, resolved).embedding

            batch_out = await _call_with_optional_model(batch_fn, cleaned, model=resolved)
            batch_vecs = [_wrap_embedding(v, resolved).embedding for v in batch_out]
        else:
            # legacy singleton path
            query_emb_task = _emb_client.create_embedding(query_text, resolved)  # type: ignore
            batch_emb_task = _emb_client.create_embeddings_batch(cleaned, resolved)  # type: ignore
            query_emb, batch_embs = await asyncio.gather(query_emb_task, batch_emb_task)
            query_vec = query_emb.embedding
            batch_vecs = [e.embedding for e in batch_embs]

        # Similarities (provider if available, else cosine)
        sim_fn = getattr(_emb_client, "calculate_similarity", None)
        scores = []
        if callable(sim_fn):
            for i, v in enumerate(batch_vecs):
                s = float(await _maybe_async(sim_fn, query_vec, v))
                scores.append({"index": i, "text": cleaned[i], "similarity_score": s, "text_length": len(cleaned[i])})
        else:
            for i, v in enumerate(batch_vecs):
                s = _cosine(query_vec, v)
                scores.append({"index": i, "text": cleaned[i], "similarity_score": s, "text_length": len(cleaned[i])})

        scores.sort(key=lambda x: x["similarity_score"], reverse=True)
        k = top_k or min(S.top_k_default, len(scores))
        return {
            "query_text": query_text,
            "query_text_length": len(query_text),
            "model": resolved,
            "total_comparisons": len(cleaned),
            "returned_results": min(k, len(scores)),
            "results": scores[:k],
            "timestamp": _utcnow_iso(),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Batch comparison failed")
        raise _http_500()




@router.get("/cluster-analysis")
async def analyze_text_clusters_get(
    text: str = Query(..., description="Text to analyze"),
    model: Optional[str] = Query(default=None, description="Embedding model to use"),
    similarity_threshold: float = Query(default=None, ge=0.0, le=1.0, description="Similarity threshold for clustering"),
):
    # For GET requests, we'll use the provided text and a duplicate of it
    # This ensures we have at least 2 texts for clustering
    # Handle potential URL encoding issues by ensuring the text is properly decoded
    try:
        # Create at least 2 different texts for clustering
        # Split by period to create variation if possible
        parts = text.split('.')
        if len(parts) > 1 and all(part.strip() for part in parts[:2]):
            texts = [parts[0].strip(), parts[1].strip()]
        else:
            # If we can't split meaningfully, duplicate with slight variation
            texts = [text, text + " (duplicate)"]
    except Exception as e:
        logger.warning(f"Error processing text for clustering: {e}")
        texts = [text, text]  # Fallback to simple duplication
        
    return await _analyze_clusters_impl(texts, model, similarity_threshold)

async def _analyze_clusters_impl(
    texts: List[str],
    model: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
):
    """
    Analyze clusters using pairwise similarity (O(n^2)).
    Limits and defaults come from embedding settings.
    """
    if not texts:
        raise _http_400("Texts list cannot be empty")

    # Clean & dedupe
    cleaned: List[str] = []
    seen = set()
    for t in texts:
        s = (t or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)

    if len(cleaned) < 2:
        raise _http_400("At least 2 valid texts required for clustering")
    if len(cleaned) > S.cluster_max_texts:
        raise _http_400(f"Maximum {S.cluster_max_texts} texts allowed for clustering analysis")

    threshold = S.similarity_threshold_default if similarity_threshold is None else float(similarity_threshold)

    try:
        resolved = _resolve_model_name(model)
        logger.info("Cluster-analysis | model=%s texts=%d threshold=%.3f", resolved, len(cleaned), threshold)

        # Embed all in one shot if possible
        batch_fn = getattr(_emb_client, "create_embeddings_batch", None) or getattr(_emb_client, "embed_texts", None)
        if callable(batch_fn):
            outs = await _call_with_optional_model(batch_fn, cleaned, model=resolved)
            embs = [_wrap_embedding(o, resolved).embedding for o in outs]
        else:
            # legacy singleton
            resp = await _emb_client.create_embeddings_batch(cleaned, resolved)  # type: ignore
            embs = [r.embedding for r in resp]

        n = len(embs)
        similarity_matrix: List[Dict[str, Any]] = []
        clusters: List[Dict[str, Any]] = []
        processed = set()

        # Similarity primitive
        sim_fn = getattr(_emb_client, "calculate_similarity", None)

        async def _sim(a: List[float], b: List[float]) -> float:
            if callable(sim_fn):
                return float(await _maybe_async(sim_fn, a, b))
            return _cosine(a, b)

        for i in range(n):
            if i in processed:
                continue
            cluster = [{"index": i, "text": cleaned[i]}]
            processed.add(i)

            for j in range(i + 1, n):
                if j in processed:
                    continue
                s = await _sim(embs[i], embs[j])
                if len(similarity_matrix) < S.similarity_matrix_limit:
                    similarity_matrix.append({"text1_index": i, "text2_index": j, "similarity": float(s)})
                if s >= threshold:
                    cluster.append({"index": j, "text": cleaned[j]})
                    processed.add(j)

            clusters.append(
                {
                    "cluster_id": len(clusters),
                    "size": len(cluster),
                    "texts": cluster,
                    "avg_length": sum(len(item["text"]) for item in cluster) / len(cluster),
                }
            )

        return {
            "model": resolved,
            "similarity_threshold": threshold,
            "total_texts": len(cleaned),
            "num_clusters": len(clusters),
            "clusters": clusters,
            "similarity_matrix": similarity_matrix,
            "timestamp": _utcnow_iso(),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Cluster analysis failed")
        raise _http_500()


@router.get("/health")
async def embedding_health_check():
    """Check health using provider method if available; otherwise return OK."""
    try:
        hc = getattr(_emb_client, "health_check", None)
        if callable(hc):
            return await _maybe_async(hc)
        # Minimal probe: list models or embed tiny text
        lister = getattr(_emb_client, "list_models", None)
        if callable(lister):
            _ = await _maybe_async(lister)
            return {"status": "ok", "provider": "embeddings", "timestamp": _utcnow_iso()}
        probe_fn = getattr(_emb_client, "embed_query", None) or getattr(_emb_client, "create_embedding", None)
        if callable(probe_fn):
            _ = await _call_with_optional_model(probe_fn, "health probe", model=_resolve_model_name(None))
            return {"status": "ok", "provider": "embeddings", "timestamp": _utcnow_iso()}
        # Legacy fallback:
        if hasattr(_emb_client, "health_check"):
            return await _emb_client.health_check()  # type: ignore
        return {"status": "ok", "provider": "embeddings", "timestamp": _utcnow_iso()}
    except Exception:
        logger.exception("Embedding health check failed")
        raise _http_500()
