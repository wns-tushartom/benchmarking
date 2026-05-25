# source/routers/llm.py
# type: ignore
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import List, Optional, Dict, Any, Callable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from source.core.llm_settings import get_llm_settings
from source.models.schemas import LLMRequest, LLMResponse
from source.services.query_analyzer import query_analyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/llm", tags=["llm"])
S = get_llm_settings()

# -------------------- prefer new factories; fall back to legacy --------------------
_use_new_llm = False
_use_new_embeddings = False
_use_new_vector = False

# LLM client
# try:
#     from source.services.llm_client import llm_client  # unified adapter
#     _llm = llm_client
#     _use_new_llm = True
# except Exception:
from source.services.llm_service import llm_service  # legacy
_llm = llm_service
_use_new_llm = True

# Embeddings (for simple RAG helpers)
try:
    from source.services.embedding_service import get_embeddings
    _emb = get_embeddings()
    _use_new_embeddings = True
except Exception:
    from source.services.embedding_service import embedding_service  # type: ignore
    _emb = embedding_service
    _use_new_embeddings = False

# Vector store (for simple RAG helpers)
try:
    from source.services.vector_service import get_vector
    _vec_fn = get_vector
    _use_new_vector = True
except Exception:
    from source.services.vector_service import get_vector_service  # type: ignore
    _vec_fn = get_vector_service
    _use_new_vector = False


# --------------------------------- helpers ---------------------------------
def _http_400(msg: str) -> HTTPException:
    return HTTPException(status_code=400, detail=msg)

def _http_500(msg: str = "Internal server error") -> HTTPException:
    return HTTPException(status_code=500, detail=msg)

async def _maybe_async(fn: Callable, *args, **kwargs):
    """
    Call a function that might be sync or async; if it returns a coroutine, await it.
    Always returns the resolved value (never a coroutine).
    """
    if asyncio.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    res = fn(*args, **kwargs)
    if asyncio.iscoroutine(res):
        return await res
    return res

def _resolve_model_name(requested: Optional[str]) -> str:
    desired = (requested or S.default_model).strip()
    if not getattr(S, "enable_model_resolver", True):
        return desired
    try:
        resolver = getattr(_llm, "resolve_model", None) or getattr(_llm, "_resolve_model", None)
        if callable(resolver):
            return resolver(desired)
    except Exception:
        pass
    return desired

def _cap_tokens(n: Optional[int]) -> int:
    return min(n or S.max_tokens_default, S.max_tokens_cap)

def _score_of(x: Any) -> float:
    return float(getattr(x, "similarity_score", None) or getattr(x, "score", 0.0) or 0.0)

async def _embed_text(text: str) -> List[float]:
    """Provider-agnostic single text embedding to vector list[float]."""
    for meth in ("create_embedding", "embed_query"):
        fn = getattr(_emb, meth, None)
        if callable(fn):
            out = await _maybe_async(fn, text)
            if isinstance(out, dict) and "embedding" in out:
                return [float(v) for v in out["embedding"]]
            if hasattr(out, "embedding"):
                return [float(v) for v in out.embedding]
            if isinstance(out, list):
                return [float(v) for v in out]
    if hasattr(_emb, "create_embeddings_batch"):
        out = await _emb.create_embeddings_batch([text])  # type: ignore
        vec = out[0].embedding if out and hasattr(out[0], "embedding") else out[0]
        return [float(v) for v in vec]
    raise RuntimeError("No embedding method available")

def _get_system_prompt() -> str:
    """Fetch system prompt via prompt_manager; safe fallback."""
    try:
        from source.services.prompt_manager import prompt_manager
        from source.core.prompt_manager_settings import get_prompt_manager_settings
        ps = get_prompt_manager_settings()
        typ = getattr(ps, "default_system_prompt_type", None) or "chat_without_rag"
        return prompt_manager.get_system_prompt(typ) or "You are a helpful assistant."
    except Exception:
        return "You are a helpful assistant."

def _extract_text_from_chat_completion(resp: Any) -> str:
    """Robustly extract assistant text from an OpenAI-like ChatCompletion."""
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        try:
            return getattr(resp, "content", "") or str(resp)
        except Exception:
            return str(resp)


# ------------------------------- core LLM calls ---------------------------------
async def _chat(messages: List[Dict[str, str]], **kwargs) -> str:
    """
    Provider-agnostic chat completion returning the assistant text.
    Preference order:
      1) llm_client.create_completion(stream=False)
      2) llm_client.chat(...)
      3) llm_client.complete(prompt)
      4) legacy llm_service.generate_response(...)
    """
    # (1) Single call site through llm_client adapter
    fn = getattr(_llm, "create_completion", None)
    if callable(fn):
        resp = await _maybe_async(fn, messages=messages, stream=False, **kwargs)
        return _extract_text_from_chat_completion(resp)

    # (2) unified .chat
    fn = getattr(_llm, "chat", None)
    if callable(fn):
        resp = await _maybe_async(fn, messages, **kwargs)
        return _extract_text_from_chat_completion(resp)

    # (3) unified .complete(prompt)
    fn = getattr(_llm, "complete", None)
    if callable(fn):
        prompt = "\n".join(m["content"] for m in messages if m.get("content"))
        out = await _maybe_async(fn, prompt, **kwargs)
        return out if isinstance(out, str) else str(out)

    # (4) legacy llm_service.generate_response (returns LLMResponse)
    if hasattr(_llm, "generate_response"):
        resp = await _llm.generate_response(  # type: ignore
            message=messages[-1]["content"],
            use_rag=False,
            model=kwargs.get("model"),
            max_tokens=kwargs.get("max_tokens", S.max_tokens_default),
            temperature=kwargs.get("temperature", S.temperature_default),
            context_k=S.context_k_default,
        )
        return resp.response

    raise RuntimeError("No chat/completion method available on LLM client")


async def _stream(messages: List[Dict[str, str]], **kwargs):
    """
    Provider-agnostic streaming generator yielding dict frames:
      {"content": "...", "finished": bool}
    Preference order:
      1) llm_client.create_completion(stream=True)
      2) llm_client.stream_chat(...)
      3) legacy llm_service.generate_streaming_response(...)
      4) fallback: single non-stream frame.
    """
    # (1) llm_client single call site with stream=True
    fn = getattr(_llm, "create_completion", None)
    if callable(fn):
        stream_obj = await _maybe_async(fn, messages=messages, stream=True, **kwargs)
        # handle async or sync iterators
        if hasattr(stream_obj, "__aiter__"):
            async for chunk in stream_obj:  # type: ignore
                content = ""
                try:
                    # OpenAI chunk shape
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None  # type: ignore
                    content = getattr(delta, "content", "") or ""
                except Exception:
                    content = str(getattr(chunk, "content", "") or "")
                yield {"content": content, "finished": False}
            yield {"content": "", "finished": True}
            return
        if hasattr(stream_obj, "__iter__"):
            for chunk in stream_obj:  # type: ignore
                content = ""
                try:
                    delta = chunk.choices[0].delta if getattr(chunk, "choices", None) else None  # type: ignore
                    content = getattr(delta, "content", "") or ""
                except Exception:
                    content = str(getattr(chunk, "content", "") or "")
                yield {"content": content, "finished": False}
            yield {"content": "", "finished": True}
            return

    # (2) unified stream_chat
    fn = getattr(_llm, "stream_chat", None)
    if callable(fn):
        async for chunk in fn(messages=messages, **kwargs):  # type: ignore
            yield {"content": str(chunk.get("content", "")), "finished": bool(chunk.get("finished", False))}
        return

    # (3) legacy streaming
    fn = getattr(_llm, "generate_streaming_response", None)
    if callable(fn):
        async for chunk in fn(**kwargs):  # type: ignore
            yield {"content": str(chunk.get("content", "")), "finished": bool(chunk.get("finished", False))}
        return

    # (4) fallback one-shot
    text = await _chat(messages, **kwargs)
    yield {"content": text, "finished": True}


# ----------------------------------- routes -----------------------------------

@router.post("/", response_model=LLMResponse)
async def llm(request: LLMRequest):
    if not request.message or not request.message.strip():
        raise _http_400("Message cannot be empty")

    try:
        # --- analyze query & choose model ---
        analysis = await query_analyzer.analyze_query(request.message)
        is_complex = str(getattr(analysis, "complexity", "")).lower() in {"complex", "research_intensive"}

        requested_model = (request.model or S.default_model)
        if is_complex and requested_model in {"llama3.3-70b-wns-airesearch-domain", "meta/llama-3.3-70b-instruct"}:
            model_name = _resolve_model_name(getattr(S, "complex_model", None) or requested_model)
        else:
            model_name = _resolve_model_name(requested_model)

        logger.info(
            "LLM request: use_rag=%s model=%s max_tokens=%s temp=%s context_k=%s",
            request.use_rag, model_name, request.max_tokens, request.temperature, request.context_k
        )

        # Legacy llm_service handles RAG path directly
        resp = await _llm.generate_response(  # type: ignore
            message=request.message,
            use_rag=request.use_rag,
            model=model_name,
            max_tokens=_cap_tokens(request.max_tokens),
            temperature=request.temperature,
            context_k=request.context_k,
        )
        return resp

    except HTTPException:
        raise
    except Exception:
        logger.exception("LLM failed")
        raise _http_500()


@router.post("/simple")
async def simple_llm(
    message: str,
    use_rag: bool = Query(False),
    model: Optional[str] = None,
    max_tokens: int = Query(default=None, ge=1, le=S.max_tokens_cap),
):
    if not message or not message.strip():
        raise _http_400("Message cannot be empty")

    try:
        fast_fallback = getattr(S, "fast_model", None) or S.default_model
        chosen_model = _resolve_model_name(model or fast_fallback)
        chosen_tokens = _cap_tokens(max_tokens)

        if not _use_new_llm and hasattr(_llm, "generate_response"):
            resp = await _llm.generate_response(  # type: ignore
                message=message, use_rag=use_rag, model=chosen_model,
                max_tokens=chosen_tokens, temperature=S.temperature_default, context_k=S.context_k_default
            )
            return {
                "response": resp.response,
                "sources_used": len(resp.sources),
                "tokens_used": resp.tokens_used,
                "model": chosen_model,
                "max_tokens": chosen_tokens,
            }

        messages = [
            {"role": "system", "content": _get_system_prompt()},
            {"role": "user", "content": message},
        ]
        text = await _chat(messages, model=chosen_model, max_tokens=chosen_tokens, temperature=S.temperature_default)
        return {"response": text, "sources_used": 0, "tokens_used": 0, "model": chosen_model, "max_tokens": chosen_tokens}
    except Exception:
        logger.exception("Simple llm failed")
        raise _http_500()


@router.post("/with-document", response_model=LLMResponse)
async def llm_with_document(
    document_id: str,
    message: str,
    model: Optional[str] = None,
    max_tokens: int = Query(default=None, ge=1, le=S.max_tokens_cap),
):
    if not message or not message.strip():
        raise _http_400("Message cannot be empty")

    try:
        resolved = _resolve_model_name(model or S.default_model)
        tokens = _cap_tokens(max_tokens)

        if not _use_new_llm and hasattr(_llm, "llm_with_document"):
            return await _llm.llm_with_document(  # type: ignore
                document_id=document_id, message=message, model=resolved, max_tokens=tokens
            )

        vs = _vec_fn()
        q_vec = await _embed_text(message)
        filter_expr = f'document_id == "{document_id}"'
        topk = getattr(S, "context_k_default", 5)
        results = await vs.search_similar(query_embedding=q_vec, top_k=topk, filter_expr=filter_expr)

        if not results:
            context = ""
            sources = []
        else:
            results.sort(key=_score_of, reverse=True)
            sources = results[:topk]
            parts = []
            for i, s in enumerate(sources, 1):
                page = getattr(s, "page_number", None)
                score = _score_of(s)
                parts.append(
                    f"[Source {i}] (Document: {getattr(s, 'document_id', '')}"
                    f"{', Page: ' + str(page) if page is not None else ''}, "
                    f"Score: {score:.3f}):\n{getattr(s, 'content', '')}\n"
                )
            context = "\n".join(parts)

        sys = "You answer questions based on the provided document context. Cite [Source N] when using specific facts."
        user = f"Context:\n{context}\n\nQuestion: {message}\nProvide a grounded answer; say if info is missing."
        start = time.time()
        text = await _chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            model=resolved, max_tokens=tokens, temperature=S.temperature_default
        )
        processing_time = time.time() - start

        return LLMResponse(
            response=text,
            sources=sources,
            model=resolved,
            tokens_used=0,
            processing_time=processing_time,
            context_used=bool(sources),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Document llm failed")
        raise _http_500()


def _validate_summary_type(summary_type: str) -> str:
    if summary_type not in S.summary_types:
        raise _http_400(f"Invalid summary_type. Allowed: {', '.join(S.summary_types)}")
    return summary_type


@router.post("/summarize/{document_id}")
async def summarize_document(
    document_id: str,
    summary_type: str = Query(default=S.summary_default),
    max_tokens: int = Query(default=S.max_tokens_default, ge=1, le=S.max_tokens_cap),
):
    try:
        summary_type = _validate_summary_type(summary_type)
        logger.info("Summarizing document %s (%s)", document_id, summary_type)

        if hasattr(_llm, "summarize_document"):
            summary = await _llm.summarize_document(  # type: ignore
                document_id=document_id, summary_type=summary_type, max_tokens=_cap_tokens(max_tokens)
            )
            return {"document_id": document_id, "summary_type": summary_type, "summary": summary, "max_tokens": _cap_tokens(max_tokens)}

        vs = _vec_fn()
        filter_expr = f'document_id == "{document_id}"'
        list_by_filter = getattr(vs, "list_by_filter", None)
        if callable(list_by_filter):
            chunks = await list_by_filter(filter_expr=filter_expr, limit=S.summary_context_cap)
        else:
            q_vec = await _embed_text(f"summary:{summary_type}")
            chunks = await vs.search_similar(query_embedding=q_vec, top_k=S.summary_context_cap, filter_expr=filter_expr)

        if not chunks:
            raise HTTPException(status_code=404, detail="No content available to summarize")

        chunks.sort(key=_score_of, reverse=True)
        ctx = "\n\n".join(getattr(c, "content", "") for c in chunks[:S.summary_context_cap])

        sys = "You create concise, faithful summaries."
        user = f"Summary type: {summary_type}\n\nCONTENT START\n{ctx}\nCONTENT END\n\nWrite a clear {summary_type} summary."
        text = await _chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            model=_resolve_model_name(S.default_model),
            max_tokens=_cap_tokens(max_tokens),
            temperature=S.temperature_default,
        )
        return {"document_id": document_id, "summary_type": summary_type, "summary": text, "max_tokens": _cap_tokens(max_tokens)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Document summarization failed")
        raise _http_500()


@router.get("/stream")
async def llm_stream_get(
    message: str = Query(..., min_length=1),
    use_rag: bool = Query(True),
    model: str = Query(S.default_model),
    max_tokens: int = Query(S.max_tokens_default, ge=1, le=S.max_tokens_cap),
    temperature: float = Query(S.temperature_default, ge=0.0, le=2.0),
    context_k: int = Query(S.context_k_default, ge=1),
):
    resolved_model = _resolve_model_name(model)

    async def generate_stream():
        logger.info("SSE STREAM start")
        last_beat = time.monotonic()
        try:
            sys_prompt = _get_system_prompt()
            messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": message}]
            async for chunk in _stream(
                messages,
                model=resolved_model,
                max_tokens=_cap_tokens(max_tokens),
                temperature=temperature,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
                if S.sse_heartbeat_ms > 0:
                    now = time.monotonic()
                    if (now - last_beat) * 1000 >= S.sse_heartbeat_ms:
                        yield ": heartbeat\n\n"
                        last_beat = now
            yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("SSE STREAM crashed")
            yield f"data: {json.dumps({'error': 'streaming_error'})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": S.sse_headers_cache_control,
            "Connection": S.sse_headers_connection,
            "X-Accel-Buffering": S.sse_headers_accel_buffering,
        },
    )


@router.post("/multi-document", response_model=LLMResponse)
async def llm_with_multiple_documents(
    document_ids: List[str],
    message: str,
    model: Optional[str] = None,
    max_tokens: int = Query(default=None, ge=1, le=S.max_tokens_cap),
    context_k_per_doc: int = Query(default=S.context_k_per_doc_default, ge=1),
):
    if not message or not message.strip():
        raise _http_400("Message cannot be empty")
    if not document_ids:
        raise _http_400("At least one document ID required")
    if len(document_ids) > S.max_documents_per_request:
        raise _http_400(f"Maximum {S.max_documents_per_request} documents allowed")

    try:
        logger.info("Multi-document llm with %d documents", len(document_ids))
        vs = _vec_fn()
        q_vec = await _embed_text(message)

        all_sources = []
        for doc_id in document_ids:
            try:
                doc_sources = await vs.search_similar(
                    query_embedding=q_vec,
                    top_k=context_k_per_doc,
                    filter_expr=f'document_id == "{doc_id}"',
                )
                all_sources.extend(doc_sources)
            except Exception as e:
                logger.warning("Context fetch failed for doc %s: %s", doc_id, e)

        if not all_sources:
            raise HTTPException(status_code=404, detail="No relevant content found in the specified documents")

        all_sources.sort(key=_score_of, reverse=True)
        selected_sources = all_sources[: context_k_per_doc * len(document_ids)]

        parts = []
        for i, s in enumerate(selected_sources, 1):
            page = getattr(s, "page_number", None)
            score = _score_of(s)
            parts.append(
                f"[Source {i}] (Document: {getattr(s, 'document_id', '')}"
                f"{', Page: ' + str(page) if page is not None else ''}, "
                f"Score: {score:.3f}):\n{getattr(s, 'content', '')}\n"
            )
        context = "\n".join(parts)

        resolved_model = _resolve_model_name(model or S.default_model)
        sys = "You answer based on multi-document context. Cite [Source N] when using specifics."
        user = f"Context:\n{context}\n\nQuestion: {message}"
        start = time.time()
        text = await _chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": user}],
            model=resolved_model,
            max_tokens=_cap_tokens(max_tokens),
            temperature=S.temperature_default,
        )
        processing_time = time.time() - start

        return LLMResponse(
            response=text,
            sources=selected_sources,
            model=resolved_model,
            tokens_used=0,
            processing_time=processing_time,
            context_used=True,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Multi-document llm failed")
        raise _http_500()


@router.post("/follow-up", response_model=LLMResponse)
async def follow_up_llm(
    original_message: str,
    follow_up_message: str,
    model: Optional[str] = None,
    max_tokens: int = Query(default=None, ge=1, le=S.max_tokens_cap),
):
    if not follow_up_message or not follow_up_message.strip():
        raise _http_400("Follow-up message cannot be empty")

    try:
        combined_query = f"Original question: {original_message}\n\nFollow-up question: {follow_up_message}"
        q_vec = await _embed_text(combined_query)
        vs = _vec_fn()
        sources = await vs.search_similar(query_embedding=q_vec, top_k=S.followup_top_k)

        convo_ctx = (
            f"Previous question: {original_message}\n"
            f"Current question: {follow_up_message}\n\n"
            "This is a follow-up. Consider the conversation history and return a contextual answer."
        )
        if sources:
            doc_ctx_parts = []
            for i, s in enumerate(sources, 1):
                sc = _score_of(s)
                doc_ctx_parts.append(
                    f"[Source {i}] (Document: {getattr(s, 'document_id', '')}, Score: {sc:.3f}):\n"
                    f"{getattr(s, 'content', '')}\n"
                )
            convo_ctx += "\n\nRelevant document content:\n" + "\n".join(doc_ctx_parts)

        resolved_model = _resolve_model_name(model or S.default_model)
        start = time.time()
        text = await _chat(
            [{"role": "system", "content": _get_system_prompt()},
             {"role": "user", "content": convo_ctx}],
            model=resolved_model,
            max_tokens=_cap_tokens(max_tokens),
            temperature=S.temperature_default,
        )
        processing_time = time.time() - start

        return LLMResponse(
            response=text,
            sources=sources,
            model=resolved_model,
            tokens_used=0,
            processing_time=processing_time,
            context_used=bool(sources),
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Follow-up llm failed")
        raise _http_500()


@router.get("/models")
async def list_available_models():
    try:
        lister = getattr(_llm, "list_models", None)
        if callable(lister):
            models = await _maybe_async(lister)
            return {
                "models": models,
                "default_model": S.default_model,
                "recommended_for_rag": S.recommended_for_rag,
                "note": "Model name must match provider/deployment id where applicable.",
            }
    except Exception:
        logger.warning("Falling back to settings.available_models", exc_info=True)

    models = [{"model": name, **meta} for name, meta in S.available_models.items()]
    return {
        "models": models,
        "default_model": S.default_model,
        "recommended_for_rag": S.recommended_for_rag,
        "note": "Model name must match provider/deployment id where applicable.",
    }
