# source/routers/llm_client.py
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Literal, Callable

from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/llm-client", tags=["LLM Client"])

# ---------------- client wiring: prefer factory, fallback to legacy singleton ----------------
_use_factory = False
try:
    # New style: a factory returning a provider-agnostic client
    from source.services.llm_client import get_llm_client  # type: ignore
    _client = get_llm_client()
    _use_factory = True
except Exception:
    # Legacy: directly import a configured client/singleton
    from source.services.llm_client import llm_client as _client  # type: ignore
    _use_factory = False


# ---------------- helpers ----------------
def _http_500(msg: str) -> HTTPException:
    return HTTPException(status_code=500, detail=msg)

async def _maybe_await(fn: Callable[..., Any], *args, **kwargs) -> Any:
    """Call a callable that may be sync or async; also handle coroutines returned by sync wrappers."""
    res = fn(*args, **kwargs)
    if hasattr(res, "__await__"):
        return await res  # type: ignore
    return res

def _resolve_model(name: Optional[str]) -> str:
    # Prefer client resolver, fall back to passthrough
    for attr in ("resolve_model", "_resolve_model", "resolve"):
        fn = getattr(_client, attr, None)
        if callable(fn):
            try:
                return fn(name)
            except Exception:
                break
    return (name or "").strip() or getattr(_client, "default_model", "")

def _client_type() -> str:
    for attr in ("get_client_type", "client_type", "provider_name"):
        val = getattr(_client, attr, None)
        if callable(val):
            try:
                return str(val())
            except Exception:
                continue
        if isinstance(val, str):
            return val
    return "generic"

def _list_models() -> Any:
    fn = getattr(_client, "list_available_models", None) or getattr(_client, "list_models", None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            pass
    # fallback to a static default if exposed
    return getattr(_client, "available_models", [])

def _cap_max_tokens(requested: int) -> int:
    cap = getattr(_client, "max_tokens_cap", None)
    default = getattr(_client, "max_tokens_default", 1000)
    if cap is None:
        return min(requested or default, 4096)  # safe ceiling
    return min(requested or default, int(cap))

def _normalize_completion(resp: Any) -> Dict[str, Any]:
    """
    Normalize provider responses to {content, tokens_used, finish_reason}.
    Supports:
      - OpenAI-style: resp.choices[0].message.content, resp.usage.total_tokens
      - Anthropic-like: resp.content / resp.completion
      - Simple string: resp
      - Dicts with 'content'/'usage'
    """
    content = ""
    tokens = 0
    finish = None

    if isinstance(resp, str):
        content = resp
    elif isinstance(resp, dict):
        content = resp.get("content") or resp.get("text") or ""
        usage = resp.get("usage") or {}
        tokens = usage.get("total_tokens", usage.get("tokens", 0)) or 0
        finish = resp.get("finish_reason")
        # OpenAI-like in dict form
        try:
            if not content and "choices" in resp and resp["choices"]:
                content = resp["choices"][0].get("message", {}).get("content") \
                          or resp["choices"][0].get("text", "") \
                          or resp["choices"][0].get("content", "")
                finish = finish or resp["choices"][0].get("finish_reason")
        except Exception:
            pass
    else:
        # Try OpenAI-like object
        try:
            choices = getattr(resp, "choices", None)
            if choices:
                ch0 = choices[0]
                msg = getattr(ch0, "message", None)
                content = (getattr(msg, "content", None) if msg else None) \
                          or getattr(ch0, "text", None) \
                          or getattr(ch0, "content", None) \
                          or ""
                finish = getattr(ch0, "finish_reason", None)
            usage = getattr(resp, "usage", None)
            if usage:
                tokens = getattr(usage, "total_tokens", 0) or getattr(usage, "tokens", 0) or 0
        except Exception:
            # Anthropic-like
            content = getattr(resp, "content", None) or getattr(resp, "completion", "") or str(resp)

    return {"content": content or "", "tokens_used": int(tokens or 0), "finish_reason": finish}


# ---------------- Pydantic I/O models ----------------
RoleLiteral = Literal["system", "user", "assistant"]

class ChatMessage(BaseModel):
    role: RoleLiteral
    content: str

class CompletionRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="OpenAI chat-style message list")
    model: Optional[str] = Field(None, description="Model name or provider-specific deployment")
    max_tokens: int = Field(1000, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0, le=2)
    # Extra provider kwargs; e.g., tools, stop, top_p, response_format...
    extra: Dict[str, Any] = Field(default_factory=dict)

class CompletionResponse(BaseModel):
    model: str
    content: str
    tokens_used: int
    finish_reason: Optional[str] = None


# ---------------- Routes ----------------

@router.get("/ping")
async def ping() -> Dict[str, str]:
    return {"status": "ok", "service": "llm-client"}

@router.get("/type")
async def client_type() -> Dict[str, str]:
    return {"client_type": _client_type()}

@router.get("/resolve-model")
async def resolve_model(name: Optional[str] = Query(None, description="Optional model hint")) -> Dict[str, str]:
    return {"resolved": _resolve_model(name)}

@router.get("/models")
async def list_models() -> Dict[str, Any]:
    return {"models": _list_models()}

@router.get("/health")
async def health() -> Dict[str, Any]:
    # Prefer async health checker; tolerate sync
    checker = getattr(_client, "health_check", None)
    if callable(checker):
        return await _maybe_await(checker)
    return {"status": "ok"}

@router.post("/reconnect")
async def reconnect() -> Dict[str, str]:
    try:
        # Prefer explicit reconnect/reset; else re-run the factory
        rec = getattr(_client, "reconnect", None) or getattr(_client, "reset", None)
        if callable(rec):
            await _maybe_await(rec)
        elif _use_factory:
            # Recreate the client from factory if no explicit method
            from source.services.llm_client import get_llm_client  # type: ignore
            globals()["_client"] = get_llm_client()
        return {"status": "reinitialized"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reconnect failed: {e}")

@router.post("/completions", response_model=CompletionResponse)
async def create_completion(req: CompletionRequest) -> CompletionResponse:
    """
    Provider-agnostic chat completion (non-streaming).
    Accepts OpenAI-style 'messages' and forwards extra kwargs unmodified.
    """
    try:
        model = _resolve_model(req.model)
        mtoks = _cap_max_tokens(req.max_tokens)
        messages = [m.model_dump() for m in req.messages]

        # Preferred: client.chat or client.create_completion (non-stream)
        for meth in ("chat", "create_completion", "complete"):
            fn = getattr(_client, meth, None)
            if callable(fn):
                resp = await _maybe_await(
                    fn,
                    messages=messages if meth != "complete" else None,
                    prompt="\n".join(m["content"] for m in messages) if meth == "complete" else None,
                    model=model,
                    max_tokens=mtoks,
                    temperature=req.temperature,
                    stream=False,
                    **(req.extra or {}),
                )
                norm = _normalize_completion(resp)
                return CompletionResponse(
                    model=model,
                    content=norm["content"],
                    tokens_used=norm["tokens_used"],
                    finish_reason=norm["finish_reason"],
                )

        # Fallback: direct OpenAI SDK style if exposed as attribute
        client = getattr(_client, "client", None)
        if client and hasattr(client, "chat") and hasattr(client.chat, "completions"):
            resp = await _maybe_await(
                client.chat.completions.create,  # type: ignore
                model=model, messages=messages, max_tokens=mtoks, temperature=req.temperature, stream=False, **(req.extra or {}),
            )
            norm = _normalize_completion(resp)
            return CompletionResponse(model=model, content=norm["content"], tokens_used=norm["tokens_used"], finish_reason=norm["finish_reason"])

        raise RuntimeError("No completion method available on LLM client")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Completion failed: {e}")


# ---------------- Streaming (SSE) ----------------

class StreamRequest(CompletionRequest):
    heartbeat_ms: int = Field(15000, ge=0, description="Interval for SSE heartbeats (0 = disabled)")

@router.post("/completions/stream")
async def create_completion_stream(req: StreamRequest):
    """
    SSE stream of chat completion chunks.
    Frames:  data: {"content":"...", "finish_reason":null, "finished":false}\n\n
    Ends with: data: [DONE]\n\n
    """
    model = _resolve_model(req.model)
    mtoks = _cap_max_tokens(req.max_tokens)
    messages = [m.model_dump() for m in req.messages]

    async def gen():
        last_beat = time.monotonic()
        try:
            # Preferred: client.stream_chat / client.create_completion(..., stream=True)
            stream_fn = getattr(_client, "stream_chat", None) or getattr(_client, "create_completion", None)
            if callable(stream_fn):
                # If using create_completion, pass stream=True to get an async iterator if supported
                kwargs = dict(model=model, max_tokens=mtoks, temperature=req.temperature, **(req.extra or {}))
                if stream_fn.__name__ == "create_completion":
                    kwargs["stream"] = True
                    kwargs["messages"] = messages
                else:
                    kwargs["messages"] = messages

                stream = await _maybe_await(stream_fn, **kwargs)

                # If provider yields dicts/chunks
                async def _aiter(obj):
                    # If the object itself is async-iterable
                    if hasattr(obj, "__aiter__"):
                        async for ch in obj:
                            yield ch
                    else:
                        # Some clients return a sync iterator even in async contexts
                        for ch in obj:
                            yield ch

                async for chunk in _aiter(stream):
                    # Normalize chunk content
                    text = ""
                    finish = None
                    if isinstance(chunk, dict):
                        text = chunk.get("content") or chunk.get("delta") or chunk.get("text") or ""
                        finish = chunk.get("finish_reason")
                    else:
                        # OpenAI-like stream delta
                        try:
                            choice = getattr(chunk, "choices", [None])[0]
                            if choice is not None:
                                delta = getattr(choice, "delta", None)
                                text = (getattr(delta, "content", None) if delta else None) \
                                       or getattr(choice, "text", None) or ""
                                finish = getattr(choice, "finish_reason", None)
                        except Exception:
                            text = str(chunk)

                    if text:
                        yield f"data: {json.dumps({'content': text, 'finish_reason': finish, 'finished': False})}\n\n"

                    # heartbeat
                    if req.heartbeat_ms > 0:
                        now = time.monotonic()
                        if (now - last_beat) * 1000 >= req.heartbeat_ms:
                            yield ": heartbeat\n\n"
                            last_beat = now

                yield "data: [DONE]\n\n"
                return

            # Fallback: one-shot completion if no streaming API
            one_shot = await create_completion(CompletionRequest(**req.model_dump(exclude={"heartbeat_ms"})))
            yield f"data: {json.dumps({'content': one_shot.content, 'finish_reason': one_shot.finish_reason, 'finished': True})}\n\n"
            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            # client closed connection
            return
        except Exception as e:
            yield f"data: {json.dumps({'error': f'stream_error: {e}'})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx
        },
    )
