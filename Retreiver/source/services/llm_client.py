# source/services/llm_client.py
# type: ignore
from __future__ import annotations

"""
LLM Client Service (adapter-first, config-driven)

- Single call site for all LLM usage: create_completion(...)
- Providers:
    * Azure OpenAI (AzureOpenAI SDK)
    * OpenAI (OpenAI SDK)
    * Custom HTTP completions endpoint (e.g., fine-tuned Llama)
- Model aliasing & provider/deployment mapping from config
- Compatible with existing routers/services (sync return values; stream yields)
- No environment reads required, but tolerated as fallback

Expected config (supports both dict and list formats for available_models):

llm_client:
  http_timeout_sec: 60
  http_retries: 3
  default_model: "gpt-4o"
  default_provider: "azure"
  available_models:
    gpt-4o:
      provider: "azure"
      deployment: "gpt-4o"
    gpt-4o-mini:
      provider: "azure"
      deployment: "gpt-4o-mini"
    llama3.3-70b-wns-airesearch-domain:
      provider: "custom"
      api_url: "http://3.212.184.57:5001/v1/completions"
      model_id: "llama3.3-70b-wns-airesearch-domain"

cloud_services:
  # Azure
  azure_api_key: "..."
  azure_endpoint_url: "https://<your-azure>.openai.azure.com/"
  api_version: "2024-02-15-preview"

  # OpenAI
  openai_api_key: "..."
"""

import os
import time
import logging
from types import SimpleNamespace
from typing import Dict, List, Any, Optional, Tuple, Callable, Iterable

import httpx
from openai import OpenAI, AzureOpenAI  # type: ignore

# ----------------------------- Config ----------------------------- #
from source.config import (
    load_cloud_services,
    load_llm_client,
    load_llm_service,  # legacy merge
)

_cloud = load_cloud_services() or {}
_llm_client_cfg = {}
try:
    _llm_client_cfg = load_llm_client() or {}
except Exception:
    _llm_client_cfg = {}

# Back-compat: some values might live under llm_service; merge in (last wins)
try:
    _llm_service_cfg = load_llm_service() or {}
except Exception:
    _llm_service_cfg = {}

def _merge(*dicts):
    out: Dict[str, Any] = {}
    for d in dicts:
        if isinstance(d, dict):
            out.update(d)
    return out

LLM_CFG: Dict[str, Any] = _merge(
    {
        "http_timeout_sec": 60,
        "http_retries": 3,
        "default_model": "gpt-4o",
        "default_provider": "azure",
        "available_models": {},
        # Convenience values some routers read:
        "max_tokens_cap": 128000,
        "max_tokens_default": 1000,
    },
    _llm_service_cfg,   # legacy values (if any)
    _llm_client_cfg,    # preferred
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# ----------------------------- HTTP ----------------------------- #

def _http_client() -> httpx.Client:
    """
    Shared HTTP client (timeout; redirects). We do manual retries in create_completion.
    """
    timeout_sec = float(LLM_CFG.get("http_timeout_sec", 60))
    return httpx.Client(timeout=timeout_sec, follow_redirects=True)


# ----------------------------- Provider Shims ----------------------------- #

class ProviderError(RuntimeError):
    pass


class _Provider:
    """Base provider shim."""
    name: str = "base"

    def __init__(self):
        self.client = None

    def init(self) -> None:
        raise NotImplementedError

    def supports_stream(self) -> bool:
        return True

    def chat(self, **kwargs):
        """
        Providers return either:
          - OpenAI/Azure ChatCompletion-like object for non-stream
          - An iterator/async-iterator for stream=True
        """
        if self.client is None:
            self.init()
        return self.client.chat.completions.create(**kwargs)


class _AzureProvider(_Provider):
    name = "azure"

    def init(self) -> None:
        api_key = _cloud.get("azure_api_key") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        endpoint = _cloud.get("azure_endpoint_url") or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("ENDPOINT_URL")
        api_version = _cloud.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("API_VERSION") or "2024-02-15-preview"
        if not api_key or not endpoint:
            raise ProviderError("Azure credentials missing (azure_api_key, azure_endpoint_url).")
        self.client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


class _OpenAIProvider(_Provider):
    name = "openai"

    def init(self) -> None:
        api_key = _cloud.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OpenAI API key missing (cloud_services.openai_api_key or OPENAI_API_KEY).")
        self.client = OpenAI(api_key=api_key)


class _CustomProvider(_Provider):
    """
    Simple HTTP completions adapter for custom fine-tuned models.

    Expected request JSON:
      {
        "model": "<model_id>",
        "prompt": "<combined prompt from messages>",
        "max_tokens": 1000,
        "temperature": 0.7,
        "stream": false
      }

    Expected response JSON (non-stream):
      {
        "choices": [{"text": "..."}],
        "usage": {"total_tokens": <int>}
      }

    For stream=True, if the endpoint doesn't truly stream, we emulate a
    single-chunk iterator so upstream streaming still works.
    """
    name = "custom"

    def __init__(self):
        super().__init__()
        self.http: Optional[httpx.Client] = None

    def init(self) -> None:
        self.http = _http_client()
        self.client = self  # sentinel to satisfy base class expectations

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are an expert assistant who answers questions using ONLY the following context."
            "If the context does not contain the full answer, summarize the relevant points from the context without adding unsupported info. Do notmention the source number in reply.\n\n"
            "<context>\n"
            f"{context}\n"
            "</context>\n\n"
            "<question>\n"
            f"{query}\n"
            "</question>\n\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    def _combine_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Combine system + user/assistant turns into one text prompt for text-only
        completion endpoints.
        """
        query, content = messages[0]['query'], messages[0]['content']
        prompt = self._build_prompt(query, content)
        return prompt
        '''
        parts: List[str] = []
        system = [m for m in messages if m.get("role") == "system"]
        users = [m for m in messages if m.get("role") == "user"]
        assists = [m for m in messages if m.get("role") == "assistant"]

        if system:
            parts.append(f"[System]\n{system[-1].get('content','').strip()}\n")

        for u in users:
            txt = (u.get("content") or "").strip()
            if txt:
                parts.append(f"[User]\n{txt}\n")

        # (optional) last assistant context if provided
        if assists:
            a = assists[-1].get("content", "").strip()
            if a:
                parts.append(f"[Assistant]\n{a}\n")

        return "\n".join(parts).strip()
        '''

    def _as_chat_like(self, text: str, total_tokens: int = 0):
        """Return an OpenAI ChatCompletion-like object."""
        choice = SimpleNamespace()
        choice.message = SimpleNamespace()
        choice.message.content = text
        result = SimpleNamespace()
        result.choices = [choice]
        result.usage = SimpleNamespace(total_tokens=total_tokens)
        return result

    def _stream_from_text(self, text: str) -> Iterable[Dict[str, Any]]:
        """
        Emulate a streaming iterator by yielding small chunks as dicts
        that the routers can understand.
        """
        # yield in ~200 char pieces
        step = 200
        for i in range(0, len(text), step):
            yield {"content": text[i:i + step], "finish_reason": None}
        yield {"content": "", "finish_reason": "stop"}

    def chat(self, *, model: str, messages: List[Dict[str, str]], meta: Dict[str, Any] | None = None, **kwargs):
        if self.http is None:
            self.init()

        meta = meta or {}
        api_url = meta.get("api_url")
        model_id = meta.get("model_id") or model
        if not api_url:
            raise ProviderError(f"Custom provider: api_url missing for model '{model_id}'")

        prompt = self._combine_messages_to_prompt(messages)
        payload = {
            "model": model_id,
            "prompt": prompt,
            "max_tokens": int(kwargs.get("max_tokens", 1000)),
            "temperature": float(kwargs.get("temperature", 0)),
            "top_p" : 1,
            "n": 1,
            "stream": bool(kwargs.get("stream", False)),
        }

        # Basic manual retry for transient failures
        retries = int(LLM_CFG.get("http_retries", 3))
        backoff = 0.6
        last_exc: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                print(payload)
                resp = self.http.post(api_url, json=payload, headers={"Content-Type": "application/json", "Accept": "application/json"})
                resp.raise_for_status()
                data = resp.json()
                print(data)
                if payload["stream"]:
                    # If the endpoint doesn't truly stream, emulate streaming from the full text
                    text = ""
                    if isinstance(data, dict):
                        if "choices" in data and data["choices"]:
                            text = str(
                                data["choices"][0].get("text", "")
                                or data["choices"][0].get("message", {}).get("content", "")
                            )
                        else:
                            text = str(data.get("text", ""))

                    return self._stream_from_text(text)

                # Non-stream: return ChatCompletion-like
                total_tokens = 0
                if isinstance(data, dict):
                    usage = data.get("usage") or {}
                    total_tokens = int(usage.get("total_tokens") or usage.get("tokens") or 0)
                    if "choices" in data and data["choices"]:
                        text = str(
                            data["choices"][0].get("text", "")
                            or data["choices"][0].get("message", {}).get("content", "")
                        )
                        return self._as_chat_like(text, total_tokens=total_tokens)
                    # fallback
                    return self._as_chat_like(str(data.get("text", "")), total_tokens=total_tokens)

                # last fallback: stringify
                return self._as_chat_like(str(data), total_tokens=0)

            except Exception as e:
                last_exc = e
                if attempt >= retries:
                    logger.error(f"Custom provider call failed after {attempt+1} attempts: {e}")
                    raise
                logger.warning(f"Custom provider error (attempt {attempt+1}/{retries}): {e}; retrying in {backoff}s")
                time.sleep(backoff)

        # Should never reach here
        if last_exc:
            raise last_exc
        raise ProviderError("Custom provider call failed (unknown error).")


_PROVIDER_FACTORIES: Dict[str, Callable[[], _Provider]] = {
    "azure": lambda: _AzureProvider(),
    "openai": lambda: _OpenAIProvider(),
    "custom": lambda: _CustomProvider(),
}


# ----------------------------- Model Resolution ----------------------------- #

def _find_model_meta(alias_or_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Return (meta, alias) for a given model key. Supports:
      - dict-style llm_client.available_models
      - list-style llm_settings.available_models
    """
    avail = LLM_CFG.get("available_models", {})
    if isinstance(avail, dict):
        meta = avail.get(alias_or_id)
        if isinstance(meta, dict):
            return meta, alias_or_id
        return None, None
    elif isinstance(avail, list):
        for item in avail:
            if isinstance(item, dict) and item.get("model") == alias_or_id:
                m = dict(item)
                if "provider" not in m and LLM_CFG.get("default_provider"):
                    m["provider"] = LLM_CFG["default_provider"]
                return m, alias_or_id
    return None, None


def _resolve_model_and_provider(model: Optional[str] = None,
                                provider: Optional[str] = None) -> Tuple[str, str, Dict[str, Any]]:
    """
    Resolve (string_for_sdk, provider_name, meta):
      - If provider is explicitly passed, it wins.
      - If model maps to a meta entry, use its provider.
      - Azure returns the DEPLOYMENT NAME.
      - OpenAI returns the MODEL ID.
      - Custom returns the model alias/id but also meta (api_url, model_id).
    """
    default_provider = str(LLM_CFG.get("default_provider", "azure")).strip().lower()
    default_model = str(LLM_CFG.get("default_model", "gpt-4o")).strip()

    requested = (model or default_model).strip()
    meta, alias = _find_model_meta(requested)

    prov = (provider or (meta or {}).get("provider") or default_provider).strip().lower()

    # Choose the string sent to the SDK:
    if prov == "azure":
        # SDK expects DEPLOYMENT name
        deployment = (meta or {}).get("deployment") or requested
        return deployment, "azure", (meta or {})
    elif prov == "openai":
        # SDK expects model id
        model_id = (meta or {}).get("model_id") or requested
        return model_id, "openai", (meta or {})
    elif prov == "custom":
        # We will hand meta to the provider
        model_name = (meta or {}).get("deployment") or requested
        return model_name, "custom", (meta or {})
    else:
        # Fallback to OpenAI semantics
        return requested, "openai", (meta or {})


# ----------------------------- LLM Client Service ----------------------------- #

class LLMClientService:
    """
    Central LLM client with multi-provider routing + simple retry/backoff.

    Public surface (backward-compatible with your routers):
      - get_client()
      - get_client_type()
      - resolve_model()
      - create_completion()
      - chat(messages=..., **kwargs)
      - complete(prompt=..., **kwargs)
      - stream_chat(messages=..., **kwargs)
      - list_available_models()
      - reconnect() / reset()
      - health_check()
      - attributes: default_model, max_tokens_cap, max_tokens_default
    """

    def __init__(self) -> None:
        self._providers: Dict[str, _Provider] = {}
        self._last_provider: Optional[str] = None

        # Helpful attributes some routers read directly
        self.default_model: str = str(LLM_CFG.get("default_model", "gpt-4o"))
        self.max_tokens_cap: int = int(LLM_CFG.get("max_tokens_cap", 128000))
        self.max_tokens_default: int = int(LLM_CFG.get("max_tokens_default", 1000))
        self.available_models = LLM_CFG.get("available_models", {})

    # ---- Provider management ----
    def _get_provider(self, name: str) -> _Provider:
        pname = (name or LLM_CFG.get("default_provider", "azure")).strip().lower()
        if pname not in _PROVIDER_FACTORIES:
            raise ProviderError(f"Unsupported provider '{pname}'. Supported: {sorted(_PROVIDER_FACTORIES)}")
        if pname not in self._providers:
            self._providers[pname] = _PROVIDER_FACTORIES[pname]()  # lazy
        self._last_provider = pname
        # ensure initialized SDK client
        if self._providers[pname].client is None:
            self._providers[pname].init()
        return self._providers[pname]

    # ---- Back-compat helpers ----
    def get_client(self):
        """Return the last-used provider client (or initialize the default)."""
        prov_name = self._last_provider or str(LLM_CFG.get("default_provider", "azure"))
        return self._get_provider(prov_name).client

    def get_client_type(self) -> str:
        """Return the last-used provider name, or the default provider."""
        return self._last_provider or str(LLM_CFG.get("default_provider", "azure"))

    # ---- Model resolution ----
    def resolve_model(self, model: Optional[str] = None) -> str:
        resolved_model, _, _ = _resolve_model_and_provider(model, provider=None)
        logger.debug("Resolved model '%s' -> '%s'", model, resolved_model)
        return resolved_model

    # ---- Model listing ----
    def list_available_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        try:
            avail = LLM_CFG.get("available_models", {})
            if isinstance(avail, list):
                # llm_settings.available_models shape
                for item in avail:
                    if isinstance(item, dict) and "model" in item:
                        models.append(dict(item))
            elif isinstance(avail, dict):
                # llm_client.available_models shape
                for alias, meta in avail.items():
                    row = {"alias": alias}
                    if isinstance(meta, dict):
                        row.update(meta)
                    models.append(row)
        except Exception as e:
            logger.warning("Failed to read available models: %s", e)
        return models

    # ---- Core completion (single call site) ----
    def create_completion(
        self,
        *,
        messages: List[Dict[str, str]] | None = None,
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stream: bool = False,
        provider: Optional[str] = None,
        retries: int = 2,
        backoff_sec: float = 0.8,
        **kwargs,
    ):
        """
        Create a chat completion.

        Returns:
          - Non-stream: ChatCompletion-like object (with .choices[0].message.content and .usage.total_tokens)
          - Stream: an iterator/async-iterator yielding OpenAI-like chunks, or dicts with {"content": "...", "finish_reason": ...}

        Notes:
          - Azure: 'model' is the DEPLOYMENT name
          - OpenAI: 'model' is the MODEL ID
          - Custom: we pass meta (api_url, model_id) to the provider
        """
        messages = messages or []
        resolved_model, prov_name, meta = _resolve_model_and_provider(model, provider)
        max_tokens=500
        # --- Sanitize response_format to avoid Azure 400s ("response_format.type" missing) ---
        if "response_format" in kwargs:
            rf = kwargs["response_format"]
            # Allow shorthand strings like "json_object"
            if isinstance(rf, str):
                rf = {"type": rf}
            # Drop if falsy or missing 'type'
            if not rf or not (isinstance(rf, dict) and rf.get("type")):
                kwargs.pop("response_format", None)
            else:
                kwargs["response_format"] = rf
        # --------------------------------------------------------------------------------------

        prov = self._get_provider(prov_name)

        attempt = 0
        while True:
            try:
                if prov_name == "custom":
                    print(messages)
                    # Pass meta to custom provider
                    return  prov.chat(
                        model=resolved_model,
                        messages=messages,
                        meta=meta,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream,
                        **kwargs,
                    )

                # Azure/OpenAI path
                return prov.chat(
                    model=resolved_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=stream,
                    **kwargs,
                )

            except Exception as e:
                attempt += 1
                if attempt > max(retries, 0):
                    logger.error("Completion failed after %s attempts: %s", attempt, e)
                    raise
                logger.warning(
                    "Completion error (attempt %s/%s): %s. Retrying in %.1fs...",
                    attempt, retries, e, backoff_sec
                )
                time.sleep(backoff_sec)

    # ---- Friendly wrappers expected by some code paths ----
    def chat(self, *, messages: List[Dict[str, str]], **kwargs):
        return self.create_completion(messages=messages, **kwargs)

    def complete(self, *, prompt: str, **kwargs):
        """
        Back-compat: convert a prompt string into a single user message
        and delegate to chat.
        """
        return self.create_completion(messages=[{"role": "user", "content": prompt}], **kwargs)

    def stream_chat(self, *, messages: List[Dict[str, str]], **kwargs):
        """Explicit streaming wrapper for callers that prefer a named stream API."""
        kwargs["stream"] = True
        return self.create_completion(messages=messages, **kwargs)

    # ---- Maintenance ----
    def reconnect(self) -> None:
        """Drop cached clients; re-init on next call."""
        logger.info("Reconnecting all LLM providers...")
        for p in self._providers.values():
            p.client = None
        self._last_provider = None

    # alias for some callers
    reset = reconnect

    # ---- Health ----
    def health_check(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "default_provider": str(LLM_CFG.get("default_provider", "azure")),
            "default_model": str(LLM_CFG.get("default_model", "")),
        }


# ----------------------------- Global instance & factory ----------------------------- #

_llm_singleton = LLMClientService()
llm_client = _llm_singleton  # preferred name in the rest of the app

def get_llm_client() -> LLMClientService:
    """
    Factory for code paths that prefer a getter.
    Returns the global singleton by default to share clients/HTTP pools.
    """
    return _llm_singleton


# ----------------------------- Standalone smoke test ----------------------------- #

if __name__ == "__main__":
    print("LLM Client Service — smoke test")
    print("=" * 40)

    svc = get_llm_client()

    print("\nAvailable Models:")
    for m in svc.list_available_models():
        print("  -", m)

    try:
        print("\nNon-stream test:")
        resp = svc.create_completion(messages=[{"role": "user", "content": "Say hi in three words."}], max_tokens=8)
        # Normalize for display
        text = ""
        try:
            text = resp.choices[0].message.content  # type: ignore
        except Exception:
            text = str(resp)
        print("  ->", text)
    except Exception as e:
        print("Non-stream test failed:", e)

    try:
        print("\nStream test (one pass):")
        stream = svc.stream_chat(messages=[{"role": "user", "content": "Count to five."}], max_tokens=16, temperature=0.0)
        # Iterate a few chunks for demo
        for i, ch in enumerate(stream):
            piece = ""
            if isinstance(ch, dict):
                piece = ch.get("content", "")
            else:
                # OpenAI-like: ch.choices[0].delta.content
                try:
                    piece = ch.choices[0].delta.content  # type: ignore
                except Exception:
                    piece = str(ch)
            if piece:
                print(piece, end="", flush=True)
            if i > 20:
                break
        print("\n(done)")
    except Exception as e:
        print("Stream test failed:", e)
