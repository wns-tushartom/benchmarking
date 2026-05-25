# type: ignore
"""Embedding Service (provider-pluggable, config-driven)

- Removes hard-coding of NVIDIA specifics from the service surface
- Adds provider abstraction + factory (NVIDIA today; easy to extend)
- All knobs (model, base_url, batch, retries, chunking) read from config
- Safer fallbacks, clearer errors, and diagnostics
- Backward compatible public methods:
    * create_embedding
    * create_embeddings_batch
    * embed_text_chunks
    * chunk_and_embed_text
    * embed_markdown_directory
"""

from __future__ import annotations

import os
import re
import json
import glob
import time
import hashlib
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Protocol, runtime_checkable

# ------------------------------- Config -------------------------------
from source.config import (
    load_cloud_services,
    load_embedding_settings,  # NEW: use embedding_settings for tunables
)
cloud_services = load_cloud_services()
embed_cfg = load_embedding_settings() or {}

# Public response schema (keep compat with your code)
try:
    from source.models.schemas import EmbeddingResponse
except ImportError:
    @dataclass
    class EmbeddingResponse:
        embedding: List[float]
        model: str
        dimension: int
        text_length: int

# ------------------------------- Logging ------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

# ---------------------- Text processing utilities ---------------------
def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")

def split_md_sections(text: str) -> List[str]:
    text = normalize_newlines(text).strip()
    if not text:
        return []
    parts = re.split(r"(?m)^(?=#)", text)
    sections = [p.strip() for p in parts if p and p.strip()]
    return sections or [text]

def chunk_text(
    text: str,
    target_chars: int = 1500,
    overlap: int = 200,
    min_chunk_chars: int = 50
) -> List[str]:
    text = text.strip()
    if not text or len(text) < min_chunk_chars:
        return []
    if len(text) <= target_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current: List[str] = []
    length = 0

    for sent in sentences:
        s_len = len(sent)
        if length + s_len + 1 > target_chars and current:
            current_text = " ".join(current).strip()
            if len(current_text) >= min_chunk_chars:
                chunks.append(current_text)

            if overlap > 0 and current_text:
                tail = current_text[-overlap:]
                carry = []
                if ". " in tail:
                    parts = tail.split(". ")
                    for i in range(len(parts) - 1, -1, -1):
                        test = ". ".join(parts[i:])
                        if len(test) <= overlap:
                            carry = [test] if test.strip() else []
                            break
                current = carry
                length = len(" ".join(current))
            else:
                current, length = [], 0

        current.append(sent)
        length += s_len + (1 if current else 0)

    if current:
        final_text = " ".join(current).strip()
        if len(final_text) >= min_chunk_chars:
            chunks.append(final_text)

    final_chunks: List[str] = []
    for ch in chunks:
        if len(ch) <= target_chars * 2:
            final_chunks.append(ch)
        else:
            for i in range(0, len(ch), target_chars):
                part = ch[i:i + target_chars]
                if len(part) >= min_chunk_chars:
                    final_chunks.append(part)

    return [c for c in final_chunks if c.strip()]

def short_hash(*parts: str, length: int = 16) -> str:
    h = hashlib.sha256(("||".join(parts)).encode("utf-8")).hexdigest()
    return h[:length]

# ============================ Provider layer ============================
@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal interface every provider must implement."""
    model: str

    async def ensure_model(self, model: Optional[str]) -> None: ...
    async def embed_one(self, text: str) -> List[float]: ...
    async def embed_many(self, texts: List[str]) -> List[List[float]]: ...
    async def dimension(self) -> int: ...

# ------------------------ NVIDIA Provider (default) ---------------------
class NvidiaProvider:
    def __init__(self, *, model: str, api_key: str, base_url: str):
        # Lazy import so the service can still start without the package
        try:
            from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings  # type: ignore
        except ImportError as e:
            raise ImportError(
                "langchain-nvidia-ai-endpoints is required for NVIDIA provider. "
                "Install: pip install -U langchain-nvidia-ai-endpoints"
            ) from e

        if not api_key:
            raise ValueError("NVIDIA_API_KEY is required for NVIDIA embeddings")
        os.environ.setdefault("NVIDIA_API_KEY", api_key)

        self._NVIDIAEmbeddings = NVIDIAEmbeddings  # type: ignore
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = self._NVIDIAEmbeddings(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            truncate="END",
        )
        # IMPORTANT: do not cache any event loop here

    async def ensure_model(self, model: Optional[str]) -> None:
        if not model or model == self.model:
            return
        self.model = model
        self._client = self._NVIDIAEmbeddings(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            truncate="NONE",
        )
        logger.info("NVIDIA provider switched model -> %s", self.model)

    async def embed_one(self, text: str) -> List[float]:
        def _run(t: str) -> List[float]:
            return self._client.embed_query(t)
        # Use to_thread to bind to the current running loop safely
        vec = await asyncio.to_thread(_run, text)
        if not isinstance(vec, list) or not vec:
            raise ValueError("Invalid embedding response from NVIDIA")
        return vec

    async def embed_many(self, texts: List[str]) -> List[List[float]]:
        def _run(ts: List[str]) -> List[List[float]]:
            return self._client.embed_documents(ts)
        vecs = await asyncio.to_thread(_run, texts)
        if (not isinstance(vecs, list)) or (len(vecs) != len(texts)):
            raise ValueError(f"NVIDIA batch mismatch: expected {len(texts)} vectors")
        if any((not isinstance(v, list) or not v) for v in vecs):
            raise ValueError("Invalid vector in NVIDIA batch response")
        return vecs

    async def dimension(self) -> int:
        v = await self.embed_one("dimension probe")
        return len(v)

# --------------------------- Provider factory ---------------------------
def make_provider() -> EmbeddingProvider:
    """
    Decide provider from config with safe defaults.
    cloud_services.embedding_provider or embedding_settings.provider
    """
    provider_name = (
        embed_cfg.get("provider")
        or cloud_services.get("embedding_provider")
        or "nvidia"
    ).lower()

    if provider_name == "nvidia":
        model = (
            embed_cfg.get("model")
            or cloud_services.get("nvidia_embedding_model")
            or "nvidia/nv-embedqa-e5-v5"
        )
        base_url = (
            embed_cfg.get("base_url")
            or cloud_services.get("nvidia_base_url")
            or "https://integrate.api.nvidia.com/v1"
        )
        key = (
            embed_cfg.get("api_key")
            or cloud_services.get("nvidia_api_key")
            or os.getenv("NVIDIA_API_KEY", "")
        )
        return NvidiaProvider(model=model, api_key=key, base_url=base_url)

    if provider_name == "openai":
        model = (
            embed_cfg.get("model")
            or cloud_services.get("openai_embedding_model")
            or "text-embedding-3-small"
        )
        key = (
            embed_cfg.get("api_key")
            or cloud_services.get("openai_api_key")
            or os.getenv("OPENAI_API_KEY", "")
        )
        return OpenAIProvider(model=model, api_key=key)

    # Placeholders for future providers
    # elif provider_name == "hf":
    #     return HuggingFaceProvider(...)

    raise ValueError(f"Unknown embedding provider: {provider_name}")

# ========================= Embedding Service (API) =========================
class EmbeddingService:
    """
    Provider-agnostic embedding service.
    All tuning parameters come from embedding_settings with safe defaults.
    """

    def __init__(self):
        # Tunables (with fallbacks)
        self.batch_size_default = int(embed_cfg.get("batch_size", 10))
        self.max_retries = int(embed_cfg.get("max_retries", 3))
        self.retry_delay = float(embed_cfg.get("retry_delay_sec", 1.0))
        self.chunk_target_chars = int(embed_cfg.get("chunk_target_chars", 1500))
        self.chunk_overlap = int(embed_cfg.get("chunk_overlap_chars", 200))
        self.min_chunk_chars = int(embed_cfg.get("min_chunk_chars", 50))
        self.min_embed_len = int(embed_cfg.get("min_embed_length", 10))

        # Provider
        self._provider: EmbeddingProvider = make_provider()
        logger.info("EmbeddingService online (provider=%s, model=%s)", type(self._provider).__name__, self._provider.model)

    # ---------------- Core single/batch APIs ----------------
    @property
    def model(self) -> str:
        return self._provider.model

    async def create_embedding(self, text: str, model: Optional[str] = None) -> EmbeddingResponse:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        await self._provider.ensure_model(model)
        try:
            vec = await self._provider.embed_one(text.strip())
            return EmbeddingResponse(
                embedding=vec,
                model=self.model,
                dimension=len(vec),
                text_length=len(text.strip()),
            )
        except Exception as e:
            # provider-specific fallback (example: NVIDIA 404 -> fallback model)
            if "404" in str(e).lower() and "not found" in str(e).lower():
                fallback = embed_cfg.get("fallback_model") or cloud_services.get("nvidia_fallback_embedding_model")
                if fallback:
                    logger.warning("Model '%s' not found. Falling back to '%s'", self.model, fallback)
                    await self._provider.ensure_model(fallback)
                    vec = await self._provider.embed_one(text.strip())
                    return EmbeddingResponse(embedding=vec, model=self.model, dimension=len(vec), text_length=len(text.strip()))
            raise

    async def create_embeddings_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None
    ) -> List[EmbeddingResponse]:
        if not texts:
            return []

        valid_texts: List[str] = []
        for t in texts:
            t = (t or "").strip()
            if t:
                valid_texts.append(t)

        if not valid_texts:
            return []

        await self._provider.ensure_model(model)

        B = int(batch_size or self.batch_size_default)
        retries = int(max_retries or self.max_retries)
        delay = float(retry_delay or self.retry_delay)

        start = time.time()
        all_vecs: List[List[float]] = []

        for i in range(0, len(valid_texts), B):
            batch = valid_texts[i:i + B]
            for attempt in range(retries):
                try:
                    vecs = await self._provider.embed_many(batch)
                    all_vecs.extend(vecs)
                    break
                except Exception as e:
                    if attempt == retries - 1:
                        logger.error("Batch %d failed after %d attempts: %s", (i // B) + 1, retries, e)
                        raise
                    logger.warning("Batch %d attempt %d failed; retrying in %.1fs: %s", (i // B) + 1, attempt + 1, delay, e)
                    await asyncio.sleep(delay)
            # small pacing
            if i + B < len(valid_texts):
                await asyncio.sleep(0.05)

        responses = [
            EmbeddingResponse(embedding=v, model=self.model, dimension=len(v), text_length=len(t))
            for t, v in zip(valid_texts, all_vecs)
        ]
        dur = time.time() - start
        if dur > 0:
            logger.info("Embedded %d texts in %.2fs (%.1f/s)", len(responses), dur, len(responses) / dur)
        return responses

    # --------------- Convenience helpers (same signatures) ---------------
    async def embed_text_chunks(
        self,
        text_chunks: List[str],
        batch_size: Optional[int] = None,
        filter_short: bool = True,
        min_length: Optional[int] = None
    ) -> List[EmbeddingResponse]:
        if not text_chunks:
            return []
        if filter_short:
            min_len = int(min_length or self.min_embed_len)
            filtered = [c for c in text_chunks if len((c or "").strip()) >= min_len]
            if len(filtered) != len(text_chunks):
                logger.info("Filtered %d short chunks (< %d chars)", len(text_chunks) - len(filtered), min_len)
            text_chunks = filtered
        if not text_chunks:
            return []
        return await self.create_embeddings_batch(text_chunks, batch_size=batch_size)

    async def chunk_and_embed_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        batch_size: Optional[int] = None,
        min_chunk_size: Optional[int] = None
    ) -> Tuple[List[str], List[EmbeddingResponse]]:
        if not (text and text.strip()):
            return [], []
        chunks = chunk_text(
            text=text.strip(),
            target_chars=int(chunk_size or self.chunk_target_chars),
            overlap=int(overlap or self.chunk_overlap),
            min_chunk_chars=int(min_chunk_size or self.min_chunk_chars),
        )
        if not chunks:
            return [], []
        vecs = await self.embed_text_chunks(chunks, batch_size=batch_size, filter_short=True, min_length=int(min_chunk_size or self.min_chunk_chars))
        return chunks, vecs

    # --------------------------- Diagnostics ---------------------------
    async def get_embedding_dimension(self) -> int:
        try:
            return await self._provider.dimension()
        except Exception as e:
            logger.warning("Dimension probe failed: %s", e)
            return 1024  # conservative default

    async def calculate_similarity(self, v1: List[float], v2: List[float]) -> float:
        try:
            import numpy as np  # type: ignore
            a = np.array(v1)
            b = np.array(v2)
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            return 0.0 if denom == 0 else float(np.dot(a, b) / denom)
        except Exception as e:
            logger.error("Failed similarity calc: %s", e)
            return 0.0



    # ---------------- Markdown directory pipeline ----------------
    def list_markdown_files(self, root: str | Path) -> List[Path]:
        root = Path(root)
        return [Path(p) for p in glob.glob(str(root / "**" / "*.md"), recursive=True)]

    def read_text_file(self, path: Path, encoding: str = "utf-8") -> str:
        try:
            with open(path, "r", encoding=encoding, errors="ignore") as f:
                return f.read()
        except Exception as e:
            logger.error("Failed to read %s: %s", path, e)
            return ""

    async def embed_markdown_directory(
        self,
        root_dir: str | Path,
        chunk_chars: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        min_section_chars: Optional[int] = None,
        batch_size: Optional[int] = None,
        save_jsonl_path: Optional[str | Path] = None,
    ) -> List[Dict[str, Any]]:
        md_files = self.list_markdown_files(root_dir)
        if not md_files:
            logger.warning("No Markdown files under %s", root_dir)
            return []

        target = int(chunk_chars or self.chunk_target_chars)
        overlap = int(chunk_overlap or self.chunk_overlap)
        min_chars = int(min_section_chars or self.min_chunk_chars)

        all_chunks: List[str] = []
        meta_rows: List[Tuple[str, int, str]] = []

        for fp in md_files:
            content = self.read_text_file(fp)
            if not content:
                continue
            sections = [s for s in split_md_sections(content) if len(s) >= min_chars]
            if not sections:
                continue
            idx = 0
            for sec in sections:
                for ch in chunk_text(sec, target_chars=target, overlap=overlap, min_chunk_chars=min_chars):
                    if len(ch) >= min_chars:
                        all_chunks.append(ch)
                        meta_rows.append((str(fp), idx, ch))
                        idx += 1

        if not all_chunks:
            logger.warning("No chunks produced from markdown files.")
            return []

        responses = await self.create_embeddings_batch(all_chunks, batch_size=batch_size)
        if not responses:
            logger.error("Embedding batch returned no vectors")
            return []

        dim = responses[0].dimension
        out: List[Dict[str, Any]] = []
        for (fp, idx, ch), r in zip(meta_rows, responses):
            out.append({
                "id": short_hash(fp, str(idx), ch[:64]),
                "path": fp,
                "chunk_index": idx,
                "text": ch,
                "embedding": r.embedding,
                "dimension": dim,
                "text_length": len(ch),
                "model": self.model,
                "created_at": time.time(),
            })

        if save_jsonl_path:
            p = Path(save_jsonl_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                for row in out:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            logger.info("Saved embeddings JSONL -> %s", p)

        logger.info("Embedded %d chunks from %d files (dim=%d)", len(out), len(md_files), dim)
        return out

# ----------------------------- Global instance -----------------------------
embedding_service = EmbeddingService()

# ---------------------- Singleton accessor ----------------------
_embedding_service_instance: Optional[EmbeddingService] = None

def get_embeddings() -> EmbeddingService:
    """Get or create the embedding service singleton instance"""
    global _embedding_service_instance
    if _embedding_service_instance is None:
        _embedding_service_instance = EmbeddingService()
    return _embedding_service_instance

# ---------------------------------- CLI ----------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Service CLI (provider-pluggable)")

    parser.add_argument("--root", default="data/parsed_output", help="Root folder with .md files")
    parser.add_argument("--out", default="data/embeddings/markdown_embeddings.jsonl", help="Output JSONL path")
    parser.add_argument("--chunk", type=int, default=None, help="Target chars per chunk (override)")
    parser.add_argument("--overlap", type=int, default=None, help="Overlap chars (override)")
    parser.add_argument("--min-sec", type=int, default=None, help="Min section/chunk chars (override)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (override)")
    parser.add_argument("--model", default=None, help="Force model for this run")
    args = parser.parse_args()

    async def _run():
        if args.model:
            await embedding_service._provider.ensure_model(args.model)
        await embedding_service.embed_markdown_directory(
            root_dir=args.root,
            chunk_chars=args.chunk,
            chunk_overlap=args.overlap,
            min_section_chars=args.min_sec,
            batch_size=args.batch,
            save_jsonl_path=args.out,
        )

    asyncio.run(_run())
