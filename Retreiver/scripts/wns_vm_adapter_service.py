#!/usr/bin/env python3
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="WNS VM Model Adapter Service")

JINA_MODEL_NAME = os.getenv("JINA_EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")
GTE_MODEL_NAME = os.getenv("GTE_EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")
BGE_RERANK_MODEL = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-base")
QWEN_RERANK_MODEL = os.getenv("QWEN_RERANK_MODEL", "Qwen/Qwen3-Reranker-4B")
DEVICE = os.getenv("WNS_MODEL_DEVICE", os.getenv("EMBEDDING_DEVICE", "cuda"))


class EmbedRequest(BaseModel):
    model: str | None = None
    input: List[str] | str | None = None
    inputs: List[str] | str | None = None
    texts: List[str] | str | None = None


class RerankRequest(BaseModel):
    model: str | None = None
    query: str
    documents: List[str] | None = None
    texts: List[str] | None = None
    passages: List[str] | None = None
    top_k: int | None = None


def _list(value: List[str] | str | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


@lru_cache(maxsize=8)
def embedding_model(key: str):
    from sentence_transformers import SentenceTransformer

    if key == "jina":
        return SentenceTransformer(JINA_MODEL_NAME, trust_remote_code=True, device=DEVICE)
    if key == "gte":
        return SentenceTransformer(GTE_MODEL_NAME, trust_remote_code=True, device=DEVICE)
    raise ValueError(f"unknown embedding key: {key}")


@lru_cache(maxsize=8)
def cross_encoder_model(key: str):
    from sentence_transformers import CrossEncoder

    if key == "bge":
        return CrossEncoder(BGE_RERANK_MODEL, trust_remote_code=True, device=DEVICE)
    if key == "qwen":
        return CrossEncoder(QWEN_RERANK_MODEL, trust_remote_code=True, device=DEVICE)
    raise ValueError(f"unknown reranker key: {key}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "endpoints": ["/embed/jina", "/embed/gte", "/rerank/bge", "/rerank/qwen"],
    }


def embed(key: str, req: EmbedRequest):
    texts = _list(req.texts) or _list(req.inputs) or _list(req.input)
    if not texts:
        raise ValueError("embedding request needs texts, inputs, or input")
    model = embedding_model(key)
    vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
    embeddings = vectors.tolist()
    return {
        "model": JINA_MODEL_NAME if key == "jina" else GTE_MODEL_NAME,
        "dimensions": int(vectors.shape[1]),
        "embeddings": embeddings,
        "data": [{"index": i, "embedding": vec} for i, vec in enumerate(embeddings)],
    }


@app.post("/embed/jina")
def embed_jina(req: EmbedRequest):
    return embed("jina", req)


@app.post("/embed/gte")
def embed_gte(req: EmbedRequest):
    return embed("gte", req)


def rerank(key: str, req: RerankRequest):
    docs = req.documents or req.texts or req.passages or []
    if not docs:
        raise ValueError("rerank request needs documents, texts, or passages")
    model = cross_encoder_model(key)
    pairs = [(req.query, doc) for doc in docs]
    raw_scores = model.predict(pairs)
    scores = [float(s) for s in raw_scores]
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    if req.top_k:
        ranked = ranked[: req.top_k]
    return {
        "model": BGE_RERANK_MODEL if key == "bge" else QWEN_RERANK_MODEL,
        "scores": scores,
        "results": [{"index": i, "score": score, "document": docs[i]} for i, score in ranked],
    }


@app.post("/rerank/bge")
def rerank_bge(req: RerankRequest):
    return rerank("bge", req)


@app.post("/rerank/qwen")
def rerank_qwen(req: RerankRequest):
    return rerank("qwen", req)
