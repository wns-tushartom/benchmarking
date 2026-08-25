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
QWEN_RERANK_MODEL = os.getenv("QWEN_RERANK_MODEL", "tomaarsen/Qwen3-Reranker-4B-seq-cls")
QWEN_RERANK_INSTRUCTION = os.getenv(
    "QWEN_RERANK_INSTRUCTION",
    "Given a WNS airline support query, retrieve relevant policy or process passages that answer the query.",
)
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


def format_qwen_query(query: str) -> str:
    """Format Qwen3 reranker input with its expected instruction fields."""
    prefix = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n"
        "<|im_start|>user\n"
    )
    return f"{prefix}<Instruct>: {QWEN_RERANK_INSTRUCTION}\n<Query>: {query}\n"


def format_qwen_document(document: str) -> str:
    suffix = "<|im_end|>\n<|im_start|>assistant\n\n\n"
    return f"<Document>: {document}{suffix}"


def score_float(value) -> float:
    try:
        return float(value)
    except Exception:
        try:
            return float(value[0])
        except Exception:
            return 0.0


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
        import torch
        model = CrossEncoder(
            QWEN_RERANK_MODEL,
            trust_remote_code=True,
            device=DEVICE,
            model_kwargs={"torch_dtype": torch.float16},
        )
        if getattr(model, "tokenizer", None) is not None and getattr(model.tokenizer, "pad_token", None) is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        if getattr(model, "model", None) is not None and getattr(model.model, "config", None) is not None:
            model.model.config.pad_token_id = getattr(model.tokenizer, "eos_token_id", None)
        return model
    raise ValueError(f"unknown reranker key: {key}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "models": {
            "jina": JINA_MODEL_NAME,
            "gte": GTE_MODEL_NAME,
            "bge_reranker": BGE_RERANK_MODEL,
            "qwen_reranker": QWEN_RERANK_MODEL,
        },
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
    if key == "qwen":
        if os.getenv("QWEN_RERANK_FORMATTED", "1") == "0":
            pairs = [(req.query, doc) for doc in docs]
        else:
            formatted_query = format_qwen_query(req.query)
            pairs = [(formatted_query, format_qwen_document(doc)) for doc in docs]
        raw_scores = model.predict(
            pairs,
            batch_size=int(os.getenv("QWEN_RERANK_BATCH_SIZE", "8")),
        )
    else:
        pairs = [(req.query, doc) for doc in docs]
        raw_scores = model.predict(pairs)
    scores = [score_float(s) for s in raw_scores]
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
