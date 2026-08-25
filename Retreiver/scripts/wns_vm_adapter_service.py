#!/usr/bin/env python3
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="WNS VM Model Adapter Service")


GTE_CANDIDATE_MODEL = "Alibaba-NLP/gte-multilingual-base"
NEMOTRON_CANDIDATE_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
GTE_MODERNBERT_CANDIDATE_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"


def candidate_model_env(name: str, expected: str) -> str:
    configured = os.getenv(name, expected).strip()
    if configured != expected:
        raise RuntimeError(
            f"{name} must be {expected!r} for the isolated candidate lane; got {configured!r}"
        )
    return configured


JINA_MODEL_NAME = os.getenv("JINA_EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")
GTE_MODEL_NAME = candidate_model_env("GTE_EMBEDDING_MODEL", GTE_CANDIDATE_MODEL)
BGE_RERANK_MODEL = os.getenv("BGE_RERANKER_MODEL", "BAAI/bge-reranker-base")
QWEN_RERANK_MODEL = os.getenv("QWEN_RERANK_MODEL", "tomaarsen/Qwen3-Reranker-4B-seq-cls")
NEMOTRON_RERANK_MODEL = candidate_model_env("NEMOTRON_RERANK_MODEL", NEMOTRON_CANDIDATE_MODEL)
GTE_MODERNBERT_RERANK_MODEL = candidate_model_env(
    "GTE_MODERNBERT_RERANK_MODEL", GTE_MODERNBERT_CANDIDATE_MODEL
)
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


def format_nemotron_pair(query: str, document: str) -> str:
    """Use NVIDIA's documented single-sequence prompt for Nemotron Rerank."""
    return f"question:{query} \n \n passage:{document}"


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
        model = CrossEncoder(QWEN_RERANK_MODEL, trust_remote_code=True, device=DEVICE)
        if getattr(model, "tokenizer", None) is not None and getattr(model.tokenizer, "pad_token", None) is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        if getattr(model, "model", None) is not None and getattr(model.model, "config", None) is not None:
            model.model.config.pad_token_id = getattr(model.tokenizer, "eos_token_id", None)
        return model
    if key == "gte-modernbert":
        return CrossEncoder(
            GTE_MODERNBERT_RERANK_MODEL,
            trust_remote_code=True,
            device=DEVICE,
            max_length=8192,
            automodel_args={"torch_dtype": "auto"},
        )
    raise ValueError(f"unknown reranker key: {key}")


@lru_cache(maxsize=1)
def nemotron_model():
    """Load Nemotron through its documented transformers sequence-classifier path."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(NEMOTRON_RERANK_MODEL, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        NEMOTRON_RERANK_MODEL,
        trust_remote_code=True,
        torch_dtype="auto",
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def nemotron_scores(query: str, documents: List[str]) -> List[float]:
    import torch

    tokenizer, model = nemotron_model()
    encoded = tokenizer(
        [format_nemotron_pair(query, document) for document in documents],
        padding=True,
        truncation=True,
        max_length=8192,
        return_tensors="pt",
    )
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits
    if len(logits.shape) > 1 and logits.shape[-1] > 1:
        logits = logits[:, -1]
    return [float(value) for value in logits.reshape(-1).detach().float().cpu().tolist()]


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
            "nemotron_reranker": NEMOTRON_RERANK_MODEL,
            "gte_modernbert_reranker": GTE_MODERNBERT_RERANK_MODEL,
        },
        "endpoints": [
            "/embed/jina",
            "/embed/gte",
            "/rerank/bge",
            "/rerank/qwen",
            "/rerank/nemotron",
            "/rerank/gte-modernbert",
        ],
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
    if key == "nemotron":
        raw_scores = nemotron_scores(req.query, docs)
    else:
        model = cross_encoder_model(key)
        if key == "qwen":
            if os.getenv("QWEN_RERANK_FORMATTED", "1") == "0":
                pairs = [(req.query, doc) for doc in docs]
            else:
                formatted_query = format_qwen_query(req.query)
                pairs = [(formatted_query, format_qwen_document(doc)) for doc in docs]
            raw_scores = model.predict(pairs)
        else:
            pairs = [(req.query, doc) for doc in docs]
            raw_scores = model.predict(pairs)
    scores = [score_float(s) for s in raw_scores]
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    if req.top_k:
        ranked = ranked[: req.top_k]
    return {
        "model": {
            "bge": BGE_RERANK_MODEL,
            "qwen": QWEN_RERANK_MODEL,
            "nemotron": NEMOTRON_RERANK_MODEL,
            "gte-modernbert": GTE_MODERNBERT_RERANK_MODEL,
        }[key],
        "scores": scores,
        "results": [{"index": i, "score": score, "document": docs[i]} for i, score in ranked],
    }


@app.post("/rerank/bge")
def rerank_bge(req: RerankRequest):
    return rerank("bge", req)


@app.post("/rerank/qwen")
def rerank_qwen(req: RerankRequest):
    return rerank("qwen", req)


@app.post("/rerank/nemotron")
def rerank_nemotron(req: RerankRequest):
    return rerank("nemotron", req)


@app.post("/rerank/gte-modernbert")
def rerank_gte_modernbert(req: RerankRequest):
    return rerank("gte-modernbert", req)
