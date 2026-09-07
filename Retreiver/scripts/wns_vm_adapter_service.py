#!/usr/bin/env python3
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="WNS VM Model Adapter Service")


GTE_CANDIDATE_MODEL = "Alibaba-NLP/gte-multilingual-base"
NEMOTRON_CANDIDATE_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
GTE_MODERNBERT_CANDIDATE_MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"

PROFILE_ENDPOINTS = {
    "jina": ("/embed/jina",),
    "gte": ("/embed/gte",),
    "bge": ("/rerank/bge",),
    "qwen": ("/rerank/qwen",),
    "nemotron-rerank": ("/rerank/nemotron",),
    "gte-modernbert": ("/rerank/gte-modernbert",),
    "legacy-unified": (
        "/embed/jina",
        "/embed/gte",
        "/rerank/bge",
        "/rerank/qwen",
        "/rerank/nemotron",
        "/rerank/gte-modernbert",
    ),
}


def validated_profile(value: str) -> str:
    profile = value.strip()
    if profile not in PROFILE_ENDPOINTS:
        raise RuntimeError(
            "WNS_ADAPTER_PROFILE must be one of " + ", ".join(sorted(PROFILE_ENDPOINTS))
        )
    return profile


ACTIVE_PROFILE = validated_profile(os.getenv("WNS_ADAPTER_PROFILE", "legacy-unified"))


def require_profile(endpoint: str) -> None:
    if endpoint not in PROFILE_ENDPOINTS[ACTIVE_PROFILE]:
        raise HTTPException(
            status_code=404,
            detail=f"Endpoint is not enabled for adapter profile {ACTIVE_PROFILE!r}",
        )


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
RERANK_BATCH_SIZE = int(os.getenv("WNS_RERANK_BATCH_SIZE", "1"))
QWEN_RERANK_MAX_LENGTH = int(os.getenv("QWEN_RERANK_MAX_LENGTH", "4096"))
NEMOTRON_RERANK_MAX_LENGTH = int(os.getenv("NEMOTRON_RERANK_MAX_LENGTH", "4096"))
if RERANK_BATCH_SIZE <= 0:
    raise RuntimeError("WNS_RERANK_BATCH_SIZE must be positive")


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
        model = CrossEncoder(
            QWEN_RERANK_MODEL,
            trust_remote_code=True,
            device=DEVICE,
            max_length=QWEN_RERANK_MAX_LENGTH,
            automodel_args={"torch_dtype": "auto"},
        )
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
    scores: List[float] = []
    for start in range(0, len(documents), RERANK_BATCH_SIZE):
        batch = documents[start : start + RERANK_BATCH_SIZE]
        encoded = tokenizer(
            [format_nemotron_pair(query, document) for document in batch],
            padding=True,
            truncation=True,
            max_length=NEMOTRON_RERANK_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
        if len(logits.shape) > 1 and logits.shape[-1] > 1:
            logits = logits[:, -1]
        scores.extend(float(value) for value in logits.reshape(-1).detach().float().cpu().tolist())
    return scores


@app.get("/health")
def health():
    models = {
        "/embed/jina": ("jina", JINA_MODEL_NAME),
        "/embed/gte": ("gte", GTE_MODEL_NAME),
        "/rerank/bge": ("bge_reranker", BGE_RERANK_MODEL),
        "/rerank/qwen": ("qwen_reranker", QWEN_RERANK_MODEL),
        "/rerank/nemotron": ("nemotron_reranker", NEMOTRON_RERANK_MODEL),
        "/rerank/gte-modernbert": (
            "gte_modernbert_reranker",
            GTE_MODERNBERT_RERANK_MODEL,
        ),
    }
    endpoints = list(PROFILE_ENDPOINTS[ACTIVE_PROFILE])
    return {
        "ok": True,
        "profile": ACTIVE_PROFILE,
        "device": DEVICE,
        "models": {models[endpoint][0]: models[endpoint][1] for endpoint in endpoints},
        "endpoints": endpoints,
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
    require_profile("/embed/jina")
    return embed("jina", req)


@app.post("/embed/gte")
def embed_gte(req: EmbedRequest):
    require_profile("/embed/gte")
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
            raw_scores = model.predict(
                pairs,
                batch_size=RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )
        else:
            pairs = [(req.query, doc) for doc in docs]
            raw_scores = model.predict(
                pairs,
                batch_size=RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )
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
    require_profile("/rerank/bge")
    return rerank("bge", req)


@app.post("/rerank/qwen")
def rerank_qwen(req: RerankRequest):
    require_profile("/rerank/qwen")
    return rerank("qwen", req)


@app.post("/rerank/nemotron")
def rerank_nemotron(req: RerankRequest):
    require_profile("/rerank/nemotron")
    return rerank("nemotron", req)


@app.post("/rerank/gte-modernbert")
def rerank_gte_modernbert(req: RerankRequest):
    require_profile("/rerank/gte-modernbert")
    return rerank("gte-modernbert", req)
