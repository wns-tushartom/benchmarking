from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

app = FastAPI(title="WNS VM Embedding Service", version="1.0.0")

JINA_MODEL_NAME = os.getenv("JINA_EMBEDDING_MODEL", "jinaai/jina-embeddings-v3")
GTE_MODEL_NAME = os.getenv("GTE_EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")
DEVICE_ENV = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
OPTIONAL_API_KEY = os.getenv("WNS_EMBEDDING_API_KEY", "").strip()

models: Dict[str, SentenceTransformer] = {}
model_load_seconds: Dict[str, float] = {}


def choose_device() -> str:
    if DEVICE_ENV and DEVICE_ENV != "auto":
        return DEVICE_ENV
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = choose_device()


class EmbedRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    model: Optional[str] = None
    normalize: bool = True


class EmbedResponse(BaseModel):
    model: str
    dimensions: int
    count: int
    device: str
    latency_ms: float
    embeddings: List[List[float]]


def require_auth(authorization: Optional[str]) -> None:
    if not OPTIONAL_API_KEY:
        return
    if authorization != f"Bearer {OPTIONAL_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def get_model(key: str) -> SentenceTransformer:
    if key in models:
        return models[key]
    started = time.perf_counter()
    if key == "jina":
        model = SentenceTransformer(JINA_MODEL_NAME, trust_remote_code=True, device=DEVICE)
    elif key == "gte":
        model = SentenceTransformer(GTE_MODEL_NAME, trust_remote_code=True, device=DEVICE)
    else:
        raise HTTPException(status_code=404, detail=f"Unknown model key: {key}")
    models[key] = model
    model_load_seconds[key] = time.perf_counter() - started
    return model


def embed(key: str, req: EmbedRequest, authorization: Optional[str]) -> EmbedResponse:
    require_auth(authorization)
    model = get_model(key)
    started = time.perf_counter()
    vectors = model.encode(req.texts, batch_size=BATCH_SIZE, normalize_embeddings=req.normalize, convert_to_numpy=True, show_progress_bar=False)
    vectors = np.asarray(vectors)
    return EmbedResponse(
        model=JINA_MODEL_NAME if key == "jina" else GTE_MODEL_NAME,
        dimensions=int(vectors.shape[1]),
        count=int(vectors.shape[0]),
        device=DEVICE,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        embeddings=vectors.astype(float).tolist(),
    )


@app.get("/health")
def health() -> dict:
    return {"ok": True, "device": DEVICE, "loaded_models": list(models.keys()), "model_load_seconds": model_load_seconds, "jina_model": JINA_MODEL_NAME, "gte_model": GTE_MODEL_NAME}


@app.post("/embed/jina", response_model=EmbedResponse)
def embed_jina(req: EmbedRequest, authorization: Optional[str] = Header(default=None)) -> EmbedResponse:
    return embed("jina", req, authorization)


@app.post("/embed/gte", response_model=EmbedResponse)
def embed_gte(req: EmbedRequest, authorization: Optional[str] = Header(default=None)) -> EmbedResponse:
    return embed("gte", req, authorization)
