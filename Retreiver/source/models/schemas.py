# type: ignore
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Annotated
import mimetypes

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# ---------- Type aliases ----------
PositiveInt = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]


# ---------- Enums ----------
class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class DocumentType(str, Enum):
    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"


# ---------- Request Models ----------
class DocumentUploadRequest(BaseModel):
    """Client request indicating a document to upload."""
    model_config = ConfigDict(extra="forbid")

    filename: str
    content_type: Optional[str] = None  # inferred if omitted
    document_type: Optional[DocumentType] = None  # hint only

    @field_validator("content_type", mode="before")
    @classmethod
    def infer_mime_if_missing(cls, v, info):
        if v:
            return v
        name = info.data.get("filename", "")
        guessed, _ = mimetypes.guess_type(name)
        if not guessed and str(name).lower().endswith(".pdf"):
            return "application/pdf"
        return guessed or "application/octet-stream"

    @field_validator("document_type", mode="before")
    @classmethod
    def infer_document_type(cls, v, info) -> Optional[DocumentType]:
        if v:
            return v
        name = str(info.data.get("filename", "")).lower()
        if name.endswith(".pdf"):
            return DocumentType.PDF
        if name.endswith(".md") or name.endswith(".markdown"):
            return DocumentType.MARKDOWN
        if name.endswith(".txt"):
            return DocumentType.TEXT
        return None


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    model: Optional[str] = Field(default=None, description="Embedding model; defaults from settings")


class EmbeddingBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: List[str]
    model: Optional[str] = Field(default=None, description="Embedding model; defaults from settings")

    @field_validator("texts")
    @classmethod
    def non_empty_texts(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("texts must contain at least one item.")
        return v


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    # Upper limit will be enforced in the router from search settings; keep only ge=1 here.
    top_k: Annotated[int, Field(default=5, ge=1)]
    collection_name: Optional[str] = None
    filter_metadata: Optional[Dict[str, Any]] = None
    score_threshold: Optional[UnitInterval] = None  # 0..1 for similarity

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query cannot be blank.")
        return v


class LLMRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str
    use_rag: bool = True
    model: Optional[str] = Field(default=None, description="Chat model; defaults from settings")
    # No fixed upper bound; router will clamp to settings.max_tokens_cap
    max_tokens: Annotated[int, Field(default=1000, ge=1)]
    temperature: Annotated[float, Field(default=0.7, ge=0.0, le=2.0)]
    # No fixed upper bound; router will clamp via settings/context
    context_k: Annotated[int, Field(default=5, ge=1)]

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message cannot be blank.")
        return v


# ---------- Response Models ----------
class DocumentUploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    status: DocumentStatus
    pages_processed: NonNegativeInt
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    file_size: Optional[NonNegativeInt] = None


class DocumentContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: PositiveInt
    content: str
    images: List[str] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    formulas: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    document_type: DocumentType
    total_pages: NonNegativeInt
    content: List[DocumentContent]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def check_pages(self) -> "ParsedDocument":
        if self.total_pages and self.content:
            max_page = max((c.page_number for c in self.content), default=0)
            if max_page > self.total_pages:
                raise ValueError(
                    f"content contains page_number={max_page} exceeding total_pages={self.total_pages}"
                )
        return self


class EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding: List[float]
    model: str
    dimension: PositiveInt
    text_length: NonNegativeInt

    @field_validator("embedding")
    @classmethod
    def embedding_matches_dimension(cls, v: List[float], info):
        dim = info.data.get("dimension")
        if dim is not None and len(v) != dim:
            raise ValueError(f"embedding length {len(v)} != dimension {dim}")
        return v


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    content: str
    score: float  # distance or similarity; see score_type
    metadata: Dict[str, Any] = Field(default_factory=dict)
    page_number: Optional[PositiveInt] = None
    score_type: Literal["similarity", "distance"] = "similarity"
    similarity_score: Optional[UnitInterval] = None

    @model_validator(mode="after")
    def derive_similarity_if_possible(self) -> "SearchResult":
        if self.similarity_score is None and self.score_type == "similarity":
            self.similarity_score = max(0.0, min(1.0, float(self.score)))
        return self


class SearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: List[SearchResult]
    total_results: NonNegativeInt
    query: str
    processing_time: NonNegativeFloat


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    response: str
    sources: List[SearchResult] = Field(default_factory=list)
    model: str
    tokens_used: NonNegativeInt
    processing_time: NonNegativeFloat
    context_used: bool


# ---------- Status and Health Models ----------
class ServiceStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service_name: str
    status: Literal["ok", "degraded", "down", "unknown"] | str
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "down", "unknown"] | str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    services: List[ServiceStatus]
    collections: List[str] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    error: str
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = None


# ---------- Collection Management ----------
class CollectionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dimension: PositiveInt
    metric_type: Literal["COSINE", "IP", "L2"] | str
    description: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_entities: NonNegativeInt


class CollectionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    dimension: PositiveInt = 1024
    metric_type: Literal["COSINE", "IP", "L2"] | str = "COSINE"
    description: Optional[str] = None


# ---------- Document Management ----------
class DocumentListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: List[Dict[str, Any]]
    total: NonNegativeInt
    page: PositiveInt
    page_size: PositiveInt


class DocumentDeleteResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    status: Literal["deleted", "not_found", "error"] | str
    message: str
    vectors_deleted: NonNegativeInt


# ---------- Statistics and Analytics ----------
class SystemStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_documents: NonNegativeInt
    total_embeddings: NonNegativeInt
    total_collections: NonNegativeInt
    storage_used: str
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    pages: NonNegativeInt
    embeddings_count: NonNegativeInt
    upload_date: datetime
    last_accessed: datetime
    access_count: NonNegativeInt
