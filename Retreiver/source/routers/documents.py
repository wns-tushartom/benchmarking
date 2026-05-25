# source/routers/documents.py
# type: ignore
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile

# === Settings / Schemas ===
from source.core.document_settings import get_document_settings
from source.models.schemas import (
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentStats,
    DocumentStatus,
    DocumentUploadResponse,
    ParsedDocument,
)

# === Services ===
try:
    from source.services.embedding_service import get_embeddings
    _use_new_embeddings = True
except Exception:
    from source.services.embedding_service import embedding_service
    _use_new_embeddings = False

try:
    from source.services.vector_service import get_vector
    _use_new_vector = True
except Exception:
    from source.services.vector_service import get_vector_service
    _use_new_vector = False

from source.services.document_parser import document_parser

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

router = APIRouter(prefix="/documents", tags=["documents"])
S = get_document_settings()

# In-memory registry with persistence
document_registry: Dict[str, Dict[str, object]] = {}
_registry_lock = asyncio.Lock()

# Registry persistence path
REGISTRY_FILE = Path(S.upload_tmp_dir) / "document_registry.json"


async def _save_registry_to_disk():
    """Save the document registry to disk"""
    try:
        async with _registry_lock:
            # Convert registry to serializable format
            serializable_registry = {}
            for doc_id, info in document_registry.items():
                serializable_info = {}
                for k, v in info.items():
                    if isinstance(v, datetime):
                        serializable_info[k] = v.isoformat()
                    elif isinstance(v, Enum):
                        serializable_info[k] = v.value
                    else:
                        serializable_info[k] = v
                serializable_registry[doc_id] = serializable_info
            
            # Save to disk
            async with aiofiles.open(REGISTRY_FILE, "w") as f:
                await f.write(json.dumps(serializable_registry, indent=2))
            logger.info(f"Document registry saved to {REGISTRY_FILE}")
    except Exception:
        logger.exception(f"Failed to save document registry to {REGISTRY_FILE}")


async def _load_registry_from_disk():
    """Load the document registry from disk"""
    try:
        if not REGISTRY_FILE.exists():
            logger.info(f"Document registry file not found at {REGISTRY_FILE}")
            return
        
        async with aiofiles.open(REGISTRY_FILE, "r") as f:
            content = await f.read()
            loaded_registry = json.loads(content)
        
        async with _registry_lock:
            for doc_id, info in loaded_registry.items():
                # Convert serialized values back to their original types
                if "created_at" in info and isinstance(info["created_at"], str):
                    info["created_at"] = datetime.fromisoformat(info["created_at"])
                if "last_accessed" in info and isinstance(info["last_accessed"], str):
                    info["last_accessed"] = datetime.fromisoformat(info["last_accessed"])
                if "status" in info and isinstance(info["status"], str):
                    info["status"] = DocumentStatus(info["status"])
                document_registry[doc_id] = info
        
        logger.info(f"Loaded {len(loaded_registry)} documents from registry file")
    except Exception:
        logger.exception(f"Failed to load document registry from {REGISTRY_FILE}")


# Periodic registry saving task
async def _periodic_registry_save():
    """Periodically save the document registry to disk"""
    while True:
        try:
            # Save every 5 minutes
            await asyncio.sleep(300)
            await _save_registry_to_disk()
        except asyncio.CancelledError:
            # Save one last time before exiting
            await _save_registry_to_disk()
            break
        except Exception:
            logger.exception("Error in periodic registry save")
            # Continue the loop even if there was an error
            await asyncio.sleep(60)  # Wait a bit before retrying

# Load registry on module import and start periodic save task
asyncio.create_task(_load_registry_from_disk())
asyncio.create_task(_periodic_registry_save())


# ---------- helpers ----------
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _http_400(detail: str): return HTTPException(status_code=400, detail=detail)
def _http_404(detail: str): return HTTPException(status_code=404, detail=detail)
def _http_413(detail: str): return HTTPException(status_code=413, detail=detail)
def _http_500(detail: str = "Internal server error"): return HTTPException(status_code=500, detail=detail)


async def _stream_save_upload(file: UploadFile, dst_path: Path) -> int:
    total = 0
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = getattr(S, "upload_chunk_size", 1024 * 1024)

    async with aiofiles.open(dst_path, "wb") as out:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > S.max_file_size:
                raise _http_413(f"File too large: {total} bytes (max: {S.max_file_size} bytes)")
            await out.write(chunk)
    return total


def _pick_parser_method(file_ext: str):
    method_name = S.parser_map.get(file_ext)
    if not method_name:
        raise _http_400(f"Unsupported file type: {file_ext}. Supported: {sorted(set(S.parser_map.keys()))}")
    method = getattr(document_parser, method_name, None)
    if not callable(method):
        logger.error("Configured parser method '%s' missing on document_parser", method_name)
        raise _http_500("Parser configuration error")
    return method


def _mk_tmp_path(file_ext: str) -> Path:
    tmp_name = f"upload_{os.getpid()}_{int(datetime.now().timestamp()*1e6)}_{os.urandom(4).hex()}{file_ext}"
    return Path(S.upload_tmp_dir) / tmp_name


async def _register_parsed_doc(parsed_doc: ParsedDocument, orig_name: str, total_bytes: int) -> None:
    async with _registry_lock:
        document_registry[parsed_doc.document_id] = {
            "filename": orig_name,
            "document_id": parsed_doc.document_id,
            "status": DocumentStatus.PROCESSING,
            "created_at": _utcnow(),
            "pages_processed": parsed_doc.total_pages,
            "file_size": total_bytes,
            "embeddings_count": 0,
            "last_accessed": _utcnow(),
            "access_count": 0,
        }
    
    # Save registry to disk after modification
    await _save_registry_to_disk()


async def _save_parse_schedule(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    collection_name: Optional[str] = None,
) -> Tuple[Optional[DocumentUploadResponse], Optional[Dict[str, str]]]:
    if not file.filename or not file.filename.strip():
        return None, {"file": "(unnamed)", "reason": "missing_filename"}

    orig_name = Path(file.filename).name
    file_ext = Path(orig_name).suffix.lower()

    if file_ext not in S.parser_map:
        return None, {"file": orig_name, "reason": f"unsupported_type:{file_ext}"}

    tmp_path = _mk_tmp_path(file_ext)
    try:
        total_bytes = await _stream_save_upload(file, tmp_path)
        parse_method = _pick_parser_method(file_ext)
        parsed_doc: ParsedDocument = await parse_method(str(tmp_path), orig_name)

        await _register_parsed_doc(parsed_doc, orig_name, total_bytes)
        background_tasks.add_task(process_document_embeddings, parsed_doc, collection_name)

        resp = DocumentUploadResponse(
            document_id=parsed_doc.document_id,
            filename=orig_name,
            status=DocumentStatus.PROCESSING,
            pages_processed=parsed_doc.total_pages,
            created_at=_utcnow(),
            file_size=total_bytes,
        )
        return resp, None
    except HTTPException as e:
        return None, {"file": orig_name, "reason": str(e.detail)}
    except Exception as e:
        logger.exception("Failed to process document '%s'", orig_name)
        return None, {"file": orig_name, "reason": f"processing_failed:{e}"}
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            logger.warning("Temp file cleanup failed for %s", tmp_path, exc_info=True)


# ---------- routes ----------
@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: Optional[str] = Query(None, description="Target collection for embeddings"),
):
    if not file.filename or not file.filename.strip():
        raise _http_400("No filename provided")

    orig_name = Path(file.filename).name
    file_ext = Path(orig_name).suffix.lower()

    if file_ext not in S.parser_map:
        raise _http_400(f"Unsupported file type: {file_ext}. Supported: {sorted(set(S.parser_map.keys()))}")

    tmp_path = _mk_tmp_path(file_ext)

    try:
        total_bytes = await _stream_save_upload(file, tmp_path)
        logger.info("Processing uploaded file '%s' (%d bytes)", orig_name, total_bytes)

        parse_method = _pick_parser_method(file_ext)
        parsed_doc: ParsedDocument = await parse_method(str(tmp_path), orig_name)

        await _register_parsed_doc(parsed_doc, orig_name, total_bytes)

        background_tasks.add_task(process_document_embeddings, parsed_doc, collection)

        return DocumentUploadResponse(
            document_id=parsed_doc.document_id,
            filename=orig_name,
            status=DocumentStatus.PROCESSING,
            pages_processed=parsed_doc.total_pages,
            created_at=_utcnow(),
            file_size=total_bytes,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to process document '%s'", orig_name)
        raise _http_500("Failed to process document")
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            logger.warning("Temp file cleanup failed for %s", tmp_path, exc_info=True)


@router.post("/upload/batch")
async def upload_documents_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Upload multiple documents"),
    collection: Optional[str] = Query(None, description="Target collection for embeddings"),
):
    if not files:
        raise _http_400("No files provided")

    max_concurrency = getattr(S, "batch_upload_concurrency", 4)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _guarded_process(f: UploadFile):
        async with semaphore:
            return await _save_parse_schedule(f, background_tasks, collection)

    results = await asyncio.gather(*[_guarded_process(f) for f in files])

    successes: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    for resp, err in results:
        if resp:
            successes.append(
                {
                    "document_id": resp.document_id,
                    "filename": resp.filename,
                    "status": resp.status,
                    "pages_processed": resp.pages_processed,
                    "created_at": resp.created_at,  # datetime OK; FastAPI will serialize
                    "file_size": resp.file_size,
                }
            )
        elif err:
            failures.append(err)

    # ✅ Return a dict; FastAPI encodes datetimes/enums
    return {
        "accepted": len(successes),
        "rejected": len(failures),
        "results": successes,
        "errors": failures,
    }


async def process_document_embeddings(parsed_doc: ParsedDocument, collection_name: Optional[str] = None):
    try:
        logger.info(
            "Embedding processing start: doc_id=%s collection=%s",
            parsed_doc.document_id, collection_name or "(default)"
        )

        texts: List[str] = []
        metadata: List[Dict[str, object]] = []

        for page in parsed_doc.content:
            text = (page.content or "").strip()
            if not text:
                continue
            texts.append(text)
            metadata.append(
                {
                    "page_number": page.page_number,
                    "filename": parsed_doc.filename,
                    "document_type": parsed_doc.document_type.value,
                    "has_images": bool(page.images),
                    "has_tables": bool(page.tables),
                    "has_formulas": bool(page.formulas),
                    "text_length": len(text),
                    "created_at": parsed_doc.created_at.isoformat(),
                }
            )

        if not texts:
            logger.warning("No text content found for document %s", parsed_doc.document_id)
            async with _registry_lock:
                if parsed_doc.document_id in document_registry:
                    document_registry[parsed_doc.document_id]["status"] = DocumentStatus.FAILED
            return

        if _use_new_embeddings:
            emb = get_embeddings()
            create_batch = getattr(emb, "create_embeddings_batch", None)
            if asyncio.iscoroutinefunction(create_batch):
                embedding_responses = await create_batch(texts)
            elif callable(create_batch):
                embedding_responses = create_batch(texts)
            else:
                embed_texts = getattr(emb, "embed_texts", None)
                if not callable(embed_texts):
                    raise RuntimeError("Embeddings client missing batch methods")
                vectors = embed_texts(texts)
                embedding_responses = [{"embedding": v} for v in vectors]
        else:
            embedding_responses = await embedding_service.create_embeddings_batch(texts)

        embeddings = [resp["embedding"] if isinstance(resp, dict) else resp.embedding for resp in embedding_responses]

        vs = get_vector() if _use_new_vector else get_vector_service()

        await vs.store_document_embeddings(
            document_id=parsed_doc.document_id,
            embeddings=embeddings,
            texts=texts,
            metadata=metadata,
            collection_name=collection_name,
        )

        async with _registry_lock:
            if parsed_doc.document_id in document_registry:
                document_registry[parsed_doc.document_id]["status"] = DocumentStatus.PROCESSED
                document_registry[parsed_doc.document_id]["embeddings_count"] = len(embeddings)
        
        # Save registry to disk after modification
        await _save_registry_to_disk()

        logger.info(
            "Embedding processing done: doc_id=%s, vectors=%d, collection=%s",
            parsed_doc.document_id, len(embeddings), collection_name or "(default)"
        )

    except Exception:
        logger.exception("Embedding processing failed: doc_id=%s", parsed_doc.document_id)
        async with _registry_lock:
            if parsed_doc.document_id in document_registry:
                document_registry[parsed_doc.document_id]["status"] = DocumentStatus.FAILED
        
        # Save registry to disk after modification
        await _save_registry_to_disk()


@router.get("/{document_id}", response_model=ParsedDocument)
async def get_document(document_id: str):
    async with _registry_lock:
        if document_id not in document_registry:
            raise _http_404("Document not found")
    raise HTTPException(
        status_code=501,
        detail="Full document retrieval not implemented. Use search endpoints to query document content.",
    )


@router.get("/{document_id}/status")
async def get_document_status(document_id: str):
    async with _registry_lock:
        info = document_registry.get(document_id)
        if not info:
            raise _http_404("Document not found")
    # ✅ Return a dict; FastAPI encodes datetimes/enums
    return {
        "document_id": document_id,
        "status": info["status"],
        "filename": info["filename"],
        "pages_processed": info["pages_processed"],
        "created_at": info["created_at"],
        "file_size": info.get("file_size"),
        "embeddings_count": info.get("embeddings_count", 0),
    }


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(document_id: str):
    async with _registry_lock:
        if document_id not in document_registry:
            raise _http_404("Document not found")

    try:
        vs = get_vector() if _use_new_vector else get_vector_service()
        vectors_deleted = await vs.delete_document(document_id)

        async with _registry_lock:
            document_registry.pop(document_id, None)
        
        # Save registry to disk after modification
        await _save_registry_to_disk()

        logger.info("Deleted document %s: %d vectors removed", document_id, vectors_deleted)

        return DocumentDeleteResponse(
            document_id=document_id,
            status="deleted",
            message=f"Document and {vectors_deleted} associated vectors deleted successfully",
            vectors_deleted=vectors_deleted,
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to delete document %s", document_id)
        raise _http_500("Failed to delete document")


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    status_filter: Optional[DocumentStatus] = Query(default=None),
):
    async with _registry_lock:
        items = []
        for doc_id, info in document_registry.items():
            if status_filter is None or info["status"] == status_filter:
                items.append(
                    {
                        "document_id": doc_id,
                        "filename": info["filename"],
                        "status": info["status"],
                        "pages_processed": info["pages_processed"],
                        "created_at": info["created_at"],
                        "file_size": info.get("file_size"),
                        "embeddings_count": info.get("embeddings_count", 0),
                    }
                )

    items.sort(key=lambda x: x["created_at"], reverse=True)
    start = (page - 1) * page_size
    end = start + page_size
    return DocumentListResponse(
        documents=items[start:end],  # Pydantic will serialize datetimes
        total=len(items),
        page=page,
        page_size=page_size,
    )


@router.get("/{document_id}/stats", response_model=DocumentStats)
async def get_document_stats(document_id: str):
    async with _registry_lock:
        info = document_registry.get(document_id)
        if not info:
            raise _http_404("Document not found")

    try:
        vs = get_vector() if _use_new_vector else get_vector_service()
        count_method = getattr(vs, "count_document_embeddings", None)
        if callable(count_method):
            embeddings_count = await count_method(document_id)
        else:
            embeddings_count = int(info.get("embeddings_count") or info["pages_processed"])

        return DocumentStats(
            document_id=document_id,
            filename=str(info["filename"]),
            pages=int(info["pages_processed"]),
            embeddings_count=int(embeddings_count),
            upload_date=info["created_at"],
            last_accessed=_utcnow(),
            access_count=1,
        )
    except Exception:
        logger.exception("Failed to get stats for %s", document_id)
        raise _http_500("Failed to get document stats")


@router.get("/supported-formats")
async def get_supported_formats():
    try:
        supported_formats = await document_parser.get_supported_formats()
    except Exception:
        logger.warning("document_parser.get_supported_formats failed; falling back to settings", exc_info=True)
        supported_formats = sorted(set(S.parser_map.keys()))

    # ✅ Return a dict; FastAPI encodes everything
    return {
        "supported_formats": supported_formats,
        "max_file_size": S.max_file_size,
        "max_file_size_mb": round(S.max_file_size / (1024 * 1024), 2),
        "supported_mime_types": S.supported_mime_types,
    }

# Add a duplicate route to handle the case where the prefix is incorrectly included
@router.get("/documents/supported-formats")
async def get_supported_formats_alt():
    """Alternative route to handle incorrect URL with double prefix"""
    return await get_supported_formats()

# Add a duplicate route to handle the case where the prefix is incorrectly included
@router.get("/documents/supported-formats")
async def get_supported_formats_alt():
    """Alternative route to handle incorrect URL with double prefix"""
    return await get_supported_formats()
