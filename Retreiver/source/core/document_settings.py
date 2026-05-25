# app/core/document_settings.py
from __future__ import annotations

import mimetypes
from functools import lru_cache
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from source.config import load_document_settings, load_cloud_services

# ---- sane defaults (override via YAML) ----
_DEFAULT_PARSER_MAP: Dict[str, str] = {
    ".pdf": "parse_pdf",
    ".txt": "parse_text_file",
    ".md": "parse_text_file",
    ".markdown": "parse_text_file",
}
_DEFAULT_SUPPORTED_FILE_TYPES: List[str] = [".pdf", ".txt", ".md", ".markdown"]
_EXT_TO_MIME = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}
_DEFAULT_SUPPORTED_MIME_TYPES = [
    "application/pdf",
    "text/plain",
    "text/markdown",
]


class DocumentSettings(BaseModel):
    # Uploads / IO
    upload_tmp_dir: str = "/tmp/app_uploads"
    upload_chunk_size: int = Field(default=1024 * 1024, ge=64 * 1024)  # 1 MiB
    max_file_size: int = Field(default=50 * 1024 * 1024, ge=1)         # 50 MiB
    batch_upload_concurrency: int = Field(default=4, ge=1)

    # File types & parser routing
    supported_file_types: List[str] = Field(default_factory=lambda: list(_DEFAULT_SUPPORTED_FILE_TYPES))
    supported_mime_types: List[str] = Field(default_factory=lambda: list(_DEFAULT_SUPPORTED_MIME_TYPES))
    parser_map: Dict[str, str] = Field(default_factory=lambda: dict(_DEFAULT_PARSER_MAP))

    # Embeddings (used by background pipeline chunking, etc.)
    embedding_batch_size: int = Field(default=64, ge=1)  # override in YAML if you want


def _normalize_ext_list(value: Optional[object]) -> List[str]:
    """
    Accepts ".pdf,.md" or [".pdf", "md"] and returns extensions like [".pdf", ".md"].
    """
    if not value:
        return list(_DEFAULT_SUPPORTED_FILE_TYPES)
    if isinstance(value, str):
        items = [it.strip() for it in value.split(",") if it.strip()]
    elif isinstance(value, list):
        items = [str(it).strip() for it in value if str(it).strip()]
    else:
        return list(_DEFAULT_SUPPORTED_FILE_TYPES)
    # ensure leading dot
    return [it if it.startswith(".") else f".{it}" for it in items]


def _mimes_for_exts(exts: List[str]) -> List[str]:
    out: List[str] = []
    for ext in exts:
        mime = _EXT_TO_MIME.get(ext)
        if not mime:
            mime, _ = mimetypes.guess_type(f"file{ext}")
        if mime:
            out.append(mime)
    return out or list(_DEFAULT_SUPPORTED_MIME_TYPES)


@lru_cache(maxsize=1)
def get_document_settings() -> DocumentSettings:
    """
    Load YAML → merge with defaults → return a cached, immutable settings object.
    Never KeyErrors on missing keys.
    """
    dcfg = load_document_settings() or {}
    ccfg = load_cloud_services() or {}

    # Build kwargs with safe fallbacks
    upload_tmp_dir = dcfg.get("upload_tmp_dir") or "/tmp/app_uploads"
    upload_chunk_size = int(dcfg.get("upload_chunk_size") or 1024 * 1024)
    max_file_size = int(dcfg.get("max_file_size") or ccfg.get("max_file_size") or 50 * 1024 * 1024)
    batch_upload_concurrency = int(dcfg.get("batch_upload_concurrency") or 4)
    embedding_batch_size = int(
        dcfg.get("embedding_batch_size")
        or ccfg.get("embedding_batch_size")
        or 64
    )

    # Parser map (overrideable)
    pmap = dcfg.get("parser_map")
    parser_map = dict(_DEFAULT_PARSER_MAP)
    if isinstance(pmap, dict) and pmap:
        # normalize keys to lowercase extensions with dot
        parser_map = {}
        for k, v in pmap.items():
            ext = str(k).strip().lower()
            if ext and not ext.startswith("."):
                ext = f".{ext}"
            parser_map[ext] = str(v)

    # Supported file types (prefer document_settings, else cloud_services)
    supported_file_types = _normalize_ext_list(
        dcfg.get("supported_file_types", ccfg.get("supported_file_types"))
    )

    # Supported MIME types (explicit list beats derived)
    smt = dcfg.get("supported_mime_types")
    if isinstance(smt, list) and smt:
        supported_mime_types = smt
    else:
        supported_mime_types = _mimes_for_exts(supported_file_types)

    return DocumentSettings(
        upload_tmp_dir=upload_tmp_dir,
        upload_chunk_size=upload_chunk_size,
        max_file_size=max_file_size,
        batch_upload_concurrency=batch_upload_concurrency,
        embedding_batch_size=embedding_batch_size,
        parser_map=parser_map,
        supported_file_types=supported_file_types,
        supported_mime_types=supported_mime_types,
    )
