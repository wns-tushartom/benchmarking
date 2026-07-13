from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Callable, Mapping

from benchmarking.core.schemas import Chunk
from source.services.project_documents import ProjectDocument


_ENTITY_VERSION = "entity_heuristic_v1"
_HEADING_VERSION = "heading_sections_v1"
_FIXED_VERSION = "fixed_token_v1"
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class ProjectChunkerSpec:
    algorithm: str
    version: str
    params: Mapping[str, int]


CHUNKER_SPECS: dict[str, ProjectChunkerSpec] = {
    "entity_heuristic_w6": ProjectChunkerSpec("entity_heuristic", _ENTITY_VERSION, {"window_size": 6}),
    "entity_heuristic_w5": ProjectChunkerSpec("entity_heuristic", _ENTITY_VERSION, {"window_size": 5}),
    "entity_heuristic_w4": ProjectChunkerSpec("entity_heuristic", _ENTITY_VERSION, {"window_size": 4}),
    "Heading_sections_l2": ProjectChunkerSpec("heading_sections", _HEADING_VERSION, {"level": 2}),
    "fixed_tok1200_ov150": ProjectChunkerSpec(
        "fixed_token",
        _FIXED_VERSION,
        {"chunk_tokens": 1200, "overlap_tokens": 150},
    ),
}
OFFICIAL_PROJECT_CHUNKERS = tuple(CHUNKER_SPECS)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_entities(text: str) -> list[str]:
    words = re.findall(r"\b\w+\b", text)
    entities: list[str] = []
    for word in words:
        if word[0].isupper() and len(word) > 1:
            entities.append(word)
        elif word.isdigit():
            entities.append(word)
        elif any(character.isupper() for character in word) and any(
            character.islower() for character in word
        ):
            entities.append(word)
    return entities


def entity_heuristic_chunking(
    text: str,
    window_size: int = 6,
    min_chunk_size: int = 50,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return []
    chunks: list[str] = []
    current_chunk: list[str] = []
    entity_window: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_entities = extract_entities(sentence)
        if entity_window:
            overlap = len(set(sentence_entities) & set(entity_window))
            overlap_ratio = (
                overlap / min(len(entity_window), len(sentence_entities))
                if sentence_entities
                else 0
            )
            if overlap_ratio < 0.3 and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text) >= min_chunk_size:
                    chunks.append(chunk_text)
                current_chunk = []
                entity_window = []
        current_chunk.append(sentence)
        entity_window.extend(sentence_entities)
        entity_window = entity_window[-window_size:]
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if len(chunk_text) >= min_chunk_size:
            chunks.append(chunk_text)
    return chunks if chunks else [text]


def heading_sections_chunking(
    text: str,
    level: int = 2,
    min_chunk_size: int = 50,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if level == 2:
        heading_pattern = r"(?m)^(?:(?:\d+\.(?:\d+\.?)?|\w+\s+\d+:?)\s+[A-Z]|[A-Z\s]{10,}$|^[A-Z][^.!?]*:$)"
    else:
        heading_pattern = r"(?m)^(?:\d+\.\s+[A-Z]|[A-Z\s]{15,}$)"
    headings = list(re.finditer(heading_pattern, text))
    if not headings:
        chunks = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
        return [chunk for chunk in chunks if len(chunk) >= min_chunk_size] or [text]
    chunks: list[str] = []
    for index, heading_match in enumerate(headings):
        start = heading_match.start()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_size:
            chunks.append(chunk)
    return chunks if chunks else [text]


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    normalized = str(text).replace("\x00", " ")
    normalized = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F]", " ", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def detokenize(tokens: list[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)])", r"\1", text)
    text = re.sub(r"([(])\s+", r"\1", text)
    return normalize_text(text)


def fixed_token_chunks(
    text: str,
    chunk_tokens: int = 1200,
    overlap_tokens: int = 150,
    min_chars: int = 50,
) -> list[str]:
    text = normalize_text(text)
    tokens = tokenize(text)
    if not tokens:
        return []
    if len(tokens) <= chunk_tokens:
        return [text] if len(text) >= min_chars else []
    chunks: list[str] = []
    step = max(1, chunk_tokens - overlap_tokens)
    start = 0
    while start < len(tokens):
        window = tokens[start : start + chunk_tokens]
        chunk = detokenize(window)
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        if start + chunk_tokens >= len(tokens):
            break
        start += step
    return chunks


def _execute(spec: ProjectChunkerSpec, text: str) -> list[str]:
    implementations: dict[str, Callable[..., list[str]]] = {
        "entity_heuristic": entity_heuristic_chunking,
        "heading_sections": heading_sections_chunking,
        "fixed_token": fixed_token_chunks,
    }
    return implementations[spec.algorithm](text, **dict(spec.params))


def chunk_project_documents(
    documents: list[ProjectDocument],
    method: str,
) -> list[Chunk]:
    try:
        spec = CHUNKER_SPECS[method]
    except KeyError:
        raise ValueError(f"Unknown project chunker: {method}") from None
    if not documents:
        raise ValueError("project documents are required")
    if not all(isinstance(document, ProjectDocument) for document in documents):
        raise ValueError("invalid project document")
    ordered_documents = sorted(
        documents,
        key=lambda document: (
            document.source_name,
            int(document.metadata["page_number"]),
            document.content_sha256,
        ),
    )
    chunks: list[Chunk] = []
    for document in ordered_documents:
        input_sha256 = document.content_sha256
        for paragraph in _execute(spec, document.text):
            if not paragraph.strip():
                continue
            metadata = {
                **dict(document.metadata),
                "source_id": document.source_id,
                "chunker_id": method,
                "algorithm": spec.algorithm,
                "algorithm_version": spec.version,
                "algorithm_params": dict(spec.params),
                "input_sha256": input_sha256,
                "output_sha256": _sha256(paragraph),
            }
            chunks.append(
                Chunk(
                    id=len(chunks) + 1,
                    pdf_name=document.source_name,
                    paragraph=paragraph,
                    parent_id=document.source_id,
                    metadata=metadata,
                )
            )
    if not chunks:
        raise ValueError(f"Project chunker produced no chunks: {method}")
    return chunks
