from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import unicodedata
from typing import Iterable

from benchmarking.core.schemas import Chunk, SearchHit
from source.services.project_questions import ProjectQuestion


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_STOPWORDS = _REPO_ROOT / "configs" / "project_relevance_stopwords_v1.txt"
_DEFAULT_STOPWORDS_PIN = _REPO_ROOT / "configs" / "project_relevance_stopwords_v1.sha256"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTEXT_MIN_TOKENS = 3
_CONTEXT_RECALL_THRESHOLD = 0.80


class ProjectRelevanceError(RuntimeError):
    """Raised when deterministic relevance inputs cannot be verified."""


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _read_control_bytes(path: Path) -> bytes:
    path = Path(path)
    if not path.is_absolute():
        raise ProjectRelevanceError("pinned stopword paths must be absolute")
    try:
        for component in (path, *path.parents):
            metadata = component.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise ProjectRelevanceError("stopword paths must not contain symlinks")
        parent = path.parent
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_fd = os.open(parent, flags)
        try:
            opened_parent = os.fstat(parent_fd)
            lexical_parent = parent.lstat()
            if (
                not stat.S_ISDIR(opened_parent.st_mode)
                or not stat.S_ISDIR(lexical_parent.st_mode)
                or (opened_parent.st_dev, opened_parent.st_ino)
                != (lexical_parent.st_dev, lexical_parent.st_ino)
                or parent.resolve(strict=True) != parent
            ):
                raise ProjectRelevanceError("stopword paths must not contain symlinks")
            leaf = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(leaf.st_mode) or not stat.S_ISREG(leaf.st_mode):
                raise ProjectRelevanceError("stopword files must not be symlinks")
            file_fd = os.open(
                path.name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent_fd,
            )
            try:
                opened_file = os.fstat(file_fd)
                if (
                    not stat.S_ISREG(opened_file.st_mode)
                    or (opened_file.st_dev, opened_file.st_ino)
                    != (leaf.st_dev, leaf.st_ino)
                ):
                    raise ProjectRelevanceError("pinned stopword files changed during loading")
                chunks: list[bytes] = []
                while True:
                    chunk = os.read(file_fd, 1024 * 1024)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)
            finally:
                os.close(file_fd)
        finally:
            os.close(parent_fd)
    except ProjectRelevanceError:
        raise
    except OSError:
        raise ProjectRelevanceError("pinned stopword files are unavailable") from None


def _load_pinned_stopwords_cached(
    stopword_path_text: str,
    pin_path_text: str,
) -> tuple[frozenset[str], str]:
    stopword_path = Path(stopword_path_text)
    pin_path = Path(pin_path_text)
    try:
        content = _read_control_bytes(stopword_path)
        expected = _read_control_bytes(pin_path).decode("ascii").strip()
    except ProjectRelevanceError:
        raise
    except UnicodeError:
        raise ProjectRelevanceError("pinned stopword files are unavailable") from None
    if _SHA256_RE.fullmatch(expected) is None:
        raise ProjectRelevanceError("stopword hash pin is invalid")
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected:
        raise ProjectRelevanceError(
            f"stopword hash mismatch (expected {expected}, got {actual})"
        )
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeError:
        raise ProjectRelevanceError("stopword file must be UTF-8") from None
    words = [_normalize_text(line) for line in lines if line.strip()]
    if not words or any(len(_TOKEN_RE.findall(word)) != 1 for word in words):
        raise ProjectRelevanceError("stopword file contains an invalid entry")
    if len(words) != len(set(words)):
        raise ProjectRelevanceError("stopword file contains duplicates")
    return frozenset(words), actual


def load_pinned_stopwords(
    stopword_path: Path = _DEFAULT_STOPWORDS,
    pin_path: Path = _DEFAULT_STOPWORDS_PIN,
) -> tuple[frozenset[str], str]:
    """Load the independently pinned stopword set used by wns_context_tokens_v1."""
    return _load_pinned_stopwords_cached(str(Path(stopword_path)), str(Path(pin_path)))


def wns_context_tokens_v1(text: str) -> frozenset[str]:
    """NFKC + casefold + punctuation split + stopword removal + set semantics."""
    if not isinstance(text, str):
        raise ProjectRelevanceError("context text must be text")
    stopwords, _ = load_pinned_stopwords()
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return frozenset(token for token in _TOKEN_RE.findall(normalized) if token not in stopwords)


def _eligible_contexts(question: ProjectQuestion) -> tuple[tuple[str, frozenset[str]], ...]:
    eligible: list[tuple[str, frozenset[str]]] = []
    for context in question.labels.reference_contexts:
        tokens = wns_context_tokens_v1(context)
        if len(tokens) >= _CONTEXT_MIN_TOKENS:
            eligible.append((_normalize_text(context), tokens))
    return tuple(eligible)


def _label_applies_verified(question: ProjectQuestion, chunker_id: str) -> bool:
    if question.labels.source_ids:
        return True
    if any(reference.chunker == chunker_id for reference in question.labels.chunk_refs):
        return True
    return bool(_eligible_contexts(question))


def label_applies(question: ProjectQuestion, chunker_id: str) -> bool:
    """Return whether this question belongs in one combination's retrieval denominator."""
    load_pinned_stopwords()
    return _label_applies_verified(question, chunker_id)


def _chunk_is_relevant_verified(
    question: ProjectQuestion,
    chunk: Chunk,
    chunker_id: str,
) -> bool:
    metadata = chunk.metadata or {}
    source_id = metadata.get("source_id")
    if isinstance(source_id, str) and source_id in question.labels.source_ids:
        return True

    chunk_id = str(chunk.id)
    if any(
        reference.chunker == chunker_id and reference.chunk_id == chunk_id
        for reference in question.labels.chunk_refs
    ):
        return True

    normalized_chunk = _normalize_text(chunk.paragraph)
    chunk_tokens = wns_context_tokens_v1(chunk.paragraph)
    for normalized_context, context_tokens in _eligible_contexts(question):
        if normalized_context in normalized_chunk:
            return True
        recall = len(context_tokens & chunk_tokens) / len(context_tokens)
        if recall >= _CONTEXT_RECALL_THRESHOLD:
            return True
    return False


def chunk_is_relevant(
    question: ProjectQuestion,
    chunk: Chunk,
    chunker_id: str,
) -> bool:
    """Evaluate one canonical chunk using the project relevance contract."""
    load_pinned_stopwords()
    return _chunk_is_relevant_verified(question, chunk, chunker_id)


def hit_is_relevant(
    question: ProjectQuestion,
    hit: SearchHit,
    chunker_id: str,
) -> bool:
    """Evaluate one search hit using the same canonical chunk predicate."""
    load_pinned_stopwords()
    return _chunk_is_relevant_verified(question, hit.chunk, chunker_id)


def relevant_corpus_count(
    question: ProjectQuestion,
    chunks: Iterable[Chunk],
    chunker_id: str,
) -> int:
    """Count canonical corpus chunks relevant to one question and chunker."""
    load_pinned_stopwords()
    if not _label_applies_verified(question, chunker_id):
        return 0
    return sum(
        1 for chunk in chunks if _chunk_is_relevant_verified(question, chunk, chunker_id)
    )


def metric_denominator(
    questions: Iterable[ProjectQuestion],
    chunker_id: str,
) -> int:
    """Count only questions carrying retrieval labels applicable to this chunker."""
    load_pinned_stopwords()
    return sum(1 for question in questions if _label_applies_verified(question, chunker_id))
