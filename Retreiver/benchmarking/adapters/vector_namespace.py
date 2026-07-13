"""Backend-safe physical namespaces for isolated project matrix runs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat


_NAMESPACE_DIGEST_LENGTH = 48


def namespace_digest(identity: str) -> str:
    """Return a bounded digest for an opaque project/run/combination identity."""
    if not isinstance(identity, str) or not identity or len(identity) > 4096:
        raise ValueError("namespace must be non-empty bounded text")
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:_NAMESPACE_DIGEST_LENGTH]


def safe_lower_namespace(identity: str) -> str:
    """Return a Qdrant/PGVector-safe lowercase identifier."""
    return f"project_{namespace_digest(identity)}"


def safe_weaviate_namespace(identity: str) -> str:
    """Return a Weaviate class name: uppercase first, then alphanumerics only."""
    return f"Project{namespace_digest(identity)}"


def create_faiss_namespace(index_root: str | Path, identity: str) -> Path:
    """Exclusively reserve a derived FAISS namespace under a no-follow root FD."""
    root = validated_index_root(index_root)
    component = safe_lower_namespace(identity)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, flags)
    created = False
    try:
        opened_root = os.fstat(root_fd)
        lexical_root = root.lstat()
        if (opened_root.st_dev, opened_root.st_ino) != (
            lexical_root.st_dev,
            lexical_root.st_ino,
        ):
            raise ValueError("index_root changed during namespace creation")
        os.mkdir(component, mode=0o700, dir_fd=root_fd)
        created = True
        os.fsync(root_fd)
    except Exception:
        if created:
            try:
                os.rmdir(component, dir_fd=root_fd)
                os.fsync(root_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(root_fd)
    namespace = root / component
    metadata = namespace.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("FAISS namespace is not a regular directory")
    return namespace


def validated_index_root(value: str | Path) -> Path:
    """Validate an existing, canonical, non-symlink directory for FAISS indexes."""
    root = Path(value)
    if not root.is_absolute():
        raise ValueError("index_root must be an absolute directory")
    try:
        metadata = root.lstat()
    except OSError:
        raise ValueError("index_root must be an existing directory") from None
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("index_root must be a non-symlink directory")
    try:
        resolved = root.resolve(strict=True)
    except OSError:
        raise ValueError("index_root must be an existing directory") from None
    if resolved != root:
        raise ValueError("index_root must not contain symlink components")
    return root
