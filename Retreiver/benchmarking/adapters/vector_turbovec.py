"""Isolated TurboVec 4-bit embedded vector-index adapter.

TurboVec is an in-process compressed index, not a network vector database.  The
optional dependency is imported only when this adapter is instantiated so the
official benchmark modules remain importable without TurboVec installed.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib
import json
import os
import secrets
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from benchmarking.core.schemas import Chunk, SearchHit


_INDEX_FILENAME = "index.tvim"
_CHUNKS_FILENAME = "chunks.json"
_MANIFEST_FILENAME = "manifest.json"
_PERSISTED_FILENAMES = (_INDEX_FILENAME, _CHUNKS_FILENAME, _MANIFEST_FILENAME)
_UINT64_MAX = np.iinfo(np.uint64).max
_RENAME_EXCHANGE = 2
_CHUNK_FIELDS = {"id", "pdf_name", "paragraph", "parent_id", "metadata"}


class TurboVecVectorStoreAdapter:
    """TurboVec ``IdMapIndex`` adapter for the isolated 4-bit candidate lane."""

    def __init__(
        self,
        name: str,
        index_dir: str | None = None,
        load_existing: bool = False,
        index_type: str = "TurboQuant4bit",
        bits: int = 4,
        **_: Any,
    ) -> None:
        self.name = name
        self.index_type = str(index_type or "")
        if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)):
            raise ValueError("TurboVec candidate supports only bits=4")
        self.bits = int(bits)
        if self.index_type != "TurboQuant4bit":
            raise ValueError("TurboVec candidate supports only index_type='TurboQuant4bit'")
        if self.bits != 4:
            raise ValueError("TurboVec candidate supports only bits=4")

        try:
            self.turbovec = importlib.import_module("turbovec")
        except Exception as exc:
            raise RuntimeError(
                "turbovec==1.0.0 is required for TurboVecVectorStoreAdapter"
            ) from exc
        if getattr(self.turbovec, "__version__", None) != "1.0.0":
            raise RuntimeError("turbovec==1.0.0 is required for TurboVecVectorStoreAdapter")
        if not hasattr(self.turbovec, "IdMapIndex"):
            raise RuntimeError("turbovec==1.0.0 does not expose IdMapIndex")

        self.index_dir = Path(index_dir) if index_dir else None
        if self.index_dir is not None:
            self._validate_namespace(require_exists=False)
        self.collection = str(self.index_dir) if self.index_dir else f"turbovec_{int(time.time())}"
        self.physical_namespace = self.collection
        self.index: Any | None = None
        self.chunks: List[Chunk] = []
        self.dimensions = 0
        self._chunks_by_id: dict[int, Chunk] = {}
        if load_existing:
            self.load()

    @staticmethod
    def _chunk_id(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("TurboVec chunk IDs must be unsigned 64-bit integers")
        external_id = int(value)
        if external_id < 0 or external_id > int(_UINT64_MAX):
            raise ValueError("TurboVec chunk IDs must be unsigned 64-bit integers")
        return external_id

    @staticmethod
    def _normalized(vectors: Any, *, subject: str) -> np.ndarray:
        try:
            array = np.asarray(vectors, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"TurboVec {subject} must be a 2D vector array") from exc
        if array.ndim != 2:
            raise ValueError(
                f"TurboVec {subject} must be a 2D vector array, got shape={array.shape}"
            )
        if array.shape[1] == 0:
            raise ValueError(f"TurboVec {subject} has an empty dimension")
        if not np.isfinite(array).all():
            raise ValueError(f"TurboVec {subject} contains non-finite values")
        norms = np.linalg.norm(array, axis=1)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ValueError(f"TurboVec {subject} contains a zero-norm vector")
        normalized = np.ascontiguousarray(array / norms[:, None], dtype=np.float32)
        normalized_norms = np.linalg.norm(normalized.astype(np.float64), axis=1)
        if (
            not np.isfinite(normalized).all()
            or not np.isfinite(normalized_norms).all()
            or np.any(normalized_norms == 0)
        ):
            raise ValueError(f"TurboVec {subject} cannot be normalized safely")
        return normalized

    def _validate_namespace(self, *, require_exists: bool) -> None:
        if self.index_dir is None:
            raise RuntimeError("index_dir is required for TurboVec persistence")
        path = self.index_dir
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            if require_exists:
                raise FileNotFoundError(f"Missing TurboVec index namespace: {path}") from None
            self._reject_symlink_parents(path)
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("TurboVec index namespace must not be a symlink")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("TurboVec index namespace must be a directory")
        self._reject_symlink_parents(path)

    @staticmethod
    def _reject_symlink_parents(path: Path) -> None:
        absolute = path.absolute()
        for parent in (absolute, *absolute.parents):
            try:
                metadata = parent.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError("TurboVec index namespace must not contain symlink components")

    @staticmethod
    def _require_regular_file(path: Path) -> None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing TurboVec persisted file: {path}") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"TurboVec persisted file must not be a symlink: {path.name}")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"TurboVec persisted file must be regular: {path.name}")

    @staticmethod
    def _reject_existing_symlink(path: Path) -> None:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"TurboVec persisted file must not be a symlink: {path.name}")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"TurboVec persisted file must be regular: {path.name}")

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _open_namespace_fd(self) -> int:
        assert self.index_dir is not None
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.index_dir, flags)
        except OSError as exc:
            raise ValueError("TurboVec index namespace must be a real directory") from exc
        try:
            actual = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            if actual != self.index_dir.absolute():
                raise ValueError("TurboVec index namespace must not contain symlink components")
            metadata = os.fstat(descriptor)
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("TurboVec index namespace must be a directory")
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _open_parent_fd(path: Path) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ValueError("TurboVec persistence parent must be a real directory") from exc
        try:
            actual = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
            if actual != path.absolute() or not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ValueError("TurboVec persistence parent must be a real directory")
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _open_directory_at(parent_fd: int, name: str) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ValueError("TurboVec generation namespace must be a real directory") from exc
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise ValueError("TurboVec generation namespace must be a directory")
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _rename_exchange(parent_fd: int, first: str, second: str) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOSYS, "renameat2 is required for atomic TurboVec generation updates")
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_fd,
            os.fsencode(first),
            parent_fd,
            os.fsencode(second),
            _RENAME_EXCHANGE,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))

    @classmethod
    def _remove_directory_at(cls, parent_fd: int, name: str) -> None:
        try:
            directory_fd = cls._open_directory_at(parent_fd, name)
        except (FileNotFoundError, ValueError):
            return
        try:
            for entry in os.listdir(directory_fd):
                metadata = os.stat(entry, dir_fd=directory_fd, follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("TurboVec staged generation contains an unsafe entry")
                os.unlink(entry, dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rmdir(name, dir_fd=parent_fd)

    @classmethod
    def _validate_generation_at(cls, parent_fd: int, name: str) -> None:
        directory_fd = cls._open_directory_at(parent_fd, name)
        try:
            entries = set(os.listdir(directory_fd))
            unexpected = entries - set(_PERSISTED_FILENAMES)
            if unexpected:
                raise ValueError("TurboVec generation contains unexpected persisted files")
            for filename in entries:
                cls._reject_unsafe_entry_at(directory_fd, filename)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _open_regular_at(directory_fd: int, name: str) -> int:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing TurboVec persisted file: {name}") from None
        except OSError as exc:
            raise ValueError(f"TurboVec persisted file must be regular and not a symlink: {name}") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"TurboVec persisted file must be regular: {name}")
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    @staticmethod
    def _read_descriptor(descriptor: int) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                return b"".join(blocks)
            blocks.append(block)

    @classmethod
    def _sha256_descriptor(cls, descriptor: int) -> str:
        return hashlib.sha256(cls._read_descriptor(descriptor)).hexdigest()

    @staticmethod
    def _reject_unsafe_entry_at(directory_fd: int, name: str) -> None:
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"TurboVec persisted file must not be a symlink: {name}")
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"TurboVec persisted file must be regular: {name}")

    @staticmethod
    def _atomic_bytes_at(directory_fd: int, name: str, payload: bytes) -> str:
        temporary = f".{name}.{secrets.token_hex(12)}.tmp"
        descriptor = -1
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            return hashlib.sha256(payload).hexdigest()
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass

    def _clear_memory(self) -> None:
        self.index = None
        self.chunks = []
        self.dimensions = 0
        self._chunks_by_id = {}

    def reset_collection(self, schema: Any = None) -> None:
        del schema
        self._clear_memory()
        if self.index_dir is None:
            return
        self._validate_namespace(require_exists=False)
        if not self.index_dir.exists():
            self.index_dir.mkdir(parents=True, mode=0o700)
            self._validate_namespace(require_exists=True)
            return
        paths = [self.index_dir / filename for filename in _PERSISTED_FILENAMES]
        for path in paths:
            self._reject_existing_symlink(path)
        for path in paths:
            if path.exists():
                path.unlink()

    def drop_namespace(self) -> None:
        """Idempotently remove this adapter's owned persistence directory."""
        self._clear_memory()
        if self.index_dir is None:
            return
        try:
            self._validate_namespace(require_exists=True)
        except FileNotFoundError:
            return
        for entry in self.index_dir.iterdir():
            if entry.name not in _PERSISTED_FILENAMES:
                raise ValueError(f"Refusing to remove TurboVec namespace with unexpected file: {entry.name}")
            self._require_regular_file(entry)
        shutil.rmtree(self.index_dir)

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> Dict[str, float]:
        array = self._normalized(vectors, subject="vector array")
        if array.shape[0] != len(chunks):
            raise ValueError(
                f"TurboVec chunk/vector mismatch: chunks={len(chunks)} vectors={array.shape[0]}"
            )

        external_ids = [self._chunk_id(chunk.id) for chunk in chunks]
        if len(set(external_ids)) != len(external_ids):
            raise ValueError("TurboVec duplicate chunk IDs are not allowed")

        start = time.perf_counter()
        dimensions = int(array.shape[1])
        index = self.turbovec.IdMapIndex(dim=dimensions, bit_width=4)
        ids = np.ascontiguousarray(external_ids, dtype=np.uint64)
        index.add_with_ids(array, ids)

        self.index = index
        self.chunks = list(chunks)
        self.dimensions = dimensions
        self._chunks_by_id = dict(zip(external_ids, self.chunks))
        if self.index_dir is not None:
            self.save()
        return {
            "upsert_latency_s": time.perf_counter() - start,
            "vector_count": len(chunks),
        }

    def search(self, query_vector: List[float], top_k: int) -> List[SearchHit]:
        if self.index is None:
            raise RuntimeError("TurboVec index is not loaded. Call upsert() or load() first.")
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or int(top_k) <= 0:
            raise ValueError("TurboVec top_k must be a positive integer")

        query = self._normalized([query_vector], subject="query vector")
        if query.shape[1] != self.dimensions:
            raise ValueError(
                f"TurboVec query dimension mismatch: expected={self.dimensions} actual={query.shape[1]}"
            )
        requested = min(int(top_k), len(self.chunks))
        if requested == 0:
            return []
        scores, external_ids = self.index.search(query, k=requested)
        raw_score_array = np.asarray(scores)
        if raw_score_array.dtype.kind not in {"f", "i", "u"}:
            raise ValueError("TurboVec search returned invalid non-numeric scores")
        score_array = np.asarray(raw_score_array, dtype=np.float64)
        id_array = np.asarray(external_ids)
        if score_array.ndim != 2 or id_array.ndim != 2 or score_array.shape != id_array.shape:
            raise ValueError("TurboVec search returned invalid result shapes")
        if score_array.shape[0] != 1:
            raise ValueError("TurboVec search returned invalid result shapes")
        if score_array.shape[1] != requested:
            raise ValueError("TurboVec search returned invalid result count")
        if not np.isfinite(score_array).all():
            raise ValueError("TurboVec search returned non-finite scores")
        if id_array.dtype.kind not in {"i", "u"}:
            raise ValueError("TurboVec search returned invalid external IDs")
        result_ids = [self._chunk_id(value) for value in id_array[0].tolist()]
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("TurboVec search returned duplicate external IDs")

        hits: List[SearchHit] = []
        for score, chunk_id in zip(score_array[0].tolist(), result_ids):
            chunk = self._chunks_by_id.get(chunk_id)
            if chunk is None:
                raise ValueError(f"TurboVec search returned unknown external ID: {chunk_id}")
            metadata = dict(chunk.metadata or {})
            metadata["store"] = "TurboVec"
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        id=chunk.id,
                        pdf_name=chunk.pdf_name,
                        paragraph=chunk.paragraph,
                        parent_id=chunk.parent_id,
                        metadata=metadata,
                    ),
                    score=float(score),
                )
            )
        return hits

    def save(self) -> None:
        if self.index_dir is None:
            return
        if self.index is None:
            raise RuntimeError("Cannot save TurboVec index before upsert/load")
        self._validate_namespace(require_exists=False)
        if self.index_dir.exists():
            for filename in _PERSISTED_FILENAMES:
                self._reject_existing_symlink(self.index_dir / filename)
        parent = self.index_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        self._reject_symlink_parents(parent)
        parent_fd = self._open_parent_fd(parent)
        generation_name = f".{self.index_dir.name}.stage.{secrets.token_hex(12)}"
        generation_fd = -1
        try:
            try:
                os.mkdir(generation_name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError as exc:
                raise ValueError("TurboVec staging namespace already exists") from exc
            generation_fd = self._open_directory_at(parent_fd, generation_name)

            temporary_index = f".{secrets.token_hex(12)}.{_INDEX_FILENAME}"
            index_fd = -1
            try:
                self.index.sync(f"/proc/self/fd/{generation_fd}/{temporary_index}")
                index_fd = self._open_regular_at(generation_fd, temporary_index)
                os.fchmod(index_fd, 0o600)
                os.fsync(index_fd)
                index_sha256 = self._sha256_descriptor(index_fd)
                os.replace(
                    temporary_index,
                    _INDEX_FILENAME,
                    src_dir_fd=generation_fd,
                    dst_dir_fd=generation_fd,
                )
            finally:
                if index_fd >= 0:
                    os.close(index_fd)
                try:
                    os.unlink(temporary_index, dir_fd=generation_fd)
                except FileNotFoundError:
                    pass

            chunks_payload = json.dumps(
                [
                    {
                        "id": self._chunk_id(chunk.id),
                        "pdf_name": chunk.pdf_name,
                        "paragraph": chunk.paragraph,
                        "parent_id": chunk.parent_id,
                        "metadata": dict(chunk.metadata or {}),
                    }
                    for chunk in self.chunks
                ],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            chunks_sha256 = self._atomic_bytes_at(
                generation_fd, _CHUNKS_FILENAME, chunks_payload
            )
            manifest_payload = json.dumps(
                {
                    "format_version": 1,
                    "index_type": self.index_type,
                    "bits": self.bits,
                    "dimensions": self.dimensions,
                    "vector_count": len(self.chunks),
                    "files": {
                        _INDEX_FILENAME: index_sha256,
                        _CHUNKS_FILENAME: chunks_sha256,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
            self._atomic_bytes_at(generation_fd, _MANIFEST_FILENAME, manifest_payload)
            os.fsync(generation_fd)
            os.close(generation_fd)
            generation_fd = -1

            try:
                target_metadata = os.stat(
                    self.index_dir.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                os.rename(
                    generation_name,
                    self.index_dir.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                )
                generation_name = ""
            else:
                if stat.S_ISLNK(target_metadata.st_mode) or not stat.S_ISDIR(target_metadata.st_mode):
                    raise ValueError("TurboVec index namespace must be a real directory")
                self._validate_generation_at(parent_fd, self.index_dir.name)
                self._rename_exchange(parent_fd, generation_name, self.index_dir.name)
            os.fsync(parent_fd)
        finally:
            if generation_fd >= 0:
                os.close(generation_fd)
            if generation_name:
                self._remove_directory_at(parent_fd, generation_name)
            os.close(parent_fd)

    def load(self) -> None:
        self._validate_namespace(require_exists=True)
        assert self.index_dir is not None
        paths = [self.index_dir / filename for filename in _PERSISTED_FILENAMES]
        for path in paths:
            self._require_regular_file(path)

        directory_fd = self._open_namespace_fd()
        descriptors: dict[str, int] = {}
        try:
            for filename in _PERSISTED_FILENAMES:
                descriptors[filename] = self._open_regular_at(directory_fd, filename)
            try:
                manifest = json.loads(
                    self._read_descriptor(descriptors[_MANIFEST_FILENAME]).decode("utf-8")
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid TurboVec metadata manifest") from exc
            if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
                raise ValueError("Unsupported TurboVec metadata manifest format")
            if manifest.get("index_type") != "TurboQuant4bit" or manifest.get("bits") != 4:
                raise ValueError("Persisted TurboVec index is not TurboQuant4bit bits=4")
            files = manifest.get("files")
            if not isinstance(files, dict) or set(files) != {_INDEX_FILENAME, _CHUNKS_FILENAME}:
                raise ValueError("Invalid TurboVec metadata manifest file hashes")
            for filename in (_INDEX_FILENAME, _CHUNKS_FILENAME):
                expected = files.get(filename)
                if (
                    not isinstance(expected, str)
                    or self._sha256_descriptor(descriptors[filename]) != expected
                ):
                    raise ValueError(f"TurboVec hash mismatch for {filename}")

            try:
                raw_chunks = json.loads(
                    self._read_descriptor(descriptors[_CHUNKS_FILENAME]).decode("utf-8")
                )
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError("Invalid TurboVec chunk metadata") from exc
            if not isinstance(raw_chunks, list):
                raise ValueError("Invalid TurboVec chunk metadata")

            chunks: list[Chunk] = []
            external_ids: list[int] = []
            for item in raw_chunks:
                if not isinstance(item, dict) or set(item) != _CHUNK_FIELDS:
                    raise ValueError("Invalid TurboVec chunk metadata entry")
                external_id = self._chunk_id(item["id"])
                metadata = item["metadata"]
                if (
                    not isinstance(item["pdf_name"], str)
                    or not isinstance(item["paragraph"], str)
                    or not isinstance(item["parent_id"], str)
                    or not isinstance(metadata, dict)
                ):
                    raise ValueError("Invalid TurboVec chunk metadata entry")
                chunks.append(
                    Chunk(
                        id=external_id,
                        pdf_name=item["pdf_name"],
                        paragraph=item["paragraph"],
                        parent_id=item["parent_id"],
                        metadata=dict(metadata),
                    )
                )
                external_ids.append(external_id)
            if len(set(external_ids)) != len(external_ids):
                raise ValueError("TurboVec persisted chunk metadata contains duplicate chunk IDs")

            dimensions = manifest.get("dimensions")
            vector_count = manifest.get("vector_count")
            if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
                raise ValueError("Invalid TurboVec persisted dimensions")
            if isinstance(vector_count, bool) or not isinstance(vector_count, int):
                raise ValueError("Invalid TurboVec persisted vector count")
            if vector_count != len(chunks):
                raise ValueError(
                    f"TurboVec vector count mismatch: manifest={vector_count} chunks={len(chunks)}"
                )

            index = self.turbovec.IdMapIndex.load(
                f"/proc/self/fd/{descriptors[_INDEX_FILENAME]}"
            )
            try:
                index_count = len(index)
            except Exception as exc:
                raise ValueError("Loaded TurboVec index does not expose its vector count") from exc
            if index_count != vector_count:
                raise ValueError(
                    f"TurboVec vector count mismatch: manifest={vector_count} index={index_count}"
                )
            index_dimension = getattr(index, "dim", getattr(index, "d", dimensions))
            if int(index_dimension) != dimensions:
                raise ValueError(
                    f"TurboVec dimension mismatch: manifest={dimensions} index={index_dimension}"
                )
            index_bit_width = getattr(index, "bit_width", None)
            if isinstance(index_bit_width, bool) or index_bit_width != 4:
                raise ValueError(
                    f"TurboVec loaded index bit width is not 4: {index_bit_width}"
                )
            contains = getattr(type(index), "__contains__", None)
            if contains is not None:
                missing = [external_id for external_id in external_ids if external_id not in index]
                if missing:
                    raise ValueError("TurboVec persisted chunk IDs do not match the loaded index")

            self.index = index
            self.chunks = chunks
            self.dimensions = dimensions
            self._chunks_by_id = dict(zip(external_ids, chunks))
        finally:
            for descriptor in descriptors.values():
                os.close(descriptor)
            os.close(directory_fd)
