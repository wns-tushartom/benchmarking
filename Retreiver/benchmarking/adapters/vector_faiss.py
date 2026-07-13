from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from benchmarking.core.schemas import Chunk, SearchHit
from benchmarking.adapters.vector_namespace import create_faiss_namespace


class FaissVectorStoreAdapter:
    """FAISS-backed vector-store adapter.

    FAISS is an in-process vector index, not a network database. The adapter supports
    both ephemeral use inside the config-driven benchmark runner and persisted use
    for the long ingestion + later retrieval-smoke scripts.
    """

    def __init__(
        self,
        name: str,
        index_dir: str | None = None,
        load_existing: bool = False,
        index_type: str = "HNSW",
        hnsw_m: int = 16,
        ef_construction: int = 100,
        ef_search: int = 64,
        namespace: str | None = None,
        index_root: str | None = None,
        **_: Any,
    ):
        try:
            import faiss  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError("faiss-cpu is required for FaissVectorStoreAdapter. Install requirements-benchmark.txt.") from exc

        self.faiss = faiss
        self.name = name
        self.index_type = str(index_type or "HNSW")
        self.hnsw_m = int(hnsw_m)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self._explicit_namespace = namespace is not None
        if namespace is not None:
            if index_dir is not None or index_root is None:
                raise ValueError(
                    "explicit FAISS namespace requires index_root and forbids index_dir"
                )
            self.index_dir = create_faiss_namespace(index_root, namespace)
        else:
            if index_root is not None:
                raise ValueError("index_root requires an explicit FAISS namespace")
            self.index_dir = Path(index_dir) if index_dir else None
        self.collection = str(self.index_dir) if self.index_dir else f"faiss_{int(time.time())}"
        self.physical_namespace = self.collection
        self.index: Any | None = None
        self.chunks: List[Chunk] = []
        self.dimensions = 0
        if load_existing:
            self.load()

    def reset_collection(self, schema: Any = None) -> None:
        self.index = None
        self.chunks = []
        self.dimensions = 0
        if self.index_dir:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("index.faiss", "chunks.json"):
                path = self.index_dir / filename
                if path.exists():
                    path.unlink()

    def drop_namespace(self) -> None:
        """Idempotently remove this adapter's persisted index namespace."""
        self.index = None
        self.chunks = []
        self.dimensions = 0
        if not self.index_dir or not self.index_dir.exists():
            return
        if self.index_dir.is_symlink():
            raise ValueError("FAISS index namespace must not be a symlink")
        if self._explicit_namespace:
            shutil.rmtree(self.index_dir)
            return
        for filename in ("index.faiss", "chunks.json"):
            path = self.index_dir / filename
            if path.exists() and not path.is_symlink():
                path.unlink()

    def _normalized(self, vectors: List[List[float]]) -> np.ndarray:
        arr = np.asarray(vectors, dtype="float32")
        if arr.ndim != 2:
            raise ValueError(f"FAISS expects a 2D vector array, got shape={arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError("FAISS vector array contains non-finite values")
        self.faiss.normalize_L2(arr)
        return arr

    def _new_index(self, dimensions: int):
        kind = self.index_type.lower()
        if kind == "flat":
            return self.faiss.IndexFlatIP(dimensions)
        if kind == "hnsw":
            try:
                index = self.faiss.IndexHNSWFlat(dimensions, self.hnsw_m, self.faiss.METRIC_INNER_PRODUCT)
            except TypeError:
                index = self.faiss.IndexHNSWFlat(dimensions, self.hnsw_m)
                index.metric_type = self.faiss.METRIC_INNER_PRODUCT
            index.hnsw.efConstruction = self.ef_construction
            index.hnsw.efSearch = self.ef_search
            return index
        raise ValueError(f"Unsupported FAISS index_type={self.index_type!r}. Use HNSW or Flat.")

    def upsert(self, chunks: List[Chunk], vectors: List[List[float]]) -> Dict[str, float]:
        if not vectors:
            return {"upsert_latency_s": 0.0, "vector_count": 0}
        if len(chunks) != len(vectors):
            raise ValueError(f"FAISS chunk/vector mismatch: chunks={len(chunks)} vectors={len(vectors)}")
        start = time.perf_counter()
        arr = self._normalized(vectors)
        self.dimensions = int(arr.shape[1])
        index = self._new_index(self.dimensions)
        index.add(arr)
        self.index = index
        self.chunks = list(chunks)
        if self.index_dir:
            self.save()
        return {"upsert_latency_s": time.perf_counter() - start, "vector_count": len(vectors)}

    def save(self) -> None:
        if not self.index_dir:
            return
        if self.index is None:
            raise RuntimeError("Cannot save FAISS index before upsert/load")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss.write_index(self.index, str(self.index_dir / "index.faiss"))
        payload = []
        for chunk in self.chunks:
            payload.append({
                "id": chunk.id,
                "pdf_name": chunk.pdf_name,
                "paragraph": chunk.paragraph,
                "parent_id": chunk.parent_id,
                "metadata": chunk.metadata or {},
            })
        (self.index_dir / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load(self) -> None:
        if not self.index_dir:
            raise RuntimeError("index_dir is required when load_existing=True")
        index_path = self.index_dir / "index.faiss"
        chunks_path = self.index_dir / "chunks.json"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Missing FAISS persisted files under {self.index_dir}")
        index = self.faiss.read_index(str(index_path))
        if hasattr(index, "hnsw"):
            index.hnsw.efSearch = self.ef_search
        self.index = index
        raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [
            Chunk(
                id=int(item.get("id") or i),
                pdf_name=str(item.get("pdf_name") or ""),
                paragraph=str(item.get("paragraph") or ""),
                parent_id=str(item.get("parent_id") or item.get("id") or i),
                metadata=dict(item.get("metadata") or {}),
            )
            for i, item in enumerate(raw_chunks, 1)
        ]
        self.dimensions = int(getattr(self.index, "d", 0) or 0)

    def search(self, query_vector: List[float], top_k: int) -> List[SearchHit]:
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded. Call upsert() or load() first.")
        query = self._normalized([query_vector])
        scores, labels = self.index.search(query, int(top_k))
        hits: List[SearchHit] = []
        for score, label in zip(scores[0].tolist(), labels[0].tolist()):
            idx = int(label)
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            metadata = dict(chunk.metadata or {})
            metadata.setdefault("store", "FAISS")
            hits.append(SearchHit(chunk=Chunk(id=chunk.id, pdf_name=chunk.pdf_name, paragraph=chunk.paragraph, parent_id=chunk.parent_id, metadata=metadata), score=float(score)))
        return hits
