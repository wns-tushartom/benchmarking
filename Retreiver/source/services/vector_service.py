# type: ignore
"""
Vector Service (Milvus adapter, config-driven, hardened)

Key improvements:
- Lazy connection (never at import)
- Robust config/env precedence
- Safe JSON filter builder (no `== {}`; uses EXISTS/ArrayContains*/JSONContains*)
- Protects against bad collection name defaults (e.g., "string")
- Creates collection with JSON field when available; falls back to VARCHAR for metadata
- Consistent similarity normalization across COSINE/IP/L2
- Backwards-compatible with existing VARCHAR(metadata=json-dumped) collections
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional, Iterable, Tuple

from source.config import load_cloud_services, load_embedding_settings

# --------------------------- Config ---------------------------
cloud_services = load_cloud_services() or {}
embed_cfg = load_embedding_settings() or {}

DEFAULT_COLLECTION = os.getenv(
    "MILVUS_DEFAULT_COLLECTION",
    cloud_services.get("default_collection", "documents"),
)

EMBED_DIM = int(
    os.getenv(
        "EMBEDDING_DIM",
        str(cloud_services.get("embedding_dim") or embed_cfg.get("embedding_dim") or 1024),
    )
)

MILVUS_HOST = os.getenv("MILVUS_HOST", cloud_services.get("milvus_host", "localhost"))
MILVUS_PORT = int(os.getenv("MILVUS_PORT", str(cloud_services.get("milvus_port", 19530))))
MILVUS_URI = os.getenv("MILVUS_URI", cloud_services.get("milvus_uri", "")).strip()
MILVUS_DB = os.getenv("MILVUS_DB", cloud_services.get("milvus_db", "default"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", cloud_services.get("milvus_token", ""))  # Zilliz: ":{api_key}"
METRIC_TYPE = os.getenv("MILVUS_METRIC", cloud_services.get("metric_type", "COSINE")).upper()
DISABLE_VECTOR = os.getenv("DISABLE_VECTOR", "0") == "1"

# --------------------------- Milvus Imports ---------------------------
try:
    from pymilvus import MilvusClient, DataType, connections
    from pymilvus.exceptions import MilvusException
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

    class MilvusException(Exception):
        pass

    class FallbackMilvusClient:
        def list_collections(self) -> List[str]: return []
        def has_collection(self, collection_name: str) -> bool: return False
        def create_collection(self, **kwargs: Any) -> None: return None
        def insert(self, **kwargs: Any) -> Dict[str, Any]: return {"ids": []}
        def drop_collection(self, collection_name: str) -> None: return None
        def search(self, **kwargs: Any) -> List[Dict[str, Any]]: return []
        def delete(self, **kwargs: Any) -> Dict[str, Any]: return {"delete_count": 0}
        def flush(self, collection_name: str) -> None: return None
        def create_schema(self, **kwargs: Any): return None
        def prepare_index_params(self): return None

    MilvusClient = FallbackMilvusClient  # type: ignore
    DataType = None  # type: ignore
    connections = None  # type: ignore

# --------------------------- Schemas ---------------------------
try:
    from source.models.schemas import SearchResult
except ImportError:
    class SearchResult:  # minimal fallback for standalone testing
        def __init__(
            self,
            id: str = "",
            score: float = 0.0,
            text: str = "",
            content: str = "",
            document_id: str = "",
            metadata: Dict[str, Any] = None,
            page_number: Optional[int] = None,
            similarity_score: Optional[float] = None,
            **kwargs
        ):
            self.id = id
            self.score = score
            self.text = text
            self.content = content or text
            self.document_id = document_id
            self.metadata = metadata or {}
            self.page_number = page_number
            self.similarity_score = similarity_score or score

# --------------------------- Logging ---------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")


# --------------------------- Helpers ---------------------------
def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON at line {i} in {path}")


def _clean_collection_name(name: Optional[str]) -> str:
    """
    Guard against leaks like collection='string' from OpenAPI defaults.
    """
    if not name or name.strip().lower() in {"string", "none", "null"}:
        return DEFAULT_COLLECTION
    return name.strip()


def _similarity_from_raw(metric: str, score: Optional[float], distance: Optional[float]) -> float:
    """
    Normalize similarity to [0..1] when possible.
    - COSINE: pymilvus usually returns 'score' in [0..1]; prefer score; fallback 1 - distance
    - IP: score can be >1; we clamp into [0..1] for thresholding convenience
    - L2: lower distance is better; map via 1 / (1 + distance)
    """
    metric = (metric or "COSINE").upper()
    if metric == "COSINE":
        if score is not None:
            return max(0.0, min(1.0, float(score)))
        if distance is not None:
            return max(0.0, min(1.0, 1.0 - float(distance)))
        return 0.0
    if metric in {"IP", "INNER_PRODUCT"}:
        if score is None:
            return 0.0
        # simple squash: assume typical range ~[0..1.2]; clamp
        return max(0.0, min(1.0, float(score)))
    # L2 or others
    if distance is None:
        return 0.0
    try:
        d = float(distance)
        return 1.0 / (1.0 + max(0.0, d))
    except Exception:
        return 0.0


def _to_milvus_literal(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return '"' + v.replace('"', r'\"') + '"'
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_to_milvus_literal(x) for x in v)
        return f"[{inner}]"
    raise ValueError(f"Unsupported literal for Milvus expr: {type(v)}")


def build_json_filter_expr(metadata: Optional[Dict[str, Any]]) -> str:
    """
    Convert a simple metadata dict into a Milvus JSON filter expression.
    Rules:
    - {} → "" (omit filter)
    - None → "" (omit)
    - key: None → NOT EXISTS metadata["key"]
    - key: list/tuple → ArrayContainsAny(metadata["key"], [..])
    - key: scalar (bool/int/float/str) → metadata["key"] == literal
    - key: dict (non-empty) → JSONContains(metadata, "<json>")
    - key: dict (empty) → omitted (or change to NOT EXISTS if you require that)
    """
    if not metadata:
        return ""

    clauses: List[str] = []
    for k, v in metadata.items():
        key = str(k).replace('"', r'\"')

        if v is None:
            clauses.append(f'NOT EXISTS metadata["{key}"]')
            continue

        if isinstance(v, (list, tuple)):
            clauses.append(f'ArrayContainsAny(metadata["{key}"], {_to_milvus_literal(list(v))})')
            continue

        if isinstance(v, (bool, int, float, str)):
            clauses.append(f'metadata["{key}"] == {_to_milvus_literal(v)}')
            continue

        if isinstance(v, dict):
            if v:  # non-empty object
                j = json.dumps(v, ensure_ascii=False).replace('"', r'\"')
                clauses.append(f'JSONContains(metadata, "{j}")')
            # else: empty dict → skip
            continue

        raise ValueError(f"Unsupported JSON value for key '{k}': {type(v)}")

    return " and ".join(clauses)


# --------------------------- Service ---------------------------
class VectorService:
    """
    Milvus adapter with config-driven parameters and lazy/no-op fallback.
    """

    def __init__(self):
        self.client: Optional[MilvusClient] = None
        self.connected: bool = False
        self._metadata_json_supported: Optional[bool] = None  # determined on first create

    # --------- Connection management ---------
    def ensure_connected(self) -> None:
        """Lazy-init Milvus connection; never crash module import."""
        if self.connected and self.client is not None:
            return
        if DISABLE_VECTOR:
            logger.info("Vector features disabled via DISABLE_VECTOR=1")
            self.client = None
            self.connected = False
            return
        if not MILVUS_AVAILABLE:
            logger.info("pymilvus not available; using no-op client")
            self.client = MilvusClient()
            self.connected = False
            return

        try:
            if MILVUS_URI:
                self.client = MilvusClient(uri=MILVUS_URI, token=(MILVUS_TOKEN or None), db_name=(MILVUS_DB or None))
                conn_info = f"URI={MILVUS_URI}"
            else:
                if connections:
                    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, user=None, password=None, db_name=(MILVUS_DB or None))
                self.client = MilvusClient()
                conn_info = f"{MILVUS_HOST}:{MILVUS_PORT}"

            # Probe
            _ = self.client.list_collections()
            self.connected = True
            logger.info(f"Connected to Milvus ({conn_info})")
        except Exception as e:
            logger.warning(f"Milvus unavailable: {e}")
            self.client = MilvusClient()
            self.connected = False

    def reconnect(self):
        logger.info("Reconnecting to Milvus...")
        self.connected = False
        self.client = None
        self.ensure_connected()

    # --------- Schema helpers ----------
    def _get_collection_schema(self, collection_name: str, dimension: int = EMBED_DIM) -> Tuple[Any, Any]:
        """
        Build schema with JSON metadata if supported; else fall back to VARCHAR.
        """
        if not MILVUS_AVAILABLE or not hasattr(self.client, "create_schema"):
            return None, None

        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_field=False,  # keep explicit fields stable
            description=f"Document embeddings collection: {collection_name}",
        )
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dimension)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("document_id", DataType.VARCHAR, max_length=255)

        # Prefer JSON for metadata if available
        metadata_is_json = hasattr(DataType, "JSON")
        self._metadata_json_supported = metadata_is_json
        if metadata_is_json:
            schema.add_field("metadata", DataType.JSON)
        else:
            schema.add_field("metadata", DataType.VARCHAR, max_length=65535)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type=METRIC_TYPE,
            params={"nlist": 1024},
        )
        return schema, index_params

    def _ensure_collection_exists(self, collection_name: Optional[str], dimension: int) -> bool:
        self.ensure_connected()
        if not self.connected:
            return False
        col = _clean_collection_name(collection_name)
        try:
            if self.client.has_collection(col):
                return True
            schema, index_params = self._get_collection_schema(col, dimension)
            if schema and index_params:
                self.client.create_collection(collection_name=col, schema=schema, index_params=index_params)
            else:
                # Legacy fallback
                self.client.create_collection(collection_name=col, dimension=dimension, metric_type=METRIC_TYPE)
            logger.info(f"Created collection '{col}' (dim={dimension})")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection '{col}': {e}")
            return False

    # --------- Collection management ----------
    async def list_collections(self) -> List[str]:
        self.ensure_connected()
        if not self.connected:
            return []
        try:
            return self.client.list_collections()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    async def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        self.ensure_connected()
        if not self.connected:
            return {"error": "not_connected"}
        col = _clean_collection_name(collection_name)
        try:
            if not self.client.has_collection(col):
                return {"error": "collection_not_found", "collection_name": col}
            return {"collection_name": col, "exists": True}
        except Exception as e:
            return {"error": str(e), "collection_name": col}

    async def create_collection(
        self,
        collection_name: str,
        dimension: Optional[int] = None,
        metric_type: str = METRIC_TYPE,
        description: Optional[str] = None,
    ) -> bool:
        self.ensure_connected()
        if not self.connected:
            return False
        col = _clean_collection_name(collection_name)
        try:
            if self.client.has_collection(col):
                logger.warning(f"Collection '{col}' already exists")
                return True
            dim = int(dimension or EMBED_DIM)
            schema, index_params = self._get_collection_schema(col, dim)
            if schema and index_params:
                self.client.create_collection(collection_name=col, schema=schema, index_params=index_params)
            else:
                self.client.create_collection(collection_name=col, dimension=dim, metric_type=metric_type)
            logger.info(f"Created collection '{col}' (dim={dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to create collection '{col}': {e}")
            return False

    async def drop_collection(self, collection_name: str) -> bool:
        self.ensure_connected()
        if not self.connected:
            return False
        col = _clean_collection_name(collection_name)
        try:
            if not self.client.has_collection(col):
                logger.warning(f"Collection '{col}' does not exist")
                return True
            self.client.drop_collection(col)
            logger.info(f"Dropped collection '{col}'")
            return True
        except Exception as e:
            logger.error(f"Failed to drop collection '{col}': {e}")
            return False

    # --------- Stats / IDs ----------
    async def count_documents(self, collection_name: Optional[str] = None) -> int:
        """
        Heuristic count via a wide search (pymilvus MilvusClient lacks a simple count).
        """
        self.ensure_connected()
        if not self.connected:
            return 0
        col = _clean_collection_name(collection_name)
        try:
            if not self._ensure_collection_exists(col, EMBED_DIM):
                return 0
            dummy_vector = [0.0] * EMBED_DIM
            results = self.client.search(
                collection_name=col,
                data=[dummy_vector],
                limit=16384,
                output_fields=["document_id"],
                search_params={"metric_type": METRIC_TYPE, "params": {"nprobe": 10}},
            )
            return len(results[0]) if results else 0
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0

    async def get_unique_document_ids(self, collection_name: Optional[str] = None) -> List[str]:
        self.ensure_connected()
        if not self.connected:
            return []
        col = _clean_collection_name(collection_name)
        try:
            if not self._ensure_collection_exists(col, EMBED_DIM):
                return []
            dummy_vector = [0.0] * EMBED_DIM
            results = self.client.search(
                collection_name=col,
                data=[dummy_vector],
                limit=16384,
                output_fields=["document_id"],
                search_params={"metric_type": METRIC_TYPE, "params": {"nprobe": 10}},
            )
            doc_ids = set()
            for hit in (results[0] if results else []):
                entity = hit.get("entity", {})
                doc_id = entity.get("document_id")
                if doc_id:
                    doc_ids.add(str(doc_id))
            return sorted(doc_ids)
        except Exception as e:
            logger.error(f"Failed to get document IDs: {e}")
            return []

    # --------- Insert ----------
    async def store_document_embeddings(
        self,
        document_id: str,
        embeddings: List[List[float]],
        texts: List[str],
        metadata: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
    ) -> bool:
        self.ensure_connected()
        if not self.connected:
            logger.warning("Vector service disconnected. Skipping insert.")
            return False
        if not embeddings or not texts or len(embeddings) != len(texts):
            logger.warning("No valid embeddings or mismatch between embeddings and texts")
            return False

        col = _clean_collection_name(collection_name)

        # Use incoming dim to avoid schema mismatch
        expected_dim = EMBED_DIM
        dim = len(embeddings[0])
        if expected_dim != dim:
            logger.warning(f"Configured EMBED_DIM={expected_dim} differs from data dim={dim}; using data dim for collection.")
            expected_dim = dim

        if not self._ensure_collection_exists(col, expected_dim):
            return False

        rows = []
        metas = metadata or [{}] * len(embeddings)
        for i, (emb, txt, meta) in enumerate(zip(embeddings, texts, metas)):
            if not isinstance(emb, list) or len(emb) != expected_dim:
                logger.warning(f"Skipping embedding at index {i}: invalid dimension ({len(emb) if isinstance(emb, list) else 'n/a'})")
                continue
            # Always include the document_id inside metadata as well (helpful during search)
            row_meta = {"document_id": document_id, **(meta or {})}

            # If collection uses JSON field for metadata, pass dict directly; else dump to string
            if self._metadata_json_supported:
                metadata_value = row_meta  # type: ignore
            else:
                metadata_value = json.dumps(row_meta, ensure_ascii=False)[:65535]

            rows.append({
                "vector": emb,
                "text": str(txt or "")[:65535],
                "document_id": str(document_id),
                "metadata": metadata_value,
            })

        if not rows:
            logger.warning("No valid data rows to insert")
            return False

        try:
            start = time.time()
            result = self.client.insert(collection_name=col, data=rows)
            inserted = len(result.get("ids", []))
            
            # Try to flush, but don't fail if it errors (Windows milvus-lite issue)
            try:
                self.client.flush(col)
            except Exception as flush_err:
                # Known issue on Windows with milvus-lite
                if "WinError 183" in str(flush_err) or "Cannot create a file when that file already exists" in str(flush_err):
                    logger.warning(f"Flush warning (data was inserted): {flush_err}")
                else:
                    raise
            
            logger.info(f"Inserted {inserted} vectors into '{col}' in {time.time() - start:.2f}s")
            return inserted > 0
        except Exception as e:
            logger.error(f"Insertion failed: {e}")
            return False

    # --------- Search ----------
    async def search_similar(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        collection_name: Optional[str] = None,
        # New: accept structured metadata filter; we still support raw filter_expr for power users
        metadata_filter: Optional[Dict[str, Any]] = None,
        filter_expr: Optional[str] = None,
        score_threshold: Optional[float] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Perform vector search with optional JSON filtering.
        - metadata_filter: dict to be converted via build_json_filter_expr(...)
        - filter_expr: raw Milvus expr (if provided, it's AND-ed with metadata_filter)
        - score_threshold: applied to normalized similarity in [0..1]
        """
        self.ensure_connected()
        if not self.connected:
            logger.warning("Vector service disconnected. Returning empty results.")
            return []

        col = _clean_collection_name(collection_name)
        output_fields = output_fields or ["text", "document_id", "metadata"]

        if not self._ensure_collection_exists(col, len(query_embedding) or EMBED_DIM):
            return []

        # Build safe filter
        json_expr = build_json_filter_expr(metadata_filter) if metadata_filter else ""
        final_expr = json_expr
        if filter_expr:
            final_expr = f"{json_expr} and ({filter_expr})" if json_expr else filter_expr

        try:
            params = {
                "collection_name": col,
                "data": [query_embedding],
                "limit": int(top_k),
                "output_fields": output_fields,
                "search_params": {"metric_type": METRIC_TYPE, "params": {"nprobe": 10}},
            }
            if final_expr:
                params["filter"] = final_expr

            # Debug the final expr in logs (no PII)
            if final_expr:
                logger.debug(f"Milvus filter expr: {final_expr}")

            start = time.time()
            results = self.client.search(**params)
            hits = results[0] if results else []
            out: List[SearchResult] = []

            for hit in hits:
                entity = hit.get("entity", {}) or {}
                raw_score = hit.get("score")
                raw_distance = hit.get("distance")
                sim = _similarity_from_raw(METRIC_TYPE, raw_score, raw_distance)

                if score_threshold is not None and sim < float(score_threshold):
                    continue

                # metadata may be dict (JSON) or string (dumped)
                meta_val = entity.get("metadata", {})
                if isinstance(meta_val, str):
                    try:
                        meta = json.loads(meta_val) if meta_val else {}
                    except Exception:
                        meta = {"raw_metadata": meta_val}
                else:
                    meta = meta_val or {}

                # prefer explicit text field; fallback to empty string
                text = str(entity.get("text", "") or "")
                doc_id = str(entity.get("document_id", "") or "")
                # Milvus auto_id primary key might be absent from 'entity'; keep safe
                row_id = str(entity.get("id", ""))

                # page_number (optional)
                page_num = None
                try:
                    if isinstance(meta, dict) and "page_number" in meta:
                        # ensure positive int if present
                        p = int(meta.get("page_number"))
                        page_num = p if p > 0 else None
                except Exception:
                    page_num = None

                out.append(
                    SearchResult(
                        id=row_id,
                        document_id=doc_id,
                        content=text,
                        score=sim,
                        similarity_score=sim,
                        metadata=meta if isinstance(meta, dict) else {"raw_metadata": meta},
                        page_number=page_num,
                    )
                )

            logger.debug(f"Search took {time.time() - start:.3f}s, returned {len(out)} results")
            return out
        except MilvusException as me:
            # Surface parse errors clearly (e.g., invalid JSON expr)
            logger.error(f"Milvus search error: {me}")
            return []
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    # --------- Delete ----------
    async def delete_document(self, document_id: str, collection_name: Optional[str] = None) -> int:
        self.ensure_connected()
        if not self.connected:
            return 0
        col = _clean_collection_name(collection_name)
        if not self._ensure_collection_exists(col, EMBED_DIM):
            return 0
        try:
            safe_id = str(document_id).replace('"', '\\"')
            expr = f'document_id == "{safe_id}"'
            result = self.client.delete(collection_name=col, filter=expr)
            deleted = result.get("delete_count", 0)
            if deleted > 0:
                self.client.flush(col)
                logger.info(f"Deleted {deleted} vectors for document '{document_id}'")
            return int(deleted)
        except Exception as e:
            logger.error(f"Failed to delete document '{document_id}': {e}")
            return 0

    # --------- JSONL import ----------
    async def load_embeddings_from_jsonl(
        self,
        path: Optional[str] = None,
        document_id: str = "jsonl_import",
        collection_name: Optional[str] = None,
        text_keys: Optional[List[str]] = None,
        embedding_keys: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> bool:
        self.ensure_connected()
        if not self.connected:
            logger.error("Vector service disconnected.")
            return False
        path = path or cloud_services.get("embeddings_path", "data/embeddings/markdown_embeddings.jsonl")
        col = _clean_collection_name(collection_name)
        text_keys = text_keys or ["text", "content", "chunk"]
        embedding_keys = embedding_keys or ["embedding", "vector", "values"]

        if not os.path.exists(path):
            logger.error(f"Embeddings file not found: {path}")
            return False

        logger.info(f"Loading embeddings from {path} → collection '{col}'")
        batch_embeds: List[List[float]] = []
        batch_texts: List[str] = []
        batch_meta: List[Dict[str, Any]] = []
        total = 0

        for obj in _iter_jsonl(path):
            # Embedding
            emb = None
            for k in embedding_keys:
                if k in obj:
                    emb = obj[k]
                    break
            if not isinstance(emb, list):
                continue

            # Text
            txt = ""
            for k in text_keys:
                if k in obj and isinstance(obj[k], str):
                    txt = obj[k]
                    break

            # Metadata
            meta = {
                "document_id": document_id,
                "source": os.path.basename(path),
                "chunk_index": obj.get("chunk_index", 0),
                "path": obj.get("path", ""),
                **(obj.get("metadata", {}) or {}),
            }

            batch_embeds.append(emb)
            batch_texts.append(txt)
            batch_meta.append(meta)

            if len(batch_embeds) >= batch_size:
                ok = await self.store_document_embeddings(document_id, batch_embeds, batch_texts, batch_meta, col)
                if ok:
                    total += len(batch_embeds)
                batch_embeds.clear(); batch_texts.clear(); batch_meta.clear()

        if batch_embeds:
            ok = await self.store_document_embeddings(document_id, batch_embeds, batch_texts, batch_meta, col)
            if ok:
                total += len(batch_embeds)

        logger.info(f"Loaded {total} embeddings from {path}")
        return total > 0


# ---------------------- Singleton accessor (no eager connect) ----------------------
_vector_service_instance: Optional[VectorService] = None

def get_vector_service() -> Optional[VectorService]:
    global _vector_service_instance
    if _vector_service_instance is None:
        if DISABLE_VECTOR:
            logger.info("get_vector_service: vectors disabled (DISABLE_VECTOR=1)")
            return None
        _vector_service_instance = VectorService()  # no connect here
    return _vector_service_instance


# ---------------------------------- CLI ----------------------------------
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Vector Service CLI")
    parser.add_argument("--list", action="store_true", help="List collections")
    parser.add_argument("--load-jsonl", help="Load embeddings from JSONL")
    parser.add_argument("--collection", default=None, help="Collection name")
    parser.add_argument("--doc-id", default="cli_import", help="Document ID for import")
    parser.add_argument("--count", action="store_true", help="Count documents (heuristic)")
    parser.add_argument("--doc-ids", action="store_true", help="List unique document IDs (heuristic)")
    parser.add_argument("--create", help="Create a new collection")
    parser.add_argument("--drop", help="Drop a collection")
    parser.add_argument("--dimension", type=int, default=None, help="Embedding dimension for collection creation")
    args = parser.parse_args()

    async def main():
        vs = get_vector_service()
        if vs is None:
            print(json.dumps({"status": "disabled", "timestamp": time.time()}, indent=2))
            return
        vs.ensure_connected()

        if args.list:
            print("Collections:", await vs.list_collections()); return
        if args.count:
            print("Document count:", await vs.count_documents(args.collection)); return
        if args.doc_ids:
            print("Document IDs:", await vs.get_unique_document_ids(args.collection)); return
        if args.create:
            ok = await vs.create_collection(args.create, args.dimension)
            print(f"Collection '{args.create}' creation:", "Success" if ok else "Failed"); return
        if args.drop:
            ok = await vs.drop_collection(args.drop)
            print(f"Dropped '{args.drop}':", "Success" if ok else "Failed"); return
        if args.load_jsonl:
            path = args.load_jsonl
            ok = await vs.load_embeddings_from_jsonl(path=path, document_id=args.doc_id, collection_name=args.collection)
            print(f"JSONL load from {path}:", "Success" if ok else "Failed"); return

        print("Collections:", await vs.list_collections())

    asyncio.run(main())
