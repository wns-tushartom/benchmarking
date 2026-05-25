from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, List


def load_benchmark_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("YAML config requires PyYAML. Use .json for stdlib-only runs.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def config_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def dataset_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).encode("utf-8"))
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()[:16]


def generate_matrix(config: Dict[str, Any]) -> List[Dict[str, str]]:
    matrix_cfg = config["matrix"]
    retrieval_methods = matrix_cfg.get("retrieval_methods") or matrix_cfg.get("retrievers") or ["Cosine Similarity"]
    rows: List[Dict[str, str]] = []
    run_id = 1
    for chunker, embedding, vector_store, index_type, retrieval_method, reranker, evaluator in product(
        matrix_cfg.get("chunkers", []),
        matrix_cfg.get("embeddings", []),
        matrix_cfg.get("vector_stores", []),
        matrix_cfg.get("index_types", ["HNSW"]),
        retrieval_methods,
        matrix_cfg.get("rerankers", []),
        matrix_cfg.get("evaluators", ["overlap_relevance"]),
    ):
        rows.append({
            "benchmark_run_id": f"mod_{run_id:04d}",
            "chunker": chunker,
            "embedding": embedding,
            "vector_store": vector_store,
            "index_type": index_type,
            "retrieval_method": retrieval_method,
            "retriever": retrieval_method,
            "reranker": reranker,
            "evaluator": evaluator,
        })
        run_id += 1
    return rows


def selected_config(config: Dict[str, Any], selections: Dict[str, str]) -> Dict[str, Any]:
    """Return a config narrowed to a single user-selected combination.

    Unknown or empty values are ignored so callers can still run partial matrices.
    """
    out = json.loads(json.dumps(config))
    key_map = {
        "chunker": "chunkers",
        "embedding": "embeddings",
        "vector_store": "vector_stores",
        "index_type": "index_types",
        "retrieval_method": "retrieval_methods",
        "reranker": "rerankers",
        "evaluator": "evaluators",
    }
    for selection_key, matrix_key in key_map.items():
        value = selections.get(selection_key)
        if value and value != "all":
            out.setdefault("matrix", {})[matrix_key] = [value]
    return out


def technique(config: Dict[str, Any], kind: str, name: str) -> Dict[str, Any]:
    try:
        return config["techniques"][kind][name]
    except KeyError as exc:
        raise KeyError(f"Technique not configured: {kind}.{name}") from exc
