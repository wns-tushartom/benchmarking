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


def _matrix_axes(config: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    try:
        matrix_cfg = config["matrix"]
    except (KeyError, TypeError) as exc:
        raise ValueError("config must contain a matrix object") from exc
    if not isinstance(matrix_cfg, dict):
        raise ValueError("config matrix must be an object")
    retrieval_methods = (
        matrix_cfg.get("retrieval_methods")
        or matrix_cfg.get("retrievers")
        or ["Cosine Similarity"]
    )
    return matrix_cfg, retrieval_methods


def _raw_matrix_rows(config: Dict[str, Any]) -> List[Dict[str, str]]:
    matrix_cfg, retrieval_methods = _matrix_axes(config)
    rows: List[Dict[str, str]] = []
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
            "chunker": chunker,
            "embedding": embedding,
            "vector_store": vector_store,
            "index_type": index_type,
            "retrieval_method": retrieval_method,
            "retriever": retrieval_method,
            "reranker": reranker,
            "evaluator": evaluator,
        })
    return rows


def _compatibility_mapping(config: Dict[str, Any]) -> Dict[str, List[str]] | None:
    compatibility = config.get("compatibility")
    if compatibility is None:
        return None
    if not isinstance(compatibility, dict):
        raise ValueError("compatibility must be an object")
    mapping = compatibility.get("vector_store_index_types")
    if not isinstance(mapping, dict):
        raise ValueError("compatibility.vector_store_index_types must be an object")

    matrix_cfg, _ = _matrix_axes(config)
    stores = matrix_cfg.get("vector_stores", [])
    indexes = set(matrix_cfg.get("index_types", ["HNSW"]))
    missing = [store for store in stores if store not in mapping]
    if missing:
        raise ValueError(
            "compatibility mapping is missing declared vector stores: "
            + ", ".join(missing)
        )

    normalized: Dict[str, List[str]] = {}
    for store in stores:
        allowed = mapping[store]
        if not isinstance(allowed, list) or not allowed or not all(
            isinstance(index, str) and index for index in allowed
        ):
            raise ValueError(f"compatibility indexes for {store} must be a non-empty list")
        unknown = [index for index in allowed if index not in indexes]
        if unknown:
            raise ValueError(
                f"compatibility indexes for {store} are not declared in matrix: "
                + ", ".join(unknown)
            )
        if len(set(allowed)) != len(allowed):
            raise ValueError(f"compatibility indexes for {store} contain duplicates")
        normalized[store] = list(allowed)
    return normalized


def generate_matrix_catalog(config: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """Return configured rows and explicit compatibility exclusions.

    Configs without a compatibility section retain the historical Cartesian
    behavior and have no exclusion records. Compatibility-aware rows receive
    stable canonical IDs; excluded attempts retain their axes and a stable
    reason code for auditability.
    """
    raw_rows = _raw_matrix_rows(config)
    compatibility = _compatibility_mapping(config)
    if compatibility is None:
        configured = []
        for run_id, row in enumerate(raw_rows, start=1):
            configured.append({"benchmark_run_id": f"mod_{run_id:04d}", **row})
        return {"configured": configured, "excluded": []}

    from benchmarking.core.portfolio import canonical_combination_id

    reason_code = config["compatibility"].get(
        "exclusion_reason_code", "incompatible_vector_store_index"
    )
    if not isinstance(reason_code, str) or not reason_code:
        raise ValueError("compatibility.exclusion_reason_code must be a non-empty string")

    configured: List[Dict[str, str]] = []
    excluded: List[Dict[str, str]] = []
    for row in raw_rows:
        identified = {**row, "combination_id": canonical_combination_id(row)}
        if row["index_type"] in compatibility[row["vector_store"]]:
            configured.append({
                "benchmark_run_id": f"mod_{len(configured) + 1:04d}",
                **identified,
                "state": "configured",
            })
        else:
            excluded.append({
                **identified,
                "state": "excluded",
                "reason_code": reason_code,
            })
    return {"configured": configured, "excluded": excluded}


def generate_matrix(config: Dict[str, Any]) -> List[Dict[str, str]]:
    return generate_matrix_catalog(config)["configured"]


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
