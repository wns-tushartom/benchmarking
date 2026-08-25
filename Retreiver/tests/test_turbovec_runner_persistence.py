from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarking.core import runner


def test_turbovec_runner_parameters_persist_under_batch_output(tmp_path: Path):
    row = {
        "chunker": "Heading_sections_l2",
        "embedding": "gte_multilingual_base",
        "vector_store": "TurboVec",
        "index_type": "TurboQuant4bit",
        "retrieval_method": "Dense Cosine",
        "reranker": "none",
    }
    cfg = {
        "adapter": "turbovec",
        "execution_location": "in_process",
        "index_type": "TurboQuant4bit",
        "bits": 4,
    }

    params = runner.vector_store_parameters(cfg, row, tmp_path)

    identity = {
        "chunker": row["chunker"],
        "embedding": row["embedding"],
        "vector_store": row["vector_store"],
        "index_type": row["index_type"],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert params == {
        "execution_location": "in_process",
        "bits": 4,
        "index_type": "TurboQuant4bit",
        "index_dir": str(tmp_path / "vector_indexes" / f"index_{digest}"),
    }
    assert Path(params["index_dir"]).is_relative_to(tmp_path)


def test_non_turbovec_runner_parameters_remain_backward_compatible(tmp_path: Path):
    row = {"vector_store": "FAISS", "index_type": "HNSW"}
    cfg = {"adapter": "faiss", "execution_location": "vm_only", "index_type": "HNSW", "m": 16}
    assert runner.vector_store_parameters(cfg, row, tmp_path) == {
        "execution_location": "vm_only",
        "m": 16,
        "index_type": "HNSW",
    }
