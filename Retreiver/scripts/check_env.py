#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"

GROUPS = {
    "Embeddings": {
        "OPENAI_API_KEY": "OpenAI text-embedding-3-large, required only for OpenAI embedding",
        "JINA_EMBEDDING_URL": "VM/remote endpoint for self-hosted Jina v3 embedding",
        "JINA_API_KEY": "Optional, only if using hosted/authenticated Jina endpoint",
        "GTE_EMBEDDING_URL": "VM/remote endpoint for self-hosted GTE multilingual embedding",
        "HF_TOKEN": "Optional, only if VM/Hugging Face endpoint requires auth",
    },
    "Vector DBs": {
        "QDRANT_URL": "Qdrant endpoint",
        "QDRANT_API_KEY": "Optional, Qdrant Cloud/auth",
        "PGVECTOR_DSN": "PGVector/Postgres DSN",
        "WEAVIATE_URL": "Weaviate endpoint",
        "WEAVIATE_API_KEY": "Optional, auth-enabled Weaviate",
    },
    "Rerankers": {
        "AWS_ACCESS_KEY_ID": "Amazon Rerank v1 / Bedrock",
        "AWS_SECRET_ACCESS_KEY": "Amazon Rerank v1 / Bedrock",
        "AWS_REGION": "Amazon Rerank v1 / Bedrock region",
        "AWS_DEFAULT_REGION": "Optional AWS region alias",
        "AMAZON_RERANK_MODEL_ID": "Exact Bedrock rerank model id",
        "BGE_RERANKER_MODEL": "Local BGE reranker model name",
        "QWEN_RERANK_URL": "VM/remote Qwen reranker endpoint",
        "QWEN_API_KEY": "Optional, if Qwen endpoint requires auth",
    },
}


def parse_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def is_set(key: str, env_file_values: dict[str, str]) -> bool:
    return bool(os.environ.get(key) or env_file_values.get(key))


def main() -> None:
    values = parse_env(ENV_FILE)
    print(f"Env file: {ENV_FILE} ({'found' if ENV_FILE.exists() else 'missing'})")
    for group, keys in GROUPS.items():
        print(f"\n{group}")
        for key, note in keys.items():
            mark = "OK" if is_set(key, values) else "MISSING"
            optional = key in {"JINA_API_KEY", "HF_TOKEN", "QDRANT_API_KEY", "WEAVIATE_API_KEY", "AWS_DEFAULT_REGION", "QWEN_API_KEY"}
            if optional and mark == "MISSING":
                mark = "optional"
            print(f"- {key}: {mark} - {note}")

    print("\nRecommended first real slice needs:")
    for key in ["JINA_EMBEDDING_URL", "GTE_EMBEDDING_URL", "QDRANT_URL"]:
        print(f"- {key}: {'OK' if is_set(key, values) else 'MISSING'}")
    print("\nEmbedding note: keep JINA_EMBEDDING_URL and GTE_EMBEDDING_URL as VM/remote endpoints. Do not load Jina/GTE locally on the WNS laptop CPU.")
    print("Qwen note: keep QWEN_RERANK_URL as a VM/remote endpoint. Do not load Qwen3:4B locally on the work laptop unless intentionally provisioned.")


if __name__ == "__main__":
    main()
