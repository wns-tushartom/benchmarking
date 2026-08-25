from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start_vm_stack.sh"
COMPOSE = ROOT / "docker-compose.benchmark.yml"
ENV_EXAMPLE = ROOT / ".env.example"


def test_vm_stack_launcher_uses_allowlisted_manager_and_waits_for_readiness():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in text
    assert "vm_adapter_manager.py" in text
    assert 'start_service "jina_embedding"' in text
    assert 'start_service "gte_embedding"' in text
    assert 'start_service "bge_reranker"' in text
    assert 'start_service "qwen_reranker"' in text
    assert 'start_service "pgvector"' in text
    assert 'start_service "weaviate_http"' in text
    assert 'start_service "qdrant_http"' in text
    assert "serve_benchmark_dashboard.py\" 5011 0.0.0.0" in text
    assert "http://127.0.0.1:5019/healthz" in text
    assert "http://127.0.0.1:5004/v1/.well-known/ready" in text
    assert "http://127.0.0.1:5011/api/results" in text
    assert "wait_http" in text


def test_vm_stack_launcher_never_kills_unknown_port_occupants():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "fuser -k" not in text
    assert "kill \"$pid\"" not in text
    assert "stop_pidfile" not in text
    assert "Port 5011 is occupied" in text


def test_vm_stack_launcher_writes_approved_endpoint_contract():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'JINA_EMBEDDING_URL="http://127.0.0.1:5000/embed/jina"' in text
    assert 'GTE_EMBEDDING_URL="http://127.0.0.1:5001/embed/gte"' in text
    assert 'BGE_RERANK_URL="http://127.0.0.1:5002/rerank/bge"' in text
    assert 'QWEN_RERANK_URL="http://127.0.0.1:5006/rerank/qwen"' in text
    assert 'NEMOTRON_RERANK_URL="http://127.0.0.1:5007/rerank/nemotron"' in text
    assert 'GTE_MODERNBERT_RERANK_URL="http://127.0.0.1:5008/rerank/gte-modernbert"' in text
    assert 'NEMOTRON_3_EMBED_1B_BF16_URL="http://127.0.0.1:5010/v1/embeddings"' in text
    assert 'NEMOTRON_3_EMBED_1B_NVFP4_URL="http://127.0.0.1:5012/v1/embeddings"' in text
    assert 'NEMOTRON_3_EMBED_8B_BF16_URL="http://127.0.0.1:5013/v1/embeddings"' in text
    assert 'QDRANT_URL="http://127.0.0.1:5019"' in text
    assert 'QDRANT_GRPC_URL="http://127.0.0.1:5015"' in text
    assert "5020" not in text


def test_compose_and_environment_examples_use_the_same_twenty_port_map():
    compose = COMPOSE.read_text(encoding="utf-8")
    env = ENV_EXAMPLE.read_text(encoding="utf-8")
    assert '"5015:6334"' in compose
    assert "5020" not in compose
    assert "QDRANT_GRPC_URL=http://127.0.0.1:5015" in env
    assert "GTE_EMBEDDING_URL=http://VM_HOST:5001/embed/gte" in env
    assert "BGE_RERANK_URL=http://VM_HOST:5002/rerank/bge" in env
    assert "QWEN_RERANK_URL=http://VM_HOST:5006/rerank/qwen" in env
    assert "NVIDIA_RAG_SERVER_URL=http://VM_HOST:5016" in env
    assert "NVIDIA_INGESTOR_URL=http://VM_HOST:5017" in env
    assert "NVIDIA_RAG_FRONTEND_URL=http://VM_HOST:5018" in env
