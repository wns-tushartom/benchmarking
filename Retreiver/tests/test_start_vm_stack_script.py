from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "start_vm_stack.sh"


def test_vm_stack_launcher_uses_live_split_port_layout_and_waits_for_readiness():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "set -Eeuo pipefail" in text
    assert 'CUDA_VISIBLE_DEVICES="$gpu"' in text
    assert "start_adapter 0 5000" in text
    assert "start_adapter 1 5001" in text
    assert "start_adapter 2 5002" in text
    assert "serve_benchmark_dashboard.py\" 5011 0.0.0.0" in text
    assert "http://127.0.0.1:5019/healthz" in text
    assert "http://127.0.0.1:5004/v1/.well-known/ready" in text
    assert "http://127.0.0.1:5011/api/results" in text
    assert "wait_http" in text
    assert "docker compose -f \"$ROOT/docker-compose.benchmark.yml\" up -d" in text


def test_vm_stack_launcher_preserves_current_endpoint_contract():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'BGE_RERANK_URL="http://127.0.0.1:5001/rerank/bge"' in text
    assert 'QWEN_RERANK_URL="http://127.0.0.1:5002/rerank/qwen"' in text
    assert 'QDRANT_URL="http://127.0.0.1:5019"' in text
    assert 'QDRANT_GRPC_URL="http://127.0.0.1:5020"' in text
    assert "5009" not in text
