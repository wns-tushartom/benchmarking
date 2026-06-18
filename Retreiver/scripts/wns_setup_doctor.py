#!/usr/bin/env python3
"""Diagnose WNS VM setup state and print actionable missing items as JSON."""
from __future__ import annotations

import json
import os
import socket
import sys
import urllib.request
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wns_env import load_env_files as load_wns_env_files, service_base_from_endpoint
CONFIG_PATH = ROOT / "configs" / "benchmark.local.json"
WORKBOOK_PATH = ROOT / "data" / "chunking_methods_output_v2.xlsx"
BENCHMARK_INPUT_PATH = ROOT / "data" / "benchmark_input.csv"
PDF_AUDIT_PATH = ROOT / "data" / "pdf_extraction_audit.csv"
GROUNDTRUTH_DIR = ROOT / "data" / "groundtruth"


def load_env_files() -> None:
    load_wns_env_files(ROOT)


def http_check(name: str, url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return {"name": name, "url": url, "ok": 200 <= resp.status < 300, "note": f"HTTP {resp.status}"}
    except Exception as exc:
        return {"name": name, "url": url, "ok": False, "note": repr(exc)}


def tcp_check(name: str, host: str, port: int, timeout: float = 3.0) -> dict[str, Any]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return {"name": name, "target": f"{host}:{port}", "ok": True, "note": "TCP open"}
    except Exception as exc:
        return {"name": name, "target": f"{host}:{port}", "ok": False, "note": repr(exc)}


def pgvector_auth_check() -> dict[str, Any]:
    dsn = os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL")
    target = "PGVECTOR_DSN/DATABASE_URL"
    if not dsn:
        return {"name": "pgvector", "target": target, "ok": False, "note": "PGVECTOR_DSN or DATABASE_URL is not set"}
    try:
        import psycopg  # type: ignore[import-not-found]
    except Exception as exc:
        return {"name": "pgvector", "target": target, "ok": False, "note": f"psycopg missing: {exc!r}"}
    try:
        with psycopg.connect(dsn, connect_timeout=5) as conn:
            conn.execute("SELECT 1").fetchone()
        return {"name": "pgvector", "target": target, "ok": True, "note": "SELECT 1 OK"}
    except Exception as exc:
        return {"name": "pgvector", "target": target, "ok": False, "note": repr(exc)}


def workbook_sheets(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        with zipfile.ZipFile(path) as zf:
            wb = ET.fromstring(zf.read("xl/workbook.xml"))
            ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            return [s.attrib.get("name", "") for s in wb.findall("a:sheets/a:sheet", ns)]
    except Exception:
        return []


def configured_options() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        matrix = cfg.get("matrix", {})
        return {
            "chunkers": list(matrix.get("chunkers", [])),
            "embeddings": list(matrix.get("embeddings", [])),
            "vector_stores": list(matrix.get("vector_stores", [])),
            "rerankers": list(matrix.get("rerankers", [])),
        }
    except Exception as exc:
        return {"error": repr(exc)}


def env_state() -> dict[str, str]:
    keys = [
        "MODEL_ADAPTER_URL",
        "GTE_EMBEDDING_URL",
        "JINA_EMBEDDING_URL",
        "QDRANT_URL",
        "PGVECTOR_DSN",
        "DATABASE_URL",
        "WEAVIATE_URL",
        "NVIDIA_RAG_SERVER_URL",
        "NVIDIA_INGESTOR_URL",
    ]
    state = {}
    for key in keys:
        value = os.getenv(key, "")
        if key in {"PGVECTOR_DSN", "DATABASE_URL"} and value:
            value = value.split("://", 1)[0] + "://***"
        state[key] = "set" if value and key in {"PGVECTOR_DSN", "DATABASE_URL"} else value
    return state


def main() -> int:
    load_env_files()
    options = configured_options()
    required_sheets = options.get("chunkers", []) if isinstance(options, dict) else []
    available_sheets = workbook_sheets(WORKBOOK_PATH)
    missing_sheets = [s for s in required_sheets if s not in available_sheets]
    groundtruth = sorted(GROUNDTRUTH_DIR.glob("*.csv")) + sorted(GROUNDTRUTH_DIR.glob("*.xlsx"))

    model_base = service_base_from_endpoint(os.getenv("MODEL_ADAPTER_URL", "http://127.0.0.1:5000"))
    qdrant_base = os.getenv("QDRANT_URL", "http://127.0.0.1:5001").rstrip("/")
    weaviate_base = os.getenv("WEAVIATE_URL", "http://127.0.0.1:5004").rstrip("/")

    service_checks = [
        http_check("model_adapter", model_base + "/health"),
        http_check("qdrant", qdrant_base + "/healthz"),
        pgvector_auth_check(),
        http_check("weaviate", weaviate_base + "/v1/.well-known/ready"),
        http_check("dashboard", "http://127.0.0.1:5009"),
        http_check("nvidia_rag", os.getenv("NVIDIA_RAG_SERVER_URL", "http://127.0.0.1:5006").rstrip("/") + "/v1/health?check_dependencies=true"),
        http_check("nvidia_ingestor", os.getenv("NVIDIA_INGESTOR_URL", "http://127.0.0.1:5007").rstrip("/") + "/v1/health?check_dependencies=true"),
    ]

    missing: list[str] = []
    warnings: list[str] = []
    if not BENCHMARK_INPUT_PATH.exists():
        missing.append("data/benchmark_input.csv missing. Run scripts/run_chunking_pipeline.py --mode all after MinerU is installed.")
    if not WORKBOOK_PATH.exists():
        missing.append("data/chunking_methods_output_v2.xlsx missing. Run scripts/run_chunking_pipeline.py --mode chunk-only if benchmark_input.csv exists, else --mode all.")
    elif missing_sheets:
        warnings.append("Chunking workbook missing configured sheet(s): " + ", ".join(missing_sheets) + ". Run scripts/run_chunking_pipeline.py --mode chunk-only.")
    if not groundtruth:
        missing.append("No groundtruth CSV/XLSX found under data/groundtruth/.")
    if PDF_AUDIT_PATH.exists():
        audit_text = PDF_AUDIT_PATH.read_text(encoding="utf-8", errors="ignore")[:20000]
        if "PyPDF2_fallback" in audit_text or "text_only_review" in audit_text:
            warnings.append("PDF audit contains text-only fallback rows. Use MinerU/layout extraction before claiming image/table/formula audit quality.")
    else:
        warnings.append("data/pdf_extraction_audit.csv missing. Extraction audit not available.")
    required_services = {"model_adapter", "qdrant", "pgvector", "weaviate", "dashboard"}
    nvidia_required = os.getenv("NVIDIA_RAG_REQUIRED", "0").strip().lower() in {"1", "true", "yes"}
    if nvidia_required:
        required_services.update({"nvidia_rag", "nvidia_ingestor"})
    for check in service_checks:
        if not check["ok"] and check["name"] in required_services:
            missing.append(f"{check['name']} unhealthy: {check['note']}")
        elif not check["ok"]:
            warnings.append(f"{check['name']} unhealthy: {check['note']}")

    payload = {
        "ok": not missing,
        "missing": missing,
        "warnings": warnings,
        "env": env_state(),
        "options": options,
        "workbook": {
            "path": str(WORKBOOK_PATH.relative_to(ROOT)),
            "exists": WORKBOOK_PATH.exists(),
            "available_sheets": available_sheets,
            "missing_configured_sheets": missing_sheets,
        },
        "groundtruth_files": [str(p.relative_to(ROOT)) for p in groundtruth],
        "service_checks": service_checks,
        "next_commands": [
            "VM_IP=$(hostname -I | awk '{print $1}'); bash ../setup_all_on_vm.sh --host \"$VM_IP\" --force-env --prepare-data --chunk-mode chunk-only --skip-model-smoke",
            "python scripts/run_complete_pipeline.py --preflight-only --sheets entity_heuristic_w6 --embeddings gte_multilingual_base --stores Qdrant --groundtruth data/groundtruth/groundtruth_500.csv --top-k 10 --chunk-limit 0 --query-limit 0 --fresh-run",
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
