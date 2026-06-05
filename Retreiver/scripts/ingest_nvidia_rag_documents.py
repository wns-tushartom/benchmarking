#!/usr/bin/env python3
"""Upload Project Smiley PDFs to NVIDIA RAG Blueprint ingestor.

This is intentionally batchable and fail-fast. It creates the collection if requested,
uploads files to /v1/documents, and polls /v1/status when a task_id is returned.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "nvidia_rag" / "ingestion_latest.json"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ")
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def http_json(url: str, method: str = "GET", payload: Any | None = None, timeout: float = 60.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return {"ok": 200 <= resp.status < 300, "status": resp.status, "body": json.loads(text) if text else {}}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": exc.read().decode("utf-8", errors="replace")[:2000]}
    except Exception as exc:
        return {"ok": False, "status": None, "error": str(exc)}


def discover_files(path: Path, limit: int) -> list[Path]:
    if path.is_file():
        return [path]
    exts = {".pdf", ".docx", ".txt", ".md", ".pptx"}
    files = [p for p in sorted(path.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    return files[:limit] if limit > 0 else files


def encode_multipart(files: list[Path], data_payload: dict[str, Any]) -> tuple[bytes, str]:
    boundary = f"project-smiley-{int(time.time() * 1000)}"
    parts: list[bytes] = []
    for path in files:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="documents"; filename="{path.name}"\r\n'.encode())
        parts.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        parts.append(path.read_bytes())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="data"\r\n')
    parts.append(b"Content-Type: application/json\r\n\r\n")
    parts.append(json.dumps(data_payload).encode("utf-8"))
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary


def upload_batch(base: str, files: list[Path], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body, boundary = encode_multipart(files, payload)
    req = urllib.request.Request(
        f"{base}/v1/documents",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return {"ok": 200 <= resp.status < 300, "status": resp.status, "body": json.loads(text) if text else {}}
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "error": exc.read().decode("utf-8", errors="replace")[:4000]}
    except Exception as exc:
        return {"ok": False, "status": None, "error": str(exc)}


def poll_status(base: str, task_id: str, timeout_seconds: int) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    last: dict[str, Any] = {}
    while time.time() < deadline:
        url = f"{base}/v1/status?{urllib.parse.urlencode({'task_id': task_id})}"
        last = http_json(url, timeout=30)
        body = last.get("body") or {}
        status = str(body.get("status") or body.get("state") or "").upper()
        if status in {"FINISHED", "SUCCESS", "COMPLETED", "FAILED", "ERROR", "UNKNOWN"}:
            return last
        time.sleep(5)
    last["timed_out"] = True
    return last


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--url", default="", help="Base NVIDIA ingestor URL. Defaults to NVIDIA_INGESTOR_URL or http://127.0.0.1:5007")
    parser.add_argument("--path", default=str(ROOT / "data" / "pdfs"))
    parser.add_argument("--collection", default=os.getenv("NVIDIA_RAG_COLLECTION", "multimodal_data"))
    parser.add_argument("--limit", type=int, default=5, help="0 means all files")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--create-collection", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=int(os.getenv("NVIDIA_RAG_EMBEDDING_DIMENSIONS", "2048")))
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    parser.add_argument("--generate-summary", action="store_true")
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--poll-timeout", type=int, default=21600)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    load_env_file(Path(args.env_file))
    load_env_file(ROOT / ".env.vm.generated")
    base = (args.url or os.getenv("NVIDIA_INGESTOR_URL", "http://127.0.0.1:5007")).rstrip("/")
    files = discover_files(Path(args.path), args.limit)
    result: dict[str, Any] = {"ok": True, "base_url": base, "collection": args.collection, "files": [str(p) for p in files], "batches": [], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not files:
        result.update({"ok": False, "error": f"No ingestible files found under {args.path}"})
    if args.create_collection:
        result["create_collection"] = http_json(f"{base}/v1/collection", method="POST", payload={"collection_name": args.collection, "embedding_dimension": args.embedding_dim, "metadata_schema": []})
        result["ok"] = result["ok"] and result["create_collection"].get("ok", False)
    for idx in range(0, len(files), args.batch_size):
        batch = files[idx: idx + args.batch_size]
        upload_payload = {"collection_name": args.collection, "blocking": False, "split_options": {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap}, "custom_metadata": [], "generate_summary": args.generate_summary}
        uploaded = upload_batch(base, batch, upload_payload, timeout=600)
        batch_result = {"batch": idx // args.batch_size + 1, "files": [p.name for p in batch], "upload": uploaded}
        body = uploaded.get("body") or {}
        task_id = body.get("task_id") or body.get("id") or body.get("taskId")
        if args.poll and task_id:
            batch_result["status"] = poll_status(base, str(task_id), args.poll_timeout)
        result["batches"].append(batch_result)
        result["ok"] = result["ok"] and uploaded.get("ok", False)
        if not uploaded.get("ok"):
            break
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"ok": result["ok"], "files": len(files), "batches": len(result["batches"]), "out": str(out.relative_to(ROOT)), "error": result.get("error", "")}, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
