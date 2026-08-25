"""Pure validation and service bridges for dashboard VM controls."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from benchmarking.core.portfolio import MAX_BATCH_COMBINATIONS
from scripts.vm_adapter_manager import VMAdapterManager, verify_operator_token


class DashboardControlError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.public_message = message
        self.status = status


def adapter_control_enabled(environ: Mapping[str, str] | None = None) -> bool:
    source = os.environ if environ is None else environ
    return source.get("WNS_ENABLE_ADAPTER_CONTROL") == "1"


def require_operator_control(
    headers: Mapping[str, str] | Any,
    *,
    environ: Mapping[str, str] | None = None,
    verifier: Any | None = None,
) -> None:
    source = os.environ if environ is None else environ
    if not adapter_control_enabled(source):
        raise DashboardControlError(
            "control_disabled", "Adapter control is disabled", 403
        )
    expected = source.get("WNS_ADAPTER_CONTROL_TOKEN", "")
    authorization = str(headers.get("Authorization", ""))
    prefix = "Bearer "
    supplied = authorization[len(prefix) :] if authorization.startswith(prefix) else ""
    compare = verify_operator_token if verifier is None else verifier
    if not compare(expected, supplied):
        raise DashboardControlError("unauthorized", "Operator token required", 401)


def create_adapter_manager(root: Path) -> VMAdapterManager:
    return VMAdapterManager(root, root / "configs" / "vm_adapter_ports.json")


def adapter_status_payload(
    manager: Any, *, control_enabled: bool | None = None
) -> dict[str, Any]:
    slots = manager.statuses()
    return {
        "schema_version": 1,
        "control_enabled": (
            adapter_control_enabled() if control_enabled is None else control_enabled
        ),
        "slot_count": len(slots),
        "slots": slots,
    }


_EMBEDDING_SERVICES = {
    "jina_v3": "jina_embedding",
    "gte_multilingual_base": "gte_embedding",
    "nemotron_3_embed_1b_bf16": "nemotron_embed_1b_bf16",
    "nemotron_3_embed_1b_nvfp4": "nemotron_embed_1b_nvfp4",
    "nemotron_3_embed_8b_bf16": "nemotron_embed_8b_bf16",
}
_RERANKER_SERVICES = {
    "bge-reranker-base": "bge_reranker",
    "Qwen3:4B Rerank": "qwen_reranker",
    "nemotron_rerank_1b": "nemotron_reranker",
    "gte_modernbert_base": "gte_modernbert_reranker",
}
_PORTFOLIO_VECTOR_SERVICES = ("pgvector", "weaviate_http", "qdrant_http")


def required_service_ids_for_batch(batch: Any) -> tuple[str, ...]:
    if len(batch.combination_ids) > MAX_BATCH_COMBINATIONS:
        raise DashboardControlError(
            "batch_too_large",
            f"Batch exceeds the {MAX_BATCH_COMBINATIONS}-combination maximum",
            409,
        )
    service_ids: list[str] = []
    embedding_service = _EMBEDDING_SERVICES.get(batch.embedding)
    if embedding_service:
        service_ids.append(embedding_service)
    service_ids.extend(_PORTFOLIO_VECTOR_SERVICES)
    for reranker in batch.rerankers:
        service_id = _RERANKER_SERVICES.get(reranker)
        if service_id and service_id not in service_ids:
            service_ids.append(service_id)
    return tuple(service_ids)


def _exact_payload(payload: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != fields:
        raise DashboardControlError(
            "invalid_request", "Invalid adapter control request fields", 400
        )
    return payload


def adapter_control_action(
    action: str,
    payload: Any,
    *,
    manager: Any,
    plan: Any | None = None,
) -> dict[str, Any]:
    if action in {"start", "retry", "stop"}:
        request = _exact_payload(payload, {"service_id"})
        service_id = request["service_id"]
        if not isinstance(service_id, str) or not service_id:
            raise DashboardControlError("invalid_request", "Invalid service ID", 400)
        return getattr(manager, action)(service_id)
    if action == "start-required":
        request = _exact_payload(payload, {"batch_id"})
        batch_id = request["batch_id"]
        if not isinstance(batch_id, str) or not batch_id or plan is None:
            raise DashboardControlError("invalid_request", "Invalid batch ID", 400)
        batch = next((item for item in plan.batches if item.batch_id == batch_id), None)
        if batch is None:
            raise DashboardControlError("unknown_batch", "unknown batch ID", 404)
        required = required_service_ids_for_batch(batch)
        return {
            "schema_version": 1,
            "batch_id": batch.batch_id,
            "required_service_ids": list(required),
            "services": manager.start_required(required),
        }
    raise DashboardControlError("unknown_action", "Unknown adapter action", 404)
