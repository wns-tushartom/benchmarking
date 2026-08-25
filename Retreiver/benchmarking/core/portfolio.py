from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


MAX_BATCH_COMBINATIONS = 250
COMBINATION_AXES = (
    "chunker",
    "embedding",
    "vector_store",
    "index_type",
    "retrieval_method",
    "reranker",
)
RERANKER_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "A": ("none", "bge-reranker-base", "Qwen3:4B Rerank"),
    "B": ("Amazon Rerank v1", "nemotron_rerank_1b", "gte_modernbert_base"),
})


class PortfolioState(str, Enum):
    CONFIGURED = "configured"
    EXCLUDED = "excluded"
    SELECTED = "selected"
    COMPLETED = "completed"
    FAILED = "failed"
    NOT_RUN = "not_run"


PORTFOLIO_STATES = frozenset(state.value for state in PortfolioState)


def canonical_combination_payload(combination: Mapping[str, Any]) -> str:
    """Serialize only identity axes in a fixed, dictionary-order-independent form."""
    if not isinstance(combination, Mapping):
        raise ValueError("combination must be a mapping")
    canonical: dict[str, str] = {}
    for axis in COMBINATION_AXES:
        value = combination.get(axis)
        if not isinstance(value, str) or not value:
            raise ValueError(f"combination axis {axis} must be a non-empty string")
        canonical[axis] = value
    return json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))


def canonical_combination_id(combination: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        canonical_combination_payload(combination).encode("utf-8")
    ).hexdigest()
    return f"combination_{digest}"


def compute_portfolio_hash(combinations: Iterable[Mapping[str, Any] | str]) -> str:
    """Hash a portfolio as a set of canonical IDs, independent of input ordering."""
    identifiers = []
    for combination in combinations:
        if isinstance(combination, str):
            identifier = combination
        else:
            identifier = canonical_combination_id(combination)
        if not identifier:
            raise ValueError("portfolio combination IDs must be non-empty strings")
        identifiers.append(identifier)
    if not identifiers:
        raise ValueError("portfolio cannot be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("portfolio contains duplicate combination IDs")
    payload = json.dumps(sorted(identifiers), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# A concise alias for callers that already operate on canonical rows or IDs.
portfolio_hash = compute_portfolio_hash
canonical_combination_serialization = canonical_combination_payload


def canonical_batch_id(portfolio_digest: str, embedding: str, reranker_group: str) -> str:
    if not all(
        isinstance(value, str) and value
        for value in (portfolio_digest, embedding, reranker_group)
    ):
        raise ValueError("batch identity parts must be non-empty strings")
    payload = json.dumps(
        {
            "embedding": embedding,
            "portfolio_hash": portfolio_digest,
            "reranker_group": reranker_group,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"batch_{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class PortfolioBatch:
    batch_id: str
    embedding: str
    reranker_group: str
    rerankers: tuple[str, ...]
    combination_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "embedding": self.embedding,
            "reranker_group": self.reranker_group,
            "rerankers": list(self.rerankers),
            "combination_ids": list(self.combination_ids),
            "combination_count": len(self.combination_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioBatch":
        try:
            return cls(
                batch_id=payload["batch_id"],
                embedding=payload["embedding"],
                reranker_group=payload["reranker_group"],
                rerankers=tuple(payload["rerankers"]),
                combination_ids=tuple(payload["combination_ids"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("malformed portfolio batch") from exc


@dataclass(frozen=True)
class PortfolioPlan:
    schema_version: int
    portfolio_hash: str
    max_combinations_per_batch: int
    combination_ids: tuple[str, ...]
    batches: tuple[PortfolioBatch, ...]

    @property
    def portfolio_id(self) -> str:
        return f"portfolio_{self.portfolio_hash}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "portfolio_id": self.portfolio_id,
            "portfolio_hash": self.portfolio_hash,
            "combination_count": len(self.combination_ids),
            "max_combinations_per_batch": self.max_combinations_per_batch,
            "combination_ids": list(self.combination_ids),
            "batch_count": len(self.batches),
            "batches": [batch.to_dict() for batch in self.batches],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PortfolioPlan":
        try:
            plan = cls(
                schema_version=payload["schema_version"],
                portfolio_hash=payload["portfolio_hash"],
                max_combinations_per_batch=payload["max_combinations_per_batch"],
                combination_ids=tuple(payload["combination_ids"]),
                batches=tuple(
                    PortfolioBatch.from_dict(batch) for batch in payload["batches"]
                ),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("malformed portfolio plan") from exc
        return validate_portfolio_plan(plan)

    @classmethod
    def from_json(cls, payload: str) -> "PortfolioPlan":
        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("malformed portfolio plan JSON") from exc
        if not isinstance(decoded, dict):
            raise ValueError("portfolio plan JSON must contain an object")
        return cls.from_dict(decoded)


def _positive_policy_cap(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max combinations per batch must be a positive integer")
    if value > MAX_BATCH_COMBINATIONS:
        raise ValueError(
            f"max combinations per batch cannot exceed {MAX_BATCH_COMBINATIONS}"
        )
    return value


def _validated_axis(matrix: Mapping[str, Any], name: str) -> tuple[str, ...]:
    values = matrix.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"matrix.{name} must be a non-empty list")
    if not all(isinstance(value, str) and value for value in values):
        raise ValueError(f"matrix.{name} values must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"matrix.{name} must not contain duplicates")
    return tuple(values)


def _validated_reranker_groups(
    groups: Mapping[str, Sequence[str]], rerankers: Sequence[str]
) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(groups, Mapping):
        raise ValueError("reranker groups must be a mapping")
    expected_names = tuple(RERANKER_GROUPS)
    if set(groups) != set(expected_names):
        unknown = sorted(set(groups) - set(expected_names))
        missing = sorted(set(expected_names) - set(groups))
        detail = []
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        if missing:
            detail.append("missing " + ", ".join(missing))
        raise ValueError("reranker group names are invalid: " + "; ".join(detail))

    normalized: dict[str, tuple[str, ...]] = {}
    all_members: list[str] = []
    for group_name in expected_names:
        members = groups[group_name]
        if isinstance(members, (str, bytes)) or not isinstance(members, Sequence):
            raise ValueError(f"reranker group {group_name} must be a sequence")
        normalized_members = tuple(members)
        if not normalized_members or not all(
            isinstance(member, str) and member for member in normalized_members
        ):
            raise ValueError(
                f"reranker group {group_name} must contain non-empty strings"
            )
        normalized[group_name] = normalized_members
        all_members.extend(normalized_members)

    if len(set(all_members)) != len(all_members):
        raise ValueError("duplicate reranker appears across reranker groups")
    if set(all_members) != set(rerankers):
        missing = sorted(set(rerankers) - set(all_members))
        unknown = sorted(set(all_members) - set(rerankers))
        raise ValueError(
            "reranker group coverage does not match matrix rerankers"
            f" (missing={missing}, unknown={unknown})"
        )
    return MappingProxyType(normalized)


def build_portfolio_plan(
    config: Mapping[str, Any],
    *,
    max_combinations_per_batch: int | None = None,
    reranker_groups: Mapping[str, Sequence[str]] | None = None,
) -> PortfolioPlan:
    """Build and validate the deterministic one-embedding/one-group batch plan."""
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    matrix = config.get("matrix")
    if not isinstance(matrix, Mapping):
        raise ValueError("config must contain a matrix object")

    # Validate all portfolio identity axes before asking the matrix generator to
    # expand them. This fails closed instead of quietly producing an empty plan.
    _validated_axis(matrix, "chunkers")
    embeddings = _validated_axis(matrix, "embeddings")
    _validated_axis(matrix, "vector_stores")
    _validated_axis(matrix, "index_types")
    _validated_axis(matrix, "retrieval_methods")
    rerankers = _validated_axis(matrix, "rerankers")

    portfolio_config = config.get("portfolio")
    if not isinstance(portfolio_config, Mapping):
        raise ValueError("config must contain a portfolio object")
    cap_value = (
        portfolio_config.get("max_combinations_per_batch")
        if max_combinations_per_batch is None
        else max_combinations_per_batch
    )
    cap = _positive_policy_cap(cap_value)
    group_value = (
        portfolio_config.get("reranker_groups")
        if reranker_groups is None
        else reranker_groups
    )
    groups = _validated_reranker_groups(group_value, rerankers)  # type: ignore[arg-type]

    # Local import avoids a module cycle: config imports the canonical ID helper
    # only when expanding a compatibility-aware catalog.
    from benchmarking.core.config import generate_matrix_catalog

    catalog = generate_matrix_catalog(dict(config))
    configured = catalog["configured"]
    if not configured:
        raise ValueError("portfolio compatibility produced no configured combinations")
    identifiers = tuple(row["combination_id"] for row in configured)
    digest = compute_portfolio_hash(identifiers)

    batches: list[PortfolioBatch] = []
    for embedding in embeddings:
        for group_name, group_rerankers in groups.items():
            batch_identifiers = tuple(
                row["combination_id"]
                for row in configured
                if row["embedding"] == embedding
                and row["reranker"] in group_rerankers
            )
            if not batch_identifiers:
                raise ValueError(
                    f"batch for embedding {embedding} and reranker group {group_name} is empty"
                )
            if len(batch_identifiers) > cap:
                raise ValueError(
                    f"batch for {embedding}/{group_name} has {len(batch_identifiers)} "
                    f"combinations and exceeds max {cap}"
                )
            batches.append(PortfolioBatch(
                batch_id=canonical_batch_id(digest, embedding, group_name),
                embedding=embedding,
                reranker_group=group_name,
                rerankers=group_rerankers,
                combination_ids=batch_identifiers,
            ))

    plan = PortfolioPlan(
        schema_version=1,
        portfolio_hash=digest,
        max_combinations_per_batch=cap,
        combination_ids=identifiers,
        batches=tuple(batches),
    )
    return validate_portfolio_plan(plan)


# Alternate verb used by command/API layers.
plan_portfolio = build_portfolio_plan


def validate_portfolio_plan(plan: PortfolioPlan) -> PortfolioPlan:
    if not isinstance(plan, PortfolioPlan):
        raise ValueError("plan must be a PortfolioPlan")
    if plan.schema_version != 1:
        raise ValueError("unsupported portfolio plan schema version")
    cap = _positive_policy_cap(plan.max_combinations_per_batch)
    if not plan.combination_ids:
        raise ValueError("portfolio plan has no combination coverage")
    if len(set(plan.combination_ids)) != len(plan.combination_ids):
        raise ValueError("portfolio plan combination coverage contains duplicates")
    if compute_portfolio_hash(plan.combination_ids) != plan.portfolio_hash:
        raise ValueError("portfolio hash does not match combination coverage")
    if not plan.batches:
        raise ValueError("portfolio plan has no batches")

    batch_ids = [batch.batch_id for batch in plan.batches]
    if len(set(batch_ids)) != len(batch_ids):
        raise ValueError("portfolio plan contains duplicate batch IDs")
    seen: list[str] = []
    for batch in plan.batches:
        if not batch.combination_ids:
            raise ValueError(f"batch {batch.batch_id} is empty")
        if len(batch.combination_ids) > cap:
            raise ValueError(f"batch {batch.batch_id} exceeds max {cap}")
        if len(set(batch.combination_ids)) != len(batch.combination_ids):
            raise ValueError(f"batch {batch.batch_id} contains duplicate combination IDs")
        expected_batch_id = canonical_batch_id(
            plan.portfolio_hash, batch.embedding, batch.reranker_group
        )
        if batch.batch_id != expected_batch_id:
            raise ValueError(f"batch ID does not match identity for {batch.embedding}")
        if not batch.rerankers:
            raise ValueError(f"batch {batch.batch_id} has no rerankers")
        seen.extend(batch.combination_ids)

    if len(seen) != len(set(seen)):
        raise ValueError("batch combination coverage contains duplicates")
    if set(seen) != set(plan.combination_ids):
        raise ValueError("batch combination coverage does not match portfolio coverage")
    return plan


def reconcile_run_states(
    plan: PortfolioPlan,
    *,
    selected_batch_id: str | None = None,
    receipts: Mapping[str, str | PortfolioState | Mapping[str, Any]] | None = None,
) -> Mapping[str, PortfolioState]:
    """Overlay selected/completed/failed state without mutating the immutable plan."""
    validate_portfolio_plan(plan)
    known_ids = set(plan.combination_ids)
    states = {
        combination_id: PortfolioState.NOT_RUN
        for combination_id in plan.combination_ids
    }

    if selected_batch_id is not None:
        selected = next(
            (batch for batch in plan.batches if batch.batch_id == selected_batch_id),
            None,
        )
        if selected is None:
            raise ValueError(f"unknown selected batch ID: {selected_batch_id}")
        for combination_id in selected.combination_ids:
            states[combination_id] = PortfolioState.SELECTED

    for combination_id, receipt in (receipts or {}).items():
        if combination_id not in known_ids:
            raise ValueError(f"receipt references unknown combination: {combination_id}")
        receipt_state: Any = receipt
        if isinstance(receipt, Mapping):
            receipt_state = receipt.get("state", receipt.get("status"))
        try:
            normalized = PortfolioState(receipt_state)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid receipt state for {combination_id}: {receipt_state!r}"
            ) from exc
        if normalized not in (PortfolioState.COMPLETED, PortfolioState.FAILED):
            raise ValueError(
                f"receipt state must be completed or failed, got {normalized.value}"
            )
        states[combination_id] = normalized

    return MappingProxyType(states)
