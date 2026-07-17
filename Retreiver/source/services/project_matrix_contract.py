"""Strict selected-matrix requests and request-bound confirmation tokens."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import stat
import time
from typing import Any, Mapping

from source.services.project_workspace import validate_resource_id


SCHEMA_VERSION = 1
WARNING_COMBINATION_THRESHOLD = 50
CONFIRMATION_COMBINATION_THRESHOLD = 100
HARD_MAX_COMBINATIONS = 180
DEFAULT_CONFIRMATION_TTL_SECONDS = 300
MAX_CONFIRMATION_TTL_SECONDS = 300
_DIMENSIONS = ("chunkers", "embeddings", "vector_stores", "rerankers")
_REQUEST_KEYS = {
    "schema_version",
    "project_id",
    "top_k",
    "questions_source",
    "selections",
    "large_matrix_confirmation",
}
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NON_PRODUCTION_RE = re.compile(r"(?:^|[^a-z0-9])(none|fallback|local)(?:$|[^a-z0-9])")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOKEN_PURPOSE = "project_matrix_large_confirmation"
_TOKEN_PREFIX = "pmc1"
_TOKEN_KEYS = {
    "schema_version",
    "purpose",
    "request_fingerprint",
    "issued_at",
    "expires_at",
    "nonce",
}


class ProjectMatrixValidationError(ValueError):
    """A selected project matrix does not satisfy the product contract."""


class ConfirmationTokenError(ProjectMatrixValidationError):
    """A large-matrix confirmation token is absent or invalid."""


@dataclass(frozen=True)
class ProjectMatrixRequest:
    """Deeply immutable semantic fields used by one selected matrix request."""

    schema_version: int
    project_id: str
    top_k: int
    questions_source_type: str
    typed_query: str | None
    question_set_id: str | None
    question_set_content_sha256: str | None
    chunkers: tuple[str, ...]
    embeddings: tuple[str, ...]
    vector_stores: tuple[str, ...]
    rerankers: tuple[str, ...]
    large_matrix_confirmation: str | None = None

    @property
    def questions_source(self) -> dict[str, str]:
        if self.questions_source_type == "typed":
            assert self.typed_query is not None
            return {"type": "typed", "query": self.typed_query}
        assert self.question_set_id is not None
        assert self.question_set_content_sha256 is not None
        return {
            "type": "question_set",
            "question_set_id": self.question_set_id,
            "content_sha256": self.question_set_content_sha256,
        }

    @property
    def selections(self) -> dict[str, list[str]]:
        return {
            "chunkers": list(self.chunkers),
            "embeddings": list(self.embeddings),
            "vector_stores": list(self.vector_stores),
            "rerankers": list(self.rerankers),
        }

    def fingerprint_payload(self) -> dict[str, Any]:
        """Return the exact semantic request fields covered by the fingerprint."""
        return {
            "schema_version": self.schema_version,
            "project_id": self.project_id,
            "top_k": self.top_k,
            "questions_source": self.questions_source,
            "selections": self.selections,
        }

    def to_dict(self, *, include_confirmation: bool = True) -> dict[str, Any]:
        payload = self.fingerprint_payload()
        if include_confirmation:
            payload["large_matrix_confirmation"] = self.large_matrix_confirmation
        return payload


@dataclass(frozen=True)
class ValidatedMatrix:
    """Validated request plus exact matrix policy results."""

    request: ProjectMatrixRequest
    combination_count: int
    request_fingerprint: str
    warning: str | None
    confirmation_required: bool
    confirmation_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.request.to_dict(),
            "request_fingerprint": self.request_fingerprint,
            "combination_count": self.combination_count,
            "warning": self.warning,
            "confirmation_required": self.confirmation_required,
            "confirmation_verified": self.confirmation_verified,
        }


def _strict_json_load(path: Path) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProjectMatrixValidationError("catalog JSON contains duplicate keys")
            result[key] = value
        return result

    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ProjectMatrixValidationError("matrix catalog must be a regular file")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    except ProjectMatrixValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ProjectMatrixValidationError("matrix catalog is unreadable or invalid") from None


def _looks_non_production(value: str) -> bool:
    return _NON_PRODUCTION_RE.search(value.casefold()) is not None


def _validate_catalog(catalog_path: Path) -> dict[str, tuple[str, ...]]:
    catalog = _strict_json_load(catalog_path)
    if not isinstance(catalog, dict) or set(catalog) != {
        "schema_version",
        "matrix",
        "techniques",
    }:
        raise ProjectMatrixValidationError("matrix catalog schema is invalid")
    schema_version = catalog["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise ProjectMatrixValidationError("matrix catalog schema_version is invalid")
    matrix = catalog["matrix"]
    techniques = catalog["techniques"]
    if (
        not isinstance(matrix, dict)
        or set(matrix) != set(_DIMENSIONS)
        or not isinstance(techniques, dict)
        or set(techniques) != set(_DIMENSIONS)
    ):
        raise ProjectMatrixValidationError("matrix catalog schema is invalid")

    canonical: dict[str, tuple[str, ...]] = {}
    catalog_product = 1
    for dimension in _DIMENSIONS:
        values = matrix[dimension]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
            or len(values) != len(set(values))
        ):
            raise ProjectMatrixValidationError("matrix catalog schema is invalid")
        if any(_looks_non_production(value) for value in values):
            raise ProjectMatrixValidationError(
                "matrix catalog contains a non-production technique"
            )
        catalog_product *= len(values)
        canonical[dimension] = tuple(values)

    if catalog_product > HARD_MAX_COMBINATIONS:
        raise ProjectMatrixValidationError(
            f"matrix catalog exceeds the hard cap of {HARD_MAX_COMBINATIONS}"
        )

    for dimension in _DIMENSIONS:
        dimension_techniques = techniques[dimension]
        if (
            not isinstance(dimension_techniques, dict)
            or set(dimension_techniques) != set(canonical[dimension])
        ):
            raise ProjectMatrixValidationError("matrix catalog technique schema is invalid")
        for technique_id, configuration in dimension_techniques.items():
            if not isinstance(configuration, dict) or not configuration:
                raise ProjectMatrixValidationError(
                    "matrix catalog technique schema is invalid"
                )
            adapter = configuration.get("adapter")
            if adapter is not None and (
                not isinstance(adapter, str)
                or not adapter
                or _looks_non_production(adapter)
            ):
                raise ProjectMatrixValidationError(
                    f"matrix catalog technique {technique_id!r} is not production-only"
                )
    return canonical


def _validate_project_id(value: Any) -> str:
    try:
        return validate_resource_id(value)
    except (TypeError, ValueError):
        raise ProjectMatrixValidationError("project_id is invalid") from None


def _validate_questions_source(value: Any) -> tuple[str, str | None, str | None, str | None]:
    if not isinstance(value, dict):
        raise ProjectMatrixValidationError("questions_source schema is invalid")
    source_type = value.get("type")
    if source_type == "typed":
        if set(value) != {"type", "query"}:
            raise ProjectMatrixValidationError("questions_source schema is invalid")
        query = value["query"]
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > 4_000:
            raise ProjectMatrixValidationError("questions_source typed query is invalid")
        return "typed", query.strip(), None, None
    if source_type == "question_set":
        if set(value) != {"type", "question_set_id", "content_sha256"}:
            raise ProjectMatrixValidationError("questions_source schema is invalid")
        question_set_id = value["question_set_id"]
        content_sha256 = value["content_sha256"]
        try:
            validated_id = validate_resource_id(question_set_id)
        except (TypeError, ValueError):
            raise ProjectMatrixValidationError("questions_source question_set_id is invalid") from None
        if not validated_id.startswith("questions_") or not isinstance(
            content_sha256, str
        ) or _HEX_SHA256_RE.fullmatch(content_sha256) is None:
            raise ProjectMatrixValidationError("questions_source question set is invalid")
        return "question_set", None, validated_id, content_sha256
    raise ProjectMatrixValidationError("questions_source type is invalid")


def _validate_selections(
    value: Any, canonical: Mapping[str, tuple[str, ...]]
) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict) or set(value) != set(_DIMENSIONS):
        raise ProjectMatrixValidationError("selection schema is invalid")
    selections: dict[str, tuple[str, ...]] = {}
    for dimension in _DIMENSIONS:
        selected = value[dimension]
        if not isinstance(selected, list):
            raise ProjectMatrixValidationError(f"{dimension} selection must be an array")
        if not selected and dimension != "rerankers":
            raise ProjectMatrixValidationError(
                f"{dimension} selection must be a non-empty array"
            )
        if any(not isinstance(item, str) for item in selected):
            raise ProjectMatrixValidationError(
                f"{dimension} selection must contain canonical production IDs"
            )
        if len(selected) != len(set(selected)):
            raise ProjectMatrixValidationError(
                f"{dimension} selection contains a duplicate ID"
            )
        allowed = set(canonical[dimension])
        if any(item not in allowed or _looks_non_production(item) for item in selected):
            raise ProjectMatrixValidationError(
                f"{dimension} selection must contain canonical production IDs only"
            )
        selections[dimension] = tuple(selected)
    return selections


def _parse_request(
    payload: Any, canonical: Mapping[str, tuple[str, ...]]
) -> ProjectMatrixRequest:
    if not isinstance(payload, dict) or set(payload) != _REQUEST_KEYS:
        raise ProjectMatrixValidationError("request schema is invalid")
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise ProjectMatrixValidationError("schema_version must be integer 1")
    top_k = payload["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ProjectMatrixValidationError("top_k must be a positive integer")
    confirmation = payload["large_matrix_confirmation"]
    if confirmation is not None and (
        not isinstance(confirmation, str) or not confirmation or len(confirmation) > 4_096
    ):
        raise ProjectMatrixValidationError(
            "large_matrix_confirmation must be null or a bounded token"
        )
    source_type, typed_query, question_set_id, content_sha256 = (
        _validate_questions_source(payload["questions_source"])
    )
    selections = _validate_selections(payload["selections"], canonical)
    return ProjectMatrixRequest(
        schema_version=schema_version,
        project_id=_validate_project_id(payload["project_id"]),
        top_k=top_k,
        questions_source_type=source_type,
        typed_query=typed_query,
        question_set_id=question_set_id,
        question_set_content_sha256=content_sha256,
        chunkers=selections["chunkers"],
        embeddings=selections["embeddings"],
        vector_stores=selections["vector_stores"],
        rerankers=selections["rerankers"],
        large_matrix_confirmation=confirmation,
    )


def _combination_count(request: ProjectMatrixRequest) -> int:
    return (
        len(request.chunkers)
        * len(request.embeddings)
        * len(request.vector_stores)
        * max(1, len(request.rerankers))
    )


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_request_fingerprint(request: ProjectMatrixRequest) -> str:
    """SHA-256 of the canonical JSON semantic request, excluding its token."""
    if not isinstance(request, ProjectMatrixRequest):
        raise ProjectMatrixValidationError("request must be a ProjectMatrixRequest")
    return hashlib.sha256(_canonical_json(request.fingerprint_payload())).hexdigest()


def _validated_policy(
    request: ProjectMatrixRequest, *, confirmation_verified: bool
) -> ValidatedMatrix:
    count = _combination_count(request)
    if count > HARD_MAX_COMBINATIONS:
        raise ProjectMatrixValidationError(
            f"selected matrix exceeds the hard cap of {HARD_MAX_COMBINATIONS}"
        )
    warning = (
        f"Large matrix selection: {count} combinations"
        if count > WARNING_COMBINATION_THRESHOLD
        else None
    )
    confirmation_required = count > CONFIRMATION_COMBINATION_THRESHOLD
    return ValidatedMatrix(
        request=request,
        combination_count=count,
        request_fingerprint=canonical_request_fingerprint(request),
        warning=warning,
        confirmation_required=confirmation_required,
        confirmation_verified=confirmation_verified,
    )


def _coerce_now(now: int | None) -> int:
    if now is None:
        return int(time.time())
    if isinstance(now, bool) or not isinstance(now, int) or now < 0:
        raise ConfirmationTokenError("confirmation time is invalid")
    return now


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_confirmation_key() -> bytes:
    configured = os.environ.get("PROJECT_MATRIX_CONFIRMATION_KEY_FILE")
    if not configured:
        raise ConfirmationTokenError("confirmation key file is not configured")
    key_path = Path(configured)
    if not key_path.is_absolute():
        raise ConfirmationTokenError("confirmation key file path must be absolute")
    try:
        metadata = key_path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ConfirmationTokenError("confirmation key file must be a regular file")
        for parent in key_path.parents:
            parent_metadata = parent.lstat()
            if stat.S_ISLNK(parent_metadata.st_mode):
                raise ConfirmationTokenError("confirmation key file path must not use symlinks")
        resolved = key_path.resolve(strict=True)
        if _is_relative_to(resolved, _REPO_ROOT):
            raise ConfirmationTokenError("confirmation key file must be outside the repository")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(key_path, flags)
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
            ):
                raise ConfirmationTokenError("confirmation key file changed during loading")
            if opened.st_uid != os.geteuid():
                raise ConfirmationTokenError(
                    "confirmation key file must be owned by the service user"
                )
            if stat.S_IMODE(opened.st_mode) != 0o600:
                raise ConfirmationTokenError("confirmation key file mode must be 0600")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, 4_097 - total)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > 4_096:
                    raise ConfirmationTokenError("confirmation key file has invalid length")
            key = b"".join(chunks)
        finally:
            os.close(descriptor)
    except ConfirmationTokenError:
        raise
    except OSError:
        raise ConfirmationTokenError("confirmation key file cannot be loaded safely") from None
    if len(key) < 32:
        raise ConfirmationTokenError("confirmation key file has invalid length")
    return key


def _b64url_encode(content: bytes) -> str:
    return base64.urlsafe_b64encode(content).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str, *, signature: bool = False) -> bytes:
    try:
        if not value or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
            raise ValueError
        decoded = base64.b64decode(
            value + "=" * (-len(value) % 4), altchars=b"-_", validate=True
        )
        if _b64url_encode(decoded) != value:
            raise ValueError
        return decoded
    except (ValueError, UnicodeError):
        message = (
            "confirmation token signature is invalid"
            if signature
            else "confirmation token is malformed"
        )
        raise ConfirmationTokenError(message) from None


def _issue_token(fingerprint: str, *, now: int, ttl_seconds: int) -> str:
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, int)
        or ttl_seconds < 1
        or ttl_seconds > MAX_CONFIRMATION_TTL_SECONDS
    ):
        raise ConfirmationTokenError("confirmation token TTL is invalid")
    token_payload = {
        "schema_version": SCHEMA_VERSION,
        "purpose": _TOKEN_PURPOSE,
        "request_fingerprint": fingerprint,
        "issued_at": now,
        "expires_at": now + ttl_seconds,
        "nonce": secrets.token_hex(32),
    }
    encoded_payload = _canonical_json(token_payload)
    signature = hmac.new(_read_confirmation_key(), encoded_payload, hashlib.sha256).digest()
    return f"{_TOKEN_PREFIX}.{_b64url_encode(encoded_payload)}.{_b64url_encode(signature)}"


def _verify_token(token: str, fingerprint: str, *, now: int) -> None:
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != _TOKEN_PREFIX:
        raise ConfirmationTokenError("confirmation token is malformed")
    encoded_payload = _b64url_decode(parts[1])
    supplied_signature = _b64url_decode(parts[2], signature=True)
    expected_signature = hmac.new(
        _read_confirmation_key(), encoded_payload, hashlib.sha256
    ).digest()
    if not hmac.compare_digest(supplied_signature, expected_signature):
        raise ConfirmationTokenError("confirmation token signature is invalid")
    try:
        payload = json.loads(encoded_payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ConfirmationTokenError("confirmation token payload is invalid") from None
    if not isinstance(payload, dict) or set(payload) != _TOKEN_KEYS:
        raise ConfirmationTokenError("confirmation token payload is invalid")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or isinstance(payload["schema_version"], bool)
        or payload["purpose"] != _TOKEN_PURPOSE
        or not isinstance(payload["request_fingerprint"], str)
        or _HEX_SHA256_RE.fullmatch(payload["request_fingerprint"]) is None
        or not isinstance(payload["nonce"], str)
        or _HEX_SHA256_RE.fullmatch(payload["nonce"]) is None
    ):
        raise ConfirmationTokenError("confirmation token payload is invalid")
    issued_at = payload["issued_at"]
    expires_at = payload["expires_at"]
    if (
        isinstance(issued_at, bool)
        or not isinstance(issued_at, int)
        or isinstance(expires_at, bool)
        or not isinstance(expires_at, int)
        or issued_at < 0
        or expires_at <= issued_at
        or expires_at - issued_at > MAX_CONFIRMATION_TTL_SECONDS
        or issued_at > now
    ):
        raise ConfirmationTokenError("confirmation token time bounds are invalid")
    if now >= expires_at:
        raise ConfirmationTokenError("confirmation token has expired")
    if not hmac.compare_digest(payload["request_fingerprint"], fingerprint):
        raise ConfirmationTokenError("confirmation token does not match this request")


def issue_large_matrix_confirmation(
    payload: Any,
    *,
    catalog_path: str | Path = _REPO_ROOT / "configs" / "project_matrix_catalog.json",
    now: int | None = None,
    ttl_seconds: int = DEFAULT_CONFIRMATION_TTL_SECONDS,
) -> str:
    """Issue one short-lived token for a valid request over 100 combinations."""
    canonical = _validate_catalog(Path(catalog_path))
    request = _parse_request(payload, canonical)
    preliminary = _validated_policy(request, confirmation_verified=False)
    if not preliminary.confirmation_required:
        raise ConfirmationTokenError(
            "large-matrix confirmation is only available above 100 combinations"
        )
    if request.large_matrix_confirmation is not None:
        raise ConfirmationTokenError("request already contains a confirmation token")
    return _issue_token(
        preliminary.request_fingerprint,
        now=_coerce_now(now),
        ttl_seconds=ttl_seconds,
    )


def validate_project_matrix_request(
    payload: Any,
    *,
    catalog_path: str | Path = _REPO_ROOT / "configs" / "project_matrix_catalog.json",
    now: int | None = None,
) -> ValidatedMatrix:
    """Validate an exact JSON-like request and enforce all matrix-size policy."""
    canonical = _validate_catalog(Path(catalog_path))
    request = _parse_request(payload, canonical)
    preliminary = _validated_policy(request, confirmation_verified=False)
    token = request.large_matrix_confirmation
    if preliminary.confirmation_required:
        if token is None:
            raise ConfirmationTokenError(
                "large-matrix confirmation is required above 100 combinations"
            )
        _verify_token(token, preliminary.request_fingerprint, now=_coerce_now(now))
        return _validated_policy(request, confirmation_verified=True)
    if token is not None:
        raise ConfirmationTokenError(
            "large-matrix confirmation is not valid at 100 combinations or fewer"
        )
    return preliminary


# Concise alias for callers that already use the project-matrix namespace.
validate_matrix_request = validate_project_matrix_request
