from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
import os
from pathlib import Path

import pytest

from source.services import project_matrix_contract
from source.services.project_matrix_contract import (
    ConfirmationTokenError,
    ProjectMatrixValidationError,
    canonical_request_fingerprint,
    issue_large_matrix_confirmation,
    validate_project_matrix_request,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "configs" / "project_matrix_catalog.json"
UUID_HEX = "123e4567e89b42d3a456426614174000"
PROJECT_ID = f"refund-data_{UUID_HEX}"
QUESTION_SET_ID = f"questions_{UUID_HEX}"
CONTENT_HASH = "a" * 64


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _selections(**overrides: list[str]) -> dict[str, list[str]]:
    selections = {
        dimension: [values[0]]
        for dimension, values in _catalog()["matrix"].items()
    }
    selections.update(overrides)
    return selections


def _typed_payload(**overrides: object) -> dict:
    payload: dict[str, object] = {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "top_k": 5,
        "questions_source": {
            "type": "typed",
            "query": "How do I refund a cancelled flight?",
        },
        "selections": _selections(),
        "large_matrix_confirmation": None,
    }
    payload.update(overrides)
    return payload


def _question_set_payload(**overrides: object) -> dict:
    payload = _typed_payload(
        questions_source={
            "type": "question_set",
            "question_set_id": QUESTION_SET_ID,
            "content_sha256": CONTENT_HASH,
        }
    )
    payload.update(overrides)
    return payload


def _write_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes = b"k" * 32) -> Path:
    key_path = tmp_path / "confirmation.key"
    key_path.write_bytes(content)
    key_path.chmod(0o600)
    monkeypatch.setenv("PROJECT_MATRIX_CONFIRMATION_KEY_FILE", str(key_path))
    return key_path


def _large_selections() -> dict[str, list[str]]:
    matrix = _catalog()["matrix"]
    return {
        "chunkers": matrix["chunkers"],
        "embeddings": matrix["embeddings"][:2],
        "vector_stores": matrix["vector_stores"],
        "rerankers": matrix["rerankers"],
    }


def test_valid_typed_request_is_frozen_and_has_exact_cartesian_count():
    validated = validate_project_matrix_request(_typed_payload(), catalog_path=CATALOG_PATH)

    assert validated.combination_count == 1
    assert validated.warning is None
    assert validated.confirmation_required is False
    assert validated.confirmation_verified is False
    assert validated.request.selections == _selections()
    with pytest.raises(FrozenInstanceError):
        validated.combination_count = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        validated.request.top_k = 10  # type: ignore[misc]


@pytest.mark.parametrize("dimension", ["chunkers", "embeddings", "vector_stores"])
def test_selection_arrays_must_be_nonempty_unique_json_arrays(dimension: str):
    empty = _selections(**{dimension: []})
    with pytest.raises(ProjectMatrixValidationError, match="non-empty"):
        validate_project_matrix_request(_typed_payload(selections=empty), catalog_path=CATALOG_PATH)

    value = _catalog()["matrix"][dimension][0]
    duplicate = _selections(**{dimension: [value, value]})
    with pytest.raises(ProjectMatrixValidationError, match="duplicate"):
        validate_project_matrix_request(_typed_payload(selections=duplicate), catalog_path=CATALOG_PATH)

    not_an_array = _selections()
    not_an_array[dimension] = value  # type: ignore[assignment]
    with pytest.raises(ProjectMatrixValidationError, match="array"):
        validate_project_matrix_request(_typed_payload(selections=not_an_array), catalog_path=CATALOG_PATH)


def test_empty_rerankers_selects_one_retrieval_only_baseline_and_is_fingerprinted():
    selections = _selections(
        chunkers=_catalog()["matrix"]["chunkers"][:2],
        rerankers=[],
    )

    baseline = validate_project_matrix_request(
        _typed_payload(selections=selections), catalog_path=CATALOG_PATH
    )
    reranked = validate_project_matrix_request(
        _typed_payload(
            selections={
                **selections,
                "rerankers": [_catalog()["matrix"]["rerankers"][0]],
            }
        ),
        catalog_path=CATALOG_PATH,
    )

    assert baseline.request.rerankers == ()
    assert baseline.request.selections["rerankers"] == []
    assert baseline.combination_count == 2
    assert baseline.request_fingerprint != reranked.request_fingerprint


def test_empty_rerankers_must_still_be_an_array():
    selections = _selections()
    selections["rerankers"] = "none"  # type: ignore[assignment]

    with pytest.raises(ProjectMatrixValidationError, match="array"):
        validate_project_matrix_request(
            _typed_payload(selections=selections), catalog_path=CATALOG_PATH
        )


def test_only_exact_canonical_production_ids_are_accepted():
    for forbidden in ("unknown", "none", "fallback", "local", "FAISS-local"):
        selections = _selections(embeddings=[forbidden])
        with pytest.raises(ProjectMatrixValidationError, match="canonical production"):
            validate_project_matrix_request(
                _typed_payload(selections=selections), catalog_path=CATALOG_PATH
            )


def test_selection_and_request_schemas_are_exact_and_reject_boolean_integer_values():
    with pytest.raises(ProjectMatrixValidationError, match="request schema"):
        validate_project_matrix_request(
            {**_typed_payload(), "unexpected": True}, catalog_path=CATALOG_PATH
        )
    with pytest.raises(ProjectMatrixValidationError, match="selection schema"):
        validate_project_matrix_request(
            _typed_payload(selections={**_selections(), "extra": ["x"]}),
            catalog_path=CATALOG_PATH,
        )
    with pytest.raises(ProjectMatrixValidationError, match="top_k"):
        validate_project_matrix_request(_typed_payload(top_k=True), catalog_path=CATALOG_PATH)
    with pytest.raises(ProjectMatrixValidationError, match="schema_version"):
        validate_project_matrix_request(
            _typed_payload(schema_version=True), catalog_path=CATALOG_PATH
        )


def test_question_source_variants_have_strict_schemas_and_values():
    typed = validate_project_matrix_request(_typed_payload(), catalog_path=CATALOG_PATH)
    question_set = validate_project_matrix_request(
        _question_set_payload(), catalog_path=CATALOG_PATH
    )
    assert typed.request.questions_source["type"] == "typed"
    assert question_set.request.questions_source == {
        "type": "question_set",
        "question_set_id": QUESTION_SET_ID,
        "content_sha256": CONTENT_HASH,
    }

    invalid_sources = [
        {"type": "typed", "query": ""},
        {"type": "typed", "query": "valid", "extra": True},
        {"type": "question_set", "question_set_id": QUESTION_SET_ID},
        {
            "type": "question_set",
            "question_set_id": f"other_{UUID_HEX}",
            "content_sha256": CONTENT_HASH,
        },
        {
            "type": "question_set",
            "question_set_id": QUESTION_SET_ID,
            "content_sha256": "A" * 64,
        },
    ]
    for source in invalid_sources:
        with pytest.raises(ProjectMatrixValidationError, match="questions_source"):
            validate_project_matrix_request(
                _typed_payload(questions_source=source), catalog_path=CATALOG_PATH
            )


def test_fingerprint_is_canonical_sha256_and_bound_to_every_semantic_field():
    payload = _question_set_payload()
    validated = validate_project_matrix_request(payload, catalog_path=CATALOG_PATH)
    fingerprint_payload = {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "top_k": 5,
        "questions_source": payload["questions_source"],
        "selections": payload["selections"],
    }
    encoded = json.dumps(
        fingerprint_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert validated.request_fingerprint == hashlib.sha256(encoded).hexdigest()
    assert canonical_request_fingerprint(validated.request) == validated.request_fingerprint

    mutations = [
        _question_set_payload(project_id=f"other_{UUID_HEX}"),
        _question_set_payload(top_k=6),
        _question_set_payload(
            questions_source={
                "type": "question_set",
                "question_set_id": QUESTION_SET_ID,
                "content_sha256": "b" * 64,
            }
        ),
        _typed_payload(),
        _question_set_payload(
            selections=_selections(
                chunkers=[_catalog()["matrix"]["chunkers"][1]]
            )
        ),
    ]
    assert all(
        validate_project_matrix_request(item, catalog_path=CATALOG_PATH).request_fingerprint
        != validated.request_fingerprint
        for item in mutations
    )


def test_warning_is_emitted_only_above_fifty_combinations():
    matrix = _catalog()["matrix"]
    forty_five = {
        "chunkers": matrix["chunkers"],
        "embeddings": matrix["embeddings"],
        "vector_stores": matrix["vector_stores"][:3],
        "rerankers": matrix["rerankers"][:1],
    }
    sixty = {**forty_five, "vector_stores": matrix["vector_stores"]}

    assert validate_project_matrix_request(
        _typed_payload(selections=forty_five), catalog_path=CATALOG_PATH
    ).warning is None
    warned = validate_project_matrix_request(
        _typed_payload(selections=sixty), catalog_path=CATALOG_PATH
    )
    assert warned.combination_count == 60
    assert warned.warning is not None


def test_more_than_one_hundred_requires_a_bound_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_key(tmp_path, monkeypatch)
    payload = _typed_payload(selections=_large_selections())

    with pytest.raises(ConfirmationTokenError, match="required"):
        validate_project_matrix_request(payload, catalog_path=CATALOG_PATH, now=1_000)

    token = issue_large_matrix_confirmation(
        payload, catalog_path=CATALOG_PATH, now=1_000, ttl_seconds=60
    )
    confirmed = validate_project_matrix_request(
        {**payload, "large_matrix_confirmation": token},
        catalog_path=CATALOG_PATH,
        now=1_030,
    )
    assert confirmed.combination_count == 120
    assert confirmed.confirmation_required is True
    assert confirmed.confirmation_verified is True


def test_confirmation_rejects_request_mismatch_tampering_and_expiry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_key(tmp_path, monkeypatch)
    payload = _typed_payload(selections=_large_selections())
    token = issue_large_matrix_confirmation(
        payload, catalog_path=CATALOG_PATH, now=2_000, ttl_seconds=30
    )

    mismatched = {
        **payload,
        "top_k": 6,
        "large_matrix_confirmation": token,
    }
    with pytest.raises(ConfirmationTokenError, match="request"):
        validate_project_matrix_request(mismatched, catalog_path=CATALOG_PATH, now=2_010)

    replacement = "A" if token[-1] != "A" else "B"
    with pytest.raises(ConfirmationTokenError, match="signature"):
        validate_project_matrix_request(
            {**payload, "large_matrix_confirmation": token[:-1] + replacement},
            catalog_path=CATALOG_PATH,
            now=2_010,
        )

    with pytest.raises(ConfirmationTokenError, match="expired"):
        validate_project_matrix_request(
            {**payload, "large_matrix_confirmation": token},
            catalog_path=CATALOG_PATH,
            now=2_030,
        )


def test_confirmation_key_file_fails_closed_when_missing_symlinked_or_wrong_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    payload = _typed_payload(selections=_large_selections())
    missing = tmp_path / "missing.key"
    monkeypatch.setenv("PROJECT_MATRIX_CONFIRMATION_KEY_FILE", str(missing))
    with pytest.raises(ConfirmationTokenError, match="key file"):
        issue_large_matrix_confirmation(payload, catalog_path=CATALOG_PATH, now=1)

    real = tmp_path / "real.key"
    real.write_bytes(os.urandom(32))
    real.chmod(0o600)
    link = tmp_path / "link.key"
    link.symlink_to(real)
    monkeypatch.setenv("PROJECT_MATRIX_CONFIRMATION_KEY_FILE", str(link))
    with pytest.raises(ConfirmationTokenError, match="key file"):
        issue_large_matrix_confirmation(payload, catalog_path=CATALOG_PATH, now=1)

    monkeypatch.setenv("PROJECT_MATRIX_CONFIRMATION_KEY_FILE", str(real))
    real.chmod(0o640)
    with pytest.raises(ConfirmationTokenError, match="0600"):
        issue_large_matrix_confirmation(payload, catalog_path=CATALOG_PATH, now=1)


def test_confirmation_key_file_must_be_owned_by_the_service_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_key(tmp_path, monkeypatch)
    payload = _typed_payload(selections=_large_selections())
    service_uid = os.geteuid() + 1
    monkeypatch.setattr(
        project_matrix_contract.os,
        "geteuid",
        lambda: service_uid,
    )

    with pytest.raises(ConfirmationTokenError, match="owned by the service user"):
        issue_large_matrix_confirmation(payload, catalog_path=CATALOG_PATH, now=1)


def test_catalog_schema_version_rejects_float_where_integer_is_required(tmp_path: Path):
    catalog = _catalog()
    catalog["schema_version"] = 1.0
    catalog_path = tmp_path / "float-version.json"
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    with pytest.raises(ProjectMatrixValidationError, match="schema_version"):
        validate_project_matrix_request(_typed_payload(), catalog_path=catalog_path)


def test_catalog_product_is_the_hard_cap_and_catalog_fallbacks_fail_closed(tmp_path: Path):
    catalog = _catalog()
    catalog["matrix"]["chunkers"] = [f"chunker_{index}" for index in range(181)]
    catalog["techniques"]["chunkers"] = {
        value: {"adapter": "production"} for value in catalog["matrix"]["chunkers"]
    }
    oversized = tmp_path / "oversized.json"
    oversized.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ProjectMatrixValidationError, match="hard cap"):
        validate_project_matrix_request(_typed_payload(), catalog_path=oversized)

    fallback_catalog = _catalog()
    embedding = fallback_catalog["matrix"]["embeddings"][0]
    fallback_catalog["techniques"]["embeddings"][embedding]["adapter"] = "local_fallback"
    fallback_path = tmp_path / "fallback.json"
    fallback_path.write_text(json.dumps(fallback_catalog), encoding="utf-8")
    with pytest.raises(ProjectMatrixValidationError, match="production"):
        validate_project_matrix_request(_typed_payload(), catalog_path=fallback_path)
