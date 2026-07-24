from __future__ import annotations

import importlib


def test_safe_error_detail_keeps_exception_type_and_short_message():
    module = importlib.import_module("source.services.project_matrix_runner")
    detail = module._safe_error_detail(FileExistsError("[Errno 17] File exists: 'project_abc'"))
    assert detail.startswith("FileExistsError:")
    assert "exists" in detail.lower()
    assert len(detail) <= 240


def test_safe_error_detail_redacts_home_paths_and_secrets():
    module = importlib.import_module("source.services.project_matrix_runner")
    detail = module._safe_error_detail(
        RuntimeError("failed at /home/U481019/secret/path token=sk-abc123456789")
    )
    assert detail == "RuntimeError"
    assert "/home/" not in detail
    assert "sk-abc" not in detail


def test_safe_error_detail_omits_freeform_provider_messages():
    module = importlib.import_module("source.services.project_matrix_runner")
    detail = module._safe_error_detail(RuntimeError("second query provider failure"))
    assert detail == "RuntimeError"
    assert "provider failure" not in detail
