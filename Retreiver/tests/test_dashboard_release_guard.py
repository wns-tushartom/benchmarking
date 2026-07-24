from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "dashboard_release_contract.json"
VERIFIER_PATH = ROOT / "scripts" / "verify_dashboard_release.py"
INVENTORY_PATH = ROOT / "scripts" / "inventory_vm_source_drift.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_contract_declares_every_required_surface() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert contract["matrix"] == {
        "chunkers": 5,
        "embeddings": 3,
        "stores": 4,
        "rerankers": 3,
        "total": 180,
    }
    assert set(contract["pages"]) == {
        "overview",
        "quality",
        "compare",
        "run",
        "evidence",
        "hallucination",
        "nvidia",
        "repository",
        "upload",
        "ops",
    }
    assert {
        "globalDataset",
        "globalGroundtruth",
        "metricsSourceContext",
        "recommendationContext",
        "recommendationStrip",
        "recommendationTable",
        "tradeoffChart",
        "metricQuickView",
        "metricPointDetails",
        "metricHeatmap",
        "runSourceContext",
    }.issubset(contract["dom_ids"])
    assert {
        "/api/results",
        "/api/result-sources",
        "/api/run/preflight-complete-pipeline",
        "/api/run/complete-pipeline",
    }.issubset(contract["api_routes"])
    file_tokens = contract["file_tokens"]
    assert "20260724-canonical-dashboard-v5" in file_tokens["web/index.html"]
    assert "web/recommendation-visuals.js" in contract["required_files"]
    assert "linked_groundtruth_id" in file_tokens["scripts/dashboard_source_catalog.py"]
    assert "groundtruth_500.csv" in file_tokens["scripts/dashboard_source_catalog.py"]
    assert "OFFICIAL_GROUNDTRUTH_ID" in file_tokens["benchmarking/core/runner.py"]
    assert "renderRecommendationAnalytics" in file_tokens["web/app.js"]
    assert "allow_fallback=False" in file_tokens["source/services/project_documents.py"]


def test_release_verifier_passes_current_tracked_tree() -> None:
    result = subprocess.run(
        [sys.executable, str(VERIFIER_PATH), "--allow-dirty-tests"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DASHBOARD_RELEASE_CONTRACT_OK" in result.stdout


def test_release_verifier_rejects_duplicate_ids_in_fixture(tmp_path: Path) -> None:
    verifier = _load_module(VERIFIER_PATH, "dashboard_release_verifier")
    duplicate = tmp_path / "duplicate.html"
    duplicate.write_text('<main id="same"><section id="same"></section></main>', encoding="utf-8")
    assert verifier.duplicate_html_ids(duplicate) == ["same"]


def test_vm_inventory_classifies_source_runtime_and_unclassified_paths() -> None:
    inventory = _load_module(INVENTORY_PATH, "dashboard_vm_inventory")
    status = (
        b" M web/app.js\0"
        b"?? scripts/vm_hotfix.py\0"
        b" M source/services/document_parser.py\0"
        b" M data/evaluation/groundtruth_eval_summary.csv\0"
        b"?? logs/dashboard.log\0"
        b"?? notes/operator.txt\0"
    )
    report = inventory.classify_porcelain_status(status)
    assert report == {
        "merge_back_required": [
            "web/app.js",
            "scripts/vm_hotfix.py",
            "source/services/document_parser.py",
        ],
        "runtime_artifacts": [
            "data/evaluation/groundtruth_eval_summary.csv",
            "logs/dashboard.log",
        ],
        "unclassified": ["notes/operator.txt"],
        "safe_to_pull_source": False,
    }


def test_vm_inventory_expected_classification_and_exit_policy() -> None:
    inventory = _load_module(INVENTORY_PATH, "dashboard_vm_inventory_policy")
    paths = [
        "web/app.js",
        "scripts/vm_hotfix.py",
        "source/services/document_parser.py",
        "benchmarking/adapters/vector_faiss.py",
        "configs/benchmark.local.json",
        "tests/test_dashboard_metrics.py",
        "data/evaluation/groundtruth_eval_summary.csv",
        "logs/dashboard.log",
        "notes/operator.txt",
    ]
    report = inventory.classify_paths(paths)
    assert report["merge_back_required"] == paths[:6]
    assert report["runtime_artifacts"] == paths[6:8]
    assert report["unclassified"] == paths[8:]
    assert report["safe_to_pull_source"] is False
    assert inventory.exit_code(report) == 2


def test_vm_inventory_runtime_only_is_safe_to_pull_source() -> None:
    inventory = _load_module(INVENTORY_PATH, "dashboard_vm_inventory_runtime")
    report = inventory.classify_paths(
        ["data/vm_dashboard_snapshot.json", "logs/dashboard.log"]
    )
    assert report["merge_back_required"] == []
    assert report["runtime_artifacts"] == [
        "data/vm_dashboard_snapshot.json",
        "logs/dashboard.log",
    ]
    assert report["unclassified"] == []
    assert report["safe_to_pull_source"] is True
    assert inventory.exit_code(report) == 0
