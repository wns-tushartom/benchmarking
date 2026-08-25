from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import vm_adapter_manager as manager


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "vm_adapter_ports.json"


def new_manager(tmp_path: Path, **kwargs):
    return manager.VMAdapterManager(
        tmp_path,
        MANIFEST,
        runtime_dir=tmp_path / "data" / "adapter_runtime",
        **kwargs,
    )


def test_manifest_is_exact_contiguous_twenty_port_allowlist():
    payload = manager.load_port_manifest(MANIFEST)
    slots = payload["slots"]
    assert len(slots) == 20
    assert [slot["port"] for slot in slots] == list(range(5000, 5020))
    assert len({slot["service_id"] for slot in slots}) == 20


def test_manifest_pins_required_port_contracts():
    slots = {slot["service_id"]: slot for slot in manager.load_port_manifest(MANIFEST)["slots"]}
    assert slots["jina_embedding"]["port"] == 5000
    assert slots["gte_embedding"]["port"] == 5001
    assert slots["bge_reranker"]["port"] == 5002
    assert slots["pgvector"]["port"] == 5003
    assert slots["weaviate_http"]["port"] == 5004
    assert slots["weaviate_grpc"]["port"] == 5005
    assert slots["qwen_reranker"]["port"] == 5006
    assert slots["nemotron_reranker"]["port"] == 5007
    assert slots["gte_modernbert_reranker"]["port"] == 5008
    assert slots["jupyter_reserved"]["port"] == 5009
    assert slots["nemotron_embed_1b_bf16"]["port"] == 5010
    assert slots["dashboard"]["port"] == 5011
    assert slots["nemotron_embed_1b_nvfp4"]["port"] == 5012
    assert slots["nemotron_embed_8b_bf16"]["port"] == 5013
    assert slots["spare_reserved"]["port"] == 5014
    assert slots["qdrant_grpc"]["port"] == 5015
    assert slots["nvidia_rag"]["port"] == 5016
    assert slots["nvidia_ingestor"]["port"] == 5017
    assert slots["nvidia_frontend"]["port"] == 5018
    assert slots["qdrant_http"]["port"] == 5019


def test_manifest_rejects_duplicates_gaps_and_unknown_keys(tmp_path: Path):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["slots"][1]["port"] = 5000
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(manager.PortManifestError, match="exactly once"):
        manager.load_port_manifest(duplicate)

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["slots"][0]["command"] = ["sh", "-c", "bad"]
    arbitrary = tmp_path / "arbitrary.json"
    arbitrary.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(manager.PortManifestError, match="unsupported keys"):
        manager.load_port_manifest(arbitrary)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("service_id", "attacker_embedding"),
        ("role", "attacker"),
        ("launch_profile", "gte"),
    ],
)
def test_manifest_rejects_mutation_of_approved_slot_contract(
    tmp_path: Path, field: str, value: str
):
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["slots"][0][field] = value
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(manager.PortManifestError, match="approved 5000-5019 service mapping"):
        manager.load_port_manifest(mutated)


def test_reserved_external_and_companion_slots_are_not_startable(tmp_path: Path):
    vm = new_manager(tmp_path)
    for service_id in ("jupyter_reserved", "spare_reserved", "dashboard", "weaviate_grpc", "qdrant_grpc"):
        result = vm.start(service_id)
        assert result["ok"] is False
        assert result["status"] in {"reserved", "external", "companion"}


def test_unknown_service_and_browser_supplied_launch_fields_are_rejected(tmp_path: Path):
    vm = new_manager(tmp_path)
    with pytest.raises(manager.AdapterControlError, match="unknown service"):
        vm.start("attacker-service")
    with pytest.raises(TypeError):
        vm.start("jina_embedding", port=9999, command="rm -rf /")


def test_start_is_idempotent_when_declared_health_is_already_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vm = new_manager(tmp_path)
    monkeypatch.setattr(vm, "_probe", lambda slot: (True, "ready"))
    monkeypatch.setattr(vm, "_spawn", lambda *args, **kwargs: pytest.fail("must not spawn a duplicate"))
    result = vm.start("jina_embedding")
    assert result["ok"] is True
    assert result["status"] == "healthy"
    assert result["already_running"] is True


def test_unknown_port_occupant_is_blocked_and_never_killed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vm = new_manager(tmp_path)
    monkeypatch.setattr(vm, "_probe", lambda slot: (False, "wrong service response"))
    monkeypatch.setattr(vm, "_port_open", lambda slot: True)
    monkeypatch.setattr(vm, "_spawn", lambda *args, **kwargs: pytest.fail("must not replace unknown process"))
    monkeypatch.setattr(os, "kill", lambda *args, **kwargs: pytest.fail("must not kill unknown process"))
    result = vm.start("jina_embedding")
    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert "occupied" in result["message"].lower()


def test_process_start_uses_fixed_argv_without_shell_and_writes_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    calls = []
    fake_proc = SimpleNamespace(pid=43210)
    vm = new_manager(tmp_path, startup_attempts=1)
    monkeypatch.setattr(vm, "_probe", lambda slot: (False, "offline") if not calls else (True, "ready"))
    monkeypatch.setattr(vm, "_port_open", lambda slot: False)
    monkeypatch.setattr(vm, "_process_start_ticks", lambda pid: "12345")

    def fake_spawn(argv, env, log_path):
        calls.append((argv, env, log_path))
        return fake_proc

    monkeypatch.setattr(vm, "_spawn", fake_spawn)
    result = vm.start("jina_embedding")
    assert result["ok"] is True
    argv, env, log_path = calls[0]
    assert list(argv[:4]) == [os.fspath(manager.python_executable(vm.root)), "-u", "-m", "uvicorn"]
    assert "sh" not in argv and "bash" not in argv and "-c" not in argv
    assert env["WNS_ADAPTER_PROFILE"] == "jina"
    assert Path(log_path).parent == vm.runtime_dir
    receipt = json.loads((vm.runtime_dir / "jina_embedding.json").read_text(encoding="utf-8"))
    assert receipt["pid"] == 43210
    assert receipt["process_start_ticks"] == "12345"
    assert receipt["command_fingerprint"]
    assert "argv" not in receipt


def test_stop_refuses_stale_or_foreign_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vm = new_manager(tmp_path)
    receipt = {
        "schema_version": 1,
        "service_id": "jina_embedding",
        "port": 5000,
        "pid": 321,
        "process_start_ticks": "old",
        "command_fingerprint": "not-current",
        "started_at": "2026-08-20T00:00:00Z",
    }
    (vm.runtime_dir / "jina_embedding.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(vm, "_process_start_ticks", lambda pid: "new")
    monkeypatch.setattr(os, "kill", lambda *args, **kwargs: pytest.fail("must not kill stale PID"))
    result = vm.stop("jina_embedding")
    assert result["ok"] is False
    assert result["status"] == "blocked"
    assert "ownership" in result["message"].lower()


def test_stop_refuses_matching_pid_ticks_when_live_command_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    vm = new_manager(tmp_path)
    expected = vm._launch_spec(vm.slot("jina_embedding"))
    receipt = {
        "schema_version": 1,
        "service_id": "jina_embedding",
        "port": 5000,
        "pid": 654,
        "process_start_ticks": "same",
        "command_fingerprint": expected.command_fingerprint,
        "started_at": "2026-08-20T00:00:00Z",
    }
    (vm.runtime_dir / "jina_embedding.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(vm, "_process_start_ticks", lambda pid: "same")
    monkeypatch.setattr(vm, "_process_command_fingerprint", lambda pid: "different")
    monkeypatch.setattr(vm, "_pid_owned_by_current_user", lambda pid: True)
    monkeypatch.setattr(os, "kill", lambda *args, **kwargs: pytest.fail("must not kill PID-reused process"))
    result = vm.stop("jina_embedding")
    assert result["ok"] is False
    assert result["status"] == "blocked"


def test_status_does_not_classify_pid_reused_process_as_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    vm = new_manager(tmp_path)
    expected = vm._launch_spec(vm.slot("jina_embedding"))
    receipt = {
        "schema_version": 1,
        "service_id": "jina_embedding",
        "port": 5000,
        "pid": 654,
        "process_start_ticks": "same",
        "command_fingerprint": expected.command_fingerprint,
        "started_at": "2026-08-20T00:00:00Z",
    }
    (vm.runtime_dir / "jina_embedding.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(vm, "_probe", lambda slot: (False, "offline"))
    monkeypatch.setattr(vm, "_process_start_ticks", lambda pid: "same")
    monkeypatch.setattr(vm, "_process_command_fingerprint", lambda pid: "different")
    monkeypatch.setattr(vm, "_pid_owned_by_current_user", lambda pid: True)
    assert vm.status("jina_embedding")["status"] == "offline"


def test_stop_terminates_only_matching_owned_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vm = new_manager(tmp_path)
    expected = vm._launch_spec(vm.slot("jina_embedding"))
    receipt = {
        "schema_version": 1,
        "service_id": "jina_embedding",
        "port": 5000,
        "pid": 654,
        "process_start_ticks": "same",
        "command_fingerprint": expected.command_fingerprint,
        "started_at": "2026-08-20T00:00:00Z",
    }
    (vm.runtime_dir / "jina_embedding.json").write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(vm, "_process_start_ticks", lambda pid: "same")
    monkeypatch.setattr(vm, "_process_command_fingerprint", lambda pid: expected.command_fingerprint)
    monkeypatch.setattr(vm, "_pid_owned_by_current_user", lambda pid: True)
    monkeypatch.setattr(vm, "_wait_for_exit", lambda pid: True)
    signals = []
    closed = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid, flags=0: 77)
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda fd, sig: signals.append((fd, sig)))
    monkeypatch.setattr(os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(os, "kill", lambda *args, **kwargs: pytest.fail("PID signaling must use pidfd"))
    result = vm.stop("jina_embedding")
    assert result["ok"] is True
    assert result["status"] == "offline"
    assert signals and signals[0][0] == 77
    assert closed == [77]
    assert not (vm.runtime_dir / "jina_embedding.json").exists()


def test_runtime_receipt_symlink_is_rejected(tmp_path: Path):
    vm = new_manager(tmp_path)
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    (vm.runtime_dir / "jina_embedding.json").symlink_to(target)
    with pytest.raises(manager.AdapterControlError, match="symlink"):
        vm._read_receipt("jina_embedding")


def test_receipt_read_uses_descriptor_instead_of_following_path_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    vm = new_manager(tmp_path)
    receipt_path = vm.runtime_dir / "jina_embedding.json"
    receipt_path.write_text(
        json.dumps({"service_id": "jina_embedding", "schema_version": 1}),
        encoding="utf-8",
    )
    original = Path.read_text

    def fail_receipt_path_open(path: Path, *args, **kwargs):
        if path == receipt_path:
            pytest.fail("receipt must be read from a no-follow descriptor")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_receipt_path_open)
    loaded = vm._read_receipt("jina_embedding")
    assert loaded is not None and loaded["service_id"] == "jina_embedding"


def test_runtime_directory_symlink_is_rejected_before_resolution(tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "adapter_runtime").symlink_to(outside, target_is_directory=True)
    with pytest.raises(manager.AdapterControlError, match="runtime directory.*symlink"):
        manager.VMAdapterManager(tmp_path, MANIFEST)
    assert list(outside.iterdir()) == []


def test_dangling_log_symlink_is_rejected_without_external_write(tmp_path: Path):
    vm = new_manager(tmp_path)
    outside = tmp_path / "outside.log"
    (vm.runtime_dir / "jina_embedding.log").symlink_to(outside)
    with pytest.raises(manager.AdapterControlError, match="log.*symlink"):
        vm._spawn(("/bin/true",), os.environ.copy(), vm._log_path("jina_embedding"))
    assert not outside.exists()


def test_runtime_directory_must_be_owned_by_current_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = tmp_path / "data" / "adapter_runtime"
    runtime.mkdir(parents=True)
    monkeypatch.setattr(os, "geteuid", lambda: runtime.stat().st_uid + 1)
    with pytest.raises(manager.AdapterControlError, match="owned by the current user"):
        manager.VMAdapterManager(tmp_path, MANIFEST)


def test_vllm_prerequisites_require_module_and_cached_approved_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    vm = new_manager(tmp_path)
    slot = vm.slot("nemotron_embed_1b_bf16")
    spec = vm._launch_spec(slot)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(vm, "_python_module_available", lambda executable, module: False)
    message = vm._profile_prerequisite(slot, spec)
    assert message is not None and "vLLM module" in message

    monkeypatch.setattr(vm, "_python_module_available", lambda executable, module: True)
    monkeypatch.setattr(vm, "_cached_hf_model_available", lambda model: False)
    message = vm._profile_prerequisite(slot, spec)
    assert message is not None and "approved model is not cached" in message


def test_cli_rejects_noncanonical_root_or_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setattr(
        "sys.argv",
        ["vm_adapter_manager.py", "status", "--root", str(tmp_path)],
    )
    with pytest.raises(SystemExit) as error:
        manager.main()
    assert error.value.code == 2

    monkeypatch.setattr(
        "sys.argv",
        ["vm_adapter_manager.py", "status", "--manifest", str(tmp_path / "ports.json")],
    )
    with pytest.raises(SystemExit) as error:
        manager.main()
    assert error.value.code == 2


def test_public_status_redacts_private_paths_commands_and_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "super-secret-token")
    vm = new_manager(tmp_path)
    monkeypatch.setattr(vm, "_probe", lambda slot: (False, f"failed under {ROOT} token=super-secret-token"))
    payload = vm.status("nemotron_embed_1b_bf16")
    serialized = json.dumps(payload)
    assert str(ROOT) not in serialized
    assert "super-secret-token" not in serialized
    assert "argv" not in serialized
    assert payload["port"] == 5010


def test_public_status_redacts_checkout_root_outside_home_and_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkout_root = Path("/srv/private/Retreiver")
    monkeypatch.setattr(manager, "_MODULE_ROOT", checkout_root, raising=False)
    vm = new_manager(tmp_path)
    monkeypatch.setattr(vm, "_probe", lambda slot: (False, f"failed under {checkout_root}"))

    serialized = json.dumps(vm.status("nemotron_embed_1b_bf16"))

    assert str(checkout_root) not in serialized


def test_start_required_accepts_only_manifest_service_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vm = new_manager(tmp_path)
    called = []
    monkeypatch.setattr(vm, "start", lambda service_id: called.append(service_id) or {"service_id": service_id, "ok": True})
    results = vm.start_required(["gte_embedding", "bge_reranker", "gte_embedding"])
    assert called == ["gte_embedding", "bge_reranker"]
    assert len(results) == 2
    with pytest.raises(manager.AdapterControlError, match="unknown service"):
        vm.start_required(["gte_embedding", "not-allowlisted"])
