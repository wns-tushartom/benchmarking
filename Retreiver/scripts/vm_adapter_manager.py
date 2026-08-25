#!/usr/bin/env python3
"""Safe lifecycle controls for the fixed WNS VM port allowlist."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


_MODULE_ROOT = Path(__file__).resolve().parents[1]


class PortManifestError(ValueError):
    pass


class AdapterControlError(RuntimeError):
    pass


_TOP_LEVEL_KEYS = {
    "schema_version",
    "allowed_host",
    "allowed_port_min",
    "allowed_port_max",
    "slots",
}
_SLOT_KEYS = {
    "service_id",
    "port",
    "label",
    "role",
    "manage_mode",
    "launch_profile",
    "health",
    "endpoint_env",
}
_HEALTH_KEYS = {"kind", "path", "expected_profile", "expected_model"}
_MANAGE_MODES = {
    "python_adapter",
    "vllm",
    "docker_compose",
    "companion",
    "external",
    "reserved",
}
_SERVICE_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")
_APPROVED_MANIFEST_SHA256 = "0c6887bb4a58b23a36cade317a18c5fd267b5c5796b16c73d3916707f4e6eba9"


def _command_fingerprint(argv: Iterable[str]) -> str:
    payload = json.dumps(list(argv), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LaunchSpec:
    argv: tuple[str, ...]
    env_overrides: Mapping[str, str]

    @property
    def command_fingerprint(self) -> str:
        return _command_fingerprint(self.argv)


def _require_exact_keys(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise PortManifestError(f"{label} has unsupported keys: {', '.join(unexpected)}")


def load_port_manifest(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink():
        raise PortManifestError("port manifest must not be a symlink")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PortManifestError(f"cannot load port manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise PortManifestError("port manifest must be a JSON object")
    _require_exact_keys(payload, _TOP_LEVEL_KEYS, "port manifest")
    if payload.get("schema_version") != 1:
        raise PortManifestError("port manifest schema_version must be 1")
    if payload.get("allowed_host") != "127.0.0.1":
        raise PortManifestError("allowed_host must be 127.0.0.1")
    minimum = payload.get("allowed_port_min")
    maximum = payload.get("allowed_port_max")
    if isinstance(minimum, bool) or not isinstance(minimum, int):
        raise PortManifestError("allowed_port_min must be an integer")
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise PortManifestError("allowed_port_max must be an integer")
    if (minimum, maximum) != (5000, 5019):
        raise PortManifestError("allowed port range must be 5000 through 5019")
    slots = payload.get("slots")
    if not isinstance(slots, list) or len(slots) != 20:
        raise PortManifestError("manifest must declare exactly 20 slots")
    ports: list[int] = []
    service_ids: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(slots):
        if not isinstance(raw, dict):
            raise PortManifestError(f"slot {index} must be an object")
        _require_exact_keys(raw, _SLOT_KEYS, f"slot {index}")
        service_id = raw.get("service_id")
        if (
            not isinstance(service_id, str)
            or not service_id
            or any(char not in _SERVICE_ID_CHARS for char in service_id)
        ):
            raise PortManifestError(f"slot {index} has invalid service_id")
        port = raw.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or port < minimum or port > maximum:
            raise PortManifestError(f"slot {service_id} has invalid port")
        mode = raw.get("manage_mode")
        if mode not in _MANAGE_MODES:
            raise PortManifestError(f"slot {service_id} has invalid manage_mode")
        if mode in {"python_adapter", "vllm", "docker_compose", "companion"}:
            if not isinstance(raw.get("launch_profile"), str) or not raw["launch_profile"]:
                raise PortManifestError(f"slot {service_id} requires launch_profile")
        elif "launch_profile" in raw:
            raise PortManifestError(f"slot {service_id} must not declare launch_profile")
        health = raw.get("health")
        if not isinstance(health, dict):
            raise PortManifestError(f"slot {service_id} requires health")
        _require_exact_keys(health, _HEALTH_KEYS, f"slot {service_id} health")
        if health.get("kind") not in {"http", "tcp"}:
            raise PortManifestError(f"slot {service_id} has invalid health kind")
        if health["kind"] == "http" and not isinstance(health.get("path"), str):
            raise PortManifestError(f"slot {service_id} HTTP health requires path")
        endpoint_env = raw.get("endpoint_env")
        if not isinstance(endpoint_env, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in endpoint_env.items()
        ):
            raise PortManifestError(f"slot {service_id} has invalid endpoint_env")
        ports.append(port)
        service_ids.append(service_id)
        normalized.append(dict(raw))
    if sorted(ports) != list(range(5000, 5020)):
        raise PortManifestError("every port from 5000 through 5019 must appear exactly once")
    if len(set(service_ids)) != 20:
        raise PortManifestError("every service_id must appear exactly once")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != _APPROVED_MANIFEST_SHA256:
        raise PortManifestError("manifest does not match the approved 5000-5019 service mapping")
    result = dict(payload)
    result["slots"] = sorted(normalized, key=lambda slot: slot["port"])
    return result


def python_executable(root: Path) -> Path:
    candidate = root / ".venv-vm" / "bin" / "python"
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else Path(sys.executable)


def verify_operator_token(expected: str, supplied: str) -> bool:
    if not expected or not supplied:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8"))


class VMAdapterManager:
    def __init__(
        self,
        root: Path,
        manifest_path: Path,
        *,
        runtime_dir: Path | None = None,
        startup_attempts: int = 30,
        startup_delay: float = 1.0,
    ) -> None:
        lexical_root = Path(root).absolute()
        if lexical_root.is_symlink() or lexical_root.resolve() != lexical_root:
            raise AdapterControlError("repository root must be a canonical real directory")
        if not lexical_root.is_dir():
            raise AdapterControlError("repository root must exist and be a directory")
        self.root = lexical_root
        self.manifest_path = manifest_path
        self.manifest = load_port_manifest(manifest_path)
        self._slots = {slot["service_id"]: slot for slot in self.manifest["slots"]}
        expected_runtime = self.root / "data" / "adapter_runtime"
        requested_runtime = Path(runtime_dir or expected_runtime).absolute()
        if requested_runtime != expected_runtime:
            raise AdapterControlError("adapter runtime directory must be the repository candidate runtime")
        self.runtime_dir = requested_runtime
        self.startup_attempts = max(1, min(int(startup_attempts), 120))
        self.startup_delay = max(0.0, min(float(startup_delay), 5.0))
        self._ensure_runtime_dir()

    def _ensure_runtime_dir(self) -> None:
        for path in (self.root, self.root / "data", self.runtime_dir):
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                path.mkdir(mode=0o700)
                metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise AdapterControlError("adapter runtime directory must not contain a symlink")
            if not stat.S_ISDIR(metadata.st_mode):
                raise AdapterControlError("adapter runtime path must be a real directory")
            if metadata.st_uid != os.geteuid():
                raise AdapterControlError("adapter runtime directory must be owned by the current user")

    @staticmethod
    def _require_owned_regular_file(path: Path, label: str) -> os.stat_result:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            raise AdapterControlError(f"adapter {label} is missing") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise AdapterControlError(f"adapter {label} must not be a symlink")
        if not stat.S_ISREG(metadata.st_mode):
            raise AdapterControlError(f"adapter {label} must be a regular file")
        if metadata.st_uid != os.geteuid():
            raise AdapterControlError(f"adapter {label} must be owned by the current user")
        return metadata

    def slot(self, service_id: str) -> dict[str, Any]:
        try:
            return self._slots[service_id]
        except KeyError as exc:
            raise AdapterControlError(f"unknown service: {service_id}") from exc

    def _receipt_path(self, service_id: str) -> Path:
        self.slot(service_id)
        return self.runtime_dir / f"{service_id}.json"

    def _log_path(self, service_id: str) -> Path:
        self.slot(service_id)
        return self.runtime_dir / f"{service_id}.log"

    def _read_receipt(self, service_id: str) -> dict[str, Any] | None:
        path = self._receipt_path(service_id)
        if not path.exists() and not path.is_symlink():
            return None
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            if path.is_symlink():
                raise AdapterControlError("adapter receipt must not be a symlink") from exc
            raise AdapterControlError("adapter receipt could not be opened safely") from exc
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            os.close(fd)
            raise AdapterControlError("adapter receipt must be a current-user-owned regular file")
        try:
            with os.fdopen(fd, "r", encoding="utf-8") as handle:
                receipt = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise AdapterControlError("adapter receipt is unreadable") from exc
        if not isinstance(receipt, dict) or receipt.get("service_id") != service_id:
            raise AdapterControlError("adapter receipt identity mismatch")
        return receipt

    def _write_receipt(self, service_id: str, receipt: Mapping[str, Any]) -> None:
        path = self._receipt_path(service_id)
        if path.exists() or path.is_symlink():
            self._require_owned_regular_file(path, "receipt")
        temporary = path.with_suffix(f".tmp-{os.getpid()}")
        if temporary.exists() or temporary.is_symlink():
            raise AdapterControlError("adapter receipt temporary path already exists")
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(dict(receipt), handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def _endpoint(self, slot: Mapping[str, Any]) -> str:
        return f"http://127.0.0.1:{slot['port']}{slot['health'].get('path', '')}"

    def _probe(self, slot: Mapping[str, Any]) -> tuple[bool, str]:
        health = slot["health"]
        if health["kind"] == "tcp":
            try:
                with socket.create_connection(("127.0.0.1", int(slot["port"])), timeout=0.6):
                    return True, "ready"
            except OSError:
                return False, "offline"
        request = urllib.request.Request(self._endpoint(slot), headers={"Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=1.5) as response:
                body = response.read(262_144)
                if response.status < 200 or response.status >= 300:
                    return False, f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            return False, f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError):
            return False, "offline"
        expected_profile = health.get("expected_profile")
        expected_model = health.get("expected_model")
        if expected_profile or expected_model:
            try:
                payload = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return False, "health response is not valid JSON"
            if expected_profile and payload.get("profile") != expected_profile:
                return False, "health profile mismatch"
            if expected_model:
                model_ids = {
                    item.get("id")
                    for item in payload.get("data", [])
                    if isinstance(item, dict)
                }
                if expected_model not in model_ids:
                    return False, "health model mismatch"
        return True, "ready"

    def _port_open(self, slot: Mapping[str, Any]) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", int(slot["port"])), timeout=0.4):
                return True
        except OSError:
            return False

    def _process_start_ticks(self, pid: int) -> str | None:
        try:
            fields = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8").split()
            return fields[21] if len(fields) > 21 else None
        except (OSError, ValueError):
            return None

    def _process_command_fingerprint(self, pid: int) -> str | None:
        try:
            raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        except (OSError, ValueError):
            return None
        argv = tuple(os.fsdecode(part) for part in raw.split(b"\0") if part)
        return _command_fingerprint(argv) if argv else None

    def _pid_owned_by_current_user(self, pid: int) -> bool:
        try:
            return Path(f"/proc/{int(pid)}").stat().st_uid == os.geteuid()
        except OSError:
            return False

    def _receipt_matches_process(
        self, slot: Mapping[str, Any], receipt: Mapping[str, Any] | None
    ) -> bool:
        if not receipt:
            return False
        try:
            pid = int(receipt["pid"])
            spec = self._launch_spec(slot)
        except (KeyError, TypeError, ValueError, AdapterControlError):
            return False
        expected_ticks = receipt.get("process_start_ticks")
        return bool(
            expected_ticks
            and self._process_start_ticks(pid) == expected_ticks
            and receipt.get("command_fingerprint") == spec.command_fingerprint
            and self._process_command_fingerprint(pid) == spec.command_fingerprint
            and self._pid_owned_by_current_user(pid)
        )

    def _launch_spec(self, slot: Mapping[str, Any]) -> LaunchSpec:
        mode = slot["manage_mode"]
        profile = slot.get("launch_profile", "")
        port = str(slot["port"])
        py = os.fspath(python_executable(self.root))
        if mode == "python_adapter":
            argv = (
                py,
                "-u",
                "-m",
                "uvicorn",
                "scripts.wns_vm_adapter_service:app",
                "--app-dir",
                os.fspath(self.root),
                "--host",
                "127.0.0.1",
                "--port",
                port,
            )
            gpu_by_profile = {
                "jina": "0",
                "gte": "0",
                "bge": "1",
                "qwen": "2",
                "nemotron-rerank": "1",
                "gte-modernbert": "2",
            }
            return LaunchSpec(
                argv,
                {
                    "WNS_ADAPTER_PROFILE": profile,
                    "WNS_MODEL_DEVICE": "cuda",
                    "CUDA_VISIBLE_DEVICES": gpu_by_profile[profile],
                },
            )
        if mode == "vllm":
            models = {
                "nemotron-embed-1b-bf16": "nvidia/Nemotron-3-Embed-1B-BF16",
                "nemotron-embed-1b-nvfp4": "nvidia/Nemotron-3-Embed-1B-NVFP4",
                "nemotron-embed-8b-bf16": "nvidia/Nemotron-3-Embed-8B-BF16",
            }
            try:
                model = models[profile]
            except KeyError as exc:
                raise AdapterControlError("unknown vLLM launch profile") from exc
            argv = (
                py,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--host",
                "127.0.0.1",
                "--port",
                port,
                "--model",
                model,
            )
            gpu_by_profile = {
                "nemotron-embed-1b-bf16": "0",
                "nemotron-embed-1b-nvfp4": "1",
                "nemotron-embed-8b-bf16": "2",
            }
            return LaunchSpec(argv, {"CUDA_VISIBLE_DEVICES": gpu_by_profile[profile]})
        if mode == "docker_compose":
            argv = (
                "docker",
                "compose",
                "-f",
                os.fspath(self.root / "docker-compose.benchmark.yml"),
                "up",
                "-d",
                profile,
            )
            return LaunchSpec(argv, {})
        raise AdapterControlError(f"service {slot['service_id']} is not startable")

    def _spawn(self, argv: Iterable[str], env: Mapping[str, str], log_path: Path):
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            log_fd = os.open(log_path, flags, 0o600)
        except OSError as exc:
            if log_path.is_symlink():
                raise AdapterControlError("adapter log must not be a symlink") from exc
            raise AdapterControlError("adapter log could not be opened safely") from exc
        metadata = os.fstat(log_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            os.close(log_fd)
            raise AdapterControlError("adapter log must be a current-user-owned regular file")
        log_handle = os.fdopen(log_fd, "ab", buffering=0)
        try:
            return subprocess.Popen(
                list(argv),
                cwd=self.root,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                shell=False,
                start_new_session=True,
            )
        finally:
            log_handle.close()

    def _public_message(self, message: str) -> str:
        redacted = str(message)
        private_roots = (
            (os.fspath(self.root), "<runtime-root>"),
            (os.fspath(_MODULE_ROOT), "<repo>"),
            (os.fspath(Path.home()), "<home>"),
        )
        for private_root, replacement in sorted(
            private_roots, key=lambda item: len(item[0]), reverse=True
        ):
            if private_root:
                redacted = redacted.replace(private_root, replacement)
        for name, value in os.environ.items():
            if value and ("TOKEN" in name or "KEY" in name or "PASSWORD" in name or "SECRET" in name):
                redacted = redacted.replace(value, "<redacted>")
        return redacted[:500]

    def _base_status(self, slot: Mapping[str, Any], status: str, message: str) -> dict[str, Any]:
        return {
            "service_id": slot["service_id"],
            "label": slot["label"],
            "role": slot["role"],
            "port": slot["port"],
            "manage_mode": slot["manage_mode"],
            "status": status,
            "message": self._public_message(message),
            "startable": slot["manage_mode"] in {"python_adapter", "vllm", "docker_compose"},
            "stoppable": slot["manage_mode"] in {"python_adapter", "vllm"},
        }

    def status(self, service_id: str) -> dict[str, Any]:
        slot = self.slot(service_id)
        healthy, note = self._probe(slot)
        if healthy:
            return {**self._base_status(slot, "healthy", note), "ok": True}
        mode = slot["manage_mode"]
        if mode == "reserved":
            return {**self._base_status(slot, "reserved", "Reserved port"), "ok": False}
        if mode == "companion":
            return {**self._base_status(slot, "companion", "Starts with its parent service"), "ok": False}
        if mode == "external":
            return {**self._base_status(slot, "external", note), "ok": False}
        receipt = self._read_receipt(service_id)
        if self._receipt_matches_process(slot, receipt):
            return {**self._base_status(slot, "unhealthy", note), "ok": False}
        return {**self._base_status(slot, "offline", note), "ok": False}

    def statuses(self) -> list[dict[str, Any]]:
        return [self.status(slot["service_id"]) for slot in self.manifest["slots"]]

    def _profile_prerequisite(self, slot: Mapping[str, Any], spec: LaunchSpec) -> str | None:
        mode = slot["manage_mode"]
        executable = spec.argv[0]
        if os.path.sep in executable:
            if not Path(executable).is_file():
                return "configured Python executable is missing"
        elif shutil.which(executable) is None:
            return f"required executable is missing: {executable}"
        if mode == "vllm":
            if not os.getenv("HF_TOKEN", "").strip():
                return "HF_TOKEN is required for this configured vLLM profile"
            if not self._python_module_available(executable, "vllm.entrypoints.openai.api_server"):
                return "configured Python does not provide the vLLM module"
            model = spec.argv[spec.argv.index("--model") + 1]
            if not self._cached_hf_model_available(model):
                return "approved model is not cached for offline startup"
        return None

    @staticmethod
    def _python_module_available(executable: str, module: str) -> bool:
        probe = "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
        try:
            completed = subprocess.run(
                [executable, "-c", probe, module],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=False,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0

    @staticmethod
    def _cached_hf_model_available(model: str) -> bool:
        cache_name = "models--" + model.replace("/", "--")
        roots: list[str] = []
        default_hf_home = Path.home() / ".cache" / "huggingface"
        for candidate in (
            os.getenv("HF_HUB_CACHE"),
            os.getenv("TRANSFORMERS_CACHE"),
            os.fspath(Path(os.getenv("HF_HOME", default_hf_home)) / "hub"),
        ):
            if candidate and candidate not in roots:
                roots.append(candidate)
        for root in roots:
            snapshots = Path(root) / cache_name / "snapshots"
            if snapshots.is_dir() and any(child.is_dir() for child in snapshots.iterdir()):
                return True
        return False

    def start(self, service_id: str) -> dict[str, Any]:
        slot = self.slot(service_id)
        mode = slot["manage_mode"]
        if mode in {"reserved", "external", "companion"}:
            return {
                **self._base_status(slot, mode, f"{mode} slot is not dashboard-startable"),
                "ok": False,
            }
        healthy, note = self._probe(slot)
        if healthy:
            return {
                **self._base_status(slot, "healthy", note),
                "ok": True,
                "already_running": True,
            }
        receipt = self._read_receipt(service_id)
        if self._port_open(slot):
            owned = self._receipt_matches_process(slot, receipt)
            message = "Owned process is listening but failed its health contract" if owned else "Port is occupied by an unrelated or unverifiable process"
            return {**self._base_status(slot, "blocked", message), "ok": False}
        spec = self._launch_spec(slot)
        prerequisite = self._profile_prerequisite(slot, spec)
        if prerequisite:
            return {**self._base_status(slot, "unconfigured", prerequisite), "ok": False}
        env = os.environ.copy()
        env.update(spec.env_overrides)
        log_path = self._log_path(service_id)
        proc = self._spawn(spec.argv, env, log_path)
        if mode != "docker_compose":
            receipt_payload = {
                "schema_version": 1,
                "service_id": service_id,
                "port": slot["port"],
                "pid": int(proc.pid),
                "process_start_ticks": self._process_start_ticks(int(proc.pid)),
                "command_fingerprint": spec.command_fingerprint,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            self._write_receipt(service_id, receipt_payload)
        for _ in range(self.startup_attempts):
            healthy, note = self._probe(slot)
            if healthy:
                return {
                    **self._base_status(slot, "healthy", note),
                    "ok": True,
                    "already_running": False,
                }
            if self.startup_delay:
                time.sleep(self.startup_delay)
        return {
            **self._base_status(slot, "unhealthy", f"Startup did not satisfy health contract: {note}"),
            "ok": False,
        }

    def _wait_for_exit(self, pid: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._process_start_ticks(pid) is None:
                return True
            time.sleep(0.05)
        return self._process_start_ticks(pid) is None

    def stop(self, service_id: str) -> dict[str, Any]:
        slot = self.slot(service_id)
        if slot["manage_mode"] not in {"python_adapter", "vllm"}:
            return {
                **self._base_status(slot, "blocked", "Only dashboard-owned model processes can be stopped"),
                "ok": False,
            }
        receipt = self._read_receipt(service_id)
        if not receipt:
            return {**self._base_status(slot, "offline", "No owned process receipt"), "ok": True}
        try:
            pid = int(receipt["pid"])
        except (KeyError, TypeError, ValueError):
            return {**self._base_status(slot, "blocked", "Process ownership receipt is invalid"), "ok": False}
        spec = self._launch_spec(slot)
        ownership_ok = (
            receipt.get("process_start_ticks")
            and self._process_start_ticks(pid) == receipt.get("process_start_ticks")
            and receipt.get("command_fingerprint") == spec.command_fingerprint
            and self._process_command_fingerprint(pid) == spec.command_fingerprint
            and self._pid_owned_by_current_user(pid)
        )
        if not ownership_ok:
            return {
                **self._base_status(slot, "blocked", "Process ownership could not be verified"),
                "ok": False,
            }
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            return {
                **self._base_status(slot, "blocked", "Safe pidfd process control is unavailable"),
                "ok": False,
            }
        try:
            pidfd = os.pidfd_open(pid, 0)
        except OSError:
            return {
                **self._base_status(slot, "blocked", "Owned process could not be pinned safely"),
                "ok": False,
            }
        try:
            if not self._receipt_matches_process(slot, receipt):
                return {
                    **self._base_status(slot, "blocked", "Process identity changed before stop"),
                    "ok": False,
                }
            signal.pidfd_send_signal(pidfd, signal.SIGTERM)
        except OSError:
            return {
                **self._base_status(slot, "blocked", "Owned process could not be signaled safely"),
                "ok": False,
            }
        finally:
            os.close(pidfd)
        if not self._wait_for_exit(pid):
            return {
                **self._base_status(slot, "unhealthy", "Owned process did not stop after SIGTERM"),
                "ok": False,
            }
        receipt_path = self._receipt_path(service_id)
        self._require_owned_regular_file(receipt_path, "receipt")
        receipt_path.unlink()
        return {**self._base_status(slot, "offline", "Owned process stopped"), "ok": True}

    def retry(self, service_id: str) -> dict[str, Any]:
        slot = self.slot(service_id)
        receipt = self._read_receipt(service_id)
        if receipt:
            stopped = self.stop(service_id)
            if not stopped.get("ok"):
                return stopped
        elif self._port_open(slot):
            return {
                **self._base_status(slot, "blocked", "Port is occupied by an unrelated process"),
                "ok": False,
            }
        return self.start(service_id)

    def start_required(self, service_ids: Iterable[str]) -> list[dict[str, Any]]:
        ordered: list[str] = []
        for service_id in service_ids:
            self.slot(service_id)
            if service_id not in ordered:
                ordered.append(service_id)
        return [self.start(service_id) for service_id in ordered]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Manage allowlisted WNS VM services")
    parser.add_argument("action", choices=["status", "start", "retry", "stop"])
    parser.add_argument("service_id", nargs="?")
    parser.add_argument("--root", default=os.fspath(Path(__file__).resolve().parents[1]))
    parser.add_argument("--manifest", default="configs/vm_adapter_ports.json")
    args = parser.parse_args()
    canonical_root = Path(__file__).resolve().parents[1]
    root = Path(args.root).absolute()
    if root.is_symlink() or root.resolve() != canonical_root or root != canonical_root:
        parser.error("--root must be the canonical repository root")
    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = root / manifest
    canonical_manifest = canonical_root / "configs" / "vm_adapter_ports.json"
    if manifest.is_symlink() or manifest.absolute() != canonical_manifest:
        parser.error("--manifest must be the canonical VM adapter manifest")
    vm = VMAdapterManager(root, manifest)
    if args.action == "status" and not args.service_id:
        payload: Any = {"slots": vm.statuses()}
    else:
        if not args.service_id:
            parser.error("service_id is required for this action")
        payload = getattr(vm, args.action)(args.service_id)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("ok", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
